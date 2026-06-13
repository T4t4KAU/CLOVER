"""Convert Logic DAGs into physical plans."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from clover.optimizer.ir import (
    DOCUMENT_REASONING_TASK_TYPE,
    TABLE_REASONING_QUERY_TASK_TYPE,
)
from clover.task_types import (
    TABLE_REASONING_ANALYZE_TASK_TYPE,
    is_table_task_type,
)
from clover.resource.preprocess.pdf_schema import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_CHUNK_SIZE_CHARS,
)


TABLE_OUTPUT_OPS = frozenset(
    {
        "Scan",
        "Filter",
        "Project",
        "Derive",
        "Aggregate",
        "Group",
        "Sort",
        "Limit",
        "Distinct",
        "Join",
        "SetOp",
        "RepeatUnion",
    }
)


class OptimizationError(ValueError):
    """Raised when a Logic DAG cannot be optimized into a physical plan."""


class OptimizationStrategy(Protocol):
    """One physical-plan optimization or annotation step."""

    def apply(
        self,
        physical_plan: dict[str, Any],
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> None:
        """Mutate physical_plan in place."""


@dataclass
class Optimizer:
    """Composable Logic DAG optimizer.

    Strategies run in order so later passes can build on earlier annotations.
    """

    strategies: list[OptimizationStrategy] = field(default_factory=list)

    @classmethod
    def default(cls) -> "Optimizer":
        return cls(
            strategies=[
                ResourceBindingStrategy(),
                NodeAnnotationStrategy(),
            ]
        )

    def optimize(
        self,
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> dict[str, Any]:
        if _is_document_dag(logic_dag):
            return self._optimize_document_reasoning(
                logic_dag=logic_dag,
                context=context,
                local_dsl=local_dsl,
            )
        if is_table_task_type(logic_dag.get("task_type")):
            if not _is_table_batch_dag(logic_dag):
                raise OptimizationError(
                    "table reasoning Logic DAG requires query_plans"
                )
            return self._optimize_table_reasoning_batch(
                logic_dag=logic_dag,
                context=context,
                local_dsl=local_dsl,
            )
        return self._optimize_single_logic_dag(
            logic_dag=logic_dag,
            context=context,
            local_dsl=local_dsl,
        )

    def _optimize_single_logic_dag(
        self,
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> dict[str, Any]:
        _validate_task_type(logic_dag, context, local_dsl)
        # Start from a structural copy of the Logic DAG; optimization passes add
        # local execution details without changing the logical dependency graph.
        physical_plan = {
            "task_type": logic_dag["task_type"],
            "resources": [],
            "resource_processing": [],
            "nodes": copy.deepcopy(logic_dag.get("nodes", [])),
            "edges": copy.deepcopy(logic_dag.get("edges", [])),
        }
        if isinstance(local_dsl.get("answer"), dict):
            physical_plan["answer"] = copy.deepcopy(local_dsl["answer"])
        if "static_collectors" in logic_dag:
            physical_plan["static_collectors"] = _validate_static_collectors(logic_dag)
        for strategy in self.strategies:
            strategy.apply(physical_plan, logic_dag, context, local_dsl)
        _validate_physical_plan(physical_plan)
        return physical_plan

    def _optimize_document_reasoning(
        self,
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> dict[str, Any]:
        _validate_task_type(logic_dag, context, local_dsl)
        map_groups = _validate_document_map_groups(logic_dag)
        logical_resource_processing = _validate_document_resource_processing(logic_dag)
        resource_views_by_id = {
            step["output"]: step for step in logical_resource_processing
        }
        resources_by_id = _resource_map(context=context, local_dsl=local_dsl)
        referenced_source_ids = _document_referenced_source_ids(
            logical_resource_processing
        )
        missing = sorted(referenced_source_ids - set(resources_by_id))
        if missing:
            raise OptimizationError(
                f"Document Logic DAG references unknown resources: {missing}"
            )

        physical_plan = {
            "task_type": DOCUMENT_REASONING_TASK_TYPE,
            "question": local_dsl.get("question"),
            "resources": [
                resources_by_id[source_id]
                for source_id in sorted(referenced_source_ids)
            ],
            "resource_processing": _document_resource_processing(
                logical_resource_processing=logical_resource_processing,
                resources_by_id=resources_by_id,
            ),
            "map_groups": _document_physical_map_groups(
                map_groups,
                resource_views_by_id=resource_views_by_id,
            ),
            "static_collectors": _validate_static_collectors(logic_dag),
            "edges": copy.deepcopy(logic_dag.get("edges", [])),
        }
        _validate_physical_plan(physical_plan)
        return physical_plan

    def _optimize_table_reasoning_batch(
        self,
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> dict[str, Any]:
        _validate_task_type(logic_dag, context, local_dsl)
        query_plans = _validate_batch_query_plans(logic_dag)
        answer_names = _validate_batch_answers(query_plans)

        physical_query_plans = []
        for query_plan in query_plans:
            physical_query_plans.append(
                {
                    "query_plan": query_plan,
                    "physical_plan": self._optimize_single_logic_dag(
                        logic_dag=_query_fragment_logic_dag(
                            query_plan,
                            task_type=logic_dag["task_type"],
                        ),
                        context=_query_context(context, task_type=logic_dag["task_type"]),
                        local_dsl=_query_local_dsl(
                            local_dsl,
                            query_plan,
                            task_type=logic_dag["task_type"],
                        ),
                    ),
                }
            )

        merged_plan = _merge_batch_physical_plans(
            physical_query_plans=physical_query_plans,
            answers=[query_plan["answer"] for query_plan in query_plans],
            answer_names=answer_names,
            task_type=logic_dag["task_type"],
        )
        for strategy in self.strategies:
            if isinstance(strategy, (ResourceBindingStrategy, NodeAnnotationStrategy)):
                continue
            strategy.apply(merged_plan, logic_dag, context, local_dsl)
        _validate_physical_plan(merged_plan)
        return merged_plan


@dataclass
class ResourceBindingStrategy:
    """Attach local resources required by the physical plan."""

    def apply(
        self,
        physical_plan: dict[str, Any],
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> None:
        resources_by_id = _resource_map(context=context, local_dsl=local_dsl)
        # Only external resource ids belong in node.input. Intermediate table
        # references live in node.dependency and are validated separately.
        referenced_resource_ids = _referenced_resource_ids(logic_dag)
        missing = sorted(referenced_resource_ids - set(resources_by_id))
        if missing:
            raise OptimizationError(f"Logic DAG references unknown resources: {missing}")

        physical_plan["resources"] = [
            resources_by_id[source_id] for source_id in sorted(referenced_resource_ids)
        ]


class InstructionTemplate(Protocol):
    """Task-specific instruction renderer for physical plan nodes."""

    def render(self, node: dict[str, Any], physical_plan: dict[str, Any]) -> str:
        """Return a Local SLM instruction for one physical node."""


class EmptyInstructionTemplate:
    """Placeholder template used until task-specific instructions are designed."""

    def render(self, node: dict[str, Any], physical_plan: dict[str, Any]) -> str:
        return ""


@dataclass
class NodeAnnotationStrategy:
    """Annotate nodes with output type and task-specific instructions."""

    instruction_templates: dict[str, "InstructionTemplate"] = field(
        default_factory=lambda: {
            TABLE_REASONING_QUERY_TASK_TYPE: EmptyInstructionTemplate(),
            TABLE_REASONING_ANALYZE_TASK_TYPE: EmptyInstructionTemplate(),
        }
    )
    default_instruction_template: "InstructionTemplate" = field(
        default_factory=EmptyInstructionTemplate
    )

    def apply(
        self,
        physical_plan: dict[str, Any],
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> None:
        task_type = physical_plan["task_type"]
        template = self.instruction_templates.get(
            task_type,
            self.default_instruction_template,
        )
        for node in physical_plan["nodes"]:
            # output_type is derived here so Logic DAGs stay task/logical only.
            node["output_type"] = infer_output_type(node, local_dsl)
            node["instruction"] = template.render(node=node, physical_plan=physical_plan)


def optimize_logic_dag_to_physical_plan(
    logic_dag: dict[str, Any],
    context: dict[str, Any],
    local_dsl: dict[str, Any],
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Build a physical plan from a Logic DAG and local resource context."""

    selected_optimizer = optimizer or Optimizer.default()
    return selected_optimizer.optimize(
        logic_dag=logic_dag,
        context=context,
        local_dsl=local_dsl,
    )


def infer_output_type(node: dict[str, Any], local_dsl: dict[str, Any]) -> str:
    """Infer node output type from the operation family."""

    op = node.get("op")
    if op in TABLE_OUTPUT_OPS:
        return "table"
    if op == "FormatAnswer":
        answer = node.get("params", {}).get("answer") or local_dsl.get("answer", {})
        return answer.get("type", "json")
    if op == "SLM":
        return "json"
    return "json"


def _is_table_batch_dag(logic_dag: dict[str, Any]) -> bool:
    return is_table_task_type(logic_dag.get("task_type")) and "query_plans" in logic_dag


def _is_document_dag(logic_dag: dict[str, Any]) -> bool:
    return logic_dag.get("task_type") == DOCUMENT_REASONING_TASK_TYPE


def _validate_document_map_groups(logic_dag: dict[str, Any]) -> list[dict[str, Any]]:
    map_groups = logic_dag.get("map_groups")
    if not isinstance(map_groups, list) or not map_groups:
        raise OptimizationError(
            "document_reasoning Logic DAG requires map_groups"
        )
    for index, group in enumerate(map_groups):
        if not isinstance(group, dict):
            raise OptimizationError(f"document map_group must be an object: {group!r}")
        if group.get("op") != "map":
            raise OptimizationError(f"document map_group {index} op must be map")
        group_id = group.get("id")
        if not isinstance(group_id, str) or not group_id:
            raise OptimizationError(f"document map_group {index} missing id")
        inputs = group.get("inputs")
        if not isinstance(inputs, dict):
            raise OptimizationError(f"document map_group {index} missing inputs")
        resource_view_id = inputs.get("resource_view")
        if not isinstance(resource_view_id, str) or not resource_view_id:
            raise OptimizationError(f"document map_group {index} missing resource_view")
        chunks = inputs.get("chunks")
        if chunks != "all" and not (
            isinstance(chunks, list)
            and chunks
            and all(isinstance(chunk, str) and chunk for chunk in chunks)
        ):
            raise OptimizationError(
                f"document map_group {index} chunks must be all or a non-empty list"
            )
        params = group.get("params")
        if not isinstance(params, dict):
            raise OptimizationError(f"document map_group {index} missing params")
        instruction = params.get("local_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise OptimizationError(
                f"document map_group {index} missing local_instruction"
            )
        replicas = group.get("replicas", 1)
        if not isinstance(replicas, int) or replicas < 1:
            raise OptimizationError(f"document map_group {index} replicas must be positive")
    return map_groups


def _validate_document_resource_processing(
    logic_dag: dict[str, Any],
) -> list[dict[str, Any]]:
    steps = logic_dag.get("resource_processing")
    if not isinstance(steps, list) or not steps:
        raise OptimizationError(
            "document_reasoning Logic DAG requires resource_processing"
        )
    seen: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise OptimizationError(
                f"document resource_processing step must be an object: {step!r}"
            )
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            raise OptimizationError(
                f"document resource_processing step {index} missing id"
            )
        output = step.get("output")
        if not isinstance(output, str) or not output:
            raise OptimizationError(
                f"document resource_processing step {step_id} missing output"
            )
        if output in seen:
            raise OptimizationError(f"duplicate document resource view output: {output}")
        seen.add(output)
        source_id = step.get("source")
        if not isinstance(source_id, str) or not source_id:
            raise OptimizationError(
                f"document resource_processing step {step_id} missing source"
            )
        op = step.get("op")
        if op not in {"chunk_by_section", "chunk_by_page"}:
            raise OptimizationError(f"unsupported document resource_processing op: {op}")
        params = step.get("params")
        if not isinstance(params, dict):
            raise OptimizationError(
                f"document resource_processing step {step_id} missing params"
            )

    for index, group in enumerate(logic_dag.get("map_groups", [])):
        resource_view_id = group.get("inputs", {}).get("resource_view")
        if resource_view_id not in seen:
            raise OptimizationError(
                f"document map_group {index} references unknown resource_view"
            )
    return steps


def _validate_static_collectors(logic_dag: dict[str, Any]) -> list[dict[str, Any]]:
    collectors = logic_dag.get("static_collectors", [])
    if collectors is None:
        return []
    if not isinstance(collectors, list):
        raise OptimizationError("Logic DAG static_collectors must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, collector in enumerate(collectors):
        if not isinstance(collector, dict):
            raise OptimizationError(f"static_collector must be an object: {collector!r}")
        collector_id = collector.get("id")
        if not isinstance(collector_id, str) or not collector_id:
            raise OptimizationError(f"static_collector {index} missing id")
        if collector_id in seen:
            raise OptimizationError(f"duplicate static_collector id: {collector_id}")
        seen.add(collector_id)
        kind = collector.get("kind")
        if not isinstance(kind, str) or not kind:
            raise OptimizationError(f"static_collector {collector_id} missing kind")
        output = collector.get("output", collector_id)
        if not isinstance(output, str) or not output:
            raise OptimizationError(f"static_collector {collector_id} missing output")
        normalized_collector = copy.deepcopy(collector)
        normalized_collector["output"] = output
        normalized.append(normalized_collector)
    return normalized


def _document_referenced_source_ids(steps: list[dict[str, Any]]) -> set[str]:
    return {step["source"] for step in steps}


def _document_resource_processing(
    *,
    logical_resource_processing: list[dict[str, Any]],
    resources_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    text_outputs: dict[str, str] = {}
    for source_id in sorted(_document_referenced_source_ids(logical_resource_processing)):
        resource = resources_by_id[source_id]
        if resource.get("type") != "document" or resource.get("source_type") != "pdf":
            raise OptimizationError(
                "document_reasoning requires PDF document resources: "
                f"{source_id}"
            )
        schema = resource.get("schema", {})
        text_extraction = schema.get("text_extraction", {})
        text_output = f"{source_id}.text"
        text_outputs[source_id] = text_output
        steps.append(
            {
                "id": f"RP{len(steps)}",
                "op": "extract_text",
                "source": source_id,
                "output": text_output,
                "params": {
                    "extractor": text_extraction.get("extractor", "pymupdf"),
                    "page_indexing": schema.get("page_indexing", "zero_based"),
                },
            }
        )

    for view in logical_resource_processing:
        steps.append(
            {
                "id": f"RP{len(steps)}",
                "op": "chunk_text",
                "source": text_outputs[view["source"]],
                "output": view["output"],
                "params": _chunk_text_params_from_view(view),
            }
        )
    return steps


def _document_physical_map_groups(
    map_groups: list[dict[str, Any]],
    *,
    resource_views_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    physical_groups: list[dict[str, Any]] = []
    for group in map_groups:
        resource_view_id = group["inputs"]["resource_view"]
        if resource_view_id not in resource_views_by_id:
            raise OptimizationError(
                f"Document map_group references unknown resource_view: {resource_view_id}"
            )
        replicas = int(group.get("replicas", 1))
        physical_group = {
            "id": group["id"],
            "op": "map",
            "input": {
                "resource_view": resource_view_id,
                "chunks": copy.deepcopy(group["inputs"]["chunks"]),
            },
            "params": copy.deepcopy(group.get("params", {})),
            "output": group.get("output", group["id"]),
            "output_type": "jsonl",
        }
        if replicas > 1:
            physical_group["replicas"] = replicas
        physical_groups.append(physical_group)
    return physical_groups


def _chunk_text_params_from_view(view: dict[str, Any]) -> dict[str, Any]:
    op = view.get("op")
    params = view.get("params", {})
    if op == "chunk_by_section":
        return {
            "strategy": "sliding_window",
            "unit": "char",
            "size": int(
                params.get("max_chunk_size", DEFAULT_CHUNK_SIZE_CHARS)
                or DEFAULT_CHUNK_SIZE_CHARS
            ),
            "overlap": int(params.get("overlap", DEFAULT_CHUNK_OVERLAP_CHARS) or 0),
            "preserve_page_spans": True,
        }
    if op == "chunk_by_page":
        return {
            "strategy": "page",
            "unit": "page",
            "size": 1,
            "overlap": 0,
            "preserve_page_spans": True,
        }
    raise OptimizationError(f"Unsupported document resource_processing op: {op}")


def _validate_batch_query_plans(logic_dag: dict[str, Any]) -> list[dict[str, Any]]:
    query_plans = logic_dag.get("query_plans")
    if not isinstance(query_plans, list) or not query_plans:
        raise OptimizationError("table_reasoning.query batch Logic DAG requires query_plans")
    for index, query_plan in enumerate(query_plans):
        if not isinstance(query_plan, dict):
            raise OptimizationError(f"batch query_plan must be an object: {query_plan!r}")
        if "logic_dag" in query_plan:
            raise OptimizationError(
                f"batch query_plan {index} must use nodes/edges, not nested logic_dag"
            )
        if "task_type" in query_plan:
            raise OptimizationError(f"batch query_plan {index} must not repeat task_type")
        if not isinstance(query_plan.get("nodes"), list) or not query_plan["nodes"]:
            raise OptimizationError(f"batch query_plan {index} missing nodes")
        if not isinstance(query_plan.get("edges", []), list):
            raise OptimizationError(f"batch query_plan {index} edges must be a list")
        if not isinstance(query_plan.get("answer"), dict):
            raise OptimizationError(f"batch query_plan {index} missing answer")
    return query_plans


def _validate_batch_answers(query_plans: list[dict[str, Any]]) -> list[str]:
    answer_names: list[str] = []
    for index, query_plan in enumerate(query_plans):
        answer_name = query_plan["answer"].get("name")
        if not isinstance(answer_name, str) or not answer_name:
            raise OptimizationError(f"batch query_plan {index} answer missing name")
        if answer_name in answer_names:
            raise OptimizationError(f"Duplicate batch answer name: {answer_name}")
        answer_names.append(answer_name)
    return answer_names


def _query_fragment_logic_dag(query_plan: dict[str, Any], *, task_type: str) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "resource_processing": [],
        "nodes": copy.deepcopy(query_plan["nodes"]),
        "edges": copy.deepcopy(query_plan.get("edges", [])),
    }


def _query_context(context: dict[str, Any], *, task_type: str) -> dict[str, Any]:
    sub_context = copy.deepcopy(context)
    sub_context["task_type"] = task_type
    return sub_context


def _query_local_dsl(
    local_dsl: dict[str, Any],
    query_plan: dict[str, Any],
    *,
    task_type: str,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "question": query_plan.get("question"),
        "sources": copy.deepcopy(local_dsl.get("sources", [])),
        "answer": copy.deepcopy(query_plan["answer"]),
    }


def _merge_batch_physical_plans(
    *,
    physical_query_plans: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    answer_names: list[str],
    task_type: str,
) -> dict[str, Any]:
    resources_by_id: dict[str, dict[str, Any]] = {}
    merged_nodes: list[dict[str, Any]] = []
    output_to_node_id: dict[str, str] = {}
    memo: dict[tuple[Any, ...], dict[str, str]] = {}
    output_counter = 0
    reused_nodes = 0
    query_outputs: list[dict[str, Any]] = []

    for item in physical_query_plans:
        query_plan = item["query_plan"]
        physical_plan = item["physical_plan"]
        for resource in physical_plan.get("resources", []):
            existing = resources_by_id.get(resource["id"])
            if existing is not None and existing != resource:
                raise OptimizationError(f"Conflicting resource binding: {resource['id']}")
            resources_by_id[resource["id"]] = copy.deepcopy(resource)

        old_to_new_output: dict[str, str] = {}
        for node in physical_plan.get("nodes", []):
            mapped_dependencies = [
                old_to_new_output[dependency]
                for dependency in node.get("dependency", [])
            ]
            key = _node_equivalence_key(node, mapped_dependencies)
            if _node_can_reuse(node) and key in memo:
                old_to_new_output[node["output"]] = memo[key]["output"]
                reused_nodes += 1
                continue

            answer_name = str(query_plan["answer"]["name"])
            output_name = (
                answer_name
                if node.get("op") == "FormatAnswer"
                else f"T{output_counter}"
            )
            if node.get("op") != "FormatAnswer":
                output_counter += 1

            merged_node = copy.deepcopy(node)
            merged_node["id"] = f"N{len(merged_nodes)}"
            merged_node["dependency"] = mapped_dependencies
            merged_node["output"] = output_name
            if output_name in output_to_node_id:
                raise OptimizationError(f"Duplicate merged output: {output_name}")

            merged_nodes.append(merged_node)
            output_to_node_id[output_name] = merged_node["id"]
            old_to_new_output[node["output"]] = output_name
            if _node_can_reuse(node):
                memo[key] = {"id": merged_node["id"], "output": output_name}

        query_outputs.append(
            {
                "id": query_plan.get("id"),
                "index": query_plan.get("index"),
                "answer": copy.deepcopy(query_plan["answer"]),
                "output": old_to_new_output.get("answer", query_plan["answer"]["name"]),
            }
        )

    merged_plan = {
        "task_type": task_type,
        "resources": [resources_by_id[key] for key in sorted(resources_by_id)],
        "resource_processing": [],
        "nodes": merged_nodes,
        "edges": _dependency_edges_from_nodes(merged_nodes, output_to_node_id),
        "answers": copy.deepcopy(answers),
        "query_outputs": query_outputs,
        "merge_stats": {
            "query_plans": len(physical_query_plans),
            "answers": len(answer_names),
            "nodes": len(merged_nodes),
            "reused_nodes": reused_nodes,
        },
    }
    return merged_plan


def _node_can_reuse(node: dict[str, Any]) -> bool:
    return node.get("op") != "FormatAnswer"


def _node_equivalence_key(
    node: dict[str, Any],
    mapped_dependencies: list[str],
) -> tuple[Any, ...]:
    return (
        node.get("op"),
        tuple(node.get("input", [])),
        tuple(mapped_dependencies),
        _canonical_payload(node.get("params", {})),
    )


def _canonical_payload(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dependency_edges_from_nodes(
    nodes: list[dict[str, Any]],
    output_to_node_id: dict[str, str],
) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in nodes:
        for dependency in node.get("dependency", []):
            from_node = output_to_node_id.get(dependency)
            if from_node is None:
                raise OptimizationError(
                    f"Merged node {node.get('id')} has unknown dependency {dependency}"
                )
            edge = (from_node, node["id"])
            if edge not in seen:
                seen.add(edge)
                edges.append({"from": from_node, "to": node["id"]})
    return edges


def _validate_task_type(
    logic_dag: dict[str, Any],
    context: dict[str, Any],
    local_dsl: dict[str, Any],
) -> None:
    task_type = logic_dag.get("task_type")
    if not task_type:
        raise OptimizationError("Logic DAG missing task_type")
    mismatches = [
        name
        for name, payload in (("context", context), ("local_dsl", local_dsl))
        if payload.get("task_type") != task_type
    ]
    if mismatches:
        raise OptimizationError(f"task_type mismatch in: {mismatches}")


def _resource_map(
    context: dict[str, Any],
    local_dsl: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source_map = context.get("source_map", {})
    base_dir = Path(context.get("base_dir", ".")).expanduser()
    resources: dict[str, dict[str, Any]] = {}

    for source in local_dsl.get("sources", []):
        source_id = source.get("id")
        if not source_id:
            raise OptimizationError(f"Local source missing id: {source}")
        mapped_source = source_map.get(source_id, {})
        path_value = source.get("path") or mapped_source.get("path")
        if path_value is None:
            raise OptimizationError(f"Local source missing path: {source_id}")
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = base_dir / path

        resource = {
            "id": source_id,
            "type": source.get("type") or mapped_source.get("type"),
            "path": str(path.resolve()),
            "format": source.get("format") or mapped_source.get("format"),
            "schema": copy.deepcopy(source.get("schema", {})),
        }
        source_type = source.get("source_type") or mapped_source.get("source_type")
        if source_type is not None:
            resource["source_type"] = source_type
        for key, value in source.items():
            if key in {
                "file",
                "format",
                "id",
                "input",
                "original_id",
                "path",
                "schema",
                "source_type",
                "type",
            }:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                resource[key] = copy.deepcopy(value)
        resources[source_id] = resource

    return resources


def _referenced_resource_ids(logic_dag: dict[str, Any]) -> set[str]:
    resource_ids: set[str] = set()
    for node in logic_dag.get("nodes", []):
        for source_id in node.get("input", []):
            resource_ids.add(source_id)
    return resource_ids


def _validate_physical_plan(physical_plan: dict[str, Any]) -> None:
    required_fields = {"task_type", "resources", "resource_processing", "edges"}
    if physical_plan.get("task_type") == DOCUMENT_REASONING_TASK_TYPE:
        required_fields.add("map_groups")
    else:
        required_fields.add("nodes")
    missing = required_fields - set(physical_plan)
    if missing:
        raise OptimizationError(f"Physical plan missing fields: {sorted(missing)}")

    resource_ids = {resource["id"] for resource in physical_plan["resources"]}
    for resource in physical_plan["resources"]:
        if not Path(resource["path"]).is_absolute():
            raise OptimizationError(f"Resource path is not absolute: {resource['id']}")

    if physical_plan.get("task_type") == DOCUMENT_REASONING_TASK_TYPE:
        _validate_document_physical_plan(physical_plan, resource_ids)
        return

    produced_outputs: set[str] = set()
    for node in physical_plan["nodes"]:
        node_id = node.get("id")
        for field_name in ("output_type", "instruction"):
            if field_name not in node:
                raise OptimizationError(f"Physical node {node_id} missing {field_name}")
        unknown_inputs = sorted(set(node.get("input", [])) - resource_ids)
        if unknown_inputs:
            raise OptimizationError(
                f"Physical node {node_id} has unknown resource inputs: "
                f"{unknown_inputs}"
            )
        invalid_dependencies = [
            dependency
            for dependency in node.get("dependency", [])
            if dependency not in produced_outputs
        ]
        if invalid_dependencies:
            raise OptimizationError(
                f"Physical node {node_id} has invalid dependencies: "
                f"{invalid_dependencies}"
            )
        output = node.get("output")
        if not isinstance(output, str) or not output:
            raise OptimizationError(f"Physical node {node_id} missing output")
        produced_outputs.add(output)


def _validate_document_physical_plan(
    physical_plan: dict[str, Any],
    resource_ids: set[str],
) -> None:
    view_outputs = set()
    for step in physical_plan["resource_processing"]:
        if not isinstance(step, dict):
            raise OptimizationError(f"Resource processing step must be an object: {step!r}")
        step_id = step.get("id")
        op = step.get("op")
        source = step.get("source")
        output = step.get("output")
        if not all(isinstance(value, str) and value for value in (step_id, op, source, output)):
            raise OptimizationError(f"Invalid resource processing step: {step}")
        if op == "extract_text":
            if source not in resource_ids:
                raise OptimizationError(
                    f"Resource processing step {step_id} has unknown source: {source}"
                )
        elif op == "chunk_text":
            if source not in view_outputs:
                raise OptimizationError(
                    f"Resource processing step {step_id} has unknown view source: {source}"
                )
        else:
            raise OptimizationError(f"Unsupported resource processing op: {op}")
        view_outputs.add(output)

    for group in physical_plan["map_groups"]:
        group_id = group.get("id")
        if group.get("op") != "map":
            raise OptimizationError(f"Document physical group {group_id} op must be map")
        group_input = group.get("input")
        if not isinstance(group_input, dict):
            raise OptimizationError(f"Document physical group {group_id} missing input")
        resource_view = group_input.get("resource_view")
        if not isinstance(resource_view, str) or resource_view not in view_outputs:
            raise OptimizationError(
                f"Document physical group {group_id} has unknown resource_view"
            )
        chunks = group_input.get("chunks")
        if chunks != "all" and not (
            isinstance(chunks, list)
            and all(isinstance(chunk, str) and chunk for chunk in chunks)
        ):
            raise OptimizationError(
                f"Document physical group {group_id} chunks must be all or a list"
            )
        if "output_type" not in group:
            raise OptimizationError(
                f"Document physical group {group_id} missing output_type"
            )
