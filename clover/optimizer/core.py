"""Convert Logic DAGs into physical plans."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


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
        if logic_dag.get("task_type") == "table_reasoning_v2":
            return self._optimize_table_reasoning_v2(
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
            "nodes": copy.deepcopy(logic_dag.get("nodes", [])),
            "edges": copy.deepcopy(logic_dag.get("edges", [])),
        }
        for strategy in self.strategies:
            strategy.apply(physical_plan, logic_dag, context, local_dsl)
        _validate_physical_plan(physical_plan)
        return physical_plan

    def _optimize_table_reasoning_v2(
        self,
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> dict[str, Any]:
        _validate_task_type(logic_dag, context, local_dsl)
        subtasks = _validate_v2_subtasks(logic_dag)
        answer_names = _validate_v2_answers(subtasks)

        subplans = []
        for subtask in subtasks:
            subplans.append(
                {
                    "subtask": subtask,
                    "physical_plan": self._optimize_single_logic_dag(
                        logic_dag=subtask["logic_dag"],
                        context=_v1_context(context),
                        local_dsl=_v1_local_dsl(local_dsl, subtask),
                    ),
                }
            )

        merged_plan = _merge_v2_physical_plans(
            subplans=subplans,
            answers=[subtask["answer"] for subtask in subtasks],
            answer_names=answer_names,
        )
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
        default_factory=lambda: {"table_reasoning_v1": EmptyInstructionTemplate()}
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


def _validate_v2_subtasks(logic_dag: dict[str, Any]) -> list[dict[str, Any]]:
    subtasks = logic_dag.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise OptimizationError("table_reasoning_v2 Logic DAG requires subtasks")
    for index, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            raise OptimizationError(f"v2 subtask must be an object: {subtask!r}")
        if not isinstance(subtask.get("logic_dag"), dict):
            raise OptimizationError(f"v2 subtask {index} missing logic_dag")
        if subtask["logic_dag"].get("task_type") != "table_reasoning_v1":
            raise OptimizationError(
                f"v2 subtask {index} logic_dag must be table_reasoning_v1"
            )
        if not isinstance(subtask.get("answer"), dict):
            raise OptimizationError(f"v2 subtask {index} missing answer")
    return subtasks


def _validate_v2_answers(subtasks: list[dict[str, Any]]) -> list[str]:
    answer_names: list[str] = []
    for index, subtask in enumerate(subtasks):
        answer_name = subtask["answer"].get("name")
        if not isinstance(answer_name, str) or not answer_name:
            raise OptimizationError(f"v2 subtask {index} answer missing name")
        if answer_name in answer_names:
            raise OptimizationError(f"Duplicate v2 answer name: {answer_name}")
        answer_names.append(answer_name)
    return answer_names


def _v1_context(context: dict[str, Any]) -> dict[str, Any]:
    sub_context = copy.deepcopy(context)
    sub_context["task_type"] = "table_reasoning_v1"
    return sub_context


def _v1_local_dsl(local_dsl: dict[str, Any], subtask: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning_v1",
        "question": subtask.get("question"),
        "sources": copy.deepcopy(local_dsl.get("sources", [])),
        "answer": copy.deepcopy(subtask["answer"]),
    }


def _merge_v2_physical_plans(
    *,
    subplans: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    answer_names: list[str],
) -> dict[str, Any]:
    resources_by_id: dict[str, dict[str, Any]] = {}
    merged_nodes: list[dict[str, Any]] = []
    output_to_node_id: dict[str, str] = {}
    memo: dict[tuple[Any, ...], dict[str, str]] = {}
    output_counter = 0
    reused_nodes = 0
    subtask_outputs: list[dict[str, Any]] = []

    for item in subplans:
        subtask = item["subtask"]
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

            answer_name = str(subtask["answer"]["name"])
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

        subtask_outputs.append(
            {
                "id": subtask.get("id"),
                "index": subtask.get("index"),
                "answer": copy.deepcopy(subtask["answer"]),
                "output": old_to_new_output.get("answer", subtask["answer"]["name"]),
            }
        )

    return {
        "task_type": "table_reasoning_v2",
        "resources": [resources_by_id[key] for key in sorted(resources_by_id)],
        "nodes": merged_nodes,
        "edges": _dependency_edges_from_nodes(merged_nodes, output_to_node_id),
        "answers": copy.deepcopy(answers),
        "subtask_outputs": subtask_outputs,
        "merge_stats": {
            "subplans": len(subplans),
            "answers": len(answer_names),
            "nodes": len(merged_nodes),
            "reused_nodes": reused_nodes,
        },
    }


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
        if not _task_type_compatible(payload.get("task_type"), task_type)
    ]
    if mismatches:
        raise OptimizationError(f"task_type mismatch in: {mismatches}")


def _task_type_compatible(payload_task_type: Any, logic_task_type: str) -> bool:
    if payload_task_type is None or payload_task_type == logic_task_type:
        return True
    return (
        payload_task_type == "table_reasoning"
        and logic_task_type in {"table_reasoning_v1", "table_reasoning_v2"}
    )


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

        resources[source_id] = {
            "id": source_id,
            "type": source.get("type") or mapped_source.get("type"),
            "path": str(path.resolve()),
            "format": source.get("format") or mapped_source.get("format"),
            "schema": copy.deepcopy(source.get("schema", {})),
        }

    return resources


def _referenced_resource_ids(logic_dag: dict[str, Any]) -> set[str]:
    resource_ids: set[str] = set()
    for node in logic_dag.get("nodes", []):
        for source_id in node.get("input", []):
            resource_ids.add(source_id)
    return resource_ids


def _validate_physical_plan(physical_plan: dict[str, Any]) -> None:
    required_fields = {"task_type", "resources", "nodes", "edges"}
    missing = required_fields - set(physical_plan)
    if missing:
        raise OptimizationError(f"Physical plan missing fields: {sorted(missing)}")

    resource_ids = {resource["id"] for resource in physical_plan["resources"]}
    for resource in physical_plan["resources"]:
        if not Path(resource["path"]).is_absolute():
            raise OptimizationError(f"Resource path is not absolute: {resource['id']}")

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
        produced_outputs.add(node.get("output"))
