"""Physical DAG executor."""

from __future__ import annotations

import time
from typing import Any

from clover.executor.agents import build_node_agent, node_agent_class_for_task
from clover.executor.context import (
    ExecutionContext,
    normalize_resources,
)
from clover.executor.errors import ExecutionError, PlanValidationError
from clover.executor.resources import ResourceLimits, ResourceStore
from clover.executor.result import (
    ExecutionResult,
    NodeExecutionRecord,
    error_payload,
)


class Executor:
    """Run physical DAGs by scheduling task-specific NodeAgents."""

    def execute(
        self,
        physical_plan: dict[str, Any],
        *,
        table_cache: dict[str, Any] | None = None,
        external_params: dict[str, Any] | None = None,
        slm_config: dict[str, Any] | None = None,
        slm_client: Any | None = None,
        agent_loop_max_iterations: int = 3,
        resource_memory_budget_bytes: int | None = None,
        resource_spill_threshold_bytes: int | None = None,
        resource_spill_root: str | None = None,
        raise_on_error: bool = False,
    ) -> ExecutionResult:
        started = time.perf_counter()
        traces: list[dict[str, Any]] = []
        context: ExecutionContext | None = None
        try:
            self._validate_plan(physical_plan)
            resource_limits = ResourceLimits(
                **{
                    key: value
                    for key, value in {
                        "memory_budget_bytes": resource_memory_budget_bytes,
                        "spill_threshold_bytes": resource_spill_threshold_bytes,
                        "spill_root": resource_spill_root,
                    }.items()
                    if value is not None
                }
            )
            resource_store = ResourceStore(
                external_resources=normalize_resources(physical_plan.get("resources", [])),
                table_cache=table_cache,
                limits=resource_limits,
            )
            context = ExecutionContext(
                task_type=physical_plan["task_type"],
                resource_store=resource_store,
                external_params=dict(external_params or {}),
                table_cache=table_cache,
                slm_config=slm_config,
                slm_client=slm_client,
                agent_loop_max_iterations=agent_loop_max_iterations,
            )
            result = self._execute_validated_plan(
                physical_plan,
                context,
                traces,
                started,
            )
        except Exception as exc:  # noqa: BLE001 - optionally converted to ExecutionResult.
            if raise_on_error:
                raise
            result = _failed_result_from_exception(
                exc,
                traces=traces,
                elapsed_ms=_elapsed_ms(started),
            )
        finally:
            if context is not None:
                context.resource_store.close_all()

        if raise_on_error and not result.ok:
            raise ExecutionError(result.error["message"] if result.error else "execution failed")
        return result

    def _execute_validated_plan(
        self,
        physical_plan: dict[str, Any],
        context: ExecutionContext,
        traces: list[dict[str, Any]],
        started: float,
    ) -> ExecutionResult:
        nodes = list(physical_plan.get("nodes", []))
        node_indexes = {node["id"]: index for index, node in enumerate(nodes)}
        remaining_consumers = _dependency_consumer_counts(nodes)
        retained_outputs = _retained_output_names(physical_plan, nodes)
        pending = {node["id"]: node for node in nodes}
        while pending:
            # v1/v2 execute one ready node at a time. Worker-level node
            # parallelism is intentionally reserved for the future v3 runtime.
            ready_nodes = sorted(
                (
                    node
                    for node in pending.values()
                    if self._node_ready(node, context)
                ),
                key=lambda item: node_indexes[item["id"]],
            )
            if not ready_nodes:
                unscheduled = sorted(pending)
                raise PlanValidationError(
                    "No schedulable nodes remain; pending nodes have "
                    f"unsatisfied dependencies or resources: {unscheduled}"
                )

            node = ready_nodes[0]
            node_context = context.node_context(node)
            try:
                agent = build_node_agent(node_context)
                record = agent.run()
            finally:
                node_context.resource_view.unpin()
            del pending[node["id"]]
            record.trace["node_index"] = node_indexes[node["id"]]
            traces.append(record.trace)
            if not record.ok:
                # Fail fast keeps downstream nodes from consuming partial or
                # semantically invalid intermediate data.
                return self._failed_result_from_record(
                    record,
                    context,
                    traces,
                    started,
                )
            if not record.output_name:
                raise PlanValidationError(
                    f"Node {record.node_id} completed without an output name"
                )
            context.resource_store.put_output(
                record.output_name,
                record.output,
                producer_node=record.node_id,
                retained=record.output_name in retained_outputs,
            )
            if (
                remaining_consumers.get(record.output_name, 0) <= 0
                and record.output_name not in retained_outputs
            ):
                context.resource_store.release(record.output_name)
            _release_consumed_dependencies(
                context,
                node,
                remaining_consumers=remaining_consumers,
                retained_outputs=retained_outputs,
            )

        outputs = context.resource_store.to_dict()
        sorted_traces = _sort_traces(traces)
        return ExecutionResult(
            ok=True,
            answer=_extract_answer(physical_plan, outputs),
            outputs=outputs,
            traces=sorted_traces,
            output_summaries=context.resource_store.summaries(),
            elapsed_ms=_elapsed_ms(started),
            fast_path_hits=_count_fast_path(sorted_traces, hit=True),
            fast_path_misses=_count_fast_path(sorted_traces, hit=False),
        )

    def _failed_result_from_record(
        self,
        record: NodeExecutionRecord,
        context: ExecutionContext,
        traces: list[dict[str, Any]],
        started: float,
    ) -> ExecutionResult:
        outputs = context.resource_store.to_dict()
        sorted_traces = _sort_traces(traces)
        return ExecutionResult(
            ok=False,
            answer=None,
            outputs=outputs,
            traces=sorted_traces,
            output_summaries=context.resource_store.summaries(),
            failing_node={
                "id": record.node_id,
                "op": record.op,
                "output": record.output_name,
            },
            error=record.error,
            elapsed_ms=_elapsed_ms(started),
            fast_path_hits=_count_fast_path(sorted_traces, hit=True),
            fast_path_misses=_count_fast_path(sorted_traces, hit=False),
        )

    def _validate_plan(self, physical_plan: dict[str, Any]) -> None:
        required_fields = {"task_type", "resources", "nodes", "edges"}
        missing = required_fields - set(physical_plan)
        if missing:
            raise PlanValidationError(f"Physical plan missing fields: {sorted(missing)}")

        task_type = physical_plan.get("task_type")
        node_agent_class_for_task(task_type)

        resources = normalize_resources(physical_plan.get("resources", []))
        nodes = physical_plan.get("nodes", [])
        if not isinstance(nodes, list):
            raise PlanValidationError("Physical plan nodes must be a list")

        seen_node_ids: set[str] = set()
        seen_outputs: set[str] = set()
        declared_outputs: set[str] = set()
        for node in nodes:
            if not isinstance(node, dict):
                raise PlanValidationError(f"Physical node must be an object: {node!r}")
            node_id = node.get("id")
            if not node_id:
                raise PlanValidationError(f"Physical node missing id: {node}")
            if node_id in seen_node_ids:
                raise PlanValidationError(f"Duplicate physical node id: {node_id}")
            seen_node_ids.add(node_id)

            for field_name in ("op", "dependency", "input", "params", "output"):
                if field_name not in node:
                    raise PlanValidationError(
                        f"Physical node {node_id} missing {field_name}"
                    )
            output_name = node.get("output")
            if not output_name:
                raise PlanValidationError(f"Physical node {node_id} missing output")
            if output_name in seen_outputs:
                raise PlanValidationError(f"Duplicate physical node output: {output_name}")
            seen_outputs.add(output_name)
            declared_outputs.add(output_name)

            if not isinstance(node.get("dependency"), list):
                raise PlanValidationError(
                    f"Physical node {node_id} dependency must be a list"
                )
            if not isinstance(node.get("input"), list):
                raise PlanValidationError(f"Physical node {node_id} input must be a list")
            missing_resources = sorted(set(node.get("input", [])) - set(resources))
            if missing_resources:
                raise PlanValidationError(
                    f"Physical node {node_id} has unknown resource inputs: "
                    f"{missing_resources}"
                )

        for node in nodes:
            # Dependencies name intermediate outputs, not producing node ids.
            missing_dependencies = sorted(
                set(node.get("dependency", [])) - declared_outputs
            )
            if missing_dependencies:
                raise PlanValidationError(
                    f"Physical node {node.get('id')} has unknown dependencies: "
                    f"{missing_dependencies}"
                )

    def _node_ready(self, node: dict[str, Any], context: ExecutionContext) -> bool:
        missing_dependencies = context.resource_store.missing(
            list(node.get("dependency", []))
        )
        if missing_dependencies:
            return False
        return all(
            context.resource_store.has_source(resource_id)
            for resource_id in node.get("input", [])
        )


def execute_physical_plan(
    physical_plan: dict[str, Any],
    *,
    table_cache: dict[str, Any] | None = None,
    external_params: dict[str, Any] | None = None,
    slm_config: dict[str, Any] | None = None,
    slm_client: Any | None = None,
    agent_loop_max_iterations: int = 3,
    resource_memory_budget_bytes: int | None = None,
    resource_spill_threshold_bytes: int | None = None,
    resource_spill_root: str | None = None,
    raise_on_error: bool = False,
) -> ExecutionResult:
    """Execute one physical plan with task-specific NodeAgents."""

    executor = Executor()
    return executor.execute(
        physical_plan,
        table_cache=table_cache,
        external_params=external_params,
        slm_config=slm_config,
        slm_client=slm_client,
        agent_loop_max_iterations=agent_loop_max_iterations,
        resource_memory_budget_bytes=resource_memory_budget_bytes,
        resource_spill_threshold_bytes=resource_spill_threshold_bytes,
        resource_spill_root=resource_spill_root,
        raise_on_error=raise_on_error,
    )


def _extract_answer(physical_plan: dict[str, Any], outputs: dict[str, Any]) -> Any:
    if physical_plan.get("task_type") == "table_reasoning_v2":
        answer_map: dict[str, Any] = {}
        for item in physical_plan.get("subtask_outputs", []):
            if not isinstance(item, dict):
                continue
            answer = item.get("answer", {})
            answer_name = answer.get("name") if isinstance(answer, dict) else None
            output_name = item.get("output") or answer_name
            if isinstance(answer_name, str) and output_name in outputs:
                answer_map[answer_name] = outputs[output_name]
        return answer_map

    # The FormatAnswer node writes to `answer`; the fallback keeps older or
    # partially-authored plans debuggable during executor development.
    if "answer" in outputs:
        return outputs["answer"]
    nodes = physical_plan.get("nodes", [])
    if nodes:
        final_output = nodes[-1].get("output")
        if final_output in outputs:
            return outputs[final_output]
    if outputs:
        return next(reversed(outputs.values()))
    return None


def _failed_result_from_exception(
    exc: Exception,
    *,
    traces: list[dict[str, Any]],
    elapsed_ms: float,
) -> ExecutionResult:
    error = error_payload(exc)
    sorted_traces = _sort_traces(traces)
    return ExecutionResult(
        ok=False,
        answer=None,
        outputs={},
        traces=sorted_traces,
        output_summaries={},
        error=error,
        elapsed_ms=elapsed_ms,
        fast_path_hits=_count_fast_path(sorted_traces, hit=True),
        fast_path_misses=_count_fast_path(sorted_traces, hit=False),
    )


def _sort_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        traces,
        key=lambda trace: (
            trace.get("node_index", 10**9),
            str(trace.get("node_id")),
        ),
    )


def _count_fast_path(traces: list[dict[str, Any]], *, hit: bool) -> int:
    return sum(1 for trace in traces if trace.get("fast_path_hit") is hit)


def _dependency_consumer_counts(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        for dependency in node.get("dependency", []):
            counts[dependency] = counts.get(dependency, 0) + 1
    return counts


def _retained_output_names(
    physical_plan: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> set[str]:
    retained = set()
    if nodes and nodes[-1].get("output"):
        retained.add(nodes[-1]["output"])
    for node in nodes:
        if node.get("op") == "FormatAnswer" and node.get("output"):
            retained.add(node["output"])
    for item in physical_plan.get("subtask_outputs", []):
        if not isinstance(item, dict):
            continue
        output_name = item.get("output")
        answer = item.get("answer")
        answer_name = answer.get("name") if isinstance(answer, dict) else None
        if isinstance(output_name, str):
            retained.add(output_name)
        if isinstance(answer_name, str):
            retained.add(answer_name)
    return retained


def _release_consumed_dependencies(
    context: ExecutionContext,
    node: dict[str, Any],
    *,
    remaining_consumers: dict[str, int],
    retained_outputs: set[str],
) -> None:
    for dependency in node.get("dependency", []):
        if dependency not in remaining_consumers:
            continue
        remaining_consumers[dependency] -= 1
        if remaining_consumers[dependency] <= 0 and dependency not in retained_outputs:
            context.resource_store.release(dependency)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
