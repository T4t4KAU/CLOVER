"""Execution-plan executor."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
import time
from typing import Any

from clover.executor.agents import build_node_agent, node_agent_class_for_task
from clover.executor.collectors import run_collectors
from clover.executor.context import ExecutionContext, normalize_resources
from clover.executor.errors import ExecutionError, NodeTimeoutError, PlanValidationError
from clover.executor.policies import NodeFailurePolicy, node_failure_policy_for_plan
from clover.executor.resources import ResourceLimits, ResourceStore
from clover.executor.result import (
    ExecutionResult,
    NodeExecutionRecord,
    error_payload,
)
from clover.executor.scheduler import ExecutionPlan, ExecutionUnit, Scheduler
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    LocalSlmSequenceDispatcher,
)
from clover.tools.table_reasoning import TABLE_REASONING_STATIC_TOOLS


class Executor:
    """Run executor-ready plans by scheduling task-specific NodeAgents."""

    def execute_execution_plan(
        self,
        execution_plan: ExecutionPlan,
        *,
        collector_context: dict[str, Any] | None = None,
        table_cache: dict[str, Any] | None = None,
        external_params: dict[str, Any] | None = None,
        slm_config: dict[str, Any] | None = None,
        slm_client: Any | None = None,
        slm_dispatcher: LocalSlmSequenceDispatcher | None = None,
        agent_loop_max_iterations: int = 3,
        resource_memory_budget_bytes: int | None = None,
        resource_spill_threshold_bytes: int | None = None,
        resource_spill_root: str | None = None,
        max_parallel_execution_units: int = 1,
        max_parallel_slm_node_jobs: int = DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
        max_parallel_slm_sequences: int = DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
        max_pending_slm_sequences: int = DEFAULT_MAX_PENDING_SLM_SEQUENCES,
        node_timeout_seconds: float | None = None,
        raise_on_error: bool = False,
    ) -> ExecutionResult:
        """Execute an executor-ready plan that may contain mixed task units."""

        started = time.perf_counter()
        traces: list[dict[str, Any]] = []
        context: ExecutionContext | None = None
        owned_slm_dispatcher: LocalSlmSequenceDispatcher | None = None
        result: ExecutionResult | None = None
        plan_context = collector_context or {
            "task_type": execution_plan.metadata.get("task_type") or "mixed",
            "resources": list(execution_plan.resources),
            "edges": [],
        }
        try:
            self._validate_execution_plan(execution_plan)
            _validate_positive_limit(
                max_parallel_execution_units,
                name="max_parallel_execution_units",
            )
            _validate_positive_limit(
                max_parallel_slm_node_jobs,
                name="max_parallel_slm_node_jobs",
            )
            _validate_positive_limit(
                max_parallel_slm_sequences,
                name="max_parallel_slm_sequences",
            )
            _validate_positive_limit(
                max_pending_slm_sequences,
                name="max_pending_slm_sequences",
            )
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
                external_resources=normalize_resources(list(execution_plan.resources)),
                table_cache=table_cache,
                limits=resource_limits,
            )
            selected_slm_dispatcher = slm_dispatcher
            if selected_slm_dispatcher is None:
                owned_slm_dispatcher = LocalSlmSequenceDispatcher(
                    slm_config=slm_config,
                    client=slm_client,
                    max_parallel_sequences=max_parallel_slm_sequences,
                    max_pending_sequences=max_pending_slm_sequences,
                )
                selected_slm_dispatcher = owned_slm_dispatcher
            context = ExecutionContext(
                task_type=str(plan_context.get("task_type") or "mixed"),
                resource_store=resource_store,
                external_params=dict(external_params or {}),
                table_cache=table_cache,
                slm_config=slm_config,
                slm_client=slm_client,
                slm_dispatcher=selected_slm_dispatcher,
                slm_node_job_slots=threading.BoundedSemaphore(
                    max_parallel_slm_node_jobs
                ),
                agent_loop_max_iterations=agent_loop_max_iterations,
                node_timeout_seconds=_resolve_node_timeout_seconds(
                    node_timeout_seconds,
                    slm_config=slm_config,
                ),
            )
            result = self._execute_validated_plan(
                execution_plan,
                plan_context,
                context,
                traces,
                started,
                max_parallel_execution_units=max_parallel_execution_units,
                max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                max_parallel_slm_sequences=max_parallel_slm_sequences,
                max_pending_slm_sequences=max_pending_slm_sequences,
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
            if owned_slm_dispatcher is not None:
                owned_slm_dispatcher.close(
                    wait=result is None or not _result_has_timeout_trace(result)
                )

        if raise_on_error and not result.ok:
            raise ExecutionError(result.error["message"] if result.error else "execution failed")
        return result

    def _execute_validated_plan(
        self,
        execution_plan: ExecutionPlan,
        collector_context: dict[str, Any],
        context: ExecutionContext,
        traces: list[dict[str, Any]],
        started: float,
        *,
        max_parallel_execution_units: int,
        max_parallel_slm_node_jobs: int,
        max_parallel_slm_sequences: int,
        max_pending_slm_sequences: int,
    ) -> ExecutionResult:
        _validate_positive_limit(
            max_parallel_execution_units,
            name="max_parallel_execution_units",
        )
        _validate_positive_limit(
            max_parallel_slm_node_jobs,
            name="max_parallel_slm_node_jobs",
        )
        _validate_positive_limit(
            max_parallel_slm_sequences,
            name="max_parallel_slm_sequences",
        )
        _validate_positive_limit(
            max_pending_slm_sequences,
            name="max_pending_slm_sequences",
        )
        scheduler = Scheduler.from_execution_plan(
            execution_plan,
            resource_state=context.resource_store,
        )
        failure_policy = node_failure_policy_for_plan(collector_context)
        retained_artifacts = scheduler.retained_artifacts
        # Configure artifact lifecycle before execution starts. The scheduler
        # knows dependency fan-out, while collectors declare which artifacts
        # must survive until final observation construction.
        context.resource_store.configure_artifact_lifecycle(
            consumers_by_artifact=scheduler.consumers_by_artifact,
            retained_artifacts=retained_artifacts,
        )
        while not scheduler.done:
            units = scheduler.next_ready_batch(max_units=max_parallel_execution_units)
            if not units:
                unscheduled = scheduler.blocked_unit_ids()
                raise PlanValidationError(
                    "No schedulable nodes remain; pending nodes have "
                    f"unsatisfied dependencies or resources: {unscheduled}"
                )

            records = self._run_execution_batch(
                units,
                context,
                max_parallel_execution_units=max_parallel_execution_units,
                failure_policy=failure_policy,
            )
            # Centralize commits so dependency release, trace order, failure
            # policy, and artifact cleanup stay consistent across static-tool
            # nodes and Local SLM agent-loop nodes.
            failed_record = self._commit_execution_batch(
                records,
                context=context,
                scheduler=scheduler,
                traces=traces,
                retained_artifacts=retained_artifacts,
                failure_policy=failure_policy,
            )
            if failed_record is not None:
                return self._failed_result_from_record(
                    failed_record,
                    context,
                    traces,
                    started,
                )

        outputs = context.resource_store.to_dict()
        sorted_traces = _sort_traces(traces)
        try:
            collector_outputs = run_collectors(
                scheduler.collectors,
                resource_store=context.resource_store,
                physical_plan=collector_context,
            )
        except Exception as exc:  # noqa: BLE001 - normalize collector failures.
            return ExecutionResult(
                ok=False,
                answer=None,
                outputs=outputs,
                collector_outputs={},
                traces=sorted_traces,
                output_summaries=context.resource_store.summaries(),
                error=error_payload(exc),
                elapsed_ms=_elapsed_ms(started),
                fast_path_hits=_count_fast_path(sorted_traces, hit=True),
                fast_path_misses=_count_fast_path(sorted_traces, hit=False),
            )
        return ExecutionResult(
            ok=True,
            answer=_collected_answer(
                collector_outputs,
                collector_context=collector_context,
            ),
            outputs=outputs,
            collector_outputs=collector_outputs,
            traces=sorted_traces,
            output_summaries=context.resource_store.summaries(),
            elapsed_ms=_elapsed_ms(started),
            fast_path_hits=_count_fast_path(sorted_traces, hit=True),
            fast_path_misses=_count_fast_path(sorted_traces, hit=False),
        )

    def _run_execution_batch(
        self,
        units: list[ExecutionUnit],
        context: ExecutionContext,
        *,
        max_parallel_execution_units: int,
        failure_policy: NodeFailurePolicy,
    ) -> list[tuple[ExecutionUnit, NodeExecutionRecord]]:
        if context.node_timeout_seconds is None and (
            len(units) == 1 or max_parallel_execution_units <= 1
        ):
            return [(unit, self._run_one_unit(unit, context)) for unit in units]

        # Prioritize Fast Path (deterministic) units so they complete
        # immediately and release thread-pool slots for SLM-heavy units.
        ordered = sorted(
            units,
            key=lambda u: (0 if u.op in TABLE_REASONING_STATIC_TOOLS else 1, u.index),
        )

        max_workers = min(max_parallel_execution_units, len(units))
        records: list[tuple[ExecutionUnit, NodeExecutionRecord]] = []
        pool = ThreadPoolExecutor(max_workers=max_workers)
        pending: set[Future[NodeExecutionRecord]] = set()
        fail_fast = False
        abandon_timed_out = False
        try:
            futures = {
                pool.submit(self._run_one_unit, unit, context): unit for unit in ordered
            }
            pending = set(futures)
            deadlines = _future_deadlines(
                pending,
                timeout_seconds=context.node_timeout_seconds,
            )
            while pending and not fail_fast:
                done, pending = wait(
                    pending,
                    timeout=_next_wait_timeout(pending, deadlines),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    timed_out = _timed_out_futures(pending, deadlines)
                    if not timed_out:
                        continue
                    for future in timed_out:
                        unit = futures[future]
                        future.cancel()
                        record = _timeout_record_for_unit(
                            unit,
                            context.node_timeout_seconds,
                            elapsed_ms=_elapsed_ms_from_deadline(
                                deadlines.get(future),
                                context.node_timeout_seconds,
                            ),
                        )
                        records.append((unit, record))
                        pending.remove(future)
                        abandon_timed_out = True
                        if not failure_policy.should_soft_fail(unit, record):
                            fail_fast = True
                    continue

                for future in done:
                    unit = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - normalize worker failures.
                        record = _failed_record_for_unit(unit, exc)
                    records.append((unit, record))
                    if (
                        not record.ok
                        and not failure_policy.should_soft_fail(unit, record)
                    ):
                        fail_fast = True

            if fail_fast:
                for future in pending:
                    future.cancel()
        finally:
            pool.shutdown(wait=not abandon_timed_out, cancel_futures=True)
        return sorted(records, key=lambda item: item[0].index)

    def _run_one_unit(
        self,
        unit: ExecutionUnit,
        context: ExecutionContext,
    ) -> NodeExecutionRecord:
        node_context = None
        started = time.perf_counter()
        try:
            node_context = context.node_context(unit)
            agent = build_node_agent(node_context)
            record = agent.run()
            timeout_seconds = context.node_timeout_seconds
            elapsed_ms = _elapsed_ms(started)
            if (
                timeout_seconds is not None
                and elapsed_ms > timeout_seconds * 1000.0
            ):
                return _timeout_record_for_unit(
                    unit,
                    timeout_seconds,
                    elapsed_ms=elapsed_ms,
                )
            return record
        except Exception as exc:  # noqa: BLE001 - executor setup failures are node failures.
            return _failed_record_for_unit(unit, exc)
        finally:
            if node_context is not None:
                node_context.resource_view.unpin()

    def _commit_execution_batch(
        self,
        records: list[tuple[ExecutionUnit, NodeExecutionRecord]],
        *,
        context: ExecutionContext,
        scheduler: Scheduler,
        traces: list[dict[str, Any]],
        retained_artifacts: set[str],
        failure_policy: NodeFailurePolicy,
    ) -> NodeExecutionRecord | None:
        first_failure: NodeExecutionRecord | None = None
        for unit, record in records:
            record.trace["node_index"] = unit.index
            traces.append(record.trace)
            if not record.ok:
                if failure_policy.should_soft_fail(unit, record):
                    record.trace["soft_failure"] = True
                    context.resource_store.put_artifact(
                        unit.output,
                        failure_policy.soft_failure_output(unit, record),
                        producer_node=record.node_id or unit.id,
                        retained=unit.output in retained_artifacts,
                    )
                    context.resource_store.mark_dependencies_consumed(
                        unit.dependencies
                    )
                    scheduler.mark_succeeded(unit.id)
                    continue
                scheduler.mark_failed(unit.id, record.error)
                if first_failure is None:
                    first_failure = record
                continue
            if not record.output_name:
                raise PlanValidationError(
                    f"Node {record.node_id} completed without an output name"
                )
            if record.output_name != unit.output:
                raise PlanValidationError(
                    f"Node {record.node_id} produced {record.output_name}, "
                    f"expected {unit.output}"
                )
            context.resource_store.put_artifact(
                record.output_name,
                record.output,
                producer_node=record.node_id,
                retained=record.output_name in retained_artifacts,
            )
            context.resource_store.mark_dependencies_consumed(unit.dependencies)
            scheduler.mark_succeeded(unit.id)
        return first_failure

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

    def _validate_execution_plan(self, execution_plan: ExecutionPlan) -> None:
        if not isinstance(execution_plan, ExecutionPlan):
            raise PlanValidationError("Executor requires an ExecutionPlan")
        seen_resource_ids: set[str] = set()
        for resource in execution_plan.resources:
            resource_id = resource.get("id") if isinstance(resource, dict) else None
            if not isinstance(resource_id, str) or not resource_id:
                raise PlanValidationError(f"Execution resource missing id: {resource}")
            if resource_id in seen_resource_ids:
                raise PlanValidationError(
                    f"Duplicate execution resource id: {resource_id}"
                )
            seen_resource_ids.add(resource_id)
        seen_unit_ids: set[str] = set()
        seen_outputs: set[str] = set()
        for unit in execution_plan.units:
            if unit.id in seen_unit_ids:
                raise PlanValidationError(f"Duplicate execution unit id: {unit.id}")
            seen_unit_ids.add(unit.id)
            if unit.output:
                if unit.output in seen_outputs:
                    raise PlanValidationError(
                        f"Duplicate execution unit output: {unit.output}"
                    )
                seen_outputs.add(unit.output)
            node_agent_class_for_task(unit.task_type)
        for unit in execution_plan.units:
            missing_dependencies = sorted(set(unit.dependencies) - seen_outputs)
            if missing_dependencies:
                raise PlanValidationError(
                    f"Execution unit {unit.id} has unknown dependencies: "
                    f"{missing_dependencies}"
                )


def _validate_positive_limit(value: int, *, name: str) -> None:
    if value <= 0:
        raise PlanValidationError(f"{name} must be positive")


def execute_execution_plan(
    execution_plan: ExecutionPlan,
    *,
    collector_context: dict[str, Any] | None = None,
    table_cache: dict[str, Any] | None = None,
    external_params: dict[str, Any] | None = None,
    slm_config: dict[str, Any] | None = None,
    slm_client: Any | None = None,
    slm_dispatcher: LocalSlmSequenceDispatcher | None = None,
    agent_loop_max_iterations: int = 3,
    resource_memory_budget_bytes: int | None = None,
    resource_spill_threshold_bytes: int | None = None,
    resource_spill_root: str | None = None,
    max_parallel_execution_units: int = 1,
    max_parallel_slm_node_jobs: int = DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    max_parallel_slm_sequences: int = DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    max_pending_slm_sequences: int = DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    node_timeout_seconds: float | None = None,
    raise_on_error: bool = False,
) -> ExecutionResult:
    """Execute an executor-ready plan, including mixed-task unit plans."""

    executor = Executor()
    return executor.execute_execution_plan(
        execution_plan,
        collector_context=collector_context,
        table_cache=table_cache,
        external_params=external_params,
        slm_config=slm_config,
        slm_client=slm_client,
        slm_dispatcher=slm_dispatcher,
        agent_loop_max_iterations=agent_loop_max_iterations,
        resource_memory_budget_bytes=resource_memory_budget_bytes,
        resource_spill_threshold_bytes=resource_spill_threshold_bytes,
        resource_spill_root=resource_spill_root,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        node_timeout_seconds=node_timeout_seconds,
        raise_on_error=raise_on_error,
    )


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


def _collected_answer(
    collector_outputs: dict[str, Any],
    *,
    collector_context: dict[str, Any],
) -> Any:
    if "answer" in collector_outputs:
        return collector_outputs["answer"]
    if collector_context.get("task_type") == "mixed":
        return dict(collector_outputs)
    return None


def _failed_record_for_unit(unit: ExecutionUnit, exc: Exception) -> NodeExecutionRecord:
    error = error_payload(exc)
    node = getattr(unit, "node", {}) or {}
    return NodeExecutionRecord(
        ok=False,
        node_id=node.get("id") or getattr(unit, "id", None),
        op=node.get("op") or getattr(unit, "op", None),
        output_name=node.get("output") or getattr(unit, "output", None),
        trace={
            "node_id": node.get("id") or getattr(unit, "id", None),
            "op": node.get("op") or getattr(unit, "op", None),
            "output": node.get("output") or getattr(unit, "output", None),
            "dependency": list(node.get("dependency", [])),
            "input": list(node.get("input", [])),
            "execution_path": "executor_setup",
            "fast_path_hit": False,
            "status": "failed",
            "error": error,
        },
        error=error,
    )


def _timeout_record_for_unit(
    unit: ExecutionUnit,
    timeout_seconds: float | None,
    *,
    elapsed_ms: float | None,
) -> NodeExecutionRecord:
    timeout_value = float(timeout_seconds or 0.0)
    message = (
        f"Node {unit.id} exceeded {timeout_value:.3g}s timeout"
        if timeout_value
        else f"Node {unit.id} exceeded its timeout"
    )
    exc = NodeTimeoutError(
        message,
        node=unit.node,
        timeout_seconds=timeout_value or None,
        elapsed_ms=elapsed_ms,
    )
    error = error_payload(exc)
    error["timeout_seconds"] = timeout_value or None
    if elapsed_ms is not None:
        error["elapsed_ms"] = elapsed_ms
    node = getattr(unit, "node", {}) or {}
    return NodeExecutionRecord(
        ok=False,
        node_id=node.get("id") or getattr(unit, "id", None),
        op=node.get("op") or getattr(unit, "op", None),
        output_name=node.get("output") or getattr(unit, "output", None),
        trace={
            "node_id": node.get("id") or getattr(unit, "id", None),
            "op": node.get("op") or getattr(unit, "op", None),
            "output": node.get("output") or getattr(unit, "output", None),
            "dependency": list(node.get("dependency", [])),
            "input": list(node.get("input", [])),
            "execution_path": "timeout",
            "fast_path_hit": False,
            "status": "failed",
            "elapsed_ms": elapsed_ms,
            "timeout_seconds": timeout_value or None,
            "error": error,
        },
        error=error,
    )


def _resolve_node_timeout_seconds(
    explicit_timeout: float | None,
    *,
    slm_config: dict[str, Any] | None,
) -> float | None:
    timeout = explicit_timeout
    if timeout is None and isinstance(slm_config, dict):
        timeout = slm_config.get("node_timeout_seconds")
    if timeout is None:
        return None
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(
            f"node_timeout_seconds must be a positive number: {timeout!r}"
        ) from exc
    if value <= 0:
        raise PlanValidationError("node_timeout_seconds must be positive")
    return value


def _future_deadlines(
    futures: set[Future[NodeExecutionRecord]],
    *,
    timeout_seconds: float | None,
) -> dict[Future[NodeExecutionRecord], float]:
    if timeout_seconds is None:
        return {}
    now = time.monotonic()
    return {future: now + timeout_seconds for future in futures}


def _next_wait_timeout(
    pending: set[Future[NodeExecutionRecord]],
    deadlines: dict[Future[NodeExecutionRecord], float],
) -> float | None:
    if not deadlines:
        return None
    deadline = min(deadlines[future] for future in pending)
    return max(0.0, deadline - time.monotonic())


def _timed_out_futures(
    pending: set[Future[NodeExecutionRecord]],
    deadlines: dict[Future[NodeExecutionRecord], float],
) -> list[Future[NodeExecutionRecord]]:
    if not deadlines:
        return []
    now = time.monotonic()
    return sorted(
        [
            future
            for future in pending
            if deadlines.get(future, float("inf")) <= now
        ],
        key=lambda future: deadlines.get(future, float("inf")),
    )


def _elapsed_ms_from_deadline(
    deadline: float | None,
    timeout_seconds: float | None,
) -> float | None:
    if deadline is None or timeout_seconds is None:
        return None
    started = deadline - timeout_seconds
    return max(0.0, (time.monotonic() - started) * 1000.0)


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


def _result_has_timeout_trace(result: ExecutionResult) -> bool:
    for trace in result.traces:
        if trace.get("execution_path") == "timeout":
            return True
        error = trace.get("error")
        if isinstance(error, dict) and error.get("type") == "NodeTimeoutError":
            return True
    return False


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
