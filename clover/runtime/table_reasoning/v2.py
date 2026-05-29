"""Synchronous table_reasoning_v2 batching runtime."""

from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from clover.executor import ExecutionResult, execute_physical_plan
from clover.planner import (
    SqlParseError,
    parse_remote_sql_to_logic_dag,
    parse_sql_list_response,
)
from clover.optimizer import optimize_logic_dag_to_physical_plan
from clover.preprocess import preprocess_task_dsl
from clover.commander import (
    render_followup_task_prompt,
    render_initial_task_prompt,
)
from clover.remote_llm import RemoteLLMSession, create_remote_llm_session
from clover.reporter import ReporterDecision, run_reporter
from clover.runtime.pipeline import CaseResult, GroupedPriorityQueue, PipelineProfiler


TASK_PENDING_REMOTE = "pending_remote"
TASK_SQL_READY = "sql_ready"
TASK_DAG_READY = "dag_ready"
TASK_EXECUTING = "executing"
TASK_REPORTER_REVIEW = "reporter_review"
TASK_SQL_REPAIR = "sql_repair"
TASK_RETRYING = "retrying"
TASK_SUCCESS = "success"
TASK_FAILED = "failed"


@dataclass
class TableReasoningCaseSpec:
    """One v1 table reasoning case submitted to the v2 runtime."""

    case_id: str
    task_dsl: dict[str, Any]
    base_dir: str | Path
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocess_result: dict[str, Any] | None = None
    answer_key: str | None = None


@dataclass
class TaskItem:
    """Internal lifecycle record for one table reasoning question."""

    case_id: str
    answer_key: str
    question: str
    answer_type: str
    source_file: str
    source_id: str
    task_dsl: dict[str, Any]
    local_dsl: dict[str, Any]
    remote_dsl: dict[str, Any]
    context: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    status: str = TASK_PENDING_REMOTE
    current_sql: str | None = None
    last_error: dict[str, Any] | None = None
    result_callback: Callable[[CaseResult], None] | None = field(
        default=None,
        repr=False,
    )


@dataclass
class SqlItem:
    """SQL ready for deterministic Planner lowering."""

    task: TaskItem
    sql: str


@dataclass
class LogicDagItem:
    """One per-question Logic DAG queued for same-table merging."""

    task: TaskItem
    sql: str
    logic_dag: dict[str, Any]


@dataclass
class TableSessionState:
    """Remote LLM session and prompt state for one source file."""

    source_file: str
    session: RemoteLLMSession
    schema_sent: bool = False
    reporter_instruction_sent: bool = False


@dataclass
class TableReasoningV2SystemResult:
    """Completed v2 runtime output."""

    case_results: list[CaseResult]
    task_items: dict[str, TaskItem]
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_results": [item.to_dict() for item in self.case_results],
            "profile": self.profile,
        }


class TableSessionManager:
    """Create and reuse one Remote LLM conversation per source file."""

    def __init__(self, remote_config: dict[str, Any], client: Any | None = None) -> None:
        self.remote_config = remote_config
        self.client = client
        self._sessions: dict[str, TableSessionState] = {}

    def get(self, source_file: str) -> TableSessionState:
        state = self._sessions.get(source_file)
        if state is None:
            state = TableSessionState(
                source_file=source_file,
                session=create_remote_llm_session(self.remote_config, client=self.client),
            )
            self._sessions[source_file] = state
        return state


class AnswerKeyAllocator:
    """Runtime-scoped global answer key allocator."""

    def __init__(self) -> None:
        self._next_index = 1

    def next(self) -> str:
        answer_key = f"answer_{self._next_index}"
        self._next_index += 1
        return answer_key


def run_table_reasoning_v2_system(
    *,
    case_specs: list[TableReasoningCaseSpec | dict[str, Any]],
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None = None,
    remote_batch_size: int = 16,
    local_batch_size: int = 4,
    max_retries: int = 1,
    client: Any | None = None,
    table_cache: dict[str, Any] | None = None,
    case_result_callback: Callable[[CaseResult], None] | None = None,
    profile_baseline: bool = False,
) -> TableReasoningV2SystemResult:
    """Run table reasoning cases through the v2 batching pipeline."""

    if remote_batch_size <= 0:
        raise ValueError("remote_batch_size must be positive")
    if local_batch_size <= 0:
        raise ValueError("local_batch_size must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    profiler = PipelineProfiler()
    session_manager = TableSessionManager(remote_config=remote_config, client=client)
    task_items = _build_task_items(case_specs, case_result_callback=case_result_callback)
    pending_remote = list(task_items.values())
    sql_items: list[SqlItem] = []
    dag_queue: GroupedPriorityQueue[LogicDagItem] = GroupedPriorityQueue()
    final_results: list[CaseResult] = []
    finalized: set[str] = set()

    while pending_remote or sql_items or dag_queue:
        if pending_remote:
            _run_one_commander_batch(
                pending_remote=pending_remote,
                sql_items=sql_items,
                final_results=final_results,
                finalized=finalized,
                session_manager=session_manager,
                remote_batch_size=remote_batch_size,
                profiler=profiler,
            )

        if sql_items:
            _run_planner(
                sql_items=sql_items,
                dag_queue=dag_queue,
                final_results=final_results,
                finalized=finalized,
                profiler=profiler,
            )

        if dag_queue:
            _run_one_execution_batch(
                dag_queue=dag_queue,
                sql_items=sql_items,
                final_results=final_results,
                finalized=finalized,
                session_manager=session_manager,
                local_batch_size=local_batch_size,
                max_retries=max_retries,
                table_cache=table_cache,
                local_slm_config=local_slm_config,
                profile_baseline=profile_baseline,
                profiler=profiler,
            )

    return TableReasoningV2SystemResult(
        case_results=final_results,
        task_items=task_items,
        profile=_profile_with_summary(profiler),
    )


def _build_task_items(
    case_specs: list[TableReasoningCaseSpec | dict[str, Any]],
    *,
    case_result_callback: Callable[[CaseResult], None] | None = None,
) -> dict[str, TaskItem]:
    allocator = AnswerKeyAllocator()
    task_items: dict[str, TaskItem] = {}
    for raw_spec in case_specs:
        spec = _normalize_case_spec(raw_spec)
        preprocess_result = spec.preprocess_result or preprocess_task_dsl(
            spec.task_dsl,
            base_dir=spec.base_dir,
        )
        answer_key = spec.answer_key or allocator.next()
        local_dsl = _with_answer_key(preprocess_result["local_dsl"], answer_key)
        remote_dsl = _with_answer_key(preprocess_result["remote_dsl"], local_dsl["answer"]["name"])
        source = local_dsl["sources"][0]
        source_file = str(Path(source["path"]).expanduser().resolve())
        task_items[answer_key] = TaskItem(
            case_id=spec.case_id,
            answer_key=answer_key,
            question=local_dsl["question"],
            answer_type=local_dsl["answer"]["type"],
            source_file=source_file,
            source_id=source["id"],
            task_dsl=copy.deepcopy(spec.task_dsl),
            local_dsl=local_dsl,
            remote_dsl=remote_dsl,
            context=copy.deepcopy(preprocess_result["context"]),
            metadata=copy.deepcopy(spec.metadata),
            result_callback=case_result_callback,
        )
    return task_items


def _normalize_case_spec(raw_spec: TableReasoningCaseSpec | dict[str, Any]) -> TableReasoningCaseSpec:
    if isinstance(raw_spec, TableReasoningCaseSpec):
        return raw_spec
    return TableReasoningCaseSpec(
        case_id=str(raw_spec["case_id"]),
        task_dsl=raw_spec["task_dsl"],
        base_dir=raw_spec["base_dir"],
        metadata=dict(raw_spec.get("metadata", {})),
        preprocess_result=raw_spec.get("preprocess_result"),
        answer_key=raw_spec.get("answer_key"),
    )


def _with_answer_key(dsl: dict[str, Any], answer_key: str) -> dict[str, Any]:
    updated = copy.deepcopy(dsl)
    updated["answer"] = copy.deepcopy(updated.get("answer", {}))
    updated["answer"]["name"] = answer_key
    return updated


def _run_one_commander_batch(
    *,
    pending_remote: list[TaskItem],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    session_manager: TableSessionManager,
    remote_batch_size: int,
    profiler: PipelineProfiler,
) -> None:
    source_file = pending_remote[0].source_file
    batch = _pop_batch_for_source(pending_remote, source_file, remote_batch_size)
    remote_dsl = _v2_remote_dsl(batch)
    state = session_manager.get(source_file)
    try:
        with profiler.measure("commander", items=len(batch)):
            prompt = (
                render_initial_task_prompt(remote_dsl)
                if not state.schema_sent
                else render_followup_task_prompt(remote_dsl)
            )
            result = state.session.generate(prompt)
            profiler.increment("commander_calls")
        state.schema_sent = True
        parsed = parse_sql_list_response(result.text, remote_dsl)
        for task, sql in zip(batch, parsed.sqls, strict=True):
            task.status = TASK_SQL_READY
            task.current_sql = sql
            sql_items.append(SqlItem(task=task, sql=sql))
    except Exception as exc:  # noqa: BLE001 - batch-level Remote failure.
        for task in batch:
            _finalize_failed(
                task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(exc),
            )


def _run_planner(
    *,
    sql_items: list[SqlItem],
    dag_queue: GroupedPriorityQueue[LogicDagItem],
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> None:
    pending = list(sql_items)
    sql_items.clear()
    with profiler.measure("planner", items=len(pending)):
        for item in pending:
            try:
                task = item.task
                logic_dag = parse_remote_sql_to_logic_dag(
                    item.sql,
                    _v1_remote_dsl(task),
                )
                task.status = TASK_DAG_READY
                dag_queue.push(
                    task.source_file,
                    LogicDagItem(task=task, sql=item.sql, logic_dag=logic_dag),
                    priority=task.retry_count,
                )
            except SqlParseError as exc:
                _finalize_failed(
                    item.task,
                    final_results=final_results,
                    finalized=finalized,
                    error=_error_payload(exc),
                )


def _run_one_execution_batch(
    *,
    dag_queue: GroupedPriorityQueue[LogicDagItem],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    session_manager: TableSessionManager,
    local_batch_size: int,
    max_retries: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    profile_baseline: bool,
    profiler: PipelineProfiler,
) -> None:
    popped = dag_queue.pop_best_group(local_batch_size)
    if popped is None:
        return
    source_file, batch = popped
    for item in batch:
        item.task.status = TASK_EXECUTING

    logic_dag = _v2_logic_dag(batch)
    local_dsl = _v2_local_dsl(batch)
    context = _v2_context(batch[0].task)
    if profile_baseline:
        _profile_one_by_one_baseline(
            batch=batch,
            table_cache=table_cache,
            local_slm_config=local_slm_config,
            profiler=profiler,
        )
    with profiler.measure("optimizer", items=len(batch)):
        physical_plan = optimize_logic_dag_to_physical_plan(
            logic_dag=logic_dag,
            context=context,
            local_dsl=local_dsl,
        )
    merge_stats = physical_plan.get("merge_stats", {})
    profiler.increment("merged_plan_count")
    profiler.increment("merged_plan_nodes", int(merge_stats.get("nodes", 0) or 0))
    profiler.increment("reused_nodes", int(merge_stats.get("reused_nodes", 0) or 0))

    with profiler.measure("executor", items=len(batch)):
        execution_result = execute_physical_plan(
            physical_plan,
            table_cache=table_cache,
            slm_config=local_slm_config,
        )

    state = session_manager.get(source_file)
    if execution_result.ok:
        _run_report_review(
            batch=batch,
            execution_result=execution_result,
            logic_dag=logic_dag,
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            session_state=state,
            max_retries=max_retries,
            profiler=profiler,
        )
        return

    affected = _affected_answer_keys(physical_plan, execution_result)
    if not affected:
        affected = {item.task.answer_key for item in batch}
    unaffected = [
        item
        for item in batch
        if item.task.answer_key not in affected
        and item.task.answer_key in execution_result.outputs
    ]
    interrupted = [
        item
        for item in batch
        if item.task.answer_key not in affected
        and item.task.answer_key not in execution_result.outputs
    ]
    for item in interrupted:
        item.task.status = TASK_DAG_READY
    if interrupted:
        # The executor fails fast, so independent branches scheduled after the
        # failing node may not have run. They are not failed answers; requeue
        # them for a later merged execution without consuming retry budget.
        for item in interrupted:
            dag_queue.push(source_file, item, priority=item.task.retry_count)

    if unaffected:
        _run_report_review(
            batch=unaffected,
            execution_result=_successful_subset_execution_result(
                execution_result,
                unaffected,
            ),
            logic_dag=_v2_logic_dag(unaffected),
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            session_state=state,
            max_retries=max_retries,
            profiler=profiler,
        )

    failed_items = [item for item in batch if item.task.answer_key in affected]
    if failed_items:
        _run_sql_repair(
            batch=failed_items,
            execution_result=execution_result,
            logic_dag=_v2_logic_dag(failed_items),
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            session_state=state,
            max_retries=max_retries,
            profiler=profiler,
        )


def _run_report_review(
    *,
    batch: list[LogicDagItem],
    execution_result: ExecutionResult,
    logic_dag: dict[str, Any],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    session_state: TableSessionState,
    max_retries: int,
    profiler: PipelineProfiler,
) -> None:
    for item in batch:
        item.task.status = TASK_REPORTER_REVIEW
    with profiler.measure("remote_reporter", items=len(batch)):
        reporter_result = run_reporter(
            local_dsl=_v2_local_dsl(batch),
            logic_dag=logic_dag,
            local_result=execution_result,
            current_sql=_sql_map(batch),
            session=session_state.session,
            include_root=not session_state.reporter_instruction_sent,
        )
        profiler.increment("remote_reporter_calls")
    session_state.reporter_instruction_sent = True
    _apply_reporter_decision(
        batch=batch,
        decision=reporter_result.decision,
        sql_items=sql_items,
        final_results=final_results,
        finalized=finalized,
        max_retries=max_retries,
    )


def _run_sql_repair(
    *,
    batch: list[LogicDagItem],
    execution_result: ExecutionResult,
    logic_dag: dict[str, Any],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    session_state: TableSessionState,
    max_retries: int,
    profiler: PipelineProfiler,
) -> None:
    eligible = [item for item in batch if item.task.retry_count < max_retries]
    exhausted = [item for item in batch if item.task.retry_count >= max_retries]
    for item in exhausted:
        _finalize_failed(
            item.task,
            final_results=final_results,
            finalized=finalized,
            error=item.task.last_error
            or execution_result.error
            or {"type": "RetryLimitExceeded", "message": "retry limit exhausted"},
        )
    if not eligible:
        return

    for item in eligible:
        item.task.status = TASK_SQL_REPAIR
    with profiler.measure("remote_reporter_sql_repair", items=len(eligible)):
        reporter_result = run_reporter(
            local_dsl=_v2_local_dsl(eligible),
            logic_dag=logic_dag,
            local_result=_failed_subset_execution_result(execution_result, eligible),
            current_sql=_sql_map(eligible),
            session=session_state.session,
            include_root=not session_state.reporter_instruction_sent,
        )
        profiler.increment("remote_reporter_sql_repair_calls")
    session_state.reporter_instruction_sent = True
    _apply_repair_decision(
        batch=eligible,
        decision=reporter_result.decision,
        sql_items=sql_items,
        final_results=final_results,
        finalized=finalized,
        max_retries=max_retries,
        fallback_error=execution_result.error,
    )


def _apply_reporter_decision(
    *,
    batch: list[LogicDagItem],
    decision: ReporterDecision,
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    max_retries: int,
) -> None:
    answer_payload = decision.answer if isinstance(decision.answer, dict) else {}
    new_sql = decision.new_sql or {}
    answer_keys = {item.task.answer_key for item in batch}

    for item in batch:
        key = item.task.answer_key
        value = answer_payload.get(key)
        if value is not None:
            _finalize_success(
                item.task,
                answer=value,
                final_results=final_results,
                finalized=finalized,
            )
            continue

        if not decision.retry:
            fallback = None
            if isinstance(decision.answer, dict):
                fallback = decision.answer.get(key)
            _finalize_success(
                item.task,
                answer=fallback,
                final_results=final_results,
                finalized=finalized,
            )
            continue

        sql = new_sql.get(key)
        if key not in answer_keys or not isinstance(sql, str):
            _finalize_failed(
                item.task,
                final_results=final_results,
                finalized=finalized,
                error={
                    "type": "ReporterProtocolError",
                    "message": f"Reporter did not return retry SQL for {key}",
                },
            )
            continue
        _enqueue_retry_sql(
            item.task,
            sql=sql,
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            max_retries=max_retries,
        )

    unexpected_keys = sorted(set(new_sql) - answer_keys)
    if unexpected_keys:
        for item in batch:
            if item.task.answer_key not in finalized:
                _finalize_failed(
                    item.task,
                    final_results=final_results,
                    finalized=finalized,
                    error={
                        "type": "ReporterProtocolError",
                        "message": f"Reporter returned unexpected SQL keys: {unexpected_keys}",
                    },
                )


def _apply_repair_decision(
    *,
    batch: list[LogicDagItem],
    decision: ReporterDecision,
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    max_retries: int,
    fallback_error: dict[str, Any] | None,
) -> None:
    new_sql = decision.new_sql or {}
    answer_keys = {item.task.answer_key for item in batch}
    for item in batch:
        key = item.task.answer_key
        sql = new_sql.get(key)
        if not decision.retry or not isinstance(sql, str):
            _finalize_failed(
                item.task,
                final_results=final_results,
                finalized=finalized,
                error=fallback_error
                or {
                    "type": "ReporterProtocolError",
                    "message": f"Reporter did not repair SQL for {key}",
                },
            )
            continue
        _enqueue_retry_sql(
            item.task,
            sql=sql,
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            max_retries=max_retries,
        )

    unexpected_keys = sorted(set(new_sql) - answer_keys)
    if unexpected_keys:
        for item in batch:
            if item.task.answer_key not in finalized:
                _finalize_failed(
                    item.task,
                    final_results=final_results,
                    finalized=finalized,
                    error={
                        "type": "ReporterProtocolError",
                        "message": f"Reporter returned unexpected SQL keys: {unexpected_keys}",
                    },
                )


def _enqueue_retry_sql(
    task: TaskItem,
    *,
    sql: str,
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    max_retries: int,
) -> None:
    if task.retry_count >= max_retries:
        _finalize_failed(
            task,
            final_results=final_results,
            finalized=finalized,
            error={"type": "RetryLimitExceeded", "message": "retry limit exhausted"},
        )
        return
    task.retry_count += 1
    task.current_sql = sql
    task.status = TASK_RETRYING
    sql_items.append(SqlItem(task=task, sql=sql))


def _finalize_success(
    task: TaskItem,
    *,
    answer: Any,
    final_results: list[CaseResult],
    finalized: set[str],
) -> None:
    if task.answer_key in finalized:
        return
    task.status = TASK_SUCCESS
    finalized.add(task.answer_key)
    result = CaseResult(
        case_id=task.case_id,
        answer_key=task.answer_key,
        status=TASK_SUCCESS,
        answer=answer,
        retry_count=task.retry_count,
        metadata=copy.deepcopy(task.metadata),
    )
    final_results.append(result)
    if task.result_callback is not None:
        task.result_callback(result)


def _finalize_failed(
    task: TaskItem,
    *,
    final_results: list[CaseResult],
    finalized: set[str],
    error: dict[str, Any],
) -> None:
    if task.answer_key in finalized:
        return
    task.status = TASK_FAILED
    task.last_error = error
    finalized.add(task.answer_key)
    result = CaseResult(
        case_id=task.case_id,
        answer_key=task.answer_key,
        status=TASK_FAILED,
        answer=None,
        error=error,
        retry_count=task.retry_count,
        metadata=copy.deepcopy(task.metadata),
    )
    final_results.append(result)
    if task.result_callback is not None:
        task.result_callback(result)


def _pop_batch_for_source(
    pending: list[TaskItem],
    source_file: str,
    max_items: int,
) -> list[TaskItem]:
    batch: list[TaskItem] = []
    remaining: list[TaskItem] = []
    for item in pending:
        if item.source_file == source_file and len(batch) < max_items:
            batch.append(item)
        else:
            remaining.append(item)
    pending[:] = remaining
    return batch


def _v2_remote_dsl(batch: list[TaskItem]) -> dict[str, Any]:
    first = batch[0]
    return {
        "task_type": "table_reasoning_v2",
        "questions": [item.question for item in batch],
        "sources": copy.deepcopy(first.remote_dsl.get("sources", [])),
        "answers": [
            {"name": item.answer_key, "type": item.answer_type}
            for item in batch
        ],
    }


def _v2_local_dsl(batch: list[LogicDagItem]) -> dict[str, Any]:
    first = batch[0].task
    return {
        "task_type": "table_reasoning_v2",
        "questions": [item.task.question for item in batch],
        "sources": copy.deepcopy(first.local_dsl.get("sources", [])),
        "answers": [
            {"name": item.task.answer_key, "type": item.task.answer_type}
            for item in batch
        ],
    }


def _v2_context(task: TaskItem) -> dict[str, Any]:
    context = copy.deepcopy(task.context)
    context["task_type"] = "table_reasoning_v2"
    return context


def _v1_remote_dsl(task: TaskItem) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning_v1",
        "question": task.question,
        "sources": copy.deepcopy(task.remote_dsl.get("sources", [])),
        "answer": {"name": task.answer_key, "type": task.answer_type},
    }


def _v1_local_dsl(task: TaskItem) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning_v1",
        "question": task.question,
        "sources": copy.deepcopy(task.local_dsl.get("sources", [])),
        "answer": {"name": task.answer_key, "type": task.answer_type},
    }


def _v1_context(task: TaskItem) -> dict[str, Any]:
    context = copy.deepcopy(task.context)
    context["task_type"] = "table_reasoning_v1"
    return context


def _profile_one_by_one_baseline(
    *,
    batch: list[LogicDagItem],
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    profiler: PipelineProfiler,
) -> None:
    for item in batch:
        try:
            with profiler.measure("baseline_optimizer", items=1):
                physical_plan = optimize_logic_dag_to_physical_plan(
                    logic_dag=item.logic_dag,
                    context=_v1_context(item.task),
                    local_dsl=_v1_local_dsl(item.task),
                )
            with profiler.measure("baseline_executor", items=1):
                execute_physical_plan(
                    physical_plan,
                    table_cache=table_cache,
                    slm_config=local_slm_config,
                )
            profiler.increment("baseline_case_count")
        except Exception:  # noqa: BLE001 - profiling must not change semantics.
            profiler.increment("baseline_profile_failures")


def _profile_with_summary(profiler: PipelineProfiler) -> dict[str, Any]:
    profile = profiler.to_dict()
    stages = profile.get("stages", {})
    executor_seconds = stages.get("executor", {}).get("total_seconds", 0.0)
    baseline_seconds = stages.get("baseline_executor", {}).get("total_seconds", 0.0)
    optimizer_seconds = stages.get("optimizer", {}).get("total_seconds", 0.0)
    baseline_optimizer_seconds = stages.get("baseline_optimizer", {}).get(
        "total_seconds",
        0.0,
    )
    summary = {
        "remote_calls": profile.get("counters", {}).get("commander_calls", 0)
        + profile.get("counters", {}).get("remote_reporter_calls", 0)
        + profile.get("counters", {}).get("remote_reporter_sql_repair_calls", 0),
        "merged_plan_count": profile.get("counters", {}).get("merged_plan_count", 0),
        "reused_nodes": profile.get("counters", {}).get("reused_nodes", 0),
    }
    if baseline_seconds:
        summary["local_executor_speedup"] = baseline_seconds / executor_seconds if executor_seconds else None
    if baseline_optimizer_seconds:
        summary["local_optimizer_speedup"] = (
            baseline_optimizer_seconds / optimizer_seconds
            if optimizer_seconds
            else None
        )
    profile["summary"] = summary
    return profile


def _v2_logic_dag(batch: list[LogicDagItem]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning_v2",
        "subtasks": [
            {
                "id": item.task.answer_key,
                "index": index,
                "question": item.task.question,
                "answer": {
                    "name": item.task.answer_key,
                    "type": item.task.answer_type,
                },
                "sql": item.sql,
                "logic_dag": copy.deepcopy(item.logic_dag),
            }
            for index, item in enumerate(batch)
        ],
    }


def _sql_map(batch: list[LogicDagItem]) -> dict[str, str]:
    return {item.task.answer_key: item.sql for item in batch}


def _affected_answer_keys(
    physical_plan: dict[str, Any],
    execution_result: ExecutionResult,
) -> set[str]:
    failing = execution_result.failing_node or {}
    failing_output = failing.get("output")
    failing_node_id = failing.get("id") or failing.get("node_id")
    output_to_node = {
        node.get("output"): node
        for node in physical_plan.get("nodes", [])
        if node.get("output")
    }
    if not failing_output and failing_node_id:
        for node in physical_plan.get("nodes", []):
            if node.get("id") == failing_node_id:
                failing_output = node.get("output")
                break
    if not failing_output:
        return {
            item["answer"]["name"]
            for item in physical_plan.get("subtask_outputs", [])
            if isinstance(item.get("answer"), dict)
        }

    affected_outputs = {failing_output}
    changed = True
    while changed:
        changed = False
        for node in physical_plan.get("nodes", []):
            dependencies = set(node.get("dependency", []))
            output = node.get("output")
            if output and dependencies & affected_outputs and output not in affected_outputs:
                affected_outputs.add(output)
                changed = True

    answer_keys = set()
    for item in physical_plan.get("subtask_outputs", []):
        answer = item.get("answer", {})
        output = item.get("output")
        if output in affected_outputs and isinstance(answer, dict):
            answer_keys.add(answer["name"])
    for output in affected_outputs:
        node = output_to_node.get(output)
        if node and node.get("op") == "FormatAnswer":
            answer_keys.add(output)
    return answer_keys


def _successful_subset_execution_result(
    execution_result: ExecutionResult,
    batch: list[LogicDagItem],
) -> ExecutionResult:
    answer = {
        item.task.answer_key: execution_result.outputs[item.task.answer_key]
        for item in batch
        if item.task.answer_key in execution_result.outputs
    }
    return ExecutionResult(
        ok=True,
        answer=answer,
        outputs=answer,
        traces=[],
        output_summaries={},
    )


def _failed_subset_execution_result(
    execution_result: ExecutionResult,
    batch: list[LogicDagItem],
) -> ExecutionResult:
    return ExecutionResult(
        ok=False,
        answer=None,
        outputs={},
        traces=[],
        output_summaries={},
        error=execution_result.error,
    )


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )[-6:],
    }
