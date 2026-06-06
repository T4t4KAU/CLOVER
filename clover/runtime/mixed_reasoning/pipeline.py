"""Task-neutral runtime for mixed table and document reasoning cases."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clover.executor import (
    ExecutionPlanBuilder,
    ExecutionResult,
    execute_execution_plan,
    slice_execution_result_by_namespace,
)
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    LocalSlmSequenceDispatcher,
)
from clover.runtime.document_reasoning import pipeline as document_pipeline
from clover.runtime.pipeline import (
    CaseResult,
    InflightCallResult,
    InflightStage,
    PipelineProfiler,
)
from clover.runtime.round_loop import RoundLoopResult, RoundLoopStep, RuntimeLoop
from clover.runtime.table_reasoning import pipeline as table_pipeline
from clover.runtime.task import (
    DocumentTaskItem,
    TableTaskItem,
    TASK_DAG_READY,
    TASK_EXECUTING,
    TASK_PENDING_REMOTE,
    TASK_SUPERVISOR_REVIEW,
)
from clover.supervisor import SupervisorAgent


TABLE_WORK_DAG = "table_dag"
TABLE_WORK_ACTION = "table_action"
DOCUMENT_WORK_PLAN = "document_plan"
REMOTE_TABLE_DECOMPOSE = "table_decompose"
REMOTE_DOCUMENT = "document"


@dataclass(frozen=True)
class MixedReasoningSystemResult:
    """Completed mixed table/document runtime output."""

    case_results: list[CaseResult]
    table_task_items: dict[str, TableTaskItem]
    document_task_items: dict[str, DocumentTaskItem]
    round_results: dict[str, RoundLoopResult]
    profile: dict[str, Any]

    @property
    def task_items(self) -> dict[str, Any]:
        return {
            **self.table_task_items,
            **self.document_task_items,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_results": [item.to_dict() for item in self.case_results],
            "profile": self.profile,
        }


@dataclass
class _MixedRemoteJob:
    kind: str
    payload: Any


@dataclass
class _MixedLocalWorkItem:
    kind: str
    payload: Any
    group_key: str
    priority: int = 0


@dataclass(frozen=True)
class _TablePlanEntry:
    namespace: str
    batch: list[table_pipeline.LogicDagItem]
    logic_dag: dict[str, Any]
    physical_plan: dict[str, Any]
    order_index: int


@dataclass(frozen=True)
class _DocumentPlanEntry:
    namespace: str
    work_item: document_pipeline._DocumentRoundWorkItem
    physical_plan: dict[str, Any]
    order_index: int


_PlanEntry = _TablePlanEntry | _DocumentPlanEntry


@dataclass
class _MixedRuntimeAdapter:
    pending_table_remote: list[TableTaskItem]
    pending_document_remote: list[document_pipeline._DocumentRemoteJob]
    remote_stage: InflightStage[_MixedRemoteJob, Any]
    table_sql_items: list[table_pipeline.SqlItem]
    document_code_items: list[document_pipeline._DocumentRoundWorkItem]
    local_items: list[_MixedLocalWorkItem]
    case_results: list[CaseResult]
    finalized: set[str]
    document_round_steps: dict[str, list[RoundLoopStep]]
    document_round_results: dict[str, RoundLoopResult]
    document_compact_observations: dict[str, dict[int, dict[str, Any]]]
    supervisor: SupervisorAgent
    remote_batch_size: int
    max_parallel_execution_units: int
    max_parallel_slm_node_jobs: int
    max_parallel_slm_sequences: int
    max_pending_slm_sequences: int
    max_retries: int
    validation_mode: str
    table_cache: dict[str, Any] | None
    local_slm_config: dict[str, Any] | None
    slm_client: Any | None
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None
    node_timeout_seconds: float | None
    profile_baseline: bool
    profiler: PipelineProfiler
    remote_turn: str = REMOTE_TABLE_DECOMPOSE

    def submit_remote_prefetch(self) -> None:
        while self.remote_stage.has_capacity:
            job = self._next_remote_job()
            if job is None:
                return
            self.remote_stage.submit(
                job,
                lambda job=job: _run_mixed_remote_job(
                    supervisor=self.supervisor,
                    job=job,
                ),
                items=_remote_job_items(job),
            )

    def _next_remote_job(self) -> _MixedRemoteJob | None:
        if not self.pending_table_remote and not self.pending_document_remote:
            return None
        prefer_document = self.remote_turn == REMOTE_DOCUMENT
        if prefer_document and self.pending_document_remote:
            self.remote_turn = REMOTE_TABLE_DECOMPOSE
            return _MixedRemoteJob(
                kind=REMOTE_DOCUMENT,
                payload=self.pending_document_remote.pop(0),
            )
        if self.pending_table_remote:
            self.remote_turn = REMOTE_DOCUMENT
            first = self.pending_table_remote[0]
            analyze = table_pipeline._is_analyze_task(first)
            batch = table_pipeline._pop_batch_for_source(
                self.pending_table_remote,
                first.group_key,
                1 if analyze else self.remote_batch_size,
                analyze=analyze,
            )
            remote_dsl = (
                table_pipeline._query_remote_dsl(batch[0])
                if analyze
                else table_pipeline._batch_remote_dsl(batch)
            )
            return _MixedRemoteJob(
                kind=REMOTE_TABLE_DECOMPOSE,
                payload=table_pipeline._RemoteDecomposeJob(
                    batch=batch,
                    remote_dsl=remote_dsl,
                ),
            )
        self.remote_turn = REMOTE_TABLE_DECOMPOSE
        return _MixedRemoteJob(
            kind=REMOTE_DOCUMENT,
            payload=self.pending_document_remote.pop(0),
        )

    def drain_remote(self, *, wait_for_one: bool) -> int:
        return self.remote_stage.drain_ready(
            lambda job, result: _finish_mixed_remote_job(
                adapter=self,
                job=job,
                call_result=result,
            ),
            wait_for_one=wait_for_one,
        )

    def parse_commands(self) -> None:
        table_pipeline._run_optimizer_parse(
            sql_items=self.table_sql_items,
            dag_queue=_TableDagLocalSink(self.local_items),
            final_results=self.case_results,
            finalized=self.finalized,
            profiler=self.profiler,
        )
        document_pending_remote: list[document_pipeline._DocumentRemoteJob] = []
        document_local_items: list[document_pipeline._DocumentRoundWorkItem] = []
        document_pipeline._parse_document_commands(
            code_items=self.document_code_items,
            local_items=document_local_items,
            pending_remote=document_pending_remote,
            case_results=self.case_results,
            finalized=self.finalized,
            round_steps=self.document_round_steps,
            round_results=self.document_round_results,
            compact_observations=self.document_compact_observations,
            max_retries=self.max_retries,
            profiler=self.profiler,
        )
        self.pending_document_remote.extend(document_pending_remote)
        for work_item in document_local_items:
            _enqueue_document_plan(self.local_items, work_item)

    def has_ready_barriers(self) -> bool:
        return False

    def advance_barriers(self) -> bool:
        return False

    def execute_local_once(self) -> bool:
        return _execute_mixed_local_once(self)

    def has_pending_remote(self) -> bool:
        return bool(self.pending_table_remote) or bool(self.pending_document_remote)

    def has_remote_inflight(self) -> bool:
        return bool(self.remote_stage)

    def has_commands(self) -> bool:
        return bool(self.table_sql_items) or bool(self.document_code_items)

    def has_local_work(self) -> bool:
        return bool(self.local_items)


class _TableDagLocalSink:
    def __init__(self, local_items: list[_MixedLocalWorkItem]) -> None:
        self._local_items = local_items

    def push(
        self,
        group_key: str,
        item: table_pipeline.LogicDagItem,
        *,
        priority: int = 0,
    ) -> None:
        self._local_items.append(
            _MixedLocalWorkItem(
                kind=TABLE_WORK_DAG,
                payload=item,
                group_key=group_key,
                priority=priority,
            )
        )


def run_mixed_reasoning_system(
    *,
    table_case_specs: list[table_pipeline.TableReasoningCaseSpec | dict[str, Any]]
    | None = None,
    document_case_specs: list[
        document_pipeline.DocumentReasoningCaseSpec | dict[str, Any]
    ]
    | None = None,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None = None,
    remote_batch_size: int = 16,
    remote_concurrency: int = 2,
    max_parallel_execution_units: int = 32,
    max_parallel_slm_node_jobs: int = 1,
    max_parallel_slm_sequences: int = DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    max_pending_slm_sequences: int = DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    max_retries: int = 1,
    max_rounds: int | None = None,
    client: Any | None = None,
    slm_client: Any | None = None,
    table_cache: dict[str, Any] | None = None,
    case_result_callback: Callable[[CaseResult], None] | None = None,
    profile_baseline: bool = False,
    validation_mode: str = table_pipeline.VALIDATION_NONE,
    node_timeout_seconds: float | None = None,
) -> MixedReasoningSystemResult:
    """Run table and document cases through one shared runtime loop."""

    if max_rounds is not None:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        max_retries = max_rounds - 1
    _validate_mixed_limits(
        remote_batch_size=remote_batch_size,
        remote_concurrency=remote_concurrency,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        max_retries=max_retries,
    )
    validation_mode = table_pipeline._normalize_validation_mode(validation_mode)

    table_specs, document_specs = _assign_mixed_answer_keys(
        table_case_specs or [],
        document_case_specs or [],
    )
    profiler = PipelineProfiler()
    supervisor = SupervisorAgent(remote_config=remote_config, client=client)
    table_task_items = table_pipeline._build_task_items(
        table_specs,
        case_result_callback=case_result_callback,
    )
    document_task_items = document_pipeline.build_document_task_items(
        document_specs,
        case_result_callback=case_result_callback,
    )

    pending_table_remote: list[TableTaskItem] = []
    pending_document_remote: list[document_pipeline._DocumentRemoteJob] = []
    table_sql_items: list[table_pipeline.SqlItem] = []
    table_action_items: list[table_pipeline.ActionGroupItem] = []
    document_code_items: list[document_pipeline._DocumentRoundWorkItem] = []
    local_items: list[_MixedLocalWorkItem] = []
    case_results: list[CaseResult] = []
    finalized: set[str] = set()
    document_round_results: dict[str, RoundLoopResult] = {}
    document_round_steps: dict[str, list[RoundLoopStep]] = {}
    document_compact_observations: dict[str, dict[int, dict[str, Any]]] = {}

    table_pipeline._initialize_table_entries(
        task_items=table_task_items,
        pending_remote=pending_table_remote,
        sql_items=table_sql_items,
        action_items=table_action_items,
        final_results=case_results,
        finalized=finalized,
        profiler=profiler,
    )
    _enqueue_table_actions(local_items, table_action_items)

    for task in document_task_items.values():
        task.status = TASK_PENDING_REMOTE
        work_item = document_pipeline._DocumentRoundWorkItem(
            task=task,
            state=document_pipeline.RoundLoopState(),
        )
        pending_document_remote.append(document_pipeline._document_decompose_job(work_item))

    local_slm_dispatcher = LocalSlmSequenceDispatcher(
        slm_config=local_slm_config,
        client=slm_client,
        max_parallel_sequences=max_parallel_slm_sequences,
        max_pending_sequences=max_pending_slm_sequences,
    )
    try:
        with InflightStage[_MixedRemoteJob, Any](
            stage_name="supervisor_remote",
            max_workers=remote_concurrency,
            profiler=profiler,
        ) as remote_stage:
            RuntimeLoop(
                _MixedRuntimeAdapter(
                    pending_table_remote=pending_table_remote,
                    pending_document_remote=pending_document_remote,
                    remote_stage=remote_stage,
                    table_sql_items=table_sql_items,
                    document_code_items=document_code_items,
                    local_items=local_items,
                    case_results=case_results,
                    finalized=finalized,
                    document_round_steps=document_round_steps,
                    document_round_results=document_round_results,
                    document_compact_observations=document_compact_observations,
                    supervisor=supervisor,
                    remote_batch_size=remote_batch_size,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                    max_retries=max_retries,
                    validation_mode=validation_mode,
                    table_cache=table_cache,
                    local_slm_config=local_slm_config,
                    slm_client=slm_client,
                    local_slm_dispatcher=local_slm_dispatcher,
                    node_timeout_seconds=node_timeout_seconds,
                    profile_baseline=profile_baseline,
                    profiler=profiler,
                )
            ).run()
    finally:
        local_slm_dispatcher.close()

    return MixedReasoningSystemResult(
        case_results=_ordered_mixed_case_results(
            case_results,
            table_task_items=table_task_items,
            document_task_items=document_task_items,
        ),
        table_task_items=table_task_items,
        document_task_items=document_task_items,
        round_results=document_round_results,
        profile=_profile_with_summary(
            profiler,
            validation_mode=validation_mode,
            table_case_count=len(table_task_items),
            document_case_count=len(document_task_items),
        ),
    )


def _validate_mixed_limits(
    *,
    remote_batch_size: int,
    remote_concurrency: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    max_retries: int,
) -> None:
    if remote_batch_size <= 0:
        raise ValueError("remote_batch_size must be positive")
    if remote_concurrency <= 0:
        raise ValueError("remote_concurrency must be positive")
    if max_parallel_execution_units <= 0:
        raise ValueError("max_parallel_execution_units must be positive")
    if max_parallel_slm_node_jobs <= 0:
        raise ValueError("max_parallel_slm_node_jobs must be positive")
    if max_parallel_slm_sequences <= 0:
        raise ValueError("max_parallel_slm_sequences must be positive")
    if max_pending_slm_sequences <= 0:
        raise ValueError("max_pending_slm_sequences must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")


def _assign_mixed_answer_keys(
    table_case_specs: list[table_pipeline.TableReasoningCaseSpec | dict[str, Any]],
    document_case_specs: list[
        document_pipeline.DocumentReasoningCaseSpec | dict[str, Any]
    ],
) -> tuple[
    list[table_pipeline.TableReasoningCaseSpec | dict[str, Any]],
    list[document_pipeline.DocumentReasoningCaseSpec | dict[str, Any]],
]:
    used: set[str] = set()
    next_index = 1
    table_specs, next_index = _assign_answer_keys_for_specs(
        table_case_specs,
        spec_class=table_pipeline.TableReasoningCaseSpec,
        used=used,
        next_index=next_index,
    )
    document_specs, _ = _assign_answer_keys_for_specs(
        document_case_specs,
        spec_class=document_pipeline.DocumentReasoningCaseSpec,
        used=used,
        next_index=next_index,
    )
    return table_specs, document_specs


def _assign_answer_keys_for_specs(
    case_specs: list[Any],
    *,
    spec_class: type[Any],
    used: set[str],
    next_index: int,
) -> tuple[list[Any], int]:
    if not case_specs:
        return [], next_index
    mixed = []
    for raw_spec in case_specs:
        answer_key = _raw_answer_key(raw_spec)
        if not answer_key:
            while f"answer_{next_index}" in used:
                next_index += 1
            answer_key = f"answer_{next_index}"
            next_index += 1
            mixed.append(_copy_spec_with_answer_key(raw_spec, spec_class, answer_key))
        else:
            if answer_key in used:
                raise ValueError(f"Duplicate mixed runtime answer_key: {answer_key}")
            mixed.append(raw_spec)
        used.add(answer_key)
    return mixed, next_index


def _raw_answer_key(raw_spec: Any) -> str | None:
    value = (
        raw_spec.get("answer_key")
        if isinstance(raw_spec, dict)
        else getattr(raw_spec, "answer_key", None)
    )
    return str(value) if value else None


def _copy_spec_with_answer_key(raw_spec: Any, spec_class: type[Any], answer_key: str) -> Any:
    if isinstance(raw_spec, dict):
        updated = dict(raw_spec)
        updated["answer_key"] = answer_key
        return updated
    if isinstance(raw_spec, spec_class):
        return spec_class(
            case_id=raw_spec.case_id,
            task_dsl=raw_spec.task_dsl,
            base_dir=raw_spec.base_dir,
            metadata=copy.deepcopy(raw_spec.metadata),
            preprocess_result=raw_spec.preprocess_result,
            answer_key=answer_key,
        )
    return raw_spec


def _run_mixed_remote_job(
    *,
    supervisor: SupervisorAgent,
    job: _MixedRemoteJob,
) -> Any:
    if job.kind == REMOTE_TABLE_DECOMPOSE:
        return supervisor.decompose(task_dsl=job.payload.remote_dsl)
    if job.kind == REMOTE_DOCUMENT:
        return document_pipeline._run_document_remote_job(
            supervisor=supervisor,
            job=job.payload,
        )
    raise ValueError(f"Unsupported mixed remote job kind: {job.kind!r}")


def _remote_job_items(job: _MixedRemoteJob) -> int:
    if job.kind == REMOTE_TABLE_DECOMPOSE:
        return len(job.payload.batch)
    return 1


def _finish_mixed_remote_job(
    *,
    adapter: _MixedRuntimeAdapter,
    job: _MixedRemoteJob,
    call_result: InflightCallResult[Any],
) -> None:
    if job.kind == REMOTE_TABLE_DECOMPOSE:
        action_items: list[table_pipeline.ActionGroupItem] = []
        table_pipeline._finish_remote_decompose_job(
            job=job.payload,
            call_result=call_result,
            sql_items=adapter.table_sql_items,
            action_items=action_items,
            final_results=adapter.case_results,
            finalized=adapter.finalized,
            profiler=adapter.profiler,
        )
        _enqueue_table_actions(adapter.local_items, action_items)
        return

    if job.kind == REMOTE_DOCUMENT:
        pending_remote: list[document_pipeline._DocumentRemoteJob] = []
        document_pipeline._finish_document_remote_job(
            job=job.payload,
            call_result=call_result,
            pending_remote=pending_remote,
            code_items=adapter.document_code_items,
            case_results=adapter.case_results,
            finalized=adapter.finalized,
            round_steps=adapter.document_round_steps,
            round_results=adapter.document_round_results,
            compact_observations=adapter.document_compact_observations,
            max_retries=adapter.max_retries,
            profiler=adapter.profiler,
        )
        adapter.pending_document_remote.extend(pending_remote)
        return

    raise ValueError(f"Unsupported mixed remote job kind: {job.kind!r}")


def _enqueue_table_actions(
    local_items: list[_MixedLocalWorkItem],
    action_items: list[table_pipeline.ActionGroupItem],
) -> None:
    for item in action_items:
        local_items.append(
            _MixedLocalWorkItem(
                kind=TABLE_WORK_ACTION,
                payload=item,
                group_key=f"table_action::{item.task.group_key}::{item.task.answer_key}",
                priority=item.task.retry_count,
            )
        )
    action_items.clear()


def _enqueue_document_plan(
    local_items: list[_MixedLocalWorkItem],
    work_item: document_pipeline._DocumentRoundWorkItem,
) -> None:
    local_items.append(
        _MixedLocalWorkItem(
            kind=DOCUMENT_WORK_PLAN,
            payload=work_item,
            group_key=f"document::{work_item.task.group_key}",
            priority=work_item.task.retry_count,
        )
    )


def _execute_mixed_local_once(adapter: _MixedRuntimeAdapter) -> bool:
    if not adapter.local_items:
        return False
    work_items = list(adapter.local_items)
    adapter.local_items.clear()

    plan_entries = _build_mixed_plan_entries(adapter, work_items)
    if plan_entries:
        _execute_mixed_plan_entries(adapter, plan_entries)

    action_items = [
        item.payload
        for item in work_items
        if item.kind == TABLE_WORK_ACTION
        and item.payload.task.answer_key not in adapter.finalized
    ]
    for action_item in action_items:
        followups = [action_item]
        table_pipeline._run_one_action_group(
            action_items=followups,
            final_results=adapter.case_results,
            finalized=adapter.finalized,
            supervisor=adapter.supervisor,
            max_retries=adapter.max_retries,
            table_cache=adapter.table_cache,
            local_slm_config=adapter.local_slm_config,
            local_slm_dispatcher=adapter.local_slm_dispatcher,
            max_parallel_execution_units=adapter.max_parallel_execution_units,
            max_parallel_slm_node_jobs=adapter.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=adapter.max_parallel_slm_sequences,
            max_pending_slm_sequences=adapter.max_pending_slm_sequences,
            profiler=adapter.profiler,
        )
        _enqueue_table_actions(adapter.local_items, followups)
    return True


def _build_mixed_plan_entries(
    adapter: _MixedRuntimeAdapter,
    work_items: list[_MixedLocalWorkItem],
) -> list[_PlanEntry]:
    table_groups: dict[tuple[str, int], list[table_pipeline.LogicDagItem]] = {}
    table_group_order: dict[tuple[str, int], int] = {}
    entries: list[_PlanEntry] = []

    for order_index, item in enumerate(work_items):
        if item.kind == TABLE_WORK_DAG:
            dag_item = item.payload
            if dag_item.task.answer_key in adapter.finalized:
                continue
            key = (item.group_key, item.priority)
            table_groups.setdefault(key, []).append(dag_item)
            table_group_order.setdefault(key, order_index)
            continue

        if item.kind == DOCUMENT_WORK_PLAN:
            work_item = item.payload
            if work_item.task.answer_key in adapter.finalized:
                continue
            work_item.task.status = TASK_EXECUTING
            entries.append(
                _DocumentPlanEntry(
                    namespace=document_pipeline._document_plan_namespace(work_item),
                    work_item=work_item,
                    physical_plan=document_pipeline._document_plan_with_question_context(
                        work_item.physical_plan or {},
                        question=work_item.task.question,
                    ),
                    order_index=order_index,
                )
            )

    for (group_key, _priority), batch in table_groups.items():
        if not batch:
            continue
        for item in batch:
            item.task.status = TASK_EXECUTING
        logic_dag = table_pipeline._batch_logic_dag(batch)
        local_dsl = table_pipeline._batch_local_dsl(batch)
        context = table_pipeline._batch_context(batch[0].task)
        if adapter.profile_baseline:
            table_pipeline._profile_one_by_one_baseline(
                batch=batch,
                table_cache=adapter.table_cache,
                local_slm_config=adapter.local_slm_config,
                profiler=adapter.profiler,
            )
        physical_plan = table_pipeline._optimize_table_logic_dag(
            logic_dag=logic_dag,
            context=context,
            local_dsl=local_dsl,
            profiler=adapter.profiler,
            stage_name="optimizer",
            items=len(batch),
        )
        merge_stats = physical_plan.get("merge_stats", {})
        adapter.profiler.increment("merged_plan_count")
        adapter.profiler.increment("merged_plan_nodes", int(merge_stats.get("nodes", 0) or 0))
        adapter.profiler.increment(
            "reused_nodes",
            int(merge_stats.get("reused_nodes", 0) or 0),
        )
        entries.append(
            _TablePlanEntry(
                namespace=_table_plan_namespace(batch),
                batch=batch,
                logic_dag=logic_dag,
                physical_plan=physical_plan,
                order_index=table_group_order[(group_key, _priority)],
            )
        )

    return sorted(entries, key=lambda entry: entry.order_index)


def _execute_mixed_plan_entries(
    adapter: _MixedRuntimeAdapter,
    entries: list[_PlanEntry],
) -> None:
    if len(entries) == 1:
        _execute_single_plan_entry(adapter, entries[0])
        return

    executor_fn = _executor_fn_for_entries(entries)
    namespaces = [entry.namespace for entry in entries]
    with adapter.profiler.measure("executor", items=_entry_item_count(entries)):
        execution_result = executor_fn(
            ExecutionPlanBuilder.default().build_many(
                [entry.physical_plan for entry in entries],
                namespaces=namespaces,
            ),
            collector_context={
                "task_type": "mixed",
                "source": "mixed_runtime",
                "namespaces": namespaces,
            },
            table_cache=adapter.table_cache,
            slm_config=adapter.local_slm_config,
            slm_client=adapter.slm_client,
            slm_dispatcher=adapter.local_slm_dispatcher,
            agent_loop_max_iterations=table_pipeline._agent_loop_max_iterations(
                adapter.local_slm_config
            ),
            max_parallel_execution_units=adapter.max_parallel_execution_units,
            max_parallel_slm_node_jobs=adapter.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=adapter.max_parallel_slm_sequences,
            max_pending_slm_sequences=adapter.max_pending_slm_sequences,
            node_timeout_seconds=adapter.node_timeout_seconds,
        )
    table_pipeline._record_executor_trace_counters(adapter.profiler, execution_result)

    for entry in entries:
        if not _namespace_has_activity(execution_result, entry.namespace):
            _requeue_entry(adapter, entry)
            continue
        task_result = slice_execution_result_by_namespace(
            execution_result,
            entry.namespace,
        )
        if isinstance(entry, _TablePlanEntry):
            table_pipeline._finish_table_execution_batch(
                batch=entry.batch,
                logic_dag=entry.logic_dag,
                physical_plan=entry.physical_plan,
                execution_result=task_result,
                requeue_interrupted=lambda item: adapter.local_items.append(
                    _MixedLocalWorkItem(
                        kind=TABLE_WORK_DAG,
                        payload=item,
                        group_key=table_pipeline._dag_group_key(item.task),
                        priority=item.task.retry_count,
                    )
                ),
                sql_items=adapter.table_sql_items,
                final_results=adapter.case_results,
                finalized=adapter.finalized,
                supervisor=adapter.supervisor,
                max_retries=adapter.max_retries,
                validation_mode=adapter.validation_mode,
                profiler=adapter.profiler,
            )
            continue
        _finish_document_plan_entry(
            adapter,
            entry=entry,
            execution_result=task_result,
        )


def _execute_single_plan_entry(
    adapter: _MixedRuntimeAdapter,
    entry: _PlanEntry,
) -> None:
    executor_fn = _executor_fn_for_entries([entry])
    collector_context = (
        entry.physical_plan
        if isinstance(entry, _TablePlanEntry)
        else {"task_type": "mixed", "source": "document_runtime"}
    )
    with adapter.profiler.measure("executor", items=_entry_item_count([entry])):
        execution_result = executor_fn(
            ExecutionPlanBuilder.default().build(entry.physical_plan),
            collector_context=collector_context,
            table_cache=adapter.table_cache,
            slm_config=adapter.local_slm_config,
            slm_client=adapter.slm_client,
            slm_dispatcher=adapter.local_slm_dispatcher,
            agent_loop_max_iterations=table_pipeline._agent_loop_max_iterations(
                adapter.local_slm_config
            ),
            max_parallel_execution_units=adapter.max_parallel_execution_units,
            max_parallel_slm_node_jobs=adapter.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=adapter.max_parallel_slm_sequences,
            max_pending_slm_sequences=adapter.max_pending_slm_sequences,
            node_timeout_seconds=adapter.node_timeout_seconds,
        )
    table_pipeline._record_executor_trace_counters(adapter.profiler, execution_result)
    if isinstance(entry, _TablePlanEntry):
        table_pipeline._finish_table_execution_batch(
            batch=entry.batch,
            logic_dag=entry.logic_dag,
            physical_plan=entry.physical_plan,
            execution_result=execution_result,
            requeue_interrupted=lambda item: adapter.local_items.append(
                _MixedLocalWorkItem(
                    kind=TABLE_WORK_DAG,
                    payload=item,
                    group_key=table_pipeline._dag_group_key(item.task),
                    priority=item.task.retry_count,
                )
            ),
            sql_items=adapter.table_sql_items,
            final_results=adapter.case_results,
            finalized=adapter.finalized,
            supervisor=adapter.supervisor,
            max_retries=adapter.max_retries,
            validation_mode=adapter.validation_mode,
            profiler=adapter.profiler,
        )
        return
    _finish_document_plan_entry(
        adapter,
        entry=entry,
        execution_result=execution_result,
    )


def _entry_item_count(entries: list[_PlanEntry]) -> int:
    total = 0
    for entry in entries:
        if isinstance(entry, _TablePlanEntry):
            total += len(entry.batch)
        else:
            total += 1
    return total


def _executor_fn_for_entries(entries: list[_PlanEntry]) -> Callable[..., ExecutionResult]:
    if any(isinstance(entry, _TablePlanEntry) for entry in entries):
        return table_pipeline.execute_execution_plan
    if all(isinstance(entry, _DocumentPlanEntry) for entry in entries):
        return document_pipeline.execute_execution_plan
    return execute_execution_plan


def _finish_document_plan_entry(
    adapter: _MixedRuntimeAdapter,
    *,
    entry: _DocumentPlanEntry,
    execution_result: ExecutionResult,
) -> None:
    work_item = entry.work_item
    task = work_item.task
    if task.answer_key in adapter.finalized:
        return
    work_item.execution_result = execution_result
    if not execution_result.ok:
        adapter.document_round_steps.setdefault(task.answer_key, []).append(
            RoundLoopStep(
                index=work_item.state.round_index,
                command_output=work_item.command or "",
                logic_dag=work_item.logic_dag or {},
                physical_plan=work_item.physical_plan or {},
                execution_result=execution_result,
                supervisor_result=None,
            )
        )
        document_pipeline._finalize_document_failed(
            task,
            case_results=adapter.case_results,
            finalized=adapter.finalized,
            round_steps=adapter.document_round_steps,
            round_results=adapter.document_round_results,
            error=execution_result.error
            or {"type": "ExecutionError", "message": "Document execution failed"},
        )
        return

    compact_observation = document_pipeline.build_compact_document_observation(
        execution_result,
        round_index=work_item.state.round_index,
        feedback=work_item.state.feedback,
        scratchpad=work_item.state.scratchpad,
    )
    compact_observation = document_pipeline._with_prior_document_evidence_memory(
        compact_observation,
        work_item.state,
    )
    work_item.compact_observation = compact_observation
    adapter.document_compact_observations.setdefault(task.answer_key, {})[
        work_item.state.round_index
    ] = copy.deepcopy(compact_observation)

    task.status = TASK_SUPERVISOR_REVIEW
    adapter.pending_document_remote.append(
        document_pipeline._document_synthesis_job(
            work_item,
            force_final_answer=work_item.state.round_index >= adapter.max_retries,
        )
    )


def _namespace_has_activity(result: ExecutionResult, namespace: str) -> bool:
    prefix = f"{namespace}__"
    if isinstance(result.failing_node, dict):
        for value in result.failing_node.values():
            if isinstance(value, str) and value.startswith(prefix):
                return True
    for payload in (result.outputs, result.collector_outputs, result.output_summaries):
        if any(isinstance(key, str) and key.startswith(prefix) for key in payload):
            return True
    if isinstance(result.answer, dict) and any(
        isinstance(key, str) and key.startswith(prefix)
        for key in result.answer
    ):
        return True
    for trace in result.traces:
        if not isinstance(trace, dict):
            continue
        for key in ("node_id", "output", "job_id", "sequence_id"):
            value = trace.get(key)
            if isinstance(value, str) and value.startswith(prefix):
                return True
    return result.ok


def _requeue_entry(adapter: _MixedRuntimeAdapter, entry: _PlanEntry) -> None:
    if isinstance(entry, _TablePlanEntry):
        for item in entry.batch:
            item.task.status = TASK_DAG_READY
            adapter.local_items.append(
                _MixedLocalWorkItem(
                    kind=TABLE_WORK_DAG,
                    payload=item,
                    group_key=table_pipeline._dag_group_key(item.task),
                    priority=item.task.retry_count,
                )
            )
        return
    entry.work_item.task.status = TASK_DAG_READY
    _enqueue_document_plan(adapter.local_items, entry.work_item)


def _table_plan_namespace(batch: list[table_pipeline.LogicDagItem]) -> str:
    raw = "table__" + "__".join(item.task.answer_key for item in batch[:3])
    if len(batch) > 3:
        raw += f"__n{len(batch)}"
    return _safe_namespace(raw)


def _safe_namespace(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "plan"


def _ordered_mixed_case_results(
    results: list[CaseResult],
    *,
    table_task_items: dict[str, TableTaskItem],
    document_task_items: dict[str, DocumentTaskItem],
) -> list[CaseResult]:
    ordered_tasks = [
        *table_task_items.values(),
        *document_task_items.values(),
    ]
    order = {
        (task.case_id, task.answer_key): index
        for index, task in enumerate(ordered_tasks)
    }
    return sorted(
        results,
        key=lambda result: order.get((result.case_id, result.answer_key), len(order)),
    )


def _profile_with_summary(
    profiler: PipelineProfiler,
    *,
    validation_mode: str,
    table_case_count: int,
    document_case_count: int,
) -> dict[str, Any]:
    profile = table_pipeline._profile_with_summary(
        profiler,
        validation_mode=validation_mode,
    )
    profile["summary"]["table_cases"] = table_case_count
    profile["summary"]["document_cases"] = document_case_count
    profile["summary"]["mixed_runtime"] = True
    return profile
