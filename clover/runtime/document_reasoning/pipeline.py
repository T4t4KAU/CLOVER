"""Document reasoning compatibility entry point and mixed-runtime helpers."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

from clover.executor import (
    ExecutionResult,
    ExecutionPlanBuilder,
    execute_execution_plan,
    slice_execution_result_by_namespace,
)
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    LocalSlmSequenceDispatcher,
)
from clover.optimizer.ir import DOCUMENT_REASONING_TASK_TYPE
from clover.optimizer import optimize_logic_dag_to_physical_plan
from clover.optimizer import parse_remote_document_code_to_logic_dag
from clover.supervisor import (
    SupervisorAgent,
    build_compact_document_observation,
    extract_token_usage,
)
from clover.resource import prepare_physical_plan_resources
from clover.runtime.items import RuntimeCommandItem, RuntimeWorkItem
from clover.runtime.final_answer import finalize_answer
from clover.runtime.pipeline import (
    CaseResult,
    InflightCallResult,
    PipelineProfiler,
)
from clover.runtime.round_loop import (
    RoundLoopResult,
    RoundLoopState,
    RoundLoopStep,
)
from clover.runtime.task import (
    DocumentTaskItem,
    RuntimeCaseSpec,
    TASK_CODE_READY,
    TASK_DAG_READY,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_SUPERVISOR_REVIEW,
    TASK_SUCCESS,
    build_runtime_task_items,
)


DOCUMENT_EVIDENCE_MEMORY_KEY = "document_evidence_memory"
DOCUMENT_EVIDENCE_MEMORY_MAX_CHARS = 16000


@dataclass
class DocumentReasoningCaseSpec(RuntimeCaseSpec):
    """One document reasoning case submitted to the document runtime."""

    pass


@dataclass
class PythonCodeItem(RuntimeCommandItem[DocumentTaskItem]):
    """Remote Python code ready for deterministic Optimizer parsing."""

    @property
    def code(self) -> str:
        return self.content


@dataclass
class DocumentLogicDagItem(RuntimeWorkItem[DocumentTaskItem]):
    """One document Logic DAG queued for later optimization/execution."""

    @property
    def code(self) -> str:
        return self.command_output


@dataclass(frozen=True)
class DocumentReasoningSystemResult:
    """Completed document reasoning runtime output."""

    case_results: list[CaseResult]
    task_items: dict[str, DocumentTaskItem]
    round_results: dict[str, RoundLoopResult]
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_results": [item.to_dict() for item in self.case_results],
            "profile": self.profile,
        }


@dataclass
class _DocumentRoundWorkItem:
    task: DocumentTaskItem
    state: RoundLoopState
    command: str | None = None
    logic_dag: dict[str, Any] | None = None
    physical_plan: dict[str, Any] | None = None
    execution_result: Any | None = None
    compact_observation: dict[str, Any] | None = None


@dataclass(frozen=True)
class _DocumentRemoteJob:
    action: str
    work_item: _DocumentRoundWorkItem
    payload: dict[str, Any]


def build_document_task_items(
    case_specs: list[DocumentReasoningCaseSpec | dict[str, Any]],
    *,
    case_result_callback: Callable[[CaseResult], None] | None = None,
) -> dict[str, DocumentTaskItem]:
    """Build document task lifecycle records from case specs."""

    return build_runtime_task_items(
        case_specs,
        task_type=DOCUMENT_REASONING_TASK_TYPE,
        case_spec_class=DocumentReasoningCaseSpec,
        task_item_class=DocumentTaskItem,
        case_result_callback=case_result_callback,
    )


def run_document_reasoning_system(
    *,
    case_specs: list[DocumentReasoningCaseSpec | dict[str, Any]],
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    client: Any | None = None,
    slm_client: Any | None = None,
    remote_concurrency: int = 64,
    max_retries: int = 1,
    max_rounds: int | None = None,
    max_parallel_execution_units: int = 64,
    max_parallel_slm_node_jobs: int = DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    max_parallel_slm_sequences: int = DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    max_pending_slm_sequences: int = DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    node_timeout_seconds: float | None = None,
    case_result_callback: Callable[[CaseResult], None] | None = None,
) -> DocumentReasoningSystemResult:
    """Run document reasoning cases through a MinionS-style supervisor loop."""

    if max_rounds is not None:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        max_retries = max_rounds - 1
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
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

    from clover.runtime.mixed_reasoning.pipeline import run_mixed_reasoning_system

    mixed_result = run_mixed_reasoning_system(
        document_case_specs=case_specs,
        remote_config=remote_config,
        synthesize_config=synthesize_config,
        local_slm_config=local_slm_config,
        client=client,
        slm_client=slm_client,
        remote_concurrency=remote_concurrency,
        max_retries=max_retries,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        node_timeout_seconds=node_timeout_seconds,
        case_result_callback=case_result_callback,
    )
    return DocumentReasoningSystemResult(
        case_results=mixed_result.case_results,
        task_items=mixed_result.document_task_items,
        round_results=mixed_result.round_results,
        profile=mixed_result.profile,
    )


def _run_document_remote_job(
    *,
    supervisor: SupervisorAgent,
    job: _DocumentRemoteJob,
) -> Any:
    if job.action == "decompose":
        return supervisor.decompose(task_dsl=job.payload["task_dsl"])
    if job.action == "synthesize":
        return supervisor.synthesize(**job.payload)
    raise ValueError(f"Unsupported document remote action: {job.action!r}")


def _finish_document_remote_job(
    *,
    job: _DocumentRemoteJob,
    call_result: InflightCallResult[Any],
    pending_remote: list[_DocumentRemoteJob],
    code_items: list[_DocumentRoundWorkItem],
    case_results: list[CaseResult],
    finalized: set[str],
    round_steps: dict[str, list[RoundLoopStep]],
    round_results: dict[str, RoundLoopResult],
    compact_observations: dict[str, dict[int, dict[str, Any]]],
    max_retries: int,
    profiler: PipelineProfiler,
) -> None:
    task = job.work_item.task
    if task.answer_key in finalized:
        return
    if call_result.error is not None:
        _finalize_document_failed(
            task,
            case_results=case_results,
            finalized=finalized,
            round_steps=round_steps,
            round_results=round_results,
            error=_error_payload(call_result.error),
        )
        return

    result = call_result.value
    if job.action == "decompose":
        _record_document_remote_result(
            profiler,
            result=result,
            stage_name="supervisor_decompose",
            counter_name="supervisor_decompose_calls",
        )
        try:
            _queue_document_code(
                job.work_item,
                code_items=code_items,
                command=result.command or "",
            )
        except Exception as exc:  # noqa: BLE001 - normalize protocol failures.
            _finalize_document_failed(
                task,
                case_results=case_results,
                finalized=finalized,
                round_steps=round_steps,
                round_results=round_results,
                error=_error_payload(exc),
            )
        return

    if job.action == "synthesize":
        _record_document_remote_result(
            profiler,
            result=result,
            stage_name="supervisor_synthesis",
            counter_name="supervisor_synthesis_calls",
        )
        _finish_document_synthesis(
            job.work_item,
            supervisor_result=result,
            pending_remote=pending_remote,
            code_items=code_items,
            case_results=case_results,
            finalized=finalized,
            round_steps=round_steps,
            round_results=round_results,
            compact_observations=compact_observations,
            max_retries=max_retries,
        )
        return

    _finalize_document_failed(
        task,
        case_results=case_results,
        finalized=finalized,
        round_steps=round_steps,
        round_results=round_results,
        error={
            "type": "ValueError",
            "message": f"Unsupported document remote action: {job.action!r}",
        },
    )


def _record_document_remote_result(
    profiler: PipelineProfiler,
    *,
    result: Any,
    stage_name: str,
    counter_name: str,
) -> None:
    profiler.increment(counter_name)
    _record_remote_token_usage(
        profiler,
        stage_name=stage_name,
        response_payload=result.response_payload,
    )


def _queue_document_code(
    work_item: _DocumentRoundWorkItem,
    *,
    code_items: list[_DocumentRoundWorkItem],
    command: str,
) -> None:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Supervisor document command is empty")
    work_item.command = command
    work_item.task.current_command = command
    work_item.task.status = TASK_CODE_READY
    code_items.append(work_item)


def _parse_document_commands(
    *,
    code_items: list[_DocumentRoundWorkItem],
    local_items: list[_DocumentRoundWorkItem],
    pending_remote: list[_DocumentRemoteJob],
    case_results: list[CaseResult],
    finalized: set[str],
    round_steps: dict[str, list[RoundLoopStep]],
    round_results: dict[str, RoundLoopResult],
    compact_observations: dict[str, dict[int, dict[str, Any]]],
    max_retries: int,
    profiler: PipelineProfiler,
) -> None:
    pending = list(code_items)
    code_items.clear()
    for work_item in pending:
        task = work_item.task
        if task.answer_key in finalized:
            continue
        command = work_item.command or ""
        remote_dsl = _round_remote_dsl(task.remote_dsl, work_item.state)
        try:
            artifact = _build_document_plan(
                command,
                task=task,
                remote_dsl=remote_dsl,
                profiler=profiler,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a normal observation.
            _queue_document_build_error_report(
                work_item,
                command=command,
                exc=exc,
                pending_remote=pending_remote,
                compact_observations=compact_observations,
                max_retries=max_retries,
            )
            continue
        work_item.command = command
        work_item.logic_dag = artifact["logic_dag"]
        work_item.physical_plan = artifact["physical_plan"]
        task.current_command = command
        task.status = TASK_DAG_READY
        local_items.append(work_item)


def _build_document_plan(
    command: str,
    *,
    task: DocumentTaskItem,
    remote_dsl: dict[str, Any],
    profiler: PipelineProfiler,
) -> dict[str, dict[str, Any]]:
    with profiler.measure("optimizer_parse", items=1):
        logic_dag = parse_remote_document_code_to_logic_dag(command, remote_dsl)

    with profiler.measure("optimizer", items=1):
        physical_plan = optimize_logic_dag_to_physical_plan(
            logic_dag=logic_dag,
            context=task.context,
            local_dsl=task.local_dsl,
        )
        physical_plan = prepare_physical_plan_resources(physical_plan)
    return {
        "logic_dag": logic_dag,
        "physical_plan": physical_plan,
    }


def _queue_document_build_error_report(
    work_item: _DocumentRoundWorkItem,
    *,
    command: str,
    exc: Exception,
    pending_remote: list[_DocumentRemoteJob],
    compact_observations: dict[str, dict[int, dict[str, Any]]],
    max_retries: int,
) -> None:
    task = work_item.task
    task.current_command = command
    task.status = TASK_SUPERVISOR_REVIEW
    error = _error_payload(exc)
    execution_result = ExecutionResult(
        ok=False,
        answer=None,
        outputs={},
        collector_outputs={},
        traces=[
            {
                "status": "error",
                "op": "build_command",
                "error": error,
            }
        ],
        output_summaries={},
        error=error,
    )
    compact_observation = build_compact_document_observation(
        execution_result,
        round_index=work_item.state.round_index,
        feedback=work_item.state.feedback,
        scratchpad=work_item.state.scratchpad,
    )
    compact_observation = _with_prior_document_evidence_memory(
        compact_observation,
        work_item.state,
    )
    work_item.command = command
    work_item.logic_dag = {}
    work_item.physical_plan = {}
    work_item.execution_result = execution_result
    work_item.compact_observation = compact_observation
    compact_observations.setdefault(task.answer_key, {})[
        work_item.state.round_index
    ] = copy.deepcopy(compact_observation)
    pending_remote.append(
        _document_synthesis_job(
            work_item,
            force_final_answer=work_item.state.round_index >= max_retries,
        )
    )


def _execute_document_plan_batch(
    *,
    local_items: list[_DocumentRoundWorkItem],
    pending_remote: list[_DocumentRemoteJob],
    case_results: list[CaseResult],
    finalized: set[str],
    round_steps: dict[str, list[RoundLoopStep]],
    round_results: dict[str, RoundLoopResult],
    compact_observations: dict[str, dict[int, dict[str, Any]]],
    local_slm_config: dict[str, Any] | None,
    slm_client: Any | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    node_timeout_seconds: float | None,
    max_retries: int,
    profiler: PipelineProfiler,
) -> bool:
    if not local_items:
        return False
    batch = list(local_items)
    local_items.clear()
    for work_item in batch:
        work_item.task.status = TASK_EXECUTING

    namespaces = [_document_plan_namespace(work_item) for work_item in batch]
    physical_plans = [
        _document_plan_with_question_context(
            work_item.physical_plan or {},
            question=work_item.task.question,
        )
        for work_item in batch
    ]
    with profiler.measure("executor", items=len(batch)):
        execution_result = execute_execution_plan(
            ExecutionPlanBuilder.default().build_many(
                physical_plans,
                namespaces=namespaces,
            ),
            collector_context={"task_type": "mixed", "source": "document_batch"},
            slm_config=local_slm_config,
            slm_client=slm_client,
            slm_dispatcher=local_slm_dispatcher,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            node_timeout_seconds=node_timeout_seconds,
        )
        _record_local_slm_token_usage(
            profiler,
            execution_result=execution_result,
        )

    for namespace, work_item in zip(namespaces, batch):
        task = work_item.task
        if task.answer_key in finalized:
            continue
        task_execution_result = slice_execution_result_by_namespace(
            execution_result,
            namespace,
        )
        work_item.execution_result = task_execution_result
        if not task_execution_result.ok:
            round_steps.setdefault(task.answer_key, []).append(
                RoundLoopStep(
                    index=work_item.state.round_index,
                    command_output=work_item.command or "",
                    logic_dag=work_item.logic_dag or {},
                    physical_plan=work_item.physical_plan or {},
                    execution_result=task_execution_result,
                    supervisor_result=None,
                )
            )
            _finalize_document_failed(
                task,
                case_results=case_results,
                finalized=finalized,
                round_steps=round_steps,
                round_results=round_results,
                error=task_execution_result.error
                or {"type": "ExecutionError", "message": "Document execution failed"},
            )
            continue
        compact_observation = build_compact_document_observation(
            task_execution_result,
            round_index=work_item.state.round_index,
            feedback=work_item.state.feedback,
            scratchpad=work_item.state.scratchpad,
        )
        compact_observation = _with_prior_document_evidence_memory(
            compact_observation,
            work_item.state,
        )
        work_item.compact_observation = compact_observation
        compact_observations.setdefault(task.answer_key, {})[
            work_item.state.round_index
        ] = copy.deepcopy(compact_observation)

        task.status = TASK_SUPERVISOR_REVIEW
        pending_remote.append(
            _document_synthesis_job(
                work_item,
                force_final_answer=work_item.state.round_index >= max_retries,
            )
        )
    return True


def _document_plan_namespace(work_item: _DocumentRoundWorkItem) -> str:
    raw = f"{work_item.task.answer_key}__r{work_item.state.round_index}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "document_plan"


def _document_plan_with_question_context(
    physical_plan: dict[str, Any],
    *,
    question: str,
) -> dict[str, Any]:
    plan = copy.deepcopy(physical_plan)
    plan["question"] = question
    for group in plan.get("map_groups", []) or []:
        if not isinstance(group, dict):
            continue
        params = group.get("params")
        if not isinstance(params, dict):
            params = {}
            group["params"] = params
        params.setdefault("question_context", question)
    for node in plan.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        params = node.get("params")
        if not isinstance(params, dict):
            params = {}
            node["params"] = params
        params.setdefault("question_context", question)
    return plan


def _finish_document_synthesis(
    work_item: _DocumentRoundWorkItem,
    *,
    supervisor_result: Any,
    pending_remote: list[_DocumentRemoteJob],
    code_items: list[_DocumentRoundWorkItem],
    case_results: list[CaseResult],
    finalized: set[str],
    round_steps: dict[str, list[RoundLoopStep]],
    round_results: dict[str, RoundLoopResult],
    compact_observations: dict[str, dict[int, dict[str, Any]]],
    max_retries: int,
) -> None:
    task = work_item.task
    step = RoundLoopStep(
        index=work_item.state.round_index,
        command_output=work_item.command or "",
        logic_dag=work_item.logic_dag or {},
        physical_plan=work_item.physical_plan or {},
        execution_result=work_item.execution_result,
        supervisor_result=supervisor_result,
    )
    steps = round_steps.setdefault(task.answer_key, [])
    steps.append(step)
    if _document_step_is_final(step, max_retries=max_retries):
        answer = _document_final_answer(
            task,
            step,
            compact_observations=compact_observations,
        )
        _finalize_document_success(
            task,
            answer=answer,
            case_results=case_results,
            finalized=finalized,
            round_steps=round_steps,
            round_results=round_results,
            final_decision=supervisor_result,
        )
        return

    try:
        next_state = _document_next_state(work_item.state, step, task=task)
    except Exception as exc:  # noqa: BLE001 - normalize next-command failures.
        _finalize_document_failed(
            task,
            case_results=case_results,
            finalized=finalized,
            round_steps=round_steps,
            round_results=round_results,
            error=_error_payload(exc),
            final_decision=supervisor_result,
        )
        return
    code_items.append(
        _DocumentRoundWorkItem(
            task=task,
            state=next_state,
            command=next_state.next_command,
        )
    )


def _document_step_is_final(
    step: RoundLoopStep,
    *,
    max_retries: int,
) -> bool:
    decision = step.supervisor_result.decision
    if decision is None:
        return False
    if step.index >= max_retries:
        return True
    if decision.sufficient is True:
        return True
    if decision.sufficient is False:
        return False
    return decision.answer is not None and not decision.retry


def _document_final_answer(
    task: DocumentTaskItem,
    step: RoundLoopStep,
    *,
    compact_observations: dict[str, dict[int, dict[str, Any]]],
) -> Any:
    decision = step.supervisor_result.decision
    if decision is None:
        return None
    return finalize_answer(
        task_type=task.task_type,
        question=task.question,
        answer=decision.answer,
        explanation=decision.explanation,
        observation=compact_observations.get(task.answer_key, {}).get(step.index),
    )


def _document_next_state(
    state: RoundLoopState,
    step: RoundLoopStep,
    *,
    task: DocumentTaskItem,
) -> RoundLoopState:
    decision = step.supervisor_result.decision
    if decision is None:
        raise ValueError("Document supervisor retry requires a decision")
    next_code = decision.next_python_code
    if not isinstance(next_code, str) or not next_code.strip():
        raise ValueError("Document supervisor retry requires next_python_code")
    previous_observation = build_compact_document_observation(
        step.execution_result,
        round_index=state.round_index,
        feedback=decision.feedback or decision.explanation or state.feedback,
        scratchpad=decision.scratchpad or state.scratchpad,
    )
    previous_observation = _with_prior_document_evidence_memory(
        previous_observation,
        state,
    )
    metadata = dict(state.metadata)
    metadata["previous_round"] = state.round_index
    metadata[DOCUMENT_EVIDENCE_MEMORY_KEY] = _next_document_evidence_memory(
        state,
        previous_observation,
    )
    task.retry_count += 1
    return RoundLoopState(
        round_index=state.round_index + 1,
        feedback=decision.feedback or decision.explanation or state.feedback,
        scratchpad=decision.scratchpad or state.scratchpad,
        previous_observation=previous_observation,
        next_command=next_code,
        metadata=metadata,
    )


def _document_decompose_job(work_item: _DocumentRoundWorkItem) -> _DocumentRemoteJob:
    return _DocumentRemoteJob(
        action="decompose",
        work_item=work_item,
        payload={"task_dsl": _round_remote_dsl(work_item.task.remote_dsl, work_item.state)},
    )


def _document_synthesis_job(
    work_item: _DocumentRoundWorkItem,
    *,
    force_final_answer: bool,
) -> _DocumentRemoteJob:
    return _DocumentRemoteJob(
        action="synthesize",
        work_item=work_item,
        payload={
            "task_dsl": work_item.task.task_dsl,
            "local_dsl": work_item.task.local_dsl,
            "logic_dag": work_item.logic_dag or {},
            "observation": work_item.compact_observation or {},
            "force_final_answer": force_final_answer,
        },
    )


def _finalize_document_success(
    task: DocumentTaskItem,
    *,
    answer: Any,
    case_results: list[CaseResult],
    finalized: set[str],
    round_steps: dict[str, list[RoundLoopStep]],
    round_results: dict[str, RoundLoopResult],
    final_decision: Any,
) -> None:
    if task.answer_key in finalized:
        return
    task.status = TASK_SUCCESS
    result = _case_success(task, answer)
    _emit_result(task, result, case_results)
    finalized.add(task.answer_key)
    round_results[task.answer_key] = RoundLoopResult(
        ok=True,
        answer=answer,
        rounds=list(round_steps.get(task.answer_key, [])),
        final_decision=final_decision,
    )


def _finalize_document_failed(
    task: DocumentTaskItem,
    *,
    case_results: list[CaseResult],
    finalized: set[str],
    round_steps: dict[str, list[RoundLoopStep]],
    round_results: dict[str, RoundLoopResult],
    error: dict[str, Any],
    final_decision: Any | None = None,
) -> None:
    if task.answer_key in finalized:
        return
    task.status = TASK_FAILED
    result = _case_failed(task, error)
    _emit_result(task, result, case_results)
    finalized.add(task.answer_key)
    round_results[task.answer_key] = RoundLoopResult(
        ok=False,
        answer=None,
        rounds=list(round_steps.get(task.answer_key, [])),
        final_decision=final_decision,
        error=copy.deepcopy(error),
    )


def _round_remote_dsl(
    remote_dsl: dict[str, Any],
    state: RoundLoopState,
) -> dict[str, Any]:
    payload = copy.deepcopy(remote_dsl)
    if state.round_index <= 0:
        return payload
    payload["round_state"] = {
        "round_index": state.round_index,
        "feedback": state.feedback,
        "scratchpad": state.scratchpad,
        "previous_observation": copy.deepcopy(state.previous_observation),
    }
    return payload


def _with_prior_document_evidence_memory(
    observation: dict[str, Any],
    state: RoundLoopState,
) -> dict[str, Any]:
    payload = dict(observation)
    sections = _document_evidence_memory_sections(state)
    evidence_memory, truncated = _format_document_evidence_memory(
        sections,
        max_chars=DOCUMENT_EVIDENCE_MEMORY_MAX_CHARS,
    )
    if evidence_memory:
        payload["prior_evidence_summary"] = evidence_memory
        payload["prior_evidence_round_count"] = len(sections)
        payload["prior_evidence_truncated"] = truncated
    else:
        payload.setdefault("prior_evidence_summary", "")
        payload.setdefault("prior_evidence_round_count", 0)
        payload.setdefault("prior_evidence_truncated", False)
    return payload


def _next_document_evidence_memory(
    state: RoundLoopState,
    observation: dict[str, Any],
) -> list[str]:
    sections = list(_document_evidence_memory_sections(state))
    section = _document_evidence_section(observation)
    if section:
        sections.append(section)
    return sections


def _document_evidence_memory_sections(state: RoundLoopState) -> list[str]:
    raw = state.metadata.get(DOCUMENT_EVIDENCE_MEMORY_KEY)
    if not isinstance(raw, list):
        return []
    sections = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sections.append(item.strip())
    return sections


def _document_evidence_section(observation: dict[str, Any]) -> str | None:
    evidence = observation.get("evidence_summary")
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    round_index = observation.get("round_index")
    header = f"Round {round_index} evidence"
    worker_count = observation.get("worker_count")
    included_count = observation.get("included_count")
    failed_count = observation.get("failed_count")
    stats = []
    if worker_count is not None:
        stats.append(f"workers={worker_count}")
    if included_count is not None:
        stats.append(f"included={included_count}")
    if failed_count is not None:
        stats.append(f"failed={failed_count}")
    if stats:
        header += " (" + ", ".join(stats) + ")"
    return header + ":\n" + evidence.strip()


def _format_document_evidence_memory(
    sections: list[str],
    *,
    max_chars: int,
) -> tuple[str, bool]:
    cleaned = [section.strip() for section in sections if section.strip()]
    if not cleaned or max_chars <= 0:
        return "", bool(cleaned)
    joined = "\n\n".join(cleaned)
    if len(joined) <= max_chars:
        return joined, False

    kept: list[str] = []
    used = 0
    separator = "\n\n"
    prefix = "...[older evidence truncated]\n\n"
    budget = max(0, max_chars - len(prefix))
    for section in reversed(cleaned):
        addition = len(section) + (len(separator) if kept else 0)
        if used + addition > budget:
            break
        kept.append(section)
        used += addition
    if not kept:
        suffix = "\n...[truncated]"
        kept_chars = max(0, max_chars - len(suffix))
        return joined[-kept_chars:].lstrip() + suffix if kept_chars else suffix, True
    kept.reverse()
    return prefix + "\n\n".join(kept), True


def _emit_result(
    task: DocumentTaskItem,
    result: CaseResult,
    case_results: list[CaseResult],
) -> None:
    case_results.append(result)
    if task.result_callback is not None:
        task.result_callback(result)


def _case_success(task: DocumentTaskItem, answer: Any) -> CaseResult:
    return CaseResult(
        case_id=task.case_id,
        answer_key=task.answer_key,
        status=TASK_SUCCESS,
        answer=answer,
        retry_count=task.retry_count,
        metadata=copy.deepcopy(task.metadata),
    )


def _case_failed(task: DocumentTaskItem, error: dict[str, Any]) -> CaseResult:
    task.last_error = copy.deepcopy(error)
    return CaseResult(
        case_id=task.case_id,
        answer_key=task.answer_key,
        status=TASK_FAILED,
        error=copy.deepcopy(error),
        retry_count=task.retry_count,
        metadata=copy.deepcopy(task.metadata),
    )


def _profile_with_summary(profiler: PipelineProfiler) -> dict[str, Any]:
    profile = profiler.to_dict()
    counters = profile.get("counters", {})
    profile["summary"] = {
        "remote_calls": counters.get("supervisor_decompose_calls", 0)
        + counters.get("supervisor_synthesis_calls", 0),
        "remote_token_usage": _remote_token_usage_summary(counters),
        "local_slm_calls": counters.get("local_slm_calls", 0),
        "local_slm_token_usage": _local_slm_token_usage_summary(counters),
    }
    return profile


def _record_remote_token_usage(
    profiler: PipelineProfiler,
    *,
    stage_name: str,
    response_payload: dict[str, Any],
) -> None:
    usage = extract_token_usage(response_payload)
    for key, value in usage.items():
        if value <= 0:
            continue
        profiler.increment(f"{stage_name}_{key}", value)
        profiler.increment(f"remote_{key}", value)


def _remote_token_usage_summary(counters: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(counters.get("remote_input_tokens", 0) or 0),
        "cached_input_tokens": int(counters.get("remote_cached_input_tokens", 0) or 0),
        "output_tokens": int(counters.get("remote_output_tokens", 0) or 0),
        "reasoning_tokens": int(counters.get("remote_reasoning_tokens", 0) or 0),
        "total_tokens": int(counters.get("remote_total_tokens", 0) or 0),
    }


def _record_local_slm_token_usage(
    profiler: PipelineProfiler,
    *,
    execution_result: Any,
) -> None:
    for trace in getattr(execution_result, "traces", []) or []:
        agent_loop = trace.get("agent_loop") if isinstance(trace, dict) else None
        if not isinstance(agent_loop, dict):
            continue
        for step in agent_loop.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            usage = step.get("token_usage")
            if not isinstance(usage, dict):
                continue
            profiler.increment("local_slm_calls")
            for key, value in usage.items():
                amount = int(value or 0)
                if amount <= 0:
                    continue
                profiler.increment(f"local_slm_{key}", amount)


def _local_slm_token_usage_summary(counters: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(counters.get("local_slm_input_tokens", 0) or 0),
        "cached_input_tokens": int(counters.get("local_slm_cached_input_tokens", 0) or 0),
        "output_tokens": int(counters.get("local_slm_output_tokens", 0) or 0),
        "reasoning_tokens": int(counters.get("local_slm_reasoning_tokens", 0) or 0),
        "total_tokens": int(counters.get("local_slm_total_tokens", 0) or 0),
    }


def _truncate_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "\n...[truncated]"
    kept = max(0, max_chars - len(suffix))
    return text[:kept].rstrip() + suffix


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }
