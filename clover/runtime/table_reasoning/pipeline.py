"""Table reasoning compatibility entry point and mixed-runtime helpers."""

from __future__ import annotations

import copy
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from clover.executor import (
    ExecutionPlanBuilder,
    ExecutionResult,
    execute_execution_plan,
)
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    LocalSlmSequenceDispatcher,
)
from clover.executor.result import json_ready
from clover.optimizer.ir import TABLE_REASONING_QUERY_TASK_TYPE
from clover.reasoning_profiles import (
    HINTS_KEY,
    PROFILE_KEY,
    TABLE_REASONING_ANALYZE_PROFILE,
    table_reasoning_profile_from_dsl,
)
from clover.optimizer import (
    SqlParseError,
    extract_sql_statement,
    parse_remote_sql_to_logic_dag,
    parse_sql_list_response,
)
from clover.optimizer import optimize_logic_dag_to_physical_plan
from clover.supervisor import extract_token_usage
from clover.resource import (
    build_table_task_dsl_with_builder_agent,
    prepare_physical_plan_resources,
)
from clover.supervisor import SupervisorAction, SupervisorAgent, SupervisorDecision
from clover.runtime.items import RuntimeCommandItem, RuntimeWorkItem
from clover.runtime.pipeline import (
    CaseResult,
    InflightCallResult,
    PipelineProfiler,
)
from clover.runtime.task import (
    TASK_DAG_READY,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PENDING_REMOTE,
    TASK_SUPERVISOR_REVIEW,
    TASK_RETRYING,
    TASK_SQL_READY,
    TASK_SUCCESS,
    RuntimeCaseSpec,
    TableTaskItem,
    build_runtime_task_items,
    normalize_runtime_case_spec,
)


VALIDATION_NONE = "none"
VALIDATION_REMOTE_SUPERVISOR = "remote_supervisor"
VALIDATION_MODES = frozenset({VALIDATION_NONE, VALIDATION_REMOTE_SUPERVISOR})
_STATIC_ACTION_NO_ANSWER = object()
_STATIC_ACTION_NUMBER_TYPES = frozenset({"number", "float", "integer", "int"})
_STATIC_ACTION_BOOLEAN_TYPES = frozenset({"boolean", "bool"})
_STATIC_ACTION_TEXT_TYPES = frozenset({"string", "entity", "category"})
_ACTION_FULL_RESULT_KEY = "_clover_full_res"


@dataclass
class TableReasoningCaseSpec(RuntimeCaseSpec):
    """One table reasoning case submitted to the batching runtime."""

    pass


TaskItem = TableTaskItem


@dataclass
class SqlItem(RuntimeCommandItem[TaskItem]):
    """SQL ready for deterministic Optimizer parsing."""

    statements: tuple[str, ...] = ()

    @property
    def sql(self) -> str:
        return self.content

    @property
    def sqls(self) -> tuple[str, ...]:
        return self.statements or (self.content,)


@dataclass
class LogicDagItem(RuntimeWorkItem[TaskItem]):
    """One per-question Logic DAG queued for same-table merging."""

    statements: tuple[str, ...] = ()

    @property
    def sql(self) -> str:
        return self.command_output

    @property
    def sqls(self) -> tuple[str, ...]:
        return self.statements or (self.command_output,)


@dataclass
class _RemoteDecomposeJob:
    batch: list[TaskItem]
    remote_dsl: dict[str, Any]


@dataclass(frozen=True)
class _PlannedSqlCommand:
    op: str
    sqls: tuple[str, ...]
    answer: Any = None
    actions: tuple[SupervisorAction, ...] = ()


@dataclass
class ActionGroupItem:
    """One analyze action group executed locally before Supervisor report."""

    task: TaskItem
    actions: tuple[SupervisorAction, ...]


@dataclass(frozen=True)
class _ActionGroupRunResult:
    order_index: int
    item: ActionGroupItem
    observation: dict[str, Any] | None
    profiler: PipelineProfiler
    elapsed: float
    error: Exception | None = None


@dataclass(frozen=True)
class _ActionReportRunResult:
    order_index: int
    item: ActionGroupItem
    observation: dict[str, Any]
    supervisor_result: Any | None
    profiler: PipelineProfiler
    error: Exception | None = None


@dataclass(frozen=True)
class TableBuilderJob:
    """One table case whose DSL must be materialized inside the runtime."""

    spec: TableReasoningCaseSpec
    result_callback: Callable[[CaseResult], None] | None = None


@dataclass
class TableReasoningSystemResult:
    """Completed table reasoning runtime output."""

    case_results: list[CaseResult]
    task_items: dict[str, TaskItem]
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_results": [item.to_dict() for item in self.case_results],
            "profile": self.profile,
        }


def run_table_reasoning_system(
    *,
    case_specs: list[TableReasoningCaseSpec | dict[str, Any]],
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    remote_batch_size: int = 64,
    remote_concurrency: int = 64,
    max_parallel_execution_units: int = 64,
    max_parallel_slm_node_jobs: int = DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    max_parallel_slm_sequences: int = DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    max_pending_slm_sequences: int = DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    max_retries: int = 1,
    client: Any | None = None,
    table_cache: dict[str, Any] | None = None,
    case_result_callback: Callable[[CaseResult], None] | None = None,
    profile_baseline: bool = False,
    validation_mode: str = VALIDATION_NONE,
) -> TableReasoningSystemResult:
    """Run table reasoning cases through the batching pipeline."""

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
    validation_mode = _normalize_validation_mode(validation_mode)

    from clover.runtime.mixed_reasoning.pipeline import run_mixed_reasoning_system

    mixed_result = run_mixed_reasoning_system(
        table_case_specs=case_specs,
        remote_config=remote_config,
        synthesize_config=synthesize_config,
        local_slm_config=local_slm_config,
        remote_batch_size=remote_batch_size,
        remote_concurrency=remote_concurrency,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        max_retries=max_retries,
        client=client,
        table_cache=table_cache,
        case_result_callback=case_result_callback,
        profile_baseline=profile_baseline,
        validation_mode=validation_mode,
    )
    return TableReasoningSystemResult(
        case_results=mixed_result.case_results,
        task_items=mixed_result.table_task_items,
        profile=mixed_result.profile,
    )


def _build_task_items(
    case_specs: list[TableReasoningCaseSpec | dict[str, Any]],
    *,
    case_result_callback: Callable[[CaseResult], None] | None = None,
) -> dict[str, TaskItem]:
    if any(_builder_payload_from_raw_spec(spec) is not None for spec in case_specs):
        raise ValueError(
            "Table builder specs must be materialized by the runtime builder stage"
        )
    return build_runtime_task_items(
        case_specs,
        task_type=TABLE_REASONING_QUERY_TASK_TYPE,
        case_spec_class=TableReasoningCaseSpec,
        task_item_class=TaskItem,
        case_result_callback=case_result_callback,
    )


def _split_table_builder_specs(
    case_specs: list[TableReasoningCaseSpec | dict[str, Any]],
    *,
    case_result_callback: Callable[[CaseResult], None] | None = None,
) -> tuple[list[TableBuilderJob], list[TableReasoningCaseSpec | dict[str, Any]]]:
    builder_jobs: list[TableBuilderJob] = []
    ready_specs: list[TableReasoningCaseSpec | dict[str, Any]] = []
    for raw_spec in case_specs:
        if _builder_payload_from_raw_spec(raw_spec) is None:
            ready_specs.append(raw_spec)
            continue
        spec = normalize_runtime_case_spec(raw_spec, TableReasoningCaseSpec)
        builder_jobs.append(
            TableBuilderJob(
                spec=spec,
                result_callback=case_result_callback,
            )
        )
    return builder_jobs, ready_specs


def _builder_payload_from_raw_spec(
    raw_spec: TableReasoningCaseSpec | dict[str, Any],
) -> dict[str, Any] | None:
    value = (
        raw_spec.get("builder")
        if isinstance(raw_spec, dict)
        else getattr(raw_spec, "builder", None)
    )
    if value is None:
        metadata = (
            raw_spec.get("metadata", {})
            if isinstance(raw_spec, dict)
            else getattr(raw_spec, "metadata", {})
        )
        if isinstance(metadata, dict):
            value = metadata.get("builder")
    return value if isinstance(value, dict) else None


def _run_table_builder_job(
    *,
    job: TableBuilderJob,
    local_slm_config: dict[str, Any] | None,
    slm_client: Any | None,
) -> TaskItem:
    if local_slm_config is None:
        raise ValueError("Table BuilderAgent requires local_slm_config")
    spec = normalize_runtime_case_spec(job.spec, TableReasoningCaseSpec)
    builder = spec.builder or spec.metadata.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("Table builder spec requires builder payload")
    question = _required_builder_string(builder, "question")
    source_file = str(builder.get("source_file") or "table.csv")
    table_path = _builder_table_path(spec.base_dir, builder, source_file=source_file)
    answer_type = builder.get("answer_type")
    if answer_type is not None:
        answer_type = str(answer_type)
    task_type = str(builder.get("task_type") or "table_reasoning.analyze")
    source_id = _builder_int(builder.get("source_id"), fallback=0)

    builder_result = build_table_task_dsl_with_builder_agent(
        question=question,
        table_path=table_path,
        source_file=source_file,
        answer_type=answer_type,
        task_type=task_type,
        source_id=source_id,
        slm_config=local_slm_config,
        client=slm_client,
    )
    task_dsl = _table_builder_task_dsl(builder_result.task_dsl, builder)
    dsl_builder_metadata = _table_builder_metadata(
        builder_result=builder_result,
        task_dsl=task_dsl,
    )
    metadata = copy.deepcopy(spec.metadata)
    metadata.pop("builder", None)
    metadata["dsl_builder"] = dsl_builder_metadata
    metadata["task_answer_type"] = task_dsl.get("answer", {}).get("type")
    built_spec = TableReasoningCaseSpec(
        case_id=spec.case_id,
        task_dsl=task_dsl,
        base_dir=spec.base_dir,
        metadata=metadata,
        answer_key=spec.answer_key,
    )
    task_items = build_runtime_task_items(
        [built_spec],
        task_type=TABLE_REASONING_QUERY_TASK_TYPE,
        case_spec_class=TableReasoningCaseSpec,
        task_item_class=TaskItem,
        case_result_callback=job.result_callback,
    )
    return next(iter(task_items.values()))


def _builder_table_path(
    base_dir: str | Path,
    builder: dict[str, Any],
    *,
    source_file: str,
) -> Path:
    value = builder.get("table_path") or source_file
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base_dir).expanduser() / path
    return path.resolve()


def _required_builder_string(builder: dict[str, Any], key: str) -> str:
    value = builder.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Table builder payload requires non-empty {key}")
    return value.strip()


def _builder_int(value: Any, *, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Table builder source_id must be an integer: {value!r}") from exc


def _table_builder_task_dsl(
    task_dsl: dict[str, Any],
    builder: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(task_dsl)
    hints = builder.get("hints")
    if isinstance(hints, dict) and hints:
        updated["hints"] = copy.deepcopy(hints)
    updated["task_type"] = str(builder.get("task_type") or updated.get("task_type"))
    for key in ("profile", "metadata", "reasoning_profile", "reasoning_context"):
        updated.pop(key, None)
    return updated


def _table_builder_metadata(
    *,
    builder_result: Any,
    task_dsl: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": builder_result.builder_mode,
        "tool_call": builder_result.tool_call,
        "diagnostics": builder_result.diagnostics,
        "raw_output": builder_result.raw_output,
        "parsed_output": builder_result.parsed_output,
        "prompt_chars": len(builder_result.prompt),
        "token_usage": extract_token_usage(builder_result.response_payload),
        "task_answer_type": task_dsl.get("answer", {}).get("type"),
    }


def _record_table_builder_usage(
    profiler: PipelineProfiler,
    task: TaskItem,
) -> None:
    profiler.increment("table_builder_cases")
    builder = task.metadata.get("dsl_builder") or {}
    if builder.get("mode") != "builder_agent":
        return
    usage = builder.get("token_usage")
    if not isinstance(usage, dict):
        return
    profiler.increment("builder_agent_calls")
    profiler.increment("local_slm_calls")
    for key, value in usage.items():
        try:
            amount = int(value or 0)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        profiler.increment(f"builder_agent_{key}", amount)
        profiler.increment(f"local_slm_{key}", amount)


def _normalize_validation_mode(value: str) -> str:
    mode = str(value or VALIDATION_NONE).strip().lower()
    if mode not in VALIDATION_MODES:
        available = ", ".join(sorted(VALIDATION_MODES))
        raise ValueError(
            f"Unsupported table validation mode: {value!r}. Available: {available}"
        )
    return mode


def _initialize_table_entries(
    *,
    task_items: dict[str, TaskItem],
    pending_remote: list[TaskItem],
    sql_items: list[SqlItem],
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> None:
    for task in task_items.values():
        try:
            command = _local_entry_command(task)
        except Exception as exc:  # noqa: BLE001 - local entry is task scoped.
            _finalize_failed(
                task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(exc),
            )
            continue
        if command is None:
            task.status = TASK_PENDING_REMOTE
            pending_remote.append(task)
            continue
        _enqueue_planned_table_command(
            task=task,
            command=command,
            sql_items=sql_items,
            action_items=action_items,
            final_results=final_results,
            finalized=finalized,
        )
        profiler.increment("local_entry_commands")


def _local_entry_command(task: TaskItem) -> _PlannedSqlCommand | None:
    """Return an explicit local entry command embedded in the task, if any."""

    source = task.task_dsl
    for key in ("command", "action"):
        value = source.get(key)
        if isinstance(value, dict):
            return _planned_command_from_payload(value, task)
        if isinstance(value, str) and value.strip():
            return _planned_command_from_payload({"q": value}, task)
    if any(key in source for key in ("acts", "actions", "op", "q", "sql", "sqls")):
        return _planned_command_from_payload(source, task)
    source = task.remote_dsl
    if any(key in source for key in ("acts", "actions", "op", "q", "sql", "sqls")):
        return _planned_command_from_payload(source, task)
    return None


def _planned_command_from_payload(
    payload: dict[str, Any],
    task: TaskItem,
) -> _PlannedSqlCommand:
    payload = _unwrap_next_table_action(dict(payload))
    action_op = _payload_action_op(payload)
    if action_op == "answer":
        if "a" not in payload:
            raise SqlParseError("Local table answer command requires a")
        return _PlannedSqlCommand(op="answer", sqls=(), answer=payload.get("a"))
    actions = _normalize_table_actions(payload)
    if actions:
        sqls = tuple(action.q for action in actions if action.op == "sql" and action.q)
        if _is_analyze_task(task):
            return _PlannedSqlCommand(op="acts", sqls=sqls, actions=actions)
        if len(actions) != 1 or actions[0].op != "sql" or len(sqls) != 1:
            raise SqlParseError("Local table query command requires one SQL action")
        return _PlannedSqlCommand(op="sql", sqls=sqls)
    raise SqlParseError("Local table command must include sql, q, acts, or answer")


def _enqueue_planned_table_command(
    *,
    task: TaskItem,
    command: _PlannedSqlCommand,
    sql_items: list[SqlItem],
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
) -> None:
    if command.op == "answer":
        _finalize_success(
            task,
            answer=command.answer,
            final_results=final_results,
            finalized=finalized,
        )
        return
    if command.actions and _is_analyze_task(task):
        task.status = TASK_EXECUTING
        task.current_command = _command_content(command)
        action_items.append(ActionGroupItem(task=task, actions=command.actions))
        return
    task.status = TASK_SQL_READY
    task.current_command = _command_content(command)
    sql_items.append(
        SqlItem(
            task=task,
            content=_command_content(command),
            content_type="sql",
            statements=command.sqls,
        )
    )


def _finish_remote_decompose_job(
    *,
    job: _RemoteDecomposeJob,
    call_result: InflightCallResult[Any],
    sql_items: list[SqlItem],
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> None:
    if call_result.error is not None:
        for task in job.batch:
            _finalize_failed(
                task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(call_result.error),
        )
        return

    result = call_result.value
    try:
        profiler.increment("supervisor_decompose_calls")
        _record_remote_token_usage(
            profiler,
            stage_name="supervisor_decompose",
            response_payload=result.response_payload,
        )
        parsed = _parse_remote_decompose_output(result.command or "", job)
        for task, command in zip(job.batch, parsed, strict=True):
            _enqueue_planned_table_command(
                task=task,
                command=command,
                sql_items=sql_items,
                action_items=action_items,
                final_results=final_results,
                finalized=finalized,
            )
    except Exception as exc:  # noqa: BLE001 - batch-level parse/protocol failure.
        for task in job.batch:
            _finalize_failed(
                task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(exc),
            )


def _parse_remote_decompose_output(
    remote_output: str,
    job: _RemoteDecomposeJob,
) -> tuple[_PlannedSqlCommand, ...]:
    if _is_batch_remote_dsl(job.remote_dsl):
        parsed = parse_sql_list_response(remote_output, job.remote_dsl)
        return tuple(
            _PlannedSqlCommand(op="sql", sqls=(sql,))
            for sql in parsed.sqls
        )
    return (_parse_single_table_command(remote_output, job),)


def _parse_single_table_command(
    remote_output: str,
    job: _RemoteDecomposeJob,
) -> _PlannedSqlCommand:
    payload = _load_json_object(remote_output)
    payload = _unwrap_next_table_action(payload)
    if "final" in payload:
        raise SqlParseError("Remote table command must not include final")
    action_op = _payload_action_op(payload)
    if action_op and action_op not in {"sql", "inspect", "analyze", "answer"}:
        raise SqlParseError(f"Unsupported remote table action op: {action_op}")
    if action_op == "answer":
        if "a" not in payload:
            raise SqlParseError("Remote answer action requires a")
        return _PlannedSqlCommand(
            op="answer",
            sqls=(),
            answer=payload.get("a"),
        )
    analyze = _is_analyze_remote_dsl(job.remote_dsl)
    actions = _normalize_table_actions(payload)
    if actions:
        sqls = tuple(action.q for action in actions if action.op == "sql" and action.q)
        if not analyze:
            if len(actions) != 1 or actions[0].op != "sql" or len(sqls) != 1:
                raise SqlParseError("Remote table query action requires one SQL")
            return _PlannedSqlCommand(
                op="sql",
                sqls=sqls,
            )
        return _PlannedSqlCommand(
            op="acts",
            sqls=sqls,
            actions=actions,
        )

    raise SqlParseError("Remote table command must include sql, q, or answer")


def _unwrap_next_table_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept common wrappers but keep ReAct execution to one next action."""

    for key in ("steps", "plan"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                return _unwrap_next_table_action(first)
    return payload


def _payload_action_op(payload: dict[str, Any]) -> str:
    value = payload.get("op")
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _load_json_object(remote_output: str) -> dict[str, Any]:
    text = str(remote_output or "").strip()
    if not text:
        raise SqlParseError("Remote table command is empty")
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _load_first_json_object(text, exc)
    if not isinstance(payload, dict):
        raise SqlParseError("Remote table command must be a JSON object")
    return payload


def _load_first_json_object(text: str, original_error: json.JSONDecodeError) -> Any:
    start = text.find("{")
    if start < 0:
        raise SqlParseError(f"Unable to parse table command JSON: {original_error}") from original_error
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise SqlParseError(f"Unable to parse table command JSON: {exc}") from exc
    raise SqlParseError(f"Unable to parse table command JSON: {original_error}") from original_error


def _normalize_sqls(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (extract_sql_statement(value),)
    if not isinstance(value, list):
        raise SqlParseError("sqls must be a list of SQL strings")
    sqls: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            if "final" in item:
                raise SqlParseError(f"sqls[{index}] must not include final")
            item = item.get("sql")
        if not isinstance(item, str) or not item.strip():
            raise SqlParseError(f"sqls[{index}] must be a non-empty SQL string")
        sqls.append(extract_sql_statement(item))
    return tuple(sqls)


def _normalize_table_actions(payload: dict[str, Any]) -> tuple[SupervisorAction, ...]:
    action_list = payload.get("acts", payload.get("actions"))
    if action_list is not None:
        if not isinstance(action_list, list) or not action_list:
            raise SqlParseError("acts must be a non-empty list")
        actions: list[SupervisorAction] = []
        for index, item in enumerate(action_list):
            if not isinstance(item, dict):
                raise SqlParseError(f"acts[{index}] must be an object")
            actions.extend(_normalize_one_table_action(item, label=f"acts[{index}]"))
        return tuple(actions)
    action_op = _payload_action_op(payload)
    if action_op in {"sql", "inspect", "analyze"}:
        return tuple(_normalize_one_table_action(payload, label="action"))
    if any(key in payload for key in ("q", "sql", "sqls")):
        return tuple(_normalize_one_table_action({"op": "sql", **payload}, label="action"))
    return ()


def _normalize_one_table_action(
    payload: dict[str, Any],
    *,
    label: str,
) -> tuple[SupervisorAction, ...]:
    action_op = _payload_action_op(payload) or "sql"
    if action_op == "sql":
        sqls = _normalize_sqls(payload.get("q", payload.get("sqls", payload.get("sql"))))
        if not sqls:
            raise SqlParseError(f"{label} requires SQL q")
        return tuple(SupervisorAction(op="sql", q=sql) for sql in sqls)
    if action_op == "inspect":
        question = payload.get("q", payload.get("ask", payload.get("task")))
        if not isinstance(question, str) or not question.strip():
            raise SqlParseError(f"{label} inspect requires non-empty q")
        seed = payload.get("seed")
        seed_sql = None
        if seed is not None and seed != "":
            seed_sqls = _normalize_sqls(seed)
            if len(seed_sqls) != 1:
                raise SqlParseError(f"{label} inspect seed requires one SQL")
            seed_sql = seed_sqls[0]
        return (SupervisorAction(op="inspect", q=question.strip(), seed=seed_sql),)
    if action_op == "analyze":
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise SqlParseError(f"{label} analyze requires non-empty kind")
        seed_sqls = _normalize_sqls(payload.get("seed"))
        if len(seed_sqls) != 1:
            raise SqlParseError(f"{label} analyze requires one seed SQL")
        return (
            SupervisorAction(
                op="analyze",
                seed=seed_sqls[0],
                kind=kind.strip().lower(),
            ),
        )
    raise SqlParseError(f"Unsupported remote table action op: {action_op}")


def _command_content(command: _PlannedSqlCommand) -> str:
    if command.op == "answer":
        return json.dumps(
            {"op": "answer", "a": command.answer},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if command.actions:
        return json.dumps(
            {"acts": [_action_to_dict(action) for action in command.actions]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if len(command.sqls) == 1:
        return command.sqls[0]
    return json.dumps(list(command.sqls), ensure_ascii=False, separators=(",", ":"))


def _action_to_dict(action: SupervisorAction) -> dict[str, Any]:
    payload = {"op": action.op}
    if action.kind:
        payload["kind"] = action.kind
    if action.q is not None:
        payload["q"] = action.q
    if action.seed:
        payload["seed"] = action.seed
    return payload


def _is_batch_remote_dsl(remote_dsl: dict[str, Any]) -> bool:
    return isinstance(remote_dsl.get("questions"), list)


def _is_analyze_remote_dsl(remote_dsl: dict[str, Any]) -> bool:
    return (
        table_reasoning_profile_from_dsl(remote_dsl)
        == TABLE_REASONING_ANALYZE_PROFILE
    )


def _run_optimizer_parse(
    *,
    sql_items: list[SqlItem],
    dag_queue: Any,
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> None:
    pending = list(sql_items)
    sql_items.clear()
    with profiler.measure("optimizer_parse", items=len(pending)):
        for item in pending:
            try:
                task = item.task
                logic_dag = _parse_sql_item_to_logic_dag(item)
                task.status = TASK_DAG_READY
                dag_queue.push(
                    _dag_group_key(task),
                    LogicDagItem(
                        task=task,
                        command_output=_command_content(
                            _PlannedSqlCommand(
                                op="sql",
                                sqls=item.sqls,
                            )
                        ),
                        output_type="sql",
                        logic_dag=logic_dag,
                        statements=item.sqls,
                    ),
                    priority=task.retry_count,
                )
            except SqlParseError as exc:
                _finalize_failed(
                    item.task,
                    final_results=final_results,
                    finalized=finalized,
                    error=_error_payload(exc),
                )


def _parse_sql_item_to_logic_dag(item: SqlItem) -> dict[str, Any]:
    return parse_remote_sql_to_logic_dag(item.sql, _query_remote_dsl(item.task))


@dataclass(frozen=True)
class _SqlActionExecution:
    ok: bool
    value: Any = None
    frame: pd.DataFrame | None = None
    error: dict[str, Any] | None = None


def _run_one_action_group(
    *,
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    supervisor: SupervisorAgent,
    max_retries: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    validation_mode: str,
    profiler: PipelineProfiler,
    remote_concurrency: int = 1,
) -> bool:
    if not action_items:
        return False
    item = action_items.pop(0)
    followups: list[ActionGroupItem] = []
    ran = _run_action_groups_batch(
        action_items=[item],
        followup_action_items=followups,
        final_results=final_results,
        finalized=finalized,
        supervisor=supervisor,
        max_retries=max_retries,
        remote_concurrency=remote_concurrency,
        table_cache=table_cache,
        local_slm_config=local_slm_config,
        local_slm_dispatcher=local_slm_dispatcher,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        validation_mode=validation_mode,
        profiler=profiler,
    )
    action_items.extend(followups)
    return ran


def _run_action_groups_batch(
    *,
    action_items: list[ActionGroupItem],
    followup_action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    supervisor: SupervisorAgent,
    max_retries: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    validation_mode: str,
    profiler: PipelineProfiler,
    remote_concurrency: int = 1,
) -> bool:
    del validation_mode
    runnable = [
        item for item in action_items if item.task.answer_key not in finalized
    ]
    if not runnable:
        return False

    for item in runnable:
        item.task.status = TASK_EXECUTING

    max_workers = max(1, min(max_parallel_slm_node_jobs, len(runnable)))
    if max_workers == 1:
        results = [
            _execute_action_group_for_batch(
                order_index=index,
                item=item,
                table_cache=table_cache,
                local_slm_config=local_slm_config,
                local_slm_dispatcher=local_slm_dispatcher,
                max_parallel_execution_units=max_parallel_execution_units,
                max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                max_parallel_slm_sequences=max_parallel_slm_sequences,
                max_pending_slm_sequences=max_pending_slm_sequences,
            )
            for index, item in enumerate(runnable)
        ]
    else:
        results_by_index: dict[int, _ActionGroupRunResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _execute_action_group_for_batch,
                    order_index=index,
                    item=item,
                    table_cache=table_cache,
                    local_slm_config=local_slm_config,
                    local_slm_dispatcher=local_slm_dispatcher,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                )
                for index, item in enumerate(runnable)
            ]
            for future in as_completed(futures):
                result = future.result()
                results_by_index[result.order_index] = result
        results = [results_by_index[index] for index in range(len(runnable))]

    report_items: list[tuple[ActionGroupItem, dict[str, Any]]] = []
    for result in results:
        _merge_pipeline_profiler(profiler, result.profiler)
        profiler.record(
            "action_group",
            items=len(result.item.actions),
            elapsed=result.elapsed,
        )
        profiler.increment("action_group_calls")
        profiler.increment("action_group_actions", len(result.item.actions))
        if result.item.task.answer_key in finalized:
            continue
        if result.error is not None:
            _finalize_failed(
                result.item.task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(result.error),
            )
            continue
        observation = result.observation
        if observation is None:
            _finalize_failed(
                result.item.task,
                final_results=final_results,
                finalized=finalized,
                error={
                    "type": "ActionGroupExecutionError",
                    "message": "action group produced no observation",
                },
            )
            continue
        if _try_finalize_static_action_group_answer(
            item=result.item,
            observation=observation,
            final_results=final_results,
            finalized=finalized,
            profiler=profiler,
        ):
            continue
        report_items.append((result.item, _public_action_group_observation(observation)))
    if report_items:
        _run_supervisor_action_reports_batch(
            report_items=report_items,
            action_items=followup_action_items,
            final_results=final_results,
            finalized=finalized,
            supervisor=supervisor,
            max_retries=max_retries,
            remote_concurrency=remote_concurrency,
            profiler=profiler,
        )
    return True


def _execute_action_group_for_batch(
    *,
    order_index: int,
    item: ActionGroupItem,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
) -> _ActionGroupRunResult:
    local_profiler = PipelineProfiler()
    started = time.perf_counter()
    try:
        observation = _execute_table_action_group(
            task=item.task,
            actions=item.actions,
            table_cache=table_cache,
            local_slm_config=local_slm_config,
            local_slm_dispatcher=local_slm_dispatcher,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            profiler=local_profiler,
        )
        return _ActionGroupRunResult(
            order_index=order_index,
            item=item,
            observation=observation,
            profiler=local_profiler,
            elapsed=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - reported per task below.
        return _ActionGroupRunResult(
            order_index=order_index,
            item=item,
            observation=None,
            profiler=local_profiler,
            elapsed=time.perf_counter() - started,
            error=exc,
        )


def _merge_pipeline_profiler(
    target: PipelineProfiler,
    child: PipelineProfiler,
) -> None:
    for stage_name, stage in child.stages.items():
        target.record(
            stage_name,
            items=stage.items,
            elapsed=stage.total_seconds,
        )
    for counter_name, amount in child.counters.items():
        target.increment(counter_name, amount)


def _try_finalize_static_action_group_answer(
    *,
    item: ActionGroupItem,
    observation: dict[str, Any],
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> bool:
    answer = _static_action_group_scalar_answer(item=item, observation=observation)
    if answer is _STATIC_ACTION_NO_ANSWER:
        return False
    profiler.increment("action_group_static_answer_hits")
    answer_type = _answer_type_name(item.task)
    if answer_type in _STATIC_ACTION_NUMBER_TYPES:
        profiler.increment("action_group_static_answer_number_hits")
    elif answer_type in _STATIC_ACTION_BOOLEAN_TYPES:
        profiler.increment("action_group_static_answer_boolean_hits")
    elif answer_type in _STATIC_ACTION_TEXT_TYPES:
        profiler.increment("action_group_static_answer_text_hits")
    _finalize_success(
        item.task,
        answer=answer,
        final_results=final_results,
        finalized=finalized,
    )
    return True


def _static_action_group_scalar_answer(
    *,
    item: ActionGroupItem,
    observation: dict[str, Any],
) -> Any:
    if len(item.actions) != 1 or item.actions[0].op != "sql":
        return _STATIC_ACTION_NO_ANSWER
    answer_type = _answer_type_name(item.task)
    allowed_answer_types = (
        _STATIC_ACTION_NUMBER_TYPES
        | _STATIC_ACTION_BOOLEAN_TYPES
        | _STATIC_ACTION_TEXT_TYPES
    )
    if answer_type not in allowed_answer_types:
        return _STATIC_ACTION_NO_ANSWER
    obs = observation.get("obs")
    if not isinstance(obs, list) or len(obs) != 1:
        return _STATIC_ACTION_NO_ANSWER
    sql_observation = obs[0]
    if (
        not isinstance(sql_observation, dict)
        or sql_observation.get("op") != "sql"
        or sql_observation.get("ok") is not True
    ):
        return _STATIC_ACTION_NO_ANSWER
    sql = item.actions[0].q or ""
    result_value = sql_observation.get(
        _ACTION_FULL_RESULT_KEY,
        sql_observation.get("res"),
    )
    cell = _STATIC_ACTION_NO_ANSWER
    if answer_type in _STATIC_ACTION_NUMBER_TYPES:
        cell = _numeric_action_result_reduce_answer(
            result_value,
            task=item.task,
            sql=sql,
        )
    if cell is _STATIC_ACTION_NO_ANSWER:
        cell = _targeted_action_result_answer(
            result_value,
            task=item.task,
            sql=sql,
        )
    if cell is _STATIC_ACTION_NO_ANSWER:
        cell = _single_answerish_cell(
            result_value,
            answer_key=item.task.answer_key,
        )
    if cell is _STATIC_ACTION_NO_ANSWER:
        return _STATIC_ACTION_NO_ANSWER
    normalized = _normalize_answer_for_task(item.task, cell)
    if answer_type in _STATIC_ACTION_NUMBER_TYPES:
        if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
            try:
                if pd.isna(normalized):
                    return _STATIC_ACTION_NO_ANSWER
            except TypeError:
                pass
            return normalized
        return _STATIC_ACTION_NO_ANSWER
    if isinstance(normalized, bool):
        return normalized
    if answer_type in _STATIC_ACTION_TEXT_TYPES:
        if normalized is None:
            return _STATIC_ACTION_NO_ANSWER
        if isinstance(normalized, str) and normalized.strip():
            return normalized
    return _STATIC_ACTION_NO_ANSWER


def _public_action_group_observation(observation: dict[str, Any]) -> dict[str, Any]:
    # Build a safe copy without the full-result payloads (typically large
    # DataFrames).  A full deepcopy would double-copy those large values even
    # though they are immediately discarded.
    public: dict[str, Any] = {}
    for key, value in observation.items():
        if key == "obs" and isinstance(value, list):
            public[key] = [
                {k: v for k, v in item.items() if k != _ACTION_FULL_RESULT_KEY}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            public[key] = value
    return public


def _targeted_action_result_answer(
    value: Any,
    *,
    task: TaskItem,
    sql: str,
) -> Any:
    rows, columns = _table_rows_and_columns(value)
    if not rows:
        return _STATIC_ACTION_NO_ANSWER
    answer_type = _answer_type_name(task)
    target_column = _task_target_column(task)
    selected_column = _matching_column(columns, target_column) if target_column else None
    if len(rows) == 1:
        row = rows[0]
        if selected_column is not None:
            return _row_column_value(row, selected_column, columns)
        if len(columns) == 1:
            return _row_column_value(row, columns[0], columns)
        return _STATIC_ACTION_NO_ANSWER
    if answer_type not in _STATIC_ACTION_TEXT_TYPES:
        return _STATIC_ACTION_NO_ANSWER
    if _sql_has_limit_one(sql):
        return _STATIC_ACTION_NO_ANSWER
    if selected_column is not None:
        ranked = _targeted_metric_rank_answer(
            rows=rows,
            columns=columns,
            target_column=selected_column,
            task=task,
        )
        if ranked is not _STATIC_ACTION_NO_ANSWER:
            return ranked
    if len(columns) != 1:
        return _STATIC_ACTION_NO_ANSWER
    if selected_column is None:
        selected_column = columns[0]
    values = [
        _row_column_value(row, selected_column, columns)
        for row in rows
    ]
    text_values = [
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ]
    if len(text_values) != len(rows):
        return _STATIC_ACTION_NO_ANSWER
    return ", ".join(text_values)


def _targeted_metric_rank_answer(
    *,
    rows: list[Any],
    columns: list[str],
    target_column: str,
    task: TaskItem,
) -> Any:
    direction = _question_rank_direction(task)
    if direction is None:
        return _STATIC_ACTION_NO_ANSWER
    division_columns = _division_columns_from_question(
        task=task,
        columns=columns,
        target_column=target_column,
    )
    if division_columns is not None:
        numerator_column, denominator_column = division_columns
        scored = _score_rows_by_division(
            rows=rows,
            columns=columns,
            target_column=target_column,
            numerator_column=numerator_column,
            denominator_column=denominator_column,
        )
    else:
        score_column = _metric_column_from_question(
            task=task,
            rows=rows,
            columns=columns,
            target_column=target_column,
        )
        if score_column is None:
            return _STATIC_ACTION_NO_ANSWER
        scored = _score_rows_by_column(
            rows=rows,
            columns=columns,
            target_column=target_column,
            score_column=score_column,
        )
    if not scored:
        return _STATIC_ACTION_NO_ANSWER
    scores = [score for _, score in scored]
    best_score = max(scores) if direction == "max" else min(scores)
    best_targets = [
        target
        for target, score in scored
        if score == best_score
    ]
    if len(best_targets) != 1:
        return _STATIC_ACTION_NO_ANSWER
    target = best_targets[0]
    if target is None:
        return _STATIC_ACTION_NO_ANSWER
    text = str(target).strip()
    return text if text else _STATIC_ACTION_NO_ANSWER


def _question_rank_direction(task: TaskItem) -> str | None:
    question = str(task.question or "")
    has_max = bool(
        re.search(
            r"\b(highest|largest|greatest|maximum|max|most|top)\b",
            question,
            flags=re.IGNORECASE,
        )
    )
    has_min = bool(
        re.search(
            r"\b(lowest|smallest|minimum|min|least|bottom)\b",
            question,
            flags=re.IGNORECASE,
        )
    )
    if has_max == has_min:
        return None
    return "max" if has_max else "min"


def _division_columns_from_question(
    *,
    task: TaskItem,
    columns: list[str],
    target_column: str,
) -> tuple[str, str] | None:
    question = str(task.question or "")
    match = re.search(r"\bdivided\s+by\b", question, flags=re.IGNORECASE)
    if match is None:
        return None
    candidates = [column for column in columns if column != target_column]
    before = question[: match.start()]
    defined_matches = list(re.finditer(r"\bdefined\s+as\b", before, flags=re.IGNORECASE))
    if defined_matches:
        before = before[defined_matches[-1].end() :]
    after = question[match.end() :]
    after = re.split(r"[.;?]", after, maxsplit=1)[0]
    numerator = _unique_mentioned_column(before, candidates)
    denominator = _unique_mentioned_column(after, candidates)
    if numerator is None or denominator is None or numerator == denominator:
        return None
    return numerator, denominator


def _metric_column_from_question(
    *,
    task: TaskItem,
    rows: list[Any],
    columns: list[str],
    target_column: str,
) -> str | None:
    question = str(task.question or "")
    candidates = [
        column
        for column in columns
        if column != target_column and _numeric_values_available(rows, columns, column)
    ]
    if re.search(r"\bdivided\s+by\b", question, flags=re.IGNORECASE):
        return None
    if re.search(r"\bper\b", question, flags=re.IGNORECASE):
        candidates = [
            column
            for column in candidates
            if "per" in _compact_text(column)
        ]
    return _unique_mentioned_column(question, candidates)


def _unique_mentioned_column(text: str, columns: list[str]) -> str | None:
    matches = [
        column
        for column in columns
        if _text_mentions_column(text, column)
    ]
    return matches[0] if len(matches) == 1 else None


def _text_mentions_column(text: str, column: str) -> bool:
    compact_column = _compact_text(column)
    if not compact_column:
        return False
    compact_text = _compact_text(text)
    return compact_column in compact_text


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _numeric_values_available(rows: list[Any], columns: list[str], column: str) -> bool:
    return all(
        _coerce_static_number(_row_column_value(row, column, columns)) is not None
        for row in rows
    )


def _score_rows_by_column(
    *,
    rows: list[Any],
    columns: list[str],
    target_column: str,
    score_column: str,
) -> list[tuple[Any, float]]:
    scored: list[tuple[Any, float]] = []
    for row in rows:
        score = _coerce_static_number(_row_column_value(row, score_column, columns))
        if score is None:
            return []
        scored.append((_row_column_value(row, target_column, columns), score))
    return scored


def _score_rows_by_division(
    *,
    rows: list[Any],
    columns: list[str],
    target_column: str,
    numerator_column: str,
    denominator_column: str,
) -> list[tuple[Any, float]]:
    scored: list[tuple[Any, float]] = []
    for row in rows:
        numerator = _coerce_static_number(
            _row_column_value(row, numerator_column, columns)
        )
        denominator = _coerce_static_number(
            _row_column_value(row, denominator_column, columns)
        )
        if numerator is None or denominator in (None, 0):
            return []
        scored.append((_row_column_value(row, target_column, columns), numerator / denominator))
    return scored


def _numeric_action_result_reduce_answer(
    value: Any,
    *,
    task: TaskItem,
    sql: str,
) -> Any:
    rows, columns = _table_rows_and_columns(value)
    if len(rows) != 1:
        return _STATIC_ACTION_NO_ANSWER
    if not _question_asks_average_across_range(task) or not _sql_uses_avg(sql):
        return _STATIC_ACTION_NO_ANSWER
    if len(columns) == 1:
        divisor = _avg_summed_identifier_count(sql)
        if divisor <= 1:
            return _STATIC_ACTION_NO_ANSWER
        number = _coerce_static_number(_row_column_value(rows[0], columns[0], columns))
        if number is None:
            return _STATIC_ACTION_NO_ANSWER
        return number / divisor
    if len(columns) <= 1:
        return _STATIC_ACTION_NO_ANSWER
    numbers = _numeric_values_from_row(rows[0], columns)
    if numbers is None or len(numbers) != len(columns):
        return _STATIC_ACTION_NO_ANSWER
    return sum(numbers) / len(numbers)


def _question_asks_average_across_range(task: TaskItem) -> bool:
    question = str(task.question or "")
    if not re.search(r"\b(avg|average|mean)\b", question, flags=re.IGNORECASE):
        return False
    return bool(
        re.search(r"\bfrom\b.+\bto\b", question, flags=re.IGNORECASE)
        or re.search(r"\bbetween\b.+\band\b", question, flags=re.IGNORECASE)
    )


def _sql_uses_avg(sql: str) -> bool:
    return bool(re.search(r"\bavg\s*\(", str(sql), flags=re.IGNORECASE))


def _avg_summed_identifier_count(sql: str) -> int:
    for argument in _avg_function_arguments(sql):
        if "/" in argument or "+" not in argument:
            continue
        identifiers = re.findall(r'"(?:[^"]|"")*"', argument)
        if len(identifiers) > 1:
            return len(identifiers)
    return 0


def _avg_function_arguments(sql: str) -> list[str]:
    text = str(sql)
    arguments: list[str] = []
    for match in re.finditer(r"\bavg\s*\(", text, flags=re.IGNORECASE):
        start = match.end()
        depth = 1
        in_quote = False
        index = start
        while index < len(text):
            char = text[index]
            if char == '"':
                if in_quote and index + 1 < len(text) and text[index + 1] == '"':
                    index += 2
                    continue
                in_quote = not in_quote
            elif not in_quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        arguments.append(text[start:index])
                        break
            index += 1
    return arguments


def _numeric_values_from_row(row: Any, columns: list[str]) -> list[float] | None:
    numbers: list[float] = []
    for column in columns:
        number = _coerce_static_number(_row_column_value(row, column, columns))
        if number is None:
            return None
        numbers.append(number)
    return numbers


def _coerce_static_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _table_rows_and_columns(value: Any) -> tuple[list[Any], list[str]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records"), [str(column) for column in value.columns]
    if isinstance(value, dict):
        rows = value.get("rows")
        if rows is None:
            rows = value.get("data")
        columns = value.get("cols", value.get("columns"))
        if isinstance(rows, list):
            normalized_columns = [
                str(column)
                for column in columns
            ] if isinstance(columns, list) else _columns_from_rows(rows)
            return rows, normalized_columns
    if isinstance(value, list):
        return value, _columns_from_rows(value)
    return [], []


def _columns_from_rows(rows: list[Any]) -> list[str]:
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, dict):
        return [str(column) for column in first.keys()]
    return []


def _row_column_value(row: Any, column: str, columns: list[str]) -> Any:
    if isinstance(row, dict):
        if column in row:
            return row[column]
        matched = _matching_column([str(key) for key in row], column)
        return row.get(matched) if matched is not None else None
    if isinstance(row, (list, tuple)):
        try:
            index = columns.index(column)
        except ValueError:
            return None
        return row[index] if index < len(row) else None
    return row if len(columns) == 1 else None


def _matching_column(columns: list[str], target: str | None) -> str | None:
    if not target:
        return None
    if target in columns:
        return target
    normalized_target = target.strip().casefold()
    matches = [
        column
        for column in columns
        if column.strip().casefold() == normalized_target
    ]
    return matches[0] if len(matches) == 1 else None


def _task_target_column(task: TaskItem) -> str | None:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    builder = metadata.get("dsl_builder")
    if not isinstance(builder, dict):
        return None
    diagnostics = builder.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    target = diagnostics.get("target_column")
    return target.strip() if isinstance(target, str) and target.strip() else None


def _sql_has_limit_one(sql: str) -> bool:
    return bool(re.search(r"\blimit\s+1\b", sql, flags=re.IGNORECASE))


def _answer_type_name(task: TaskItem) -> str:
    return str(task.answer_type or "").strip().lower()


def _single_answerish_cell(value: Any, *, answer_key: str) -> Any:
    if isinstance(value, pd.DataFrame):
        if value.shape != (1, 1):
            return _STATIC_ACTION_NO_ANSWER
        column = str(value.columns[0])
        if not _is_answerish_column(column, answer_key=answer_key):
            return _STATIC_ACTION_NO_ANSWER
        return value.iloc[0, 0]
    if isinstance(value, dict):
        rows = value.get("rows")
        if rows is None:
            rows = value.get("data")
        if isinstance(rows, list):
            count = value.get("n")
            if count is not None and count != 1:
                return _STATIC_ACTION_NO_ANSWER
            return _single_answerish_row_cell(
                rows,
                columns=value.get("cols", value.get("columns")),
                answer_key=answer_key,
            )
        if len(value) == 1:
            column, cell = next(iter(value.items()))
            if _is_answerish_column(str(column), answer_key=answer_key):
                return cell
        return _STATIC_ACTION_NO_ANSWER
    if isinstance(value, list):
        return _single_answerish_row_cell(
            value,
            columns=None,
            answer_key=answer_key,
        )
    return _STATIC_ACTION_NO_ANSWER


def _single_answerish_row_cell(
    rows: list[Any],
    *,
    columns: Any,
    answer_key: str,
) -> Any:
    if len(rows) != 1:
        return _STATIC_ACTION_NO_ANSWER
    row = rows[0]
    if isinstance(row, dict):
        if isinstance(columns, list) and len(columns) == 1:
            column = str(columns[0])
            if column in row and _is_answerish_column(column, answer_key=answer_key):
                return row[column]
        if len(row) == 1:
            column, cell = next(iter(row.items()))
            if _is_answerish_column(str(column), answer_key=answer_key):
                return cell
        return _STATIC_ACTION_NO_ANSWER
    if isinstance(row, (list, tuple)):
        if len(row) != 1:
            return _STATIC_ACTION_NO_ANSWER
        if not isinstance(columns, list) or len(columns) != 1:
            return _STATIC_ACTION_NO_ANSWER
        if not _is_answerish_column(str(columns[0]), answer_key=answer_key):
            return _STATIC_ACTION_NO_ANSWER
        return row[0]
    return _STATIC_ACTION_NO_ANSWER


def _is_answerish_column(column: str, *, answer_key: str) -> bool:
    normalized = column.strip().lower()
    answer = str(answer_key or "").strip().lower()
    if normalized in {"answer", "result", "value"}:
        return True
    if answer and (normalized == answer or normalized.startswith(f"{answer}__")):
        return True
    return False


def _execute_table_action_group(
    *,
    task: TaskItem,
    actions: tuple[SupervisorAction, ...],
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    profiler: PipelineProfiler,
) -> dict[str, Any]:
    observations = []
    for index, action in enumerate(actions):
        if action.op == "sql":
            observations.append(
                _execute_sql_action_observation(
                    task=task,
                    action=action,
                    index=index,
                    table_cache=table_cache,
                    local_slm_config=local_slm_config,
                    local_slm_dispatcher=local_slm_dispatcher,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                    profiler=profiler,
                )
            )
            continue
        if action.op == "inspect":
            observations.append(
                _execute_inspect_action_observation(
                    task=task,
                    action=action,
                    index=index,
                    table_cache=table_cache,
                    local_slm_config=local_slm_config,
                    local_slm_dispatcher=local_slm_dispatcher,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                    profiler=profiler,
                )
            )
            continue
        if action.op == "analyze":
            observations.append(
                _execute_analyze_action_observation(
                    task=task,
                    action=action,
                    index=index,
                    table_cache=table_cache,
                    local_slm_config=local_slm_config,
                    local_slm_dispatcher=local_slm_dispatcher,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                    profiler=profiler,
                )
            )
            continue
        observations.append(
            {
                "i": index,
                "op": action.op,
                "ok": False,
                "err": {"type": "UnsupportedAction", "message": action.op},
            }
        )
    return {"ok": True, "answer": None, "obs": observations}


def _execute_sql_action_observation(
    *,
    task: TaskItem,
    action: SupervisorAction,
    index: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    profiler: PipelineProfiler,
) -> dict[str, Any]:
    sql = action.q or ""
    result = _execute_sql_action(
        task=task,
        sql=sql,
        action_index=index,
        table_cache=table_cache,
        local_slm_config=local_slm_config,
        local_slm_dispatcher=local_slm_dispatcher,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        profiler=profiler,
    )
    observation: dict[str, Any] = {
        "i": index,
        "op": "sql",
        "ok": result.ok,
        "q": sql,
    }
    if result.ok:
        observation["res"] = _compact_evidence_value(result.value)
        observation[_ACTION_FULL_RESULT_KEY] = result.value
    else:
        observation["err"] = result.error
    return observation


def _execute_sql_action(
    *,
    task: TaskItem,
    sql: str,
    action_index: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    profiler: PipelineProfiler,
) -> _SqlActionExecution:
    if not isinstance(sql, str) or not sql.strip():
        return _SqlActionExecution(
            ok=False,
            error={"type": "EmptySQL", "message": "sql action has empty q"},
        )
    evidence_key = f"{task.answer_key}__a{action_index + 1}"
    try:
        evidence_dsl = _query_remote_dsl_with_answer(
            task,
            {"name": evidence_key, "type": "table"},
        )
        logic_dag = parse_remote_sql_to_logic_dag(sql, evidence_dsl)
        physical_plan = _optimize_table_logic_dag(
            logic_dag=logic_dag,
            context=_batch_context(task),
            local_dsl=_query_local_dsl(task),
            profiler=profiler,
            stage_name="action_group_sql_optimizer",
            items=1,
        )
        execution_result = _execute_table_physical_plan(
            physical_plan=physical_plan,
            table_cache=table_cache,
            local_slm_config=local_slm_config,
            profiler=profiler,
            stage_name="action_group_sql_executor",
            items=1,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            local_slm_dispatcher=local_slm_dispatcher,
        )
        _record_executor_trace_counters(profiler, execution_result)
        if not execution_result.ok:
            return _SqlActionExecution(
                ok=False,
                error=execution_result.error
                or {"type": "ExecutionFailed", "message": "SQL action failed"},
            )
        value = _execution_value_for_key(execution_result, evidence_key)
        return _SqlActionExecution(ok=True, value=value, frame=_value_to_frame(value))
    except Exception as exc:  # noqa: BLE001 - surfaced as action observation.
        return _SqlActionExecution(ok=False, error=_error_payload(exc))


def _inspect_action_physical_plan(
    *,
    task: TaskItem,
    action: SupervisorAction,
    action_index: int,
    profiler: PipelineProfiler,
) -> tuple[dict[str, Any], str]:
    evidence_key = f"{task.answer_key}__a{action_index + 1}"
    source_ids = _task_source_ids(task)
    if not source_ids:
        raise ValueError("inspect action requires at least one source table")

    if action.seed:
        seed_key = f"{evidence_key}__seed"
        seed_dsl = _query_remote_dsl_with_answer(
            task,
            {"name": seed_key, "type": "table"},
        )
        logic_dag = parse_remote_sql_to_logic_dag(action.seed, seed_dsl)
        physical_plan = _optimize_table_logic_dag(
            logic_dag=logic_dag,
            context=_batch_context(task),
            local_dsl=_query_local_dsl(task),
            profiler=profiler,
            stage_name="action_group_inspect_seed_optimizer",
            items=1,
        )
        dependencies = [seed_key]
    else:
        physical_plan = prepare_physical_plan_resources(
            {
                "task_type": task.task_type,
                "resources": copy.deepcopy(task.local_dsl.get("sources", [])),
                "resource_processing": [],
                "nodes": [],
                "edges": [],
            }
        )
        dependencies = []

    inspect_node = {
        "id": f"A{action_index + 1}_inspect",
        "op": "Inspect",
        "dependency": dependencies,
        "input": source_ids,
        "params": {
            "question": task.question,
            "request": action.q,
            "need": _inspect_action_need(task),
        },
        "output": evidence_key,
        "output_type": "evidence",
        "instruction": action.q or "",
    }
    physical_plan.setdefault("nodes", []).append(inspect_node)
    physical_plan.setdefault("edges", [])
    return physical_plan, evidence_key


def _analyze_action_physical_plan(
    *,
    task: TaskItem,
    action: SupervisorAction,
    action_index: int,
    profiler: PipelineProfiler,
) -> tuple[dict[str, Any], str]:
    evidence_key = f"{task.answer_key}__a{action_index + 1}"
    seed = action.seed
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("analyze action requires seed SQL")
    kind = action.kind
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("analyze action requires kind")

    seed_key = f"{evidence_key}__seed"
    seed_dsl = _query_remote_dsl_with_answer(
        task,
        {"name": seed_key, "type": "table"},
    )
    logic_dag = parse_remote_sql_to_logic_dag(seed, seed_dsl)
    physical_plan = _optimize_table_logic_dag(
        logic_dag=logic_dag,
        context=_batch_context(task),
        local_dsl=_query_local_dsl(task),
        profiler=profiler,
        stage_name="action_group_analyze_seed_optimizer",
        items=1,
    )
    analyze_dependency = _pre_format_table_dependency(physical_plan, seed_key)
    analyze_node = {
        "id": f"A{action_index + 1}_analyze",
        "op": "AnalyzeEvidence",
        "dependency": [analyze_dependency],
        "input": [],
        "params": {"kind": kind.strip().lower()},
        "output": evidence_key,
        "output_type": "evidence",
    }
    physical_plan.setdefault("nodes", []).append(analyze_node)
    physical_plan.setdefault("edges", [])
    return physical_plan, evidence_key


def _pre_format_table_dependency(physical_plan: dict[str, Any], output_key: str) -> str:
    for node in physical_plan.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("output") != output_key or node.get("op") != "FormatAnswer":
            continue
        dependencies = node.get("dependency")
        if isinstance(dependencies, list) and dependencies:
            dependency = dependencies[0]
            if isinstance(dependency, str) and dependency:
                return dependency
    return output_key


def _inspect_action_need(task: TaskItem) -> dict[str, Any]:
    payload: dict[str, Any] = {"answer_type": task.answer_type}
    profile = table_reasoning_profile_from_dsl(task.local_dsl)
    if profile:
        payload["profile"] = profile
    return payload


def _task_source_ids(task: TaskItem) -> list[str]:
    source_ids = []
    for source in task.local_dsl.get("sources", []):
        if isinstance(source, dict) and isinstance(source.get("id"), str):
            source_ids.append(source["id"])
    return source_ids


def _inspect_action_iterations(
    execution_result: ExecutionResult,
    evidence_key: str,
) -> int | None:
    for trace in execution_result.traces:
        if not isinstance(trace, dict):
            continue
        if trace.get("output") != evidence_key or trace.get("op") != "Inspect":
            continue
        agent_loop = trace.get("agent_loop")
        if isinstance(agent_loop, dict) and isinstance(agent_loop.get("iterations"), int):
            return agent_loop["iterations"]
    return None


def _execution_value_for_key(execution_result: ExecutionResult, key: str) -> Any:
    if isinstance(execution_result.answer, dict) and key in execution_result.answer:
        return execution_result.answer[key]
    if key in execution_result.outputs:
        return execution_result.outputs[key]
    if "answer" in execution_result.outputs:
        return execution_result.outputs["answer"]
    return execution_result.answer


def _execute_inspect_action_observation(
    *,
    task: TaskItem,
    action: SupervisorAction,
    index: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    profiler: PipelineProfiler,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "i": index,
        "op": "inspect",
        "ok": False,
        "q": action.q,
    }
    if action.seed:
        observation["seed"] = action.seed
    try:
        with profiler.measure("action_group_inspect", items=1):
            physical_plan, evidence_key = _inspect_action_physical_plan(
                task=task,
                action=action,
                action_index=index,
                profiler=profiler,
            )
            execution_result = _execute_table_physical_plan(
                physical_plan=physical_plan,
                table_cache=table_cache,
                local_slm_config=local_slm_config,
                local_slm_dispatcher=local_slm_dispatcher,
                profiler=profiler,
                stage_name="action_group_inspect_executor",
                items=1,
                max_parallel_execution_units=max_parallel_execution_units,
                max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                max_parallel_slm_sequences=max_parallel_slm_sequences,
                max_pending_slm_sequences=max_pending_slm_sequences,
            )
        _record_executor_trace_counters(profiler, execution_result)
        if not execution_result.ok:
            observation["err"] = execution_result.error or {
                "type": "ExecutionFailed",
                "message": "inspect action failed",
            }
            return observation
        observation["ok"] = True
        observation["ev"] = _execution_value_for_key(execution_result, evidence_key)
        iterations = _inspect_action_iterations(execution_result, evidence_key)
        if iterations is not None:
            observation["it"] = iterations
        profiler.increment("action_group_inspect_calls")
        return observation
    except Exception as exc:  # noqa: BLE001 - surfaced as action observation.
        observation["err"] = _error_payload(exc)
        return observation


def _execute_analyze_action_observation(
    *,
    task: TaskItem,
    action: SupervisorAction,
    index: int,
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    profiler: PipelineProfiler,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "i": index,
        "op": "analyze",
        "kind": action.kind,
        "ok": False,
        "seed": action.seed,
    }
    try:
        with profiler.measure("action_group_analyze", items=1):
            physical_plan, evidence_key = _analyze_action_physical_plan(
                task=task,
                action=action,
                action_index=index,
                profiler=profiler,
            )
            execution_result = _execute_table_physical_plan(
                physical_plan=physical_plan,
                table_cache=table_cache,
                local_slm_config=local_slm_config,
                local_slm_dispatcher=local_slm_dispatcher,
                profiler=profiler,
                stage_name="action_group_analyze_executor",
                items=1,
                max_parallel_execution_units=max_parallel_execution_units,
                max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                max_parallel_slm_sequences=max_parallel_slm_sequences,
                max_pending_slm_sequences=max_pending_slm_sequences,
            )
        _record_executor_trace_counters(profiler, execution_result)
        if not execution_result.ok:
            observation["err"] = execution_result.error or {
                "type": "ExecutionFailed",
                "message": "analyze action failed",
            }
            return observation
        observation["ok"] = True
        observation["res"] = _compact_evidence_value(
            _execution_value_for_key(execution_result, evidence_key)
        )
        profiler.increment("action_group_analyze_calls")
        return observation
    except Exception as exc:  # noqa: BLE001 - surfaced as action observation.
        observation["err"] = _error_payload(exc)
        return observation


def _value_to_frame(value: Any) -> pd.DataFrame | None:
    frame = getattr(value, "frame", None)
    if isinstance(frame, pd.DataFrame):
        return frame.copy(deep=True)
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        rows = value.get("rows")
        if isinstance(rows, list):
            return pd.DataFrame(rows)
        data = value.get("data")
        if isinstance(data, list):
            return pd.DataFrame(data)
    return None


def _optimize_table_logic_dag(
    *,
    logic_dag: dict[str, Any],
    context: dict[str, Any],
    local_dsl: dict[str, Any],
    profiler: PipelineProfiler,
    stage_name: str,
    items: int,
) -> dict[str, Any]:
    with profiler.measure(stage_name, items=items):
        physical_plan = optimize_logic_dag_to_physical_plan(
            logic_dag=logic_dag,
            context=context,
            local_dsl=local_dsl,
        )
        return prepare_physical_plan_resources(physical_plan)


def _execute_table_physical_plan(
    *,
    physical_plan: dict[str, Any],
    table_cache: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    profiler: PipelineProfiler,
    stage_name: str,
    items: int,
    max_parallel_execution_units: int | None = None,
    max_parallel_slm_node_jobs: int | None = None,
    max_parallel_slm_sequences: int | None = None,
    max_pending_slm_sequences: int | None = None,
    local_slm_dispatcher: LocalSlmSequenceDispatcher | None = None,
) -> ExecutionResult:
    with profiler.measure(stage_name, items=items):
        kwargs: dict[str, Any] = {}
        if max_parallel_execution_units is not None:
            kwargs["max_parallel_execution_units"] = max_parallel_execution_units
        if max_parallel_slm_node_jobs is not None:
            kwargs["max_parallel_slm_node_jobs"] = max_parallel_slm_node_jobs
        if max_parallel_slm_sequences is not None:
            kwargs["max_parallel_slm_sequences"] = max_parallel_slm_sequences
        if max_pending_slm_sequences is not None:
            kwargs["max_pending_slm_sequences"] = max_pending_slm_sequences
        return execute_execution_plan(
            ExecutionPlanBuilder.default().build(physical_plan),
            collector_context=physical_plan,
            table_cache=table_cache,
            slm_config=local_slm_config,
            slm_dispatcher=local_slm_dispatcher,
            agent_loop_max_iterations=_agent_loop_max_iterations(local_slm_config),
            **kwargs,
        )


def _apply_action_group_decision(
    *,
    task: TaskItem,
    decision: SupervisorDecision,
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    max_retries: int,
) -> None:
    if decision.done:
        _finalize_success(
            task,
            answer=decision.answer,
            final_results=final_results,
            finalized=finalized,
        )
        return
    actions = decision.actions
    if not actions and decision.sqls:
        actions = tuple(SupervisorAction(op="sql", q=sql) for sql in decision.sqls)
    if actions:
        _enqueue_action_group(
            task,
            actions=actions,
            action_items=action_items,
            final_results=final_results,
            finalized=finalized,
            max_retries=max_retries,
        )
        return
    _finalize_failed(
        task,
        final_results=final_results,
        finalized=finalized,
        error={
            "type": "SupervisorProtocolError",
            "message": "Analyze supervisor returned neither answer nor actions",
        },
    )


def _enqueue_action_group(
    task: TaskItem,
    *,
    actions: tuple[SupervisorAction, ...],
    action_items: list[ActionGroupItem],
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
    task.current_command = json.dumps(
        {"acts": [_action_to_dict(action) for action in actions]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    task.status = TASK_RETRYING
    action_items.append(ActionGroupItem(task=task, actions=actions))


def _record_action_memory(
    *,
    task: TaskItem,
    actions: tuple[SupervisorAction, ...],
    observation: dict[str, Any],
    decision: SupervisorDecision,
) -> None:
    if decision.done is not False:
        return
    entry = {
        "act": [_action_to_dict(action) for action in actions],
        "obs": _compact_action_memory_result(observation),
    }
    if decision.feedback:
        entry["fb"] = decision.feedback
    task.memory.append(entry)
    del task.memory[:-3]


def _compact_action_memory_result(observation: dict[str, Any]) -> Any:
    obs = observation.get("obs")
    if not isinstance(obs, list):
        return None
    return json_ready(obs[:4])


def _finish_table_execution_batch(
    *,
    batch: list[LogicDagItem],
    logic_dag: dict[str, Any],
    physical_plan: dict[str, Any],
    execution_result: ExecutionResult,
    requeue_interrupted: Callable[[LogicDagItem], None],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    supervisor: SupervisorAgent,
    max_retries: int,
    validation_mode: str,
    profiler: PipelineProfiler,
) -> bool:
    if execution_result.ok:
        if _needs_supervisor_review(batch, validation_mode):
            _run_supervisor_synthesis(
                batch=batch,
                execution_result=execution_result,
                logic_dag=logic_dag,
                sql_items=sql_items,
                final_results=final_results,
                finalized=finalized,
                supervisor=supervisor,
                max_retries=max_retries,
                profiler=profiler,
            )
        else:
            _run_static_synthesis_review(
                batch=batch,
                execution_result=execution_result,
                final_results=final_results,
                finalized=finalized,
                profiler=profiler,
            )
        return True

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
            requeue_interrupted(item)

    if unaffected:
        subset_result = _successful_subset_execution_result(
            execution_result,
            unaffected,
        )
        if _needs_supervisor_review(unaffected, validation_mode):
            _run_supervisor_synthesis(
                batch=unaffected,
                execution_result=subset_result,
                logic_dag=_batch_logic_dag(unaffected),
                sql_items=sql_items,
                final_results=final_results,
                finalized=finalized,
                supervisor=supervisor,
                max_retries=max_retries,
                profiler=profiler,
            )
        else:
            _run_static_synthesis_review(
                batch=unaffected,
                execution_result=subset_result,
                final_results=final_results,
                finalized=finalized,
                profiler=profiler,
            )

    failed_items = [item for item in batch if item.task.answer_key in affected]
    if failed_items:
        if _needs_supervisor_review(failed_items, validation_mode):
            for item in failed_items:
                _run_supervisor_synthesis(
                    batch=[item],
                    execution_result=execution_result,
                    logic_dag=item.logic_dag,
                    sql_items=sql_items,
                    final_results=final_results,
                    finalized=finalized,
                    supervisor=supervisor,
                    max_retries=max_retries,
                    profiler=profiler,
                )
        else:
            _run_static_synthesis_failure(
                batch=failed_items,
                execution_result=execution_result,
                final_results=final_results,
                finalized=finalized,
                profiler=profiler,
            )
    return True


def _run_static_synthesis_review(
    *,
    batch: list[LogicDagItem],
    execution_result: ExecutionResult,
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> None:
    for item in batch:
        item.task.status = TASK_SUPERVISOR_REVIEW
    with profiler.measure("static_synthesis", items=len(batch)):
        profiler.increment("static_synthesis_calls")
        answer_map = _execution_answer_map(execution_result, batch)
        for item in batch:
            key = item.task.answer_key
            if key in answer_map:
                _finalize_success(
                    item.task,
                    answer=answer_map[key],
                    final_results=final_results,
                    finalized=finalized,
                )
                continue
            _finalize_failed(
                item.task,
                final_results=final_results,
                finalized=finalized,
                error={
                    "type": "MissingExecutionAnswer",
                    "message": f"Executor result did not include answer {key}",
                },
            )


def _needs_supervisor_review(
    batch: list[LogicDagItem],
    validation_mode: str,
) -> bool:
    del batch
    return validation_mode == VALIDATION_REMOTE_SUPERVISOR


def _run_static_synthesis_failure(
    *,
    batch: list[LogicDagItem],
    execution_result: ExecutionResult,
    final_results: list[CaseResult],
    finalized: set[str],
    profiler: PipelineProfiler,
) -> None:
    for item in batch:
        item.task.status = TASK_SUPERVISOR_REVIEW
    error = execution_result.error or {
        "type": "ExecutionFailed",
        "message": "Local execution failed and validation_mode=none disables retry",
    }
    with profiler.measure("static_synthesis", items=len(batch)):
        profiler.increment("static_synthesis_calls")
        for item in batch:
            _finalize_failed(
                item.task,
                final_results=final_results,
                finalized=finalized,
                error=copy.deepcopy(error),
            )


def _execution_answer_map(
    execution_result: ExecutionResult,
    batch: list[LogicDagItem],
) -> dict[str, Any]:
    if isinstance(execution_result.answer, dict):
        return dict(execution_result.answer)

    if len(batch) == 1 and execution_result.answer is not None:
        return {batch[0].task.answer_key: execution_result.answer}

    answer_map: dict[str, Any] = {}
    for item in batch:
        key = item.task.answer_key
        if key in execution_result.outputs:
            answer_map[key] = execution_result.outputs[key]
    return answer_map


def _run_supervisor_synthesis(
    *,
    batch: list[LogicDagItem],
    execution_result: ExecutionResult,
    logic_dag: dict[str, Any],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    supervisor: SupervisorAgent,
    max_retries: int,
    profiler: PipelineProfiler,
) -> None:
    for item in batch:
        item.task.status = TASK_SUPERVISOR_REVIEW
    try:
        observation = execution_result
        with profiler.measure("supervisor_synthesis", items=len(batch)):
            supervisor_result = supervisor.synthesize(
                local_dsl=_synthesis_local_dsl(batch),
                logic_dag=logic_dag,
                observation=observation,
                current_command=_sql_map(batch),
            )
            profiler.increment("supervisor_synthesis_calls")
            _record_remote_token_usage(
                profiler,
                stage_name="supervisor_synthesis",
                response_payload=supervisor_result.response_payload,
            )
        if supervisor_result.decision is None:
            raise ValueError("Supervisor synthesis did not return a decision")
    except Exception as exc:  # noqa: BLE001 - keep cloud/report failures task-scoped.
        profiler.increment("supervisor_synthesis_failures")
        for item in batch:
            _finalize_failed(
                item.task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(exc),
            )
        return
    _apply_supervisor_decision(
        batch=batch,
        decision=supervisor_result.decision,
        sql_items=sql_items,
        final_results=final_results,
        finalized=finalized,
        max_retries=max_retries,
    )


def _run_supervisor_action_reports_batch(
    *,
    report_items: list[tuple[ActionGroupItem, dict[str, Any]]],
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    supervisor: SupervisorAgent,
    max_retries: int,
    remote_concurrency: int,
    profiler: PipelineProfiler,
) -> None:
    runnable = [
        (index, item, observation)
        for index, (item, observation) in enumerate(report_items)
        if item.task.answer_key not in finalized
    ]
    if not runnable:
        return

    for _, item, _ in runnable:
        item.task.status = TASK_SUPERVISOR_REVIEW

    max_workers = max(1, min(remote_concurrency, len(runnable)))
    if max_workers == 1:
        results = [
            _run_one_supervisor_action_report_for_batch(
                order_index=index,
                item=item,
                observation=observation,
                supervisor=supervisor,
            )
            for index, item, observation in runnable
        ]
    else:
        results_by_index: dict[int, _ActionReportRunResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _run_one_supervisor_action_report_for_batch,
                    order_index=index,
                    item=item,
                    observation=observation,
                    supervisor=supervisor,
                )
                for index, item, observation in runnable
            ]
            for future in as_completed(futures):
                result = future.result()
                results_by_index[result.order_index] = result
        results = [results_by_index[index] for index, _, _ in runnable]

    for result in results:
        task = result.item.task
        _merge_pipeline_profiler(profiler, result.profiler)
        if task.answer_key in finalized:
            continue
        if result.error is not None or result.supervisor_result is None:
            _finalize_failed(
                task,
                final_results=final_results,
                finalized=finalized,
                error=_error_payload(
                    result.error
                    or RuntimeError("Supervisor synthesis returned no result")
                ),
            )
            continue
        decision = result.supervisor_result.decision
        _record_action_memory(
            task=task,
            actions=result.item.actions,
            observation=result.observation,
            decision=decision,
        )
        _apply_action_group_decision(
            task=task,
            decision=decision,
            action_items=action_items,
            final_results=final_results,
            finalized=finalized,
            max_retries=max_retries,
        )


def _run_one_supervisor_action_report_for_batch(
    *,
    order_index: int,
    item: ActionGroupItem,
    observation: dict[str, Any],
    supervisor: SupervisorAgent,
) -> _ActionReportRunResult:
    local_profiler = PipelineProfiler()
    try:
        with local_profiler.measure("supervisor_synthesis", items=1):
            supervisor_result = supervisor.synthesize(
                local_dsl=_query_local_dsl(item.task),
                logic_dag={"task_type": item.task.task_type, "query_plans": []},
                observation=observation,
                current_command={
                    "acts": [_action_to_dict(action) for action in item.actions]
                },
            )
            local_profiler.increment("supervisor_synthesis_calls")
            _record_remote_token_usage(
                local_profiler,
                stage_name="supervisor_synthesis",
                response_payload=supervisor_result.response_payload,
            )
        if supervisor_result.decision is None:
            raise ValueError("Supervisor synthesis did not return a decision")
        return _ActionReportRunResult(
            order_index=order_index,
            item=item,
            observation=observation,
            supervisor_result=supervisor_result,
            profiler=local_profiler,
        )
    except Exception as exc:  # noqa: BLE001 - keep cloud/report failures task-scoped.
        local_profiler.increment("supervisor_synthesis_failures")
        return _ActionReportRunResult(
            order_index=order_index,
            item=item,
            observation=observation,
            supervisor_result=None,
            profiler=local_profiler,
            error=exc,
        )


def _run_supervisor_action_report(
    *,
    item: ActionGroupItem,
    observation: dict[str, Any],
    action_items: list[ActionGroupItem],
    final_results: list[CaseResult],
    finalized: set[str],
    supervisor: SupervisorAgent,
    max_retries: int,
    profiler: PipelineProfiler,
) -> None:
    _run_supervisor_action_reports_batch(
        report_items=[(item, observation)],
        action_items=action_items,
        final_results=final_results,
        finalized=finalized,
        supervisor=supervisor,
        max_retries=max_retries,
        remote_concurrency=1,
        profiler=profiler,
    )


def _apply_supervisor_decision(
    *,
    batch: list[LogicDagItem],
    decision: SupervisorDecision,
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    max_retries: int,
) -> None:
    if decision.done is True:
        _finalize_batch_from_answer(
            batch=batch,
            answer=decision.answer,
            final_results=final_results,
            finalized=finalized,
        )
        return

    actions = decision.actions
    if not actions and decision.sqls:
        actions = tuple(SupervisorAction(op="sql", q=sql) for sql in decision.sqls)
    if actions:
        if len(batch) != 1:
            for item in batch:
                _finalize_failed(
                    item.task,
                    final_results=final_results,
                    finalized=finalized,
                    error={
                        "type": "SupervisorProtocolError",
                        "message": "Batched supervisor action must be reported per task",
                    },
                )
            return
        _enqueue_actions_for_task(
            batch[0].task,
            actions=actions,
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            max_retries=max_retries,
        )
        return

    if not decision.retry:
        _finalize_batch_from_answer(
            batch=batch,
            answer=decision.answer,
            final_results=final_results,
            finalized=finalized,
        )
        return

    for item in batch:
        _finalize_failed(
            item.task,
            final_results=final_results,
            finalized=finalized,
            error={
                "type": "SupervisorProtocolError",
                "message": "Supervisor requested another round without actions",
            },
        )


def _finalize_batch_from_answer(
    *,
    batch: list[LogicDagItem],
    answer: Any,
    final_results: list[CaseResult],
    finalized: set[str],
) -> None:
    if isinstance(answer, dict):
        for item in batch:
            key = item.task.answer_key
            if key in answer:
                _finalize_success(
                    item.task,
                    answer=answer.get(key),
                    final_results=final_results,
                    finalized=finalized,
                )
            else:
                _finalize_failed(
                    item.task,
                    final_results=final_results,
                    finalized=finalized,
                    error={
                        "type": "SupervisorProtocolError",
                        "message": f"Supervisor answer missing {key}",
                    },
                )
        return
    if len(batch) == 1:
        item = batch[0]
        _finalize_success(
            item.task,
            answer=answer,
            final_results=final_results,
            finalized=finalized,
        )
        return
    for item in batch:
        _finalize_failed(
            item.task,
            final_results=final_results,
            finalized=finalized,
            error={
                "type": "SupervisorProtocolError",
                "message": "Supervisor answer for batch must be keyed by answer name",
            },
        )


def _enqueue_actions_for_task(
    task: TaskItem,
    *,
    actions: tuple[SupervisorAction, ...],
    sql_items: list[SqlItem],
    final_results: list[CaseResult],
    finalized: set[str],
    max_retries: int,
) -> None:
    if len(actions) == 1 and actions[0].op == "sql" and actions[0].q:
        _enqueue_sqls(
            task,
            sqls=(actions[0].q,),
            sql_items=sql_items,
            final_results=final_results,
            finalized=finalized,
            max_retries=max_retries,
        )
        return
    _finalize_failed(
        task,
        final_results=final_results,
        finalized=finalized,
        error={
            "type": "UnsupportedSupervisorAction",
            "message": "Query execution report only accepts one follow-up SQL action",
        },
    )


def _enqueue_sqls(
    task: TaskItem,
    *,
    sqls: tuple[str, ...],
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
    command = _PlannedSqlCommand(
        op="sql",
        sqls=sqls,
    )
    task.current_command = _command_content(command)
    task.status = TASK_RETRYING
    sql_items.append(
        SqlItem(
            task=task,
            content=_command_content(command),
            content_type="sql",
            statements=sqls,
        )
    )


def _finalize_success(
    task: TaskItem,
    *,
    answer: Any,
    final_results: list[CaseResult],
    finalized: set[str],
) -> None:
    if task.answer_key in finalized:
        return
    answer = _normalize_answer_for_task(task, answer)
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


def _normalize_answer_for_task(task: TaskItem, answer: Any) -> Any:
    answer_type = str(task.answer_type or "").strip().lower()
    if answer_type in {"number", "float", "integer", "int"}:
        return _normalize_number_answer(answer)
    if answer_type in {"boolean", "bool"}:
        return _normalize_boolean_answer(answer)
    if answer_type in {"string", "entity"}:
        if answer is None or isinstance(answer, str):
            return answer
        return json.dumps(json_ready(answer), ensure_ascii=False, separators=(",", ":"))
    return answer


def _normalize_number_answer(answer: Any) -> Any:
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return answer
    if isinstance(answer, list) and len(answer) == 1:
        return _normalize_number_answer(answer[0])
    if isinstance(answer, dict) and len(answer) == 1:
        return _normalize_number_answer(next(iter(answer.values())))
    if isinstance(answer, str):
        stripped = answer.strip().replace(",", "")
        try:
            return float(stripped)
        except ValueError:
            return answer
    return answer


def _normalize_boolean_answer(answer: Any) -> Any:
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, str):
        normalized = answer.strip().lower()
        if normalized in {"true", "yes", "y", "1", "support", "supports"}:
            return True
        if normalized in {"false", "no", "n", "0", "refute", "refutes"}:
            return False
    return answer


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
    *,
    analyze: bool,
) -> list[TaskItem]:
    batch: list[TaskItem] = []
    remaining: list[TaskItem] = []
    for item in pending:
        if (
            item.source_file == source_file
            and _is_analyze_task(item) == analyze
            and len(batch) < max_items
        ):
            batch.append(item)
        else:
            remaining.append(item)
    pending[:] = remaining
    return batch


def _batch_remote_dsl(batch: list[TaskItem]) -> dict[str, Any]:
    first = batch[0]
    dsl = {
        "task_type": first.task_type,
        "questions": [item.question for item in batch],
        "sources": copy.deepcopy(first.remote_dsl.get("sources", [])),
        "answers": [
            {"name": item.answer_key, "type": item.answer_type}
            for item in batch
        ],
    }
    _copy_shared_optional_batch_fields(dsl, [item.remote_dsl for item in batch])
    return dsl


def _batch_local_dsl(batch: list[LogicDagItem]) -> dict[str, Any]:
    first = batch[0].task
    dsl = {
        "task_type": first.task_type,
        "questions": [item.task.question for item in batch],
        "sources": copy.deepcopy(first.local_dsl.get("sources", [])),
        "answers": [
            {"name": item.task.answer_key, "type": item.task.answer_type}
            for item in batch
        ],
    }
    _copy_shared_optional_batch_fields(dsl, [item.task.local_dsl for item in batch])
    return dsl


def _batch_context(task: TaskItem) -> dict[str, Any]:
    context = copy.deepcopy(task.context)
    context["task_type"] = task.task_type
    return context


def _query_remote_dsl(task: TaskItem) -> dict[str, Any]:
    return _query_remote_dsl_with_answer(
        task,
        {"name": task.answer_key, "type": task.answer_type},
    )


def _query_remote_dsl_with_answer(task: TaskItem, answer: dict[str, Any]) -> dict[str, Any]:
    dsl = {
        "task_type": task.task_type,
        "question": task.question,
        "sources": copy.deepcopy(task.remote_dsl.get("sources", [])),
        "answer": copy.deepcopy(answer),
    }
    _copy_optional_task_fields(task.remote_dsl, dsl)
    return dsl


def _query_local_dsl(task: TaskItem) -> dict[str, Any]:
    dsl = {
        "task_type": task.task_type,
        "question": task.question,
        "sources": copy.deepcopy(task.local_dsl.get("sources", [])),
        "answer": {"name": task.answer_key, "type": task.answer_type},
    }
    _copy_optional_task_fields(task.local_dsl, dsl)
    if isinstance(task, TableTaskItem) and task.memory:
        dsl["mem"] = copy.deepcopy(task.memory)
    return dsl


def _copy_optional_task_fields(source: dict[str, Any], target: dict[str, Any]) -> None:
    for key in (PROFILE_KEY, HINTS_KEY):
        if key in source:
            target[key] = copy.deepcopy(source[key])


def _copy_shared_optional_batch_fields(
    target: dict[str, Any],
    dsls: list[dict[str, Any]],
) -> None:
    for key in (PROFILE_KEY, HINTS_KEY):
        values = [dsl.get(key) for dsl in dsls if key in dsl]
        if not values:
            continue
        first_value = values[0]
        if all(value == first_value for value in values):
            target[key] = copy.deepcopy(first_value)


def _query_context(task: TaskItem) -> dict[str, Any]:
    context = copy.deepcopy(task.context)
    context["task_type"] = task.task_type
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
                    context=_query_context(item.task),
                    local_dsl=_query_local_dsl(item.task),
                )
                physical_plan = prepare_physical_plan_resources(physical_plan)
            with profiler.measure("baseline_executor", items=1):
                execute_execution_plan(
                    ExecutionPlanBuilder.default().build(physical_plan),
                    collector_context=physical_plan,
                    table_cache=table_cache,
                    slm_config=local_slm_config,
                    agent_loop_max_iterations=_agent_loop_max_iterations(local_slm_config),
                )
            profiler.increment("baseline_case_count")
        except Exception:  # noqa: BLE001 - profiling must not change semantics.
            profiler.increment("baseline_profile_failures")


def _agent_loop_max_iterations(local_slm_config: dict[str, Any] | None) -> int:
    if not isinstance(local_slm_config, dict):
        return 3
    value = local_slm_config.get("agent_loop_max_iterations")
    if value is None:
        return 3
    try:
        iterations = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"agent_loop_max_iterations must be a positive integer: {value!r}"
        ) from exc
    if iterations <= 0:
        raise ValueError("agent_loop_max_iterations must be positive")
    return iterations


def _record_executor_trace_counters(
    profiler: PipelineProfiler,
    execution_result: ExecutionResult,
) -> None:
    profiler.increment("executor_fast_path_hits", execution_result.fast_path_hits)
    profiler.increment("executor_fast_path_misses", execution_result.fast_path_misses)
    for trace in execution_result.traces:
        if trace.get("fast_path_hit") is False:
            reason = _counter_suffix(trace.get("fast_path_miss_reason") or "unknown")
            profiler.increment(f"executor_fast_path_miss_{reason}")
        trigger = trace.get("agent_loop_trigger")
        if trigger:
            trigger_suffix = _counter_suffix(trigger)
            profiler.increment(f"executor_agent_loop_trigger_{trigger_suffix}")
            agent_loop = trace.get("agent_loop")
            if isinstance(agent_loop, dict):
                steps = agent_loop.get("steps")
                if isinstance(steps, list):
                    profiler.increment("executor_local_slm_steps", len(steps))
                    _record_local_slm_trace_usage(profiler, steps)
        fast_path_error = trace.get("fast_path_error")
        if isinstance(fast_path_error, dict):
            error_type = _counter_suffix(fast_path_error.get("type") or "unknown")
            profiler.increment(f"executor_fast_path_error_{error_type}")
        if trace.get("execution_path") == "timeout":
            profiler.increment("executor_node_timeouts")


def _record_local_slm_trace_usage(
    profiler: PipelineProfiler,
    steps: list[Any],
) -> None:
    for step in steps:
        if not isinstance(step, dict):
            continue
        usage = step.get("token_usage")
        if not isinstance(usage, dict):
            continue
        profiler.increment("local_slm_calls")
        for key, value in usage.items():
            try:
                amount = int(value or 0)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            profiler.increment(f"local_slm_{key}", amount)


def _counter_suffix(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def _profile_with_summary(
    profiler: PipelineProfiler,
    *,
    validation_mode: str,
) -> dict[str, Any]:
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
        "validation_mode": validation_mode,
        "remote_calls": profile.get("counters", {}).get("supervisor_decompose_calls", 0)
        + profile.get("counters", {}).get("supervisor_synthesis_calls", 0),
        "local_slm_calls": profile.get("counters", {}).get("local_slm_calls", 0),
        "merged_plan_count": profile.get("counters", {}).get("merged_plan_count", 0),
        "reused_nodes": profile.get("counters", {}).get("reused_nodes", 0),
        "remote_token_usage": _remote_token_usage_summary(profile.get("counters", {})),
        "supervisor_decompose_token_usage": _remote_stage_token_usage_summary(
            profile.get("counters", {}),
            "supervisor_decompose",
        ),
        "supervisor_synthesis_token_usage": _remote_stage_token_usage_summary(
            profile.get("counters", {}),
            "supervisor_synthesis",
        ),
        "local_slm_token_usage": _local_slm_token_usage_summary(
            profile.get("counters", {})
        ),
    }
    if baseline_seconds:
        summary["local_executor_speedup"] = (
            baseline_seconds / executor_seconds if executor_seconds else None
        )
    if baseline_optimizer_seconds:
        summary["local_optimizer_speedup"] = (
            baseline_optimizer_seconds / optimizer_seconds
            if optimizer_seconds
            else None
        )
    profile["summary"] = summary
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


def _remote_stage_token_usage_summary(
    counters: dict[str, int],
    stage_name: str,
) -> dict[str, int]:
    return {
        "input_tokens": int(counters.get(f"{stage_name}_input_tokens", 0) or 0),
        "cached_input_tokens": int(
            counters.get(f"{stage_name}_cached_input_tokens", 0) or 0
        ),
        "output_tokens": int(counters.get(f"{stage_name}_output_tokens", 0) or 0),
        "reasoning_tokens": int(counters.get(f"{stage_name}_reasoning_tokens", 0) or 0),
        "total_tokens": int(counters.get(f"{stage_name}_total_tokens", 0) or 0),
    }


def _local_slm_token_usage_summary(counters: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(counters.get("local_slm_input_tokens", 0) or 0),
        "cached_input_tokens": int(counters.get("local_slm_cached_input_tokens", 0) or 0),
        "output_tokens": int(counters.get("local_slm_output_tokens", 0) or 0),
        "reasoning_tokens": int(counters.get("local_slm_reasoning_tokens", 0) or 0),
        "total_tokens": int(counters.get("local_slm_total_tokens", 0) or 0),
    }


def _is_analyze_task(task: TaskItem) -> bool:
    return (
        table_reasoning_profile_from_dsl(task.remote_dsl)
        == TABLE_REASONING_ANALYZE_PROFILE
    )


def _dag_group_key(task: TaskItem) -> str:
    if _is_analyze_task(task):
        return f"{task.group_key}::analyze::{task.answer_key}"
    return task.group_key


def _batch_logic_dag(batch: list[LogicDagItem]) -> dict[str, Any]:
    query_plans = []
    for item in batch:
        plans = item.logic_dag.get("query_plans")
        if isinstance(plans, list) and plans:
            for plan in plans:
                if isinstance(plan, dict):
                    query_plans.append(copy.deepcopy(plan))
            continue
        query_plans.append(
            {
                "id": item.task.answer_key,
                "index": len(query_plans),
                "answer": {
                    "name": item.task.answer_key,
                    "type": item.task.answer_type,
                },
                **_query_fragment(item.logic_dag),
            }
        )
    for index, query_plan in enumerate(query_plans):
        query_plan["index"] = index
    return {
        "task_type": batch[0].task.task_type,
        "query_plans": query_plans,
    }


def _query_fragment(logic_dag: dict[str, Any]) -> dict[str, Any]:
    query_plans = logic_dag.get("query_plans")
    if isinstance(query_plans, list) and len(query_plans) == 1:
        query_plan = query_plans[0]
        if isinstance(query_plan, dict):
            return {
                "nodes": copy.deepcopy(query_plan.get("nodes", [])),
                "edges": copy.deepcopy(query_plan.get("edges", [])),
            }
    return {
        "nodes": copy.deepcopy(logic_dag.get("nodes", [])),
        "edges": copy.deepcopy(logic_dag.get("edges", [])),
    }


def _synthesis_local_dsl(batch: list[LogicDagItem]) -> dict[str, Any]:
    if len(batch) == 1 and _is_analyze_task(batch[0].task):
        return _query_local_dsl(batch[0].task)
    return _batch_local_dsl(batch)


def _sql_map(batch: list[LogicDagItem]) -> dict[str, Any]:
    return {item.task.answer_key: item.sql for item in batch}


def _compact_evidence_value(value: Any, *, max_rows: int = 20) -> Any:
    ready = copy.deepcopy(value)
    if isinstance(ready, dict) and isinstance(ready.get("rows"), list):
        rows = ready["rows"]
        ready["n"] = len(rows)
        ready["rows"] = rows[:max_rows]
    return ready


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
            for item in physical_plan.get("query_outputs", [])
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
    for item in physical_plan.get("query_outputs", []):
        answer = item.get("answer", {})
        output = item.get("output")
        if output in affected_outputs and isinstance(answer, dict):
            answer_keys.add(answer["name"])
    for output in affected_outputs:
        node = output_to_node.get(output)
        if node and node.get("op") == "FormatAnswer":
            answer_keys.add(output)
    parent_keys = {
        key.split("__e", 1)[0]
        for key in answer_keys
        if isinstance(key, str) and "__e" in key
    }
    answer_keys.update(parent_keys)
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
