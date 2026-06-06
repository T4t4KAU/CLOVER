"""Generic runtime task lifecycle records shared by task families."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from clover.resource import preprocess_task_dsl
from clover.runtime.pipeline import CaseResult


TASK_PENDING_REMOTE = "pending_remote"
TASK_SQL_READY = "sql_ready"
TASK_CODE_READY = "code_ready"
TASK_DAG_READY = "dag_ready"
TASK_EXECUTING = "executing"
TASK_SUPERVISOR_REVIEW = "supervisor_review"
TASK_RETRYING = "retrying"
TASK_SUCCESS = "success"
TASK_FAILED = "failed"


@dataclass
class RuntimeCaseSpec:
    """External case input normalized before becoming a RuntimeTaskItem."""

    case_id: str
    task_dsl: dict[str, Any]
    base_dir: str | Path
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocess_result: dict[str, Any] | None = None
    answer_key: str | None = None


@dataclass
class RuntimeTaskItem:
    """Internal lifecycle record for one question/answer."""

    case_id: str
    answer_key: str
    task_type: str
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
    current_command: Any | None = None
    memory: list[dict[str, Any]] = field(default_factory=list)
    last_error: dict[str, Any] | None = None
    result_callback: Callable[[CaseResult], None] | None = field(
        default=None,
        repr=False,
    )

    @property
    def group_key(self) -> str:
        """Return the batching key for tasks that share the same source."""

        return self.source_file

    @property
    def current_sql(self) -> str | None:
        """Return the SQL text carried by the current table command, if any."""

        return _current_sql(self.current_command)


@dataclass
class TableTaskItem(RuntimeTaskItem):
    """Runtime task item for table reasoning."""

    pass


@dataclass
class DocumentTaskItem(RuntimeTaskItem):
    """Runtime task item for document reasoning."""

    pass


def _current_sql(command: Any) -> str | None:
    if command is None:
        return None
    if isinstance(command, str):
        return command
    if isinstance(command, dict):
        sql_values = []
        if isinstance(command.get("sql"), str):
            sql_values.append(command["sql"])
        if command.get("op") == "sql" and isinstance(command.get("q"), str):
            sql_values.append(command["q"])
        acts = command.get("acts")
        if isinstance(acts, list):
            for act in acts:
                nested = _current_sql(act)
                if nested:
                    sql_values.append(nested)
        return "\n".join(sql_values) if sql_values else None
    if isinstance(command, list):
        sql_values = [_current_sql(item) for item in command]
        return "\n".join(item for item in sql_values if item) or None
    return None


class AnswerKeyAllocator:
    """Runtime-scoped global answer key allocator."""

    def __init__(self) -> None:
        self._next_index = 1

    def next(self) -> str:
        answer_key = f"answer_{self._next_index}"
        self._next_index += 1
        return answer_key


CaseSpecT = TypeVar("CaseSpecT", bound=RuntimeCaseSpec)
TaskItemT = TypeVar("TaskItemT", bound=RuntimeTaskItem)


def build_runtime_task_items(
    case_specs: list[CaseSpecT | dict[str, Any]],
    *,
    task_type: str,
    case_spec_class: type[CaseSpecT],
    task_item_class: type[TaskItemT],
    case_result_callback: Callable[[CaseResult], None] | None = None,
) -> dict[str, TaskItemT]:
    """Build task lifecycle records from normalized case specs."""

    allocator = AnswerKeyAllocator()
    task_items: dict[str, TaskItemT] = {}
    for raw_spec in case_specs:
        spec = normalize_runtime_case_spec(raw_spec, case_spec_class)
        preprocess_result = spec.preprocess_result or preprocess_task_dsl(
            spec.task_dsl,
            base_dir=spec.base_dir,
        )
        answer_key = spec.answer_key or allocator.next()
        local_dsl = with_answer_key(preprocess_result["local_dsl"], answer_key)
        remote_dsl = with_answer_key(
            preprocess_result["remote_dsl"],
            local_dsl["answer"]["name"],
        )
        source = _single_source(local_dsl)
        source_file = str(Path(source["path"]).expanduser().resolve())
        context = copy.deepcopy(preprocess_result["context"])
        item_task_type = local_dsl.get("task_type", task_type)
        task_items[answer_key] = task_item_class(
            case_id=spec.case_id,
            answer_key=answer_key,
            task_type=item_task_type,
            question=local_dsl["question"],
            answer_type=local_dsl["answer"]["type"],
            source_file=source_file,
            source_id=source["id"],
            task_dsl=copy.deepcopy(spec.task_dsl),
            local_dsl=local_dsl,
            remote_dsl=remote_dsl,
            context=context,
            metadata=copy.deepcopy(spec.metadata),
            result_callback=case_result_callback,
        )
    return task_items


def normalize_runtime_case_spec(
    raw_spec: CaseSpecT | dict[str, Any],
    case_spec_class: type[CaseSpecT],
) -> CaseSpecT:
    """Normalize a case-spec mapping into the configured case spec class."""

    if isinstance(raw_spec, case_spec_class):
        return raw_spec
    return case_spec_class(
        case_id=str(raw_spec["case_id"]),
        task_dsl=raw_spec["task_dsl"],
        base_dir=raw_spec["base_dir"],
        metadata=dict(raw_spec.get("metadata", {})),
        preprocess_result=raw_spec.get("preprocess_result"),
        answer_key=raw_spec.get("answer_key"),
    )


def with_answer_key(dsl: dict[str, Any], answer_key: str) -> dict[str, Any]:
    """Return a DSL copy whose answer name is the runtime answer key."""

    updated = copy.deepcopy(dsl)
    updated["answer"] = copy.deepcopy(updated.get("answer", {}))
    updated["answer"]["name"] = answer_key
    return updated


def _single_source(local_dsl: dict[str, Any]) -> dict[str, Any]:
    sources = local_dsl.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Runtime task requires at least one local source")
    source = sources[0]
    if not isinstance(source, dict) or "path" not in source or "id" not in source:
        raise ValueError("Runtime task source requires id and path")
    return source
