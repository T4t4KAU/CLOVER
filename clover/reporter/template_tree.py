"""Prompt template tree for Reporter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clover.executor.result import json_ready
from clover.prompt_safety import strip_sensitive_prompt_fields


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class TemplateNode:
    """A node in the Reporter prompt template tree."""

    name: str
    template: str | None = None
    children: tuple["TemplateNode", ...] = ()

    def child(self, name: str) -> "TemplateNode":
        for child in self.children:
            if child.name == name:
                return child
        raise KeyError(f"Template node not found under {self.name!r}: {name!r}")

    def find_path(self, names: tuple[str, ...]) -> tuple["TemplateNode", ...]:
        if not names:
            return (self,)
        next_node = self.child(names[0])
        return (self,) + next_node.find_path(names[1:])


REPORTER_TEMPLATE_TREE = TemplateNode(
    name="root",
    template="common/root.md",
    children=(
        TemplateNode(
            name="table_reasoning",
            children=(
                TemplateNode(
                    name="v1",
                    template="table_reasoning/v1/report.md",
                ),
                TemplateNode(
                    name="v2",
                    template="table_reasoning/v2/report.md",
                ),
            ),
        ),
    ),
)


TASK_TEMPLATE_ROUTES: dict[str, tuple[str, ...]] = {
    "table_reasoning": ("table_reasoning", "v1"),
    "table_reasoning_v1": ("table_reasoning", "v1"),
    "table_reasoning_v2": ("table_reasoning", "v2"),
}

SQL_REPAIR_TEMPLATE_PATHS: dict[str, tuple[str, ...]] = {
    "table_reasoning": ("table_reasoning/v1/sql_repair.md",),
    "table_reasoning_v1": ("table_reasoning/v1/sql_repair.md",),
    "table_reasoning_v2": ("table_reasoning/v2/sql_repair.md",),
}


PUBLIC_TASK_TYPES = ("table_reasoning",)


def available_task_types() -> tuple[str, ...]:
    return PUBLIC_TASK_TYPES


def template_paths_for_task_type(task_type: str) -> tuple[str, ...]:
    return tuple(node.template for node in _route_nodes(task_type) if node.template)


def initial_report_template_paths(task_type: str) -> tuple[str, ...]:
    # The root template is sent once when a long Remote LLM session starts.
    return _root_template_paths() + template_paths_for_task_type(task_type)


def sql_repair_template_paths(task_type: str) -> tuple[str, ...]:
    try:
        return SQL_REPAIR_TEMPLATE_PATHS[task_type]
    except KeyError as exc:
        available = ", ".join(available_task_types())
        raise ValueError(
            f"Unsupported Reporter task_type: {task_type!r}. "
            f"Available task types: {available}"
        ) from exc


def render_initial_report_prompt(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    logic_dag: dict[str, Any],
    local_result: Any,
    current_sql: Any = None,
) -> str:
    task_type = _task_type(task_dsl=task_dsl, local_dsl=local_dsl, logic_dag=logic_dag)
    template_paths = _template_paths_for_local_result(
        task_type=task_type,
        local_result=local_result,
    )
    return _render_templates(
        _root_template_paths() + template_paths,
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        local_result=local_result,
        current_sql=current_sql,
    )


def render_reporter_instruction_prompt() -> str:
    """Render the reusable Reporter instruction for a maintained session."""

    return _render_templates(
        _root_template_paths(),
        task_dsl=None,
        local_dsl=None,
        local_result={},
        current_sql=None,
    )


def render_report_prompt(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    logic_dag: dict[str, Any],
    local_result: Any,
    current_sql: Any = None,
) -> str:
    task_type = _task_type(task_dsl=task_dsl, local_dsl=local_dsl, logic_dag=logic_dag)
    # Follow-up prompts omit root, mirroring Commander session reuse.
    return _render_templates(
        _template_paths_for_local_result(
            task_type=task_type,
            local_result=local_result,
        ),
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        local_result=local_result,
        current_sql=current_sql,
    )


def reporter_payload(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    local_result: Any,
    current_sql: Any = None,
) -> dict[str, Any]:
    """Build the filtered payload sent to the Reporter prompt."""

    # Reporter only needs the user-facing local result. Executor traces and
    # intermediate table summaries are internal debug artifacts and can dwarf
    # the actual answer payload in batched runs.
    return strip_sensitive_prompt_fields(
        {
            "task": _task_summary(task_dsl=task_dsl, local_dsl=local_dsl),
            "current_sql": current_sql,
            "local_results": _local_results_summary(local_result),
        }
    )


def _root_template_paths() -> tuple[str, ...]:
    if REPORTER_TEMPLATE_TREE.template is None:
        return ()
    return (REPORTER_TEMPLATE_TREE.template,)


def _template_paths_for_local_result(
    *,
    task_type: str,
    local_result: Any,
) -> tuple[str, ...]:
    if _local_result_failed(local_result):
        return sql_repair_template_paths(task_type)
    return template_paths_for_task_type(task_type)


def _render_templates(
    template_paths: tuple[str, ...],
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
    local_result: Any,
    current_sql: Any,
) -> str:
    env = _template_environment()
    payload = reporter_payload(
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        local_result=local_result,
        current_sql=current_sql,
    )
    payload_json = _to_pretty_json(payload)
    context = {
        "payload": payload,
        "payload_json": payload_json,
        "REPORT_PAYLOAD": payload_json,
        "task": payload["task"],
        "local_results": payload["local_results"],
    }

    rendered_parts = []
    for template_path in template_paths:
        rendered_parts.append(env.get_template(template_path).render(**context).strip())
    return "\n\n".join(part for part in rendered_parts if part)


def _route_nodes(task_type: str) -> tuple[TemplateNode, ...]:
    try:
        route = TASK_TEMPLATE_ROUTES[task_type]
    except KeyError as exc:
        available = ", ".join(available_task_types())
        raise ValueError(
            f"Unsupported Reporter task_type: {task_type!r}. "
            f"Available task types: {available}"
        ) from exc
    return REPORTER_TEMPLATE_TREE.find_path(route)[1:]


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _task_type(
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
    logic_dag: dict[str, Any],
) -> str:
    task_type = (
        logic_dag.get("task_type")
        or (local_dsl or {}).get("task_type")
        or (task_dsl or {}).get("task_type")
    )
    if not isinstance(task_type, str):
        raise ValueError("Reporter prompt requires task_type")
    return task_type


def _task_summary(
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
) -> dict[str, Any]:
    source = local_dsl or task_dsl or {}
    task_type = source.get("task_type")
    if task_type == "table_reasoning_v2":
        return {
            "task_type": task_type,
            "questions": json_ready(source.get("questions", [])),
            "answers": json_ready(source.get("answers", [])),
        }
    return {
        "task_type": task_type,
        "question": source.get("question"),
        "answer": json_ready(source.get("answer", {})),
    }


def _local_results_summary(local_result: Any) -> dict[str, Any]:
    if isinstance(local_result, dict):
        ok = local_result.get("ok")
        answer = local_result.get("answer")
        error = local_result.get("error")
    elif hasattr(local_result, "ok"):
        ok = getattr(local_result, "ok")
        answer = getattr(local_result, "answer", None)
        error = getattr(local_result, "error", None)
    else:
        ok = True
        answer = local_result
        error = None

    payload: dict[str, Any] = {
        "ok": ok,
        "answer": json_ready(answer),
    }
    if ok is False:
        payload["error"] = _compact_error(error)
    return strip_sensitive_prompt_fields(json_ready(payload))


def _compact_error(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, dict):
        return {
            key: error.get(key)
            for key in ("type", "message")
            if error.get(key) is not None
        }
    return {"message": str(error)}


def _local_result_failed(local_result: Any) -> bool:
    if hasattr(local_result, "ok"):
        return getattr(local_result, "ok") is False
    if isinstance(local_result, dict):
        return local_result.get("ok") is False
    return False


def _to_pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
