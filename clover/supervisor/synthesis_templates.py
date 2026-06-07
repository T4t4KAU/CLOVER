"""Prompt template tree for Supervisor synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clover.executor.result import json_ready
from clover.reasoning_profiles import (
    TABLE_REASONING_ANALYZE_PROFILE,
    short_table_reasoning_profile,
    table_reasoning_hints_from_dsl,
    table_reasoning_profile_from_dsl,
)
from clover.supervisor.observations import build_compact_document_observation
from clover.supervisor.prompt_safety import (
    sanitize_source_for_prompt,
    strip_sensitive_prompt_fields,
)
from clover.task_types import (
    DOCUMENT_REASONING_TASK_TYPE,
    is_table_task_type,
    public_task_types,
)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "synthesize"


@dataclass(frozen=True)
class TemplateNode:
    """A node in the Supervisor synthesis prompt template tree."""

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


SYNTHESIS_TEMPLATE_TREE = TemplateNode(
    name="root",
    template="common/root.md",
    children=(
        TemplateNode(
            name="table_reasoning",
            children=(
                TemplateNode(
                    name="query",
                    template="table_reasoning/query/synthesize.md",
                ),
                TemplateNode(
                    name="batch_query",
                    template="table_reasoning/batch_query/synthesize.md",
                ),
            ),
        ),
        TemplateNode(
            name="document_reasoning",
            template="document_reasoning/synthesize.md",
        ),
    ),
)


TABLE_REASONING_QUERY_ROUTE = ("table_reasoning", "query")
TABLE_REASONING_BATCH_QUERY_ROUTE = ("table_reasoning", "batch_query")
DOCUMENT_REASONING_ROUTE = ("document_reasoning",)


def available_task_types() -> tuple[str, ...]:
    return public_task_types()


def render_initial_synthesis_prompt(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    logic_dag: dict[str, Any],
    observation: Any,
    current_command: Any = None,
    force_final_answer: bool = False,
) -> str:
    template_paths = _template_paths_for_synthesis(
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        logic_dag=logic_dag,
        observation=observation,
    )
    return _render_templates(
        _root_template_paths() + template_paths,
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        observation=observation,
        current_command=current_command,
        force_final_answer=force_final_answer,
    )


def render_synthesis_prompt(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    logic_dag: dict[str, Any],
    observation: Any,
    current_command: Any = None,
    force_final_answer: bool = False,
) -> str:
    # Follow-up prompts may omit the tiny root instruction when the
    # task-specific template already carries the output contract.
    return _render_templates(
        _template_paths_for_synthesis(
            task_dsl=task_dsl,
            local_dsl=local_dsl,
            logic_dag=logic_dag,
            observation=observation,
        ),
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        observation=observation,
        current_command=current_command,
        force_final_answer=force_final_answer,
    )


def synthesis_payload(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    observation: Any,
    current_command: Any = None,
) -> dict[str, Any]:
    """Build the filtered observation payload sent to Supervisor synthesis."""

    task_type = _synthesis_task_type(task_dsl=task_dsl, local_dsl=local_dsl)
    if is_table_task_type(task_type):
        return strip_sensitive_prompt_fields(
            _table_evidence_payload(
                local_dsl=local_dsl,
                task_dsl=task_dsl,
                observation=observation,
                current_command=current_command,
            )
        )
    # Document synthesis only needs the user-facing local result. Executor traces and
    # intermediate worker details are internal debug artifacts and can dwarf
    # the actual answer payload in batched runs.
    include_sources = (
        task_type != DOCUMENT_REASONING_TASK_TYPE
        and _observation_failed(observation)
    )
    payload: dict[str, Any] = {
        "task": _task_summary(
            task_dsl=task_dsl,
            local_dsl=local_dsl,
            include_sources=include_sources,
        ),
        "observations": (
            _document_observations_summary(observation)
            if task_type == DOCUMENT_REASONING_TASK_TYPE
            else _observations_summary(observation)
        ),
    }
    if current_command is not None:
        payload["current_command"] = current_command
    return strip_sensitive_prompt_fields(
        payload
    )


def _root_template_paths() -> tuple[str, ...]:
    if SYNTHESIS_TEMPLATE_TREE.template is None:
        return ()
    return (SYNTHESIS_TEMPLATE_TREE.template,)


def _template_paths_for_synthesis(
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
    logic_dag: dict[str, Any],
    observation: Any,
) -> tuple[str, ...]:
    if _is_table_analyze_dsl(local_dsl or task_dsl):
        return _template_paths_for_route(TABLE_REASONING_QUERY_ROUTE)
    if _is_table_batch_synthesis_context(
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        logic_dag=logic_dag,
    ):
        return _template_paths_for_route(TABLE_REASONING_BATCH_QUERY_ROUTE)
    task_type = _task_type(
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        logic_dag=logic_dag,
    )
    if is_table_task_type(task_type):
        return _template_paths_for_route(TABLE_REASONING_QUERY_ROUTE)
    if task_type == DOCUMENT_REASONING_TASK_TYPE:
        return _template_paths_for_route(DOCUMENT_REASONING_ROUTE)
    available = ", ".join(available_task_types())
    raise ValueError(
        f"Unsupported Supervisor synthesis task_type: {task_type!r}. "
        f"Available task types: {available}"
    )


def _render_templates(
    template_paths: tuple[str, ...],
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
    observation: Any,
    current_command: Any,
    force_final_answer: bool = False,
) -> str:
    env = _template_environment()
    payload = synthesis_payload(
        task_dsl=task_dsl,
        local_dsl=local_dsl,
        observation=observation,
        current_command=current_command,
    )
    payload_json = _to_pretty_json(payload)
    context = {
        "payload": payload,
        "payload_json": payload_json,
        "OBSERVATION_PAYLOAD": payload_json,
        "task": payload.get("task", {}),
        "observations": payload.get("observations", {}),
        "force_final_answer": force_final_answer,
    }

    rendered_parts = []
    for template_path in template_paths:
        rendered_parts.append(env.get_template(template_path).render(**context).strip())
    return "\n\n".join(part for part in rendered_parts if part)


def _template_paths_for_route(route: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        node.template
        for node in SYNTHESIS_TEMPLATE_TREE.find_path(route)[1:]
        if node.template
    )


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
    task_type = _template_task_type(
        logic_dag=logic_dag,
        local_dsl=local_dsl,
        task_dsl=task_dsl,
    )
    if not isinstance(task_type, str):
        raise ValueError("Supervisor synthesis prompt requires task_type")
    return task_type


def _template_task_type(
    *,
    logic_dag: dict[str, Any],
    local_dsl: dict[str, Any] | None,
    task_dsl: dict[str, Any] | None,
) -> str | None:
    task_type = (
        logic_dag.get("task_type")
        or (local_dsl or {}).get("task_type")
        or (task_dsl or {}).get("task_type")
    )
    if not isinstance(task_type, str):
        return None
    return task_type


def _is_table_batch_synthesis_context(
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
    logic_dag: dict[str, Any],
) -> bool:
    if not is_table_task_type(logic_dag.get("task_type")):
        return False
    if isinstance(logic_dag.get("query_plans"), list):
        return True
    if isinstance(logic_dag.get("query_outputs"), list):
        return True
    return _is_table_batch_task(local_dsl) or _is_table_batch_task(task_dsl)


def _task_summary(
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
    include_sources: bool,
) -> dict[str, Any]:
    source = local_dsl or task_dsl or {}
    task_type = source.get("task_type") if isinstance(source.get("task_type"), str) else None
    if is_table_task_type(task_type) and _is_table_batch_task(source):
        summary = {
            "task_type": task_type,
            "questions": json_ready(source.get("questions", [])),
            "answers": json_ready(source.get("answers", [])),
        }
        _add_profile_fields(summary, source)
        sources = _source_summary(source) if include_sources else []
        if sources:
            summary["sources"] = sources
        return summary
    summary = {
        "task_type": task_type,
        "question": source.get("question"),
        "answer": json_ready(source.get("answer", {})),
    }
    _add_profile_fields(summary, source)
    sources = _source_summary(source) if include_sources else []
    if sources:
        summary["sources"] = sources
    return summary


def _add_profile_fields(summary: dict[str, Any], source: dict[str, Any]) -> None:
    if not is_table_task_type(source.get("task_type")):
        return
    profile = table_reasoning_profile_from_dsl(source)
    summary["profile"] = short_table_reasoning_profile(profile)
    hints = table_reasoning_hints_from_dsl(source)
    if hints:
        summary["hints"] = json_ready(hints)


def _source_summary(source: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in source.get("sources", []):
        if not isinstance(item, dict):
            continue
        safe_source = sanitize_source_for_prompt(item)
        source_summary: dict[str, Any] = {"id": safe_source.get("id")}
        columns = (
            safe_source.get("schema", {}).get("columns")
            if isinstance(safe_source.get("schema"), dict)
            else None
        )
        if columns:
            source_summary["schema"] = {"columns": columns}
        summaries.append(json_ready(source_summary))
    return summaries


def _table_evidence_payload(
    *,
    local_dsl: dict[str, Any] | None,
    task_dsl: dict[str, Any] | None,
    observation: Any,
    current_command: Any,
) -> dict[str, Any]:
    source = local_dsl or task_dsl or {}
    if _is_table_batch_task(source):
        payload: dict[str, Any] = {
            "qs": _batch_questions(source),
            "ev": _table_evidence_observations(observation),
        }
    else:
        payload = {
            "q": source.get("question"),
            "ty": _answer_type(source),
            "ev": _table_evidence_observations(observation),
        }
    ctx = _table_repair_context(
        source=source,
        observation=observation,
        current_command=current_command,
    )
    if ctx:
        payload["ctx"] = ctx
    mem = source.get("mem")
    if mem:
        payload["mem"] = _compact_table_memory(mem)
    return json_ready(payload)


def _batch_questions(source: dict[str, Any]) -> list[dict[str, Any]]:
    questions = source.get("questions")
    answers = source.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list):
        return []
    output = []
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            continue
        item: dict[str, Any] = {}
        name = answer.get("name")
        if isinstance(name, str) and name:
            item["id"] = name
        answer_type = answer.get("type")
        if isinstance(answer_type, str) and answer_type:
            item["ty"] = answer_type
        if index < len(questions):
            item["q"] = questions[index]
        output.append(item)
    return output


def _answer_type(source: dict[str, Any]) -> str | None:
    answer = source.get("answer")
    if not isinstance(answer, dict):
        return None
    answer_type = answer.get("type")
    return answer_type if isinstance(answer_type, str) else None


def _table_columns_map(source: dict[str, Any]) -> dict[str, list[Any]]:
    tables: dict[str, list[Any]] = {}
    for item in source.get("sources", []):
        if not isinstance(item, dict):
            continue
        table_id = item.get("id")
        if not isinstance(table_id, str) or not table_id:
            continue
        schema = item.get("schema")
        columns = schema.get("columns") if isinstance(schema, dict) else None
        tables[table_id] = list(columns) if isinstance(columns, list) else []
    return tables


def _current_actions_list(current_command: Any) -> list[dict[str, Any]]:
    if isinstance(current_command, dict):
        acts = current_command.get("acts", current_command.get("actions"))
        if isinstance(acts, list):
            output = []
            for item in acts:
                if not isinstance(item, dict):
                    continue
                action = _compact_action_dict(item)
                if action:
                    output.append(action)
            return output
        op = current_command.get("op")
        if isinstance(op, str) and op:
            action = _compact_action_dict(current_command)
            return [action] if action else []
    return [{"op": "sql", "q": sql} for sql in _current_sql_list(current_command)]


def _compact_action_dict(value: dict[str, Any]) -> dict[str, Any]:
    action: dict[str, Any] = {}
    op = value.get("op")
    if isinstance(op, str) and op:
        action["op"] = op
    kind = value.get("kind")
    if isinstance(kind, str) and kind:
        action["kind"] = kind
    q = value.get("q") or value.get("sql")
    if isinstance(q, str) and q:
        action["q"] = q
    seed = value.get("seed")
    if isinstance(seed, str) and seed:
        action["seed"] = seed
    return action


def _current_sql_list(current_command: Any) -> list[str]:
    if isinstance(current_command, dict):
        value = current_command.get(
            "seed",
            current_command.get(
                "seeds",
                current_command.get("sqls", current_command.get("sql")),
            ),
        )
    else:
        value = current_command
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _table_evidence_observations(observation: Any) -> list[Any]:
    if not isinstance(observation, dict):
        return [_table_object_observation(observation)]
    obs = observation.get("obs")
    if isinstance(obs, list):
        return [_compact_action_observation(item) for item in obs]
    if observation.get("error"):
        return [{"i": 0, "ok": False, "err": _compact_error(observation.get("error"))}]
    return [_table_object_observation(observation)]


def _table_object_observation(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "ok"):
        ok = bool(getattr(observation, "ok"))
        output: dict[str, Any] = {"ok": ok}
        if ok:
            answer = getattr(observation, "answer", None)
            if answer is not None:
                output["answer"] = json_ready(answer)
            outputs = getattr(observation, "outputs", None)
            if isinstance(outputs, dict) and outputs:
                output["outputs"] = _compact_result_payload(outputs)
        else:
            output["err"] = _compact_error(getattr(observation, "error", None))
        return output
    if isinstance(observation, dict):
        ok = observation.get("ok")
        output = {"ok": bool(ok) if ok is not None else True}
        if "answer" in observation and observation.get("answer") is not None:
            output["answer"] = json_ready(observation.get("answer"))
        if "outputs" in observation and isinstance(observation.get("outputs"), dict):
            output["outputs"] = _compact_result_payload(observation.get("outputs"))
        if "ev" in observation:
            output["ev"] = _compact_result_payload(observation.get("ev"))
        if output["ok"] is False:
            output["err"] = _compact_error(observation.get("error"))
        if len(output) > 1 or output["ok"] is False:
            return output
    return {"ok": True, "res": _table_result_value(observation)}


def _table_repair_context(
    *,
    source: dict[str, Any],
    observation: Any,
    current_command: Any,
) -> dict[str, Any]:
    if not _table_evidence_needs_repair(observation):
        return {}
    ctx: dict[str, Any] = {}
    actions = _current_actions_list(current_command)
    if actions:
        ctx["act"] = actions
    columns = _table_columns_map(source)
    if columns:
        ctx["t"] = columns
    return ctx


def _table_evidence_needs_repair(observation: Any) -> bool:
    if _observation_failed(observation):
        return True
    if isinstance(observation, dict):
        obs = observation.get("obs")
        if isinstance(obs, list):
            return any(_action_observation_needs_repair(item) for item in obs)
    return False


def _action_observation_needs_repair(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("ok") is False or value.get("err") is not None:
        return True
    result = value.get("res")
    if isinstance(result, dict):
        rows = result.get("rows")
        count = result.get("n")
        if isinstance(rows, list) and not rows:
            return True
        if count == 0:
            return True
    return False


def _compact_action_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"ok": True, "res": _table_result_value(value)}
    output: dict[str, Any] = {}
    if "i" in value:
        output["i"] = value.get("i")
    if isinstance(value.get("op"), str):
        output["op"] = value.get("op")
    if isinstance(value.get("kind"), str):
        output["kind"] = value.get("kind")
    if "ok" in value:
        output["ok"] = bool(value.get("ok"))
    if "res" in value:
        output["res"] = _table_result_value(value.get("res"))
    if "ev" in value:
        output["ev"] = _compact_result_payload(value.get("ev"))
    if "err" in value:
        output["err"] = _compact_error(value.get("err"))
    for key in ("seed_ok", "seed_err", "seed_note", "it"):
        if key not in value:
            continue
        output[key] = (
            _compact_error(value.get(key))
            if key.endswith("err")
            else json_ready(value.get(key))
        )
    return output


def _table_result_value(value: Any) -> Any:
    if isinstance(value, dict):
        error = value.get("error") or value.get("err")
        if error:
            return {"err": _compact_error(error)}
        if "ev" in value:
            return {"ev": _compact_result_payload(value.get("ev"))}
        if value.get("op") == "ev" and "evidence" in value:
            return {"ev": _compact_result_payload(value.get("evidence"))}
        rows = value.get("rows")
        if isinstance(rows, list):
            columns = _table_result_columns(value, rows)
            return {
                "n": _table_result_count(value, rows),
                "cols": columns,
                "rows": _table_result_rows(rows, columns),
            }
    return json_ready(value)


def _compact_result_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get("rows"), int) and isinstance(value.get("data"), list):
            data = value.get("data", [])
            cols = value.get("cols", [])
            if not isinstance(cols, list):
                cols = []
            return {
                "n": value.get("rows"),
                "cols": json_ready(cols[:40]),
                "rows": json_ready(data[:10]),
            }
        return {
            str(key): _compact_result_payload(item)
            for key, item in list(value.items())[:20]
            if key not in {"op", "ask", "need"}
        }
    if isinstance(value, list):
        return [_compact_result_payload(item) for item in value[:20]]
    text = str(value) if isinstance(value, str) else None
    if text is not None and len(text) > 500:
        return text[:500] + "...<truncated>"
    return json_ready(value)


def _table_result_columns(value: dict[str, Any], rows: list[Any]) -> list[Any]:
    for key in ("cols", "columns"):
        columns = value.get(key)
        if isinstance(columns, list):
            return list(columns)
    if rows and isinstance(rows[0], dict):
        return list(rows[0].keys())
    return []


def _table_result_count(value: dict[str, Any], rows: list[Any]) -> int:
    for key in ("n", "row_count", "rows_count"):
        count = value.get(key)
        if isinstance(count, int):
            return count
    return len(rows)


def _table_result_rows(rows: list[Any], columns: list[Any], max_rows: int = 10) -> list[Any]:
    output = []
    for row in rows[:max_rows]:
        if isinstance(row, dict) and columns:
            output.append([row.get(column) for column in columns])
        else:
            output.append(row)
    return output


def _compact_table_memory(mem: Any, max_items: int = 2) -> list[Any]:
    if not isinstance(mem, list):
        return []
    return json_ready(mem[-max_items:])


def _is_table_batch_task(source: dict[str, Any] | None) -> bool:
    if not isinstance(source, dict):
        return False
    return isinstance(source.get("questions"), list) and isinstance(
        source.get("answers"),
        list,
    )


def _observations_summary(observation: Any) -> dict[str, Any]:
    if isinstance(observation, dict):
        ok = observation.get("ok")
        answer = observation.get("answer")
        error = observation.get("error")
        evidence = observation.get("ev")
    elif hasattr(observation, "ok"):
        ok = getattr(observation, "ok")
        answer = getattr(observation, "answer", None)
        error = getattr(observation, "error", None)
        evidence = None
    else:
        ok = True
        answer = observation
        error = None
        evidence = None

    payload: dict[str, Any] = {
        "ok": ok,
        "answer": json_ready(answer),
    }
    if evidence is not None:
        payload["ev"] = json_ready(evidence)
    if ok is False:
        payload["error"] = _compact_error(error)
    return strip_sensitive_prompt_fields(json_ready(payload))


def _document_observations_summary(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        observation = build_compact_document_observation(observation)
    return {
        "ok": observation.get("ok", True),
        "worker_count": _non_negative_int(observation.get("worker_count")),
        "included_count": _non_negative_int(observation.get("included_count")),
        "failed_count": _non_negative_int(observation.get("failed_count")),
        "evidence_summary": _string_or_empty(observation.get("evidence_summary")),
        "evidence_truncated": bool(observation.get("evidence_truncated", False)),
        "prior_evidence_summary": _string_or_empty(
            observation.get("prior_evidence_summary")
        ),
        "prior_evidence_round_count": _non_negative_int(
            observation.get("prior_evidence_round_count")
        ),
        "prior_evidence_truncated": bool(
            observation.get("prior_evidence_truncated", False)
        ),
        "fallback_used": bool(observation.get("fallback_used", False)),
        "transform_error": _optional_string(observation.get("transform_error")),
        "error": _compact_error(observation.get("error")),
        "round_index": observation.get("round_index"),
        "feedback": _optional_string(observation.get("feedback")),
        "scratchpad": _optional_string(observation.get("scratchpad")),
    }


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


def _non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_or_empty(value: Any) -> str:
    return _optional_string(value) or ""


def _observation_failed(observation: Any) -> bool:
    if hasattr(observation, "ok"):
        return getattr(observation, "ok") is False
    if isinstance(observation, dict):
        return observation.get("ok") is False
    return False


def _synthesis_task_type(
    *,
    task_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any] | None,
) -> str | None:
    source = local_dsl or task_dsl or {}
    task_type = source.get("task_type")
    return task_type if isinstance(task_type, str) else None


def _is_table_analyze_dsl(source: dict[str, Any] | None) -> bool:
    if not isinstance(source, dict):
        return False
    if not is_table_task_type(source.get("task_type")):
        return False
    return table_reasoning_profile_from_dsl(source) == TABLE_REASONING_ANALYZE_PROFILE


def _to_pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
