"""Prompt template tree for Supervisor decomposition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clover.reasoning_profiles import (
    TABLE_REASONING_ANALYZE_PROFILE,
    TABLE_REASONING_QUERY_PROFILE,
    table_reasoning_profile_from_dsl,
)
from clover.supervisor.prompt_safety import sanitize_task_dsl_for_prompt
from clover.task_types import (
    DOCUMENT_REASONING_TASK_TYPE,
    is_table_task_type,
    public_task_types,
)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "decompose"


@dataclass(frozen=True)
class TemplateNode:
    """A node in the decomposition prompt template tree."""

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


PROMPT_TEMPLATE_TREE = TemplateNode(
    name="root",
    template="common/root.md",
    children=(
        TemplateNode(
            name="table_reasoning",
            children=(
                TemplateNode(
                    name="query",
                    template="table_reasoning/query/system.md",
                    children=(
                        TemplateNode(
                            name="sql_constraints",
                            template="table_reasoning/query/sql_constraints.md",
                            children=(
                                TemplateNode(
                                    name="sql_generation",
                                    template="table_reasoning/query/sql_generation.md",
                                ),
                            ),
                        ),
                    ),
                ),
                TemplateNode(
                    name="batch_query",
                    template="table_reasoning/batch_query/system.md",
                    children=(
                        TemplateNode(
                            name="sql_constraints",
                            template="table_reasoning/batch_query/sql_constraints.md",
                            children=(
                                TemplateNode(
                                    name="sql_generation",
                                    template="table_reasoning/batch_query/sql_generation.md",
                                ),
                            ),
                        ),
                    ),
                ),
                TemplateNode(
                    name="profiles",
                    children=(
                        TemplateNode(
                            name="query",
                            template="table_reasoning/profiles/query.md",
                        ),
                        TemplateNode(
                            name="analyze",
                            template="table_reasoning/profiles/analyze.md",
                        ),
                    ),
                ),
            ),
        ),
        TemplateNode(
            name="document_reasoning",
            template="document_reasoning/system.md",
            children=(
                TemplateNode(
                    name="python_code_contract",
                    template="document_reasoning/python_code_contract.md",
                    children=(
                        TemplateNode(
                            name="python_generation",
                            template="document_reasoning/python_generation.md",
                        ),
                    ),
                ),
            ),
        ),
    ),
)


TABLE_REASONING_QUERY_ROUTE = (
    "table_reasoning",
    "query",
    "sql_constraints",
    "sql_generation",
)

TABLE_REASONING_BATCH_QUERY_ROUTE = (
    "table_reasoning",
    "batch_query",
    "sql_constraints",
    "sql_generation",
)

TABLE_REASONING_PROFILE_ROUTES = {
    TABLE_REASONING_QUERY_PROFILE: ("table_reasoning", "profiles", "query"),
    TABLE_REASONING_ANALYZE_PROFILE: ("table_reasoning", "profiles", "analyze"),
}

DOCUMENT_REASONING_ROUTE = (
    "document_reasoning",
    "python_code_contract",
    "python_generation",
)


def available_task_types() -> tuple[str, ...]:
    return public_task_types()


def render_initial_task_prompt(task_dsl: dict[str, Any]) -> str:
    task_type = _template_task_type(task_dsl)
    if not isinstance(task_type, str):
        raise ValueError("Supervisor decomposition prompt requires task_dsl.task_type")
    return _render_templates(
        _root_template_paths() + _template_paths_for_task_dsl(task_dsl),
        task_dsl,
    )


def _root_template_paths() -> tuple[str, ...]:
    if PROMPT_TEMPLATE_TREE.template is None:
        return ()
    return (PROMPT_TEMPLATE_TREE.template,)


def _render_templates(template_paths: tuple[str, ...], task_dsl: dict[str, Any]) -> str:
    env = _template_environment()
    prompt_task_dsl = sanitize_task_dsl_for_prompt(task_dsl)
    task_dsl_json = _to_pretty_json(prompt_task_dsl)
    context = {
        "task_dsl": prompt_task_dsl,
        "task_dsl_json": task_dsl_json,
        "TASK_DSL": task_dsl_json,
        "BATCH_PAYLOAD": task_dsl_json,
        "source_count": len(prompt_task_dsl.get("sources", [])),
        "answer": prompt_task_dsl.get("answer", {}),
    }

    rendered_parts = []
    for template_path in template_paths:
        rendered_parts.append(env.get_template(template_path).render(**context).strip())
    return "\n\n".join(part for part in rendered_parts if part)


def _template_paths_for_task_dsl(task_dsl: dict[str, Any]) -> tuple[str, ...]:
    task_type = _template_task_type(task_dsl)
    if not isinstance(task_type, str):
        raise ValueError("Supervisor decomposition prompt requires task_dsl.task_type")
    if is_table_task_type(task_type):
        if _is_table_batch_task(task_dsl):
            raise ValueError(
                "Batch table decomposition is disabled; submit one question per request"
            )
        return _with_table_profile_template(
            _template_paths_for_route(TABLE_REASONING_QUERY_ROUTE),
            task_dsl,
        )
    if task_type == DOCUMENT_REASONING_TASK_TYPE:
        return _template_paths_for_route(DOCUMENT_REASONING_ROUTE)
    available = ", ".join(available_task_types())
    raise ValueError(
        f"Unsupported Supervisor decomposition task_type: {task_type!r}. "
        f"Available task types: {available}"
    )


def _template_paths_for_route(route: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        node.template
        for node in PROMPT_TEMPLATE_TREE.find_path(route)[1:]
        if node.template
    )


def _with_table_profile_template(
    template_paths: tuple[str, ...],
    task_dsl: dict[str, Any],
) -> tuple[str, ...]:
    profile_template = _table_profile_template_path(task_dsl)
    if profile_template is None:
        return template_paths
    return template_paths[:-1] + (profile_template,) + template_paths[-1:]


def _table_profile_template_path(task_dsl: dict[str, Any]) -> str | None:
    profile = table_reasoning_profile_from_dsl(
        task_dsl,
        default=TABLE_REASONING_QUERY_PROFILE,
    )
    route = TABLE_REASONING_PROFILE_ROUTES.get(profile)
    if route is None:
        return None
    nodes = PROMPT_TEMPLATE_TREE.find_path(route)
    return nodes[-1].template


def _template_task_type(task_dsl: dict[str, Any]) -> str | None:
    task_type = task_dsl.get("task_type")
    if not isinstance(task_type, str):
        return None
    return task_type


def _is_table_batch_task(task_dsl: dict[str, Any]) -> bool:
    questions = task_dsl.get("questions")
    answers = task_dsl.get("answers")
    return (
        isinstance(questions, list)
        and isinstance(answers, list)
        and (len(questions) > 1 or len(answers) > 1)
    )


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _to_pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
