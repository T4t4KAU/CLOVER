"""Prompt template tree for Commander."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clover.prompt_safety import sanitize_task_dsl_for_prompt


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class TemplateNode:
    """A node in the Commander prompt template tree."""

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
                    name="v1",
                    template="table_reasoning/v1/system.md",
                    children=(
                        TemplateNode(
                            name="sql_constraints",
                            template="table_reasoning/v1/sql_constraints.md",
                            children=(
                                TemplateNode(
                                    name="sql_generation",
                                    template="table_reasoning/v1/sql_generation.md",
                                ),
                            ),
                        ),
                    ),
                ),
                TemplateNode(
                    name="v2",
                    template="table_reasoning/v2/system.md",
                    children=(
                        TemplateNode(
                            name="sql_constraints",
                            template="table_reasoning/v2/sql_constraints.md",
                            children=(
                                TemplateNode(
                                    name="sql_generation",
                                    template="table_reasoning/v2/sql_generation.md",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


TASK_TEMPLATE_ROUTES: dict[str, tuple[str, ...]] = {
    "table_reasoning": (
        "table_reasoning",
        "v1",
        "sql_constraints",
        "sql_generation",
    ),
    "table_reasoning_v1": (
        "table_reasoning",
        "v1",
        "sql_constraints",
        "sql_generation",
    ),
    "table_reasoning_v2": (
        "table_reasoning",
        "v2",
        "sql_constraints",
        "sql_generation",
    ),
}

FOLLOWUP_TEMPLATE_PATHS: dict[str, tuple[str, ...]] = {
    "table_reasoning_v2": ("table_reasoning/v2/followup_sql_generation.md",),
}


PUBLIC_TASK_TYPES = ("table_reasoning",)


def available_task_types() -> tuple[str, ...]:
    return PUBLIC_TASK_TYPES


def template_paths_for_task_type(task_type: str) -> tuple[str, ...]:
    return tuple(node.template for node in _route_nodes(task_type) if node.template)


def initial_task_template_paths(task_type: str) -> tuple[str, ...]:
    # The root template is sent once when a long Remote LLM session starts.
    return _root_template_paths() + template_paths_for_task_type(task_type)


def render_initial_task_prompt(task_dsl: dict[str, Any]) -> str:
    task_type = task_dsl.get("task_type")
    if not isinstance(task_type, str):
        raise ValueError("Commander prompt requires task_dsl.task_type")
    return _render_templates(initial_task_template_paths(task_type), task_dsl)


def render_task_prompt(task_dsl: dict[str, Any]) -> str:
    task_type = task_dsl.get("task_type")
    if not isinstance(task_type, str):
        raise ValueError("Commander prompt requires task_dsl.task_type")
    # Follow-up prompts omit the root template to keep the session-level
    # instruction stable across tasks in the same remote conversation.
    return _render_templates(template_paths_for_task_type(task_type), task_dsl)


def render_followup_task_prompt(task_dsl: dict[str, Any]) -> str:
    """Render a task follow-up prompt for an existing Remote LLM session."""

    task_type = task_dsl.get("task_type")
    if not isinstance(task_type, str):
        raise ValueError("Commander prompt requires task_dsl.task_type")
    template_paths = FOLLOWUP_TEMPLATE_PATHS.get(task_type)
    if template_paths is None:
        return render_task_prompt(task_dsl)
    batch_payload = dict(task_dsl)
    # Follow-up prompts in a maintained table session rely on schema already
    # sent in the initial batch; only new questions and answer keys are fresh.
    batch_payload.pop("sources", None)
    return _render_templates(template_paths, batch_payload)


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


def _route_nodes(task_type: str) -> tuple[TemplateNode, ...]:
    try:
        route = TASK_TEMPLATE_ROUTES[task_type]
    except KeyError as exc:
        available = ", ".join(available_task_types())
        raise ValueError(
            f"Unsupported Commander task_type: {task_type!r}. "
            f"Available task types: {available}"
        ) from exc
    return PROMPT_TEMPLATE_TREE.find_path(route)[1:]


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _to_pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
