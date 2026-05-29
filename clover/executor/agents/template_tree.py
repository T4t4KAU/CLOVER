"""Prompt template tree for local NodeAgents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clover.executor.result import json_ready


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class TemplateNode:
    """A node in the local NodeAgent prompt template tree."""

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


NODE_AGENT_TEMPLATE_TREE = TemplateNode(
    name="root",
    template="common/root.md",
    children=(
        TemplateNode(
            name="table_reasoning",
            template="table_reasoning/agent_loop.md",
        ),
    ),
)


TASK_TEMPLATE_ROUTES: dict[str, tuple[str, ...]] = {
    "table_reasoning": ("table_reasoning",),
    "table_reasoning_v1": ("table_reasoning",),
    "table_reasoning_v2": ("table_reasoning",),
}


def render_agent_loop_prompt(
    *,
    task_type: str,
    view: dict[str, Any],
    iteration: int,
) -> str:
    """Render a local NodeAgent Agent Loop prompt."""

    return _render_templates(
        _root_template_paths() + template_paths_for_task_type(task_type),
        view=view,
        iteration=iteration,
    )


def template_paths_for_task_type(task_type: str) -> tuple[str, ...]:
    return tuple(node.template for node in _route_nodes(task_type) if node.template)


def _root_template_paths() -> tuple[str, ...]:
    if NODE_AGENT_TEMPLATE_TREE.template is None:
        return ()
    return (NODE_AGENT_TEMPLATE_TREE.template,)


def _render_templates(
    template_paths: tuple[str, ...],
    *,
    view: dict[str, Any],
    iteration: int,
) -> str:
    env = _template_environment()
    view_json = _to_pretty_json(json_ready(view))
    context = {
        "iteration": iteration,
        "view": view,
        "view_json": view_json,
        "SANDBOX_VIEW": view_json,
    }
    rendered_parts = []
    for template_path in template_paths:
        rendered_parts.append(env.get_template(template_path).render(**context).strip())
    return "\n\n".join(part for part in rendered_parts if part)


def _route_nodes(task_type: str) -> tuple[TemplateNode, ...]:
    try:
        route = TASK_TEMPLATE_ROUTES[task_type]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_TEMPLATE_ROUTES))
        raise ValueError(
            f"Unsupported NodeAgent task_type: {task_type!r}. "
            f"Available task types: {available}"
        ) from exc
    return NODE_AGENT_TEMPLATE_TREE.find_path(route)[1:]


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _to_pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
