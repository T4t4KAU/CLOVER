"""Task-neutral Node View abstractions.

Internal IR remains the execution format. A Node View is the model-facing view
of one sandboxed node or node slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clover.task_types import agent_kind_for_task_type


class NodeViewRenderError(ValueError):
    """Raised when a node cannot be rendered into a model-facing view."""


@dataclass(frozen=True)
class NodeView:
    """A compact model-facing description of one sandbox world and task."""

    kind: str
    language: str
    world: dict[str, Any]
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe shape for prompts, traces, and tests."""

        return {
            "kind": self.kind,
            "language": self.language,
            "world": dict(self.world),
            "task": self.task,
            "metadata": dict(self.metadata),
        }


def render_node_view(
    task_type: str,
    node: dict[str, Any],
    *,
    world: dict[str, Any] | None = None,
) -> NodeView:
    """Render one IR node into the task-neutral Node View interface."""

    agent_kind = agent_kind_for_task_type(task_type)
    if agent_kind == "table_reasoning":
        from clover.executor.node_views.table import render_table_node_view

        return render_table_node_view(node, world=world)
    if agent_kind == "document_reasoning":
        from clover.executor.node_views.document import render_document_node_view

        return render_document_node_view(node, world=world)
    raise NodeViewRenderError(f"Unsupported Node View task_type: {task_type!r}")
