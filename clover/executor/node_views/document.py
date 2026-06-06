"""Document reasoning Node View rendering."""

from __future__ import annotations

from typing import Any

from clover.executor.node_views.core import NodeView, NodeViewRenderError


def render_document_node_view(
    node: dict[str, Any],
    *,
    world: dict[str, Any] | None = None,
) -> NodeView:
    """Render one document map node into a chunk-local worker view."""

    if node.get("op") != "map":
        raise NodeViewRenderError(
            f"Document Node View only supports map nodes, got {node.get('op')!r}"
        )
    selected_world = dict(world or {})
    params = node.get("params", {})
    if not isinstance(params, dict):
        params = {}
    instruction = selected_world.get("local_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = params.get("local_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise NodeViewRenderError("Document Node View requires local_instruction")
    advice = selected_world.get("advice")
    if not isinstance(advice, str):
        advice = params.get("advice") or params.get("local_guidance") or ""
    chunk_text = selected_world.get("chunk_text")
    if not isinstance(chunk_text, str):
        chunk_text = ""
    return NodeView(
        kind="document_reasoning.map",
        language="json",
        world={
            "chunk_text": chunk_text,
            "chunk": dict(selected_world.get("chunk") or {}),
            "observations": _list_or_empty(selected_world.get("observations")),
        },
        task=instruction.strip(),
        metadata={
            "op": "map",
            "advice": advice.strip() if isinstance(advice, str) else "",
        },
    )


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
