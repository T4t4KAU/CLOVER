"""Document reasoning sandbox policy for chunk-local worker loops."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from clover.executor.context import NodeExecutionContext
from clover.executor.errors import NodeExecutionError
from clover.executor.node_views import NodeView
from clover.executor.node_views.document import render_document_node_view
from clover.executor.sandbox.core import SandboxActionResult


@dataclass
class DocumentReasoningSandboxState:
    """Projected view for one document chunk worker."""

    node: dict[str, Any]
    chunk_text: str
    chunk_record: dict[str, Any]
    local_instruction: str
    advice: str


class DocumentReasoningSandboxPolicy:
    """Expose only the current document chunk and local instruction."""

    def start(
        self,
        context: NodeExecutionContext,
        *,
        decision: Any,
        trigger: str,
        error: Exception | None,
    ) -> DocumentReasoningSandboxState:
        _validate_document_node(context.node)
        resource_id = context.node["input"][0]
        params = context.node.get("params", {})
        if not isinstance(params, dict):
            raise NodeExecutionError("Document map node params must be an object")
        instruction = _required_string(params, "local_instruction")
        advice = _optional_string(params.get("advice") or params.get("local_guidance"))
        question = _optional_string(
            params.get("question")
            or params.get("question_context")
            or context.external_params.get("question")
        )
        return DocumentReasoningSandboxState(
            node=copy.deepcopy(context.node),
            chunk_text=_chunk_text(context, resource_id),
            chunk_record=_chunk_record(context, resource_id),
            local_instruction=_with_question_context(instruction, question=question),
            advice=advice,
        )

    def view(
        self,
        state: DocumentReasoningSandboxState,
        observations: list[dict[str, Any]],
    ) -> NodeView:
        return render_document_node_view(
            state.node,
            world={
                "chunk_text": state.chunk_text,
                "chunk": state.chunk_record,
                "local_instruction": state.local_instruction,
                "advice": state.advice,
                "observations": copy.deepcopy(observations),
            },
        )

    def run_action(
        self,
        state: DocumentReasoningSandboxState,
        action: dict[str, Any],
    ) -> SandboxActionResult:
        if action.get("action") != "submit_worker_output":
            return SandboxActionResult(
                ok=False,
                accepted=False,
                terminal=True,
                error={
                    "type": "invalid_document_action",
                    "message": "Document worker must submit worker output",
                },
            )
        payload = action.get("output")
        if not isinstance(payload, dict):
            return SandboxActionResult(
                ok=False,
                accepted=False,
                terminal=True,
                error={
                    "type": "invalid_document_output",
                    "message": "Document worker output must be an object",
                },
            )
        output = {
            "answer": _string_or_none(payload.get("answer")),
            "citation": _string_or_none(payload.get("citation")),
            "explanation": _string_value(payload.get("explanation")),
            "chunk": _chunk_locator(state.chunk_record),
            "sample": _string_value(action.get("sample")),
        }
        return SandboxActionResult(
            ok=True,
            accepted=True,
            terminal=True,
            observation={
                "type": "submit_worker_output",
                "ok": True,
            },
            output=output,
        )

    def close(self, state: DocumentReasoningSandboxState) -> None:
        return None


def _validate_document_node(node: dict[str, Any]) -> None:
    if node.get("op") != "map":
        raise NodeExecutionError(
            f"DocumentReasoningNodeAgent only supports map nodes, got {node.get('op')}"
        )
    inputs = node.get("input")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], str):
        raise NodeExecutionError("Document map node must reference exactly one chunk")


def _chunk_text(context: NodeExecutionContext, resource_id: str) -> str:
    values = context.materialize_sources(target="text")
    if resource_id not in values:
        raise NodeExecutionError(f"Document chunk resource not available: {resource_id}")
    return str(values[resource_id])


def _chunk_record(context: NodeExecutionContext, resource_id: str) -> dict[str, Any]:
    values = context.materialize_sources(target="chunk_record")
    value = values.get(resource_id)
    if not isinstance(value, dict):
        raise NodeExecutionError(f"Document chunk record not available: {resource_id}")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise NodeExecutionError(f"Document map node params must include {key}")
    return value.strip()


def _optional_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        return stripped
    if isinstance(value, list):
        parts = [_string_or_none(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return str(value)


def _string_value(value: Any) -> str:
    normalized = _string_or_none(value)
    return normalized or ""


def _with_question_context(instruction: str, *, question: str) -> str:
    component_rule = (
        "Local extraction rule: if the task mentions multiple variables, "
        "formula components, years, numerator/denominator values, or beginning "
        "and ending balances, return any requested component found in this "
        "excerpt even when other requested components are absent."
    )
    if not question:
        return f"{component_rule}\nTask: {instruction}"
    if question in instruction or instruction.lower().startswith("question:"):
        return f"{component_rule}\nTask: {instruction}"
    return f"Question: {question}\n{component_rule}\nTask: {instruction}"


def _chunk_locator(record: dict[str, Any]) -> dict[str, Any]:
    locator: dict[str, Any] = {}
    for key in (
        "chunk_id",
        "page_start",
        "page_end",
        "page_indexing",
        "char_start",
        "char_end",
    ):
        if key in record and record[key] is not None:
            locator[key] = record[key]
    return locator
