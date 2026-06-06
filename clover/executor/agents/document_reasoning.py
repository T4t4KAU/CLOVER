"""Document reasoning NodeAgent for chunk-local worker execution."""

from __future__ import annotations

import json
import re
from typing import Any

from clover.executor.agents.base import BaseNodeAgent, FastPathDecision
from clover.executor.agents.loop import AgentLoopExecutionError, run_sandbox_agent_loop
from clover.executor.agents.template_tree import render_document_worker_prompt
from clover.executor.errors import AgentLoopNotImplementedError
from clover.executor.node_views import NodeView
from clover.task_types import DOCUMENT_REASONING_TASK_TYPE


class DocumentReasoningNodeAgent(BaseNodeAgent):
    """NodeAgent for one chunk-local document map operation."""

    backend_name = "local_slm"
    supported_task_types = frozenset({DOCUMENT_REASONING_TASK_TYPE})

    def try_fast_path(self) -> FastPathDecision:
        if self.context.task_type not in self.supported_task_types:
            return FastPathDecision(
                hit=False,
                backend=self.backend_name,
                miss_reason="unsupported_task_type",
                miss_detail=f"Unsupported task_type: {self.context.task_type}",
            )
        return FastPathDecision(
            hit=False,
            backend=self.backend_name,
            miss_reason="requires_document_worker",
            miss_detail="Document chunks are handled by a single local worker call.",
        )

    def execute_fast_path(self, decision: FastPathDecision) -> Any:
        raise AgentLoopNotImplementedError(
            "DocumentReasoningNodeAgent does not implement a Fast Path"
        )

    def should_run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str,
        error: Exception | None = None,
    ) -> bool:
        return trigger == "fast_path_miss" and error is None

    def run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str = "fast_path_miss",
        error: Exception | None = None,
    ) -> Any:
        try:
            result = run_sandbox_agent_loop(
                context=self.context,
                node=self.node,
                sandbox=self.sandbox,
                decision=decision,
                trigger=trigger,
                error=error,
                max_iterations=1,
                render_prompt=_render_document_prompt,
                parse_action=_parse_worker_action,
                prompt_kind="document_worker",
            )
        except AgentLoopExecutionError as exc:
            self.agent_loop_trace = exc.trace
            raise
        self.agent_loop_trace = result.trace
        return result.output


def _render_document_prompt(view: NodeView, iteration: int) -> str:
    del iteration
    return render_document_worker_prompt(
        chunk_text=str(view.world.get("chunk_text", "")),
        local_instruction=view.task,
        advice=str(view.metadata.get("advice", "")),
    )


def _parse_worker_action(text: str) -> dict[str, Any]:
    return {
        "action": "submit_worker_output",
        "output": _extract_worker_output(text),
        "sample": text,
    }


def _extract_worker_output(text: str) -> dict[str, Any]:
    payload = _load_json_object(text)
    if not isinstance(payload, dict):
        raise ValueError("Document worker output must be a JSON object")
    return payload


def _load_json_object(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return json.loads(_first_json_object(stripped))


def _first_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in document worker response")
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
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
                return text[start : index + 1]
    raise ValueError("Unterminated JSON object in document worker response")
