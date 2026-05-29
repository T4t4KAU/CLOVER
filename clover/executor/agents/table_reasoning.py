"""Table reasoning NodeAgent with Static Tool Fast Path and local ReAct fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from clover.executor.agents.base import BaseNodeAgent, FastPathDecision
from clover.executor.agents.template_tree import render_agent_loop_prompt
from clover.executor.errors import AgentLoopNotImplementedError, NodeExecutionError
from clover.local_slm import generate_slm_text, load_slm_config
from clover.tools import StaticToolError, build_static_tool_call
from clover.tools.table_reasoning import TABLE_REASONING_STATIC_TOOLS
from clover.tools.table_reasoning.pandas_backend import PandasTableReasoningExecutor


class TableReasoningNodeAgent(BaseNodeAgent):
    """NodeAgent for table reasoning physical DAG nodes."""

    backend_name = "pandas"
    supported_task_types = frozenset(
        {"table_reasoning", "table_reasoning_v1", "table_reasoning_v2"}
    )

    def try_fast_path(self) -> FastPathDecision:
        node = self.node
        if self.context.task_type not in self.supported_task_types:
            return FastPathDecision(
                hit=False,
                backend=self.backend_name,
                miss_reason="unsupported_task_type",
                miss_detail=f"Unsupported task_type: {self.context.task_type}",
            )

        op = node.get("op")
        if op not in TABLE_REASONING_STATIC_TOOLS:
            return FastPathDecision(
                hit=False,
                backend=self.backend_name,
                miss_reason="unsupported_op",
                miss_detail=f"No table reasoning static tool for op: {op}",
            )

        missing_dependencies = [
            dependency
            for dependency in node.get("dependency", [])
            if dependency not in self.context.upstream_outputs
        ]
        if missing_dependencies:
            return FastPathDecision(
                hit=False,
                backend=self.backend_name,
                miss_reason="dependencies_not_ready",
                miss_detail=f"Missing dependencies: {missing_dependencies}",
            )

        missing_resources = [
            resource_id
            for resource_id in node.get("input", [])
            if resource_id not in self.context.resources
        ]
        if missing_resources:
            return FastPathDecision(
                hit=False,
                backend=self.backend_name,
                miss_reason="resources_not_ready",
                miss_detail=f"Missing resources: {missing_resources}",
            )

        try:
            # The static tool layer validates node shape and turns the physical
            # node into a backend-neutral call object.
            call = build_static_tool_call(
                self.context.task_type,
                node,
                resources=self.context.resources,
                upstream_outputs={},
                external_params=self.context.external_params,
            )
        except StaticToolError as exc:
            return FastPathDecision(
                hit=False,
                backend=self.backend_name,
                miss_reason="invalid_static_tool_call",
                miss_detail=str(exc),
            )
        # Static tool declarations normalize the call payload. The concrete
        # backend should consume live upstream objects, not deep-copied tables.
        call["upstream_outputs"] = dict(self.context.upstream_outputs)

        return FastPathDecision(
            hit=True,
            call=call,
            tool=call.get("tool"),
            backend=self.backend_name,
        )

    def execute_fast_path(self, decision: FastPathDecision) -> Any:
        if decision.call is None:
            raise ValueError("Fast Path hit is missing a static tool call")
        executor = PandasTableReasoningExecutor(
            resources=self.context.resources,
            external_params=self.context.external_params,
            table_cache=self.context.table_cache,
        )
        return executor.execute_call(decision.call)

    def should_run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str,
        error: Exception | None = None,
    ) -> bool:
        if trigger == "fast_path_execution_error" and error is not None:
            return _is_recoverable_local_execution_error(error)
        return trigger == "fast_path_miss"

    def run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str = "fast_path_miss",
        error: Exception | None = None,
    ) -> Any:
        if self.sandbox is None:
            raise AgentLoopNotImplementedError(
                f"No Agent sandbox is available for task_type {self.context.task_type}"
            )

        max_iterations = max(1, int(self.context.agent_loop_max_iterations or 1))
        slm_config = self.context.slm_config or load_slm_config()
        observations: list[dict[str, Any]] = []
        trace_steps: list[dict[str, Any]] = []
        self.sandbox.start(decision=decision, trigger=trigger, error=error)

        try:
            for iteration in range(max_iterations):
                view = self.sandbox.view(observations)
                prompt = render_agent_loop_prompt(
                    task_type=self.context.task_type,
                    view=view,
                    iteration=iteration + 1,
                )
                result = generate_slm_text(
                    prompt,
                    slm_config=slm_config,
                    client=self.context.slm_client,
                )
                try:
                    action = _extract_action_json(result.text)
                except ValueError as exc:
                    observation = {
                        "type": "invalid_action_json",
                        "ok": False,
                        "error": {"message": str(exc)},
                    }
                    observations.append(observation)
                    trace_steps.append(
                        _trace_step(
                            iteration=iteration,
                            action=None,
                            observation=observation,
                            response_id=result.response_id,
                        )
                    )
                    continue

                action_result = self.sandbox.run_action(action)
                trace_steps.append(
                    _trace_step(
                        iteration=iteration,
                        action=action,
                        observation=action_result.observation,
                        response_id=result.response_id,
                        accepted=action_result.accepted,
                        terminal=action_result.terminal,
                        error=action_result.error,
                    )
                )
                if action_result.accepted:
                    self.agent_loop_trace = {
                        "trigger": trigger,
                        "iterations": iteration + 1,
                        "steps": trace_steps,
                    }
                    return action_result.output
                if action_result.terminal:
                    self.agent_loop_trace = {
                        "trigger": trigger,
                        "iterations": iteration + 1,
                        "steps": trace_steps,
                    }
                    message = (
                        action_result.error.get("message")
                        if isinstance(action_result.error, dict)
                        else "Agent Loop terminated without output"
                    )
                    raise NodeExecutionError(str(message), node=self.node)
                if action_result.observation is not None:
                    observations.append(action_result.observation)
        finally:
            self.sandbox.close()

        self.agent_loop_trace = {
            "trigger": trigger,
            "iterations": max_iterations,
            "steps": trace_steps,
        }
        raise NodeExecutionError(
            f"Agent Loop reached max_iterations={max_iterations} without valid output",
            node=self.node,
        )


def _extract_action_json(text: str) -> dict[str, Any]:
    candidates = _extract_fenced_json_blocks(text)
    candidates.append(text)
    errors = []
    for candidate in candidates:
        try:
            payload = _load_json_object(candidate)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if "action" not in payload:
            raise ValueError("Agent action JSON must include action")
        return payload
    detail = "; ".join(errors) if errors else "no JSON candidate found"
    raise ValueError(f"Unable to parse Agent action JSON: {detail}")


def _extract_fenced_json_blocks(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = _load_first_json_object(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Agent action JSON must be an object")
    return payload


def _load_first_json_object(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
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
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(str(exc)) from exc
    raise ValueError("Unclosed JSON object")


def _trace_step(
    *,
    iteration: int,
    action: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    response_id: str | None,
    accepted: bool = False,
    terminal: bool = False,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "iteration": iteration + 1,
        "response_id": response_id,
        "action": action.get("action") if isinstance(action, dict) else None,
        "ok": bool(accepted or (observation or {}).get("ok")),
        "accepted": accepted,
        "terminal": terminal,
        "observation_type": (
            observation.get("type")
            if isinstance(observation, dict)
            else None
        ),
        "error": error or ((observation or {}).get("error") if observation else None),
    }


def _is_recoverable_local_execution_error(error: Exception) -> bool:
    message = str(error).lower()
    unrecoverable_markers = (
        "unknown column",
        "unknown resource",
        "missing resource",
        "unsatisfied dependencies",
        "no such column",
        "has unknown",
    )
    return not any(marker in message for marker in unrecoverable_markers)
