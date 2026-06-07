"""Shared local Agent Loop runner for NodeAgents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clover.executor.context import NodeExecutionContext
from clover.executor.errors import AgentLoopNotImplementedError, NodeExecutionError
from clover.executor.local_slm import (
    limit_slm_request_timeout,
    load_slm_config,
)
from clover.executor.agents.template_tree import template_leaf_key_for_local_slm_prompt
from clover.executor.node_views import NodeView
from clover.executor.result import json_ready
from clover.executor.sandbox.core import AgentSandbox
from clover.executor.slm_dispatcher import LocalSlmSequenceRequest
from clover.supervisor.client import extract_token_usage


PromptRenderer = Callable[[NodeView, int, list[dict[str, Any]]], str]
ActionParser = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class AgentLoopRunResult:
    """Output and trace produced by one local Agent Loop."""

    output: Any
    trace: dict[str, Any]


class AgentLoopExecutionError(NodeExecutionError):
    """Agent Loop failure that still carries the loop trace."""

    def __init__(
        self,
        message: str,
        *,
        node: dict[str, Any],
        trace: dict[str, Any],
    ) -> None:
        super().__init__(message, node=node)
        self.trace = trace


def run_sandbox_agent_loop(
    *,
    context: NodeExecutionContext,
    node: dict[str, Any],
    sandbox: AgentSandbox | None,
    decision: Any,
    trigger: str,
    error: Exception | None,
    max_iterations: int,
    render_prompt: PromptRenderer,
    parse_action: ActionParser,
    prompt_kind: str,
) -> AgentLoopRunResult:
    """Run a short local Agent Loop over a task-specific sandbox."""

    if sandbox is None:
        raise AgentLoopNotImplementedError(
            f"No Agent sandbox is available for task_type {context.task_type}"
        )

    iterations = max(1, int(max_iterations or 1))
    slm_config = limit_slm_request_timeout(
        context.slm_config or load_slm_config(),
        node_timeout_seconds=context.node_timeout_seconds,
    )
    observations: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    prompt_steps: list[dict[str, Any]] = []
    last_error_message: str | None = None
    sandbox.start(decision=decision, trigger=trigger, error=error)

    try:
        for iteration in range(iterations):
            view = sandbox.view(observations)
            prompt = render_prompt(view, iteration + 1, prompt_steps)
            sequence_result = _generate_local_slm_sequence(
                context=context,
                node=node,
                prompt=prompt,
                prompt_kind=prompt_kind,
                iteration=iteration + 1,
                slm_config=slm_config,
            )
            result = sequence_result.llm_result
            token_usage = extract_token_usage(result.response_payload)
            try:
                action = parse_action(result.text)
            except ValueError as exc:
                last_error_message = str(exc)
                observation = {
                    "type": "invalid_action_json",
                    "ok": False,
                    "error": {"message": last_error_message},
                }
                observations.append(observation)
                prompt_steps.append(
                    _prompt_step(action=None, observation=observation)
                )
                trace_steps.append(
                    _trace_step(
                        iteration=iteration,
                        action=None,
                        observation=observation,
                        response_id=result.response_id,
                        prompt_kind=prompt_kind,
                        token_usage=token_usage,
                        sequence_trace=sequence_result.trace_metadata(),
                    )
                )
                continue

            action_result = sandbox.run_action(action)
            prompt_steps.append(
                _prompt_step(
                    action=action,
                    observation=action_result.observation,
                )
            )
            trace_steps.append(
                _trace_step(
                    iteration=iteration,
                    action=action,
                    observation=action_result.observation,
                    response_id=result.response_id,
                    prompt_kind=prompt_kind,
                    accepted=action_result.accepted,
                    terminal=action_result.terminal,
                    error=action_result.error,
                    token_usage=token_usage,
                    sequence_trace=sequence_result.trace_metadata(),
                )
            )
            if action_result.accepted:
                return AgentLoopRunResult(
                    output=action_result.output,
                    trace={
                        "trigger": trigger,
                        "iterations": iteration + 1,
                        "steps": trace_steps,
                    },
                )
            if action_result.terminal:
                message = (
                    action_result.error.get("message")
                    if isinstance(action_result.error, dict)
                    else "Agent Loop terminated without output"
                )
                raise AgentLoopExecutionError(
                    str(message),
                    node=node,
                    trace={
                        "trigger": trigger,
                        "iterations": iteration + 1,
                        "steps": trace_steps,
                    },
                )
            if action_result.observation is not None:
                observations.append(action_result.observation)
                if isinstance(action_result.observation.get("error"), dict):
                    last_error_message = str(
                        action_result.observation["error"].get("message") or ""
                    )
    finally:
        sandbox.close()

    detail = (
        f": {last_error_message}"
        if last_error_message
        else ""
    )
    raise AgentLoopExecutionError(
        f"Agent Loop reached max_iterations={iterations} without valid output{detail}",
        node=node,
        trace={
            "trigger": trigger,
            "iterations": iterations,
            "steps": trace_steps,
        },
    )


def _generate_local_slm_sequence(
    *,
    context: NodeExecutionContext,
    node: dict[str, Any],
    prompt: str,
    prompt_kind: str,
    iteration: int,
    slm_config: dict[str, Any],
) -> Any:
    dispatcher = context.slm_dispatcher
    if dispatcher is None:
        raise RuntimeError("Local SLM sequence dispatcher is not configured")
    leaf_key = template_leaf_key_for_local_slm_prompt(
        prompt_kind=prompt_kind,
        task_type=context.task_type,
        node=node,
    )
    return dispatcher.generate(
        LocalSlmSequenceRequest(
            prompt=prompt,
            leaf_key=leaf_key,
            prompt_kind=prompt_kind,
            node_id=str(node.get("id") or ""),
            job_id=str(node.get("id") or ""),
            iteration=iteration,
            slm_config=slm_config,
            client=context.slm_client,
        )
    )


def _trace_step(
    *,
    iteration: int,
    action: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    response_id: str | None,
    prompt_kind: str,
    accepted: bool = False,
    terminal: bool = False,
    error: dict[str, Any] | None = None,
    token_usage: dict[str, int] | None = None,
    sequence_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step = {
        "iteration": iteration + 1,
        "prompt_kind": prompt_kind,
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
    if token_usage is not None:
        step["token_usage"] = dict(token_usage)
    if sequence_trace is not None:
        step["sequence"] = dict(sequence_trace)
    return step


def _prompt_step(
    *,
    action: dict[str, Any] | None,
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "action": _prompt_action(action) if isinstance(action, dict) else None,
        "observation": json_ready(observation) if observation is not None else None,
    }


def _prompt_action(action: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action.get("action")}
    code = action.get("code")
    if isinstance(code, str):
        payload["code"] = code
    return payload
