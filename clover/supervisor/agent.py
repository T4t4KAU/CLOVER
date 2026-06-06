"""Remote Supervisor as a small-action-space ReAct-style agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clover.supervisor.client import generate_remote_text
from clover.supervisor.decision import (
    SupervisorDecision,
    parse_supervisor_decision,
)
from clover.supervisor.decompose import render_initial_task_prompt
from clover.supervisor.synthesis_templates import render_initial_synthesis_prompt


SUPERVISOR_ACTION_DECOMPOSE = "decompose"
SUPERVISOR_ACTION_SYNTHESIZE = "synthesize"
SUPERVISOR_ACTIONS = frozenset(
    {
        SUPERVISOR_ACTION_DECOMPOSE,
        SUPERVISOR_ACTION_SYNTHESIZE,
    }
)


@dataclass(frozen=True)
class SupervisorStepResult:
    """One Supervisor action plus the model output it produced."""

    action: str
    prompt: str
    remote_output: str
    response_payload: dict[str, Any]
    response_id: str | None
    response_status: str | None
    api_type: str
    command: str | None = None
    decision: SupervisorDecision | None = None


class SupervisorAgent:
    """Stateless Remote Supervisor driver.

    CLOVER owns loop state explicitly and includes the needed observation in
    every prompt. The provider-side model call remains stateless.
    """

    def __init__(
        self,
        *,
        remote_config: dict[str, Any],
        client: Any | None = None,
    ) -> None:
        self.remote_config = remote_config
        self.client = client

    def step(self, action: str, **kwargs: Any) -> SupervisorStepResult:
        """Run one Supervisor action."""

        if action == SUPERVISOR_ACTION_DECOMPOSE:
            return self.decompose(task_dsl=kwargs["task_dsl"])
        if action == SUPERVISOR_ACTION_SYNTHESIZE:
            return self.synthesize(**kwargs)
        available = ", ".join(sorted(SUPERVISOR_ACTIONS))
        raise ValueError(
            f"Unsupported Supervisor action: {action!r}. Available: {available}"
        )

    def decompose(self, *, task_dsl: dict[str, Any]) -> SupervisorStepResult:
        """Ask the Supervisor to produce a local command."""

        prompt = render_initial_task_prompt(task_dsl)
        llm_result = generate_remote_text(
            prompt=prompt,
            remote_config=self.remote_config,
            client=self.client,
        )
        return SupervisorStepResult(
            action=SUPERVISOR_ACTION_DECOMPOSE,
            prompt=prompt,
            remote_output=llm_result.text,
            response_payload=llm_result.response_payload,
            response_id=llm_result.response_id,
            response_status=llm_result.response_status,
            api_type=llm_result.api_type,
            command=llm_result.text,
        )

    def synthesize(
        self,
        *,
        task_dsl: dict[str, Any] | None = None,
        local_dsl: dict[str, Any] | None = None,
        logic_dag: dict[str, Any],
        observation: Any,
        current_command: Any = None,
        force_final_answer: bool = False,
    ) -> SupervisorStepResult:
        """Ask the Supervisor to judge an observation and decide next action."""

        prompt = render_initial_synthesis_prompt(
            task_dsl=task_dsl,
            local_dsl=local_dsl,
            logic_dag=logic_dag,
            observation=observation,
            current_command=current_command,
            force_final_answer=force_final_answer,
        )
        llm_result = generate_remote_text(
            prompt=prompt,
            remote_config=self.remote_config,
            client=self.client,
        )
        decision = parse_supervisor_decision(llm_result.text)
        return SupervisorStepResult(
            action=SUPERVISOR_ACTION_SYNTHESIZE,
            prompt=prompt,
            remote_output=llm_result.text,
            response_payload=llm_result.response_payload,
            response_id=llm_result.response_id,
            response_status=llm_result.response_status,
            api_type=llm_result.api_type,
            decision=decision,
        )
