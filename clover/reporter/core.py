"""Reporter model call helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clover.remote_llm import RemoteLLMSession, generate_remote_text
from clover.reporter.decision import ReporterDecision, parse_reporter_decision
from clover.reporter.template_tree import render_initial_report_prompt, render_report_prompt


@dataclass(frozen=True)
class ReporterResult:
    """Reporter call output plus parsed decision."""

    decision: ReporterDecision
    prompt: str
    remote_output: str
    response_payload: dict[str, Any]
    response_id: str | None
    response_status: str | None
    api_type: str


def run_reporter(
    *,
    task_dsl: dict[str, Any] | None = None,
    local_dsl: dict[str, Any] | None = None,
    logic_dag: dict[str, Any],
    local_result: Any,
    current_sql: Any = None,
    remote_config: dict[str, Any] | None = None,
    client: Any | None = None,
    session: RemoteLLMSession | None = None,
    include_root: bool = True,
) -> ReporterResult:
    """Render a Reporter prompt, call Remote LLM, and parse its decision."""

    if session is not None and remote_config is not None:
        raise ValueError("Reporter accepts either session or remote_config, not both")
    if session is None and remote_config is None:
        raise ValueError("Reporter requires session or remote_config")

    prompt = (
        render_initial_report_prompt(
            task_dsl=task_dsl,
            local_dsl=local_dsl,
            logic_dag=logic_dag,
            local_result=local_result,
            current_sql=current_sql,
        )
        if include_root
        else render_report_prompt(
            task_dsl=task_dsl,
            local_dsl=local_dsl,
            logic_dag=logic_dag,
            local_result=local_result,
            current_sql=current_sql,
        )
    )
    if session is not None:
        # In the normal workflow this is the same session used by Remote
        # Commander, so Reporter sees the original SQL constraints in context.
        llm_result = session.generate(prompt)
    else:
        llm_result = generate_remote_text(
            prompt=prompt,
            remote_config=remote_config,
            client=client,
        )
    decision = parse_reporter_decision(llm_result.text)
    return ReporterResult(
        decision=decision,
        prompt=prompt,
        remote_output=llm_result.text,
        response_payload=llm_result.response_payload,
        response_id=llm_result.response_id,
        response_status=llm_result.response_status,
        api_type=llm_result.api_type,
    )
