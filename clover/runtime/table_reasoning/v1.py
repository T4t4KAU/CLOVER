"""Table reasoning v1 runtime orchestration loops."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from clover.executor import ExecutionResult, execute_physical_plan
from clover.planner import SqlParseError, parse_remote_sql_to_logic_dag
from clover.optimizer import optimize_logic_dag_to_physical_plan
from clover.prompt_safety import sanitize_task_dsl_for_prompt
from clover.remote_llm import RemoteLLMSession, create_remote_llm_session
from clover.reporter import ReporterDecision, ReporterResult, run_reporter


@dataclass(frozen=True)
class RuntimeRound:
    """One local execution plus Reporter decision."""

    index: int
    sql: str | None
    logic_dag: dict[str, Any]
    physical_plan: dict[str, Any]
    execution_result: ExecutionResult
    reporter_result: ReporterResult


@dataclass(frozen=True)
class RuntimeLoopResult:
    """Result of executing Reporter-controlled retry rounds."""

    ok: bool
    answer: Any
    rounds: list[RuntimeRound]
    final_decision: ReporterDecision | None
    retry_exhausted: bool = False
    error: dict[str, Any] | None = None


def run_reporter_retry_loop(
    *,
    logic_dag: dict[str, Any],
    context: dict[str, Any],
    local_dsl: dict[str, Any],
    task_dsl: dict[str, Any] | None = None,
    remote_dsl: dict[str, Any] | None = None,
    initial_sql: str | None = None,
    session: RemoteLLMSession | None = None,
    remote_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    max_retries: int = 1,
    table_cache: dict[str, Any] | None = None,
    include_reporter_instruction: bool = True,
) -> RuntimeLoopResult:
    """Run Optimizer -> Executor -> Reporter until accepted or retries expire."""

    if session is not None and remote_config is not None:
        raise ValueError("Retry loop accepts either session or remote_config, not both")
    if session is None:
        if remote_config is None:
            raise ValueError("Retry loop requires session or remote_config")
        session = create_remote_llm_session(remote_config)
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    rounds: list[RuntimeRound] = []
    task_type = local_dsl.get("task_type") or logic_dag.get("task_type")
    current_logic_dag = copy.deepcopy(logic_dag)
    if "task_type" not in current_logic_dag and task_type:
        current_logic_dag["task_type"] = task_type
    current_sql = initial_sql
    planner_dsl = _planner_dsl(
        remote_dsl=remote_dsl,
        local_dsl=local_dsl,
        task_dsl=task_dsl,
    )

    for round_index in range(max_retries + 1):
        # Each retry SQL is lowered by Planner before reaching the
        # Optimizer, so the Executor only ever sees normal physical plans.
        physical_plan = optimize_logic_dag_to_physical_plan(
            logic_dag=current_logic_dag,
            context=context,
            local_dsl=local_dsl,
        )
        execution_result = execute_physical_plan(
            physical_plan,
            table_cache=table_cache,
            slm_config=local_slm_config,
        )
        reporter_result = run_reporter(
            task_dsl=task_dsl,
            local_dsl=local_dsl,
            logic_dag=current_logic_dag,
            local_result=execution_result,
            current_sql=current_sql,
            session=session,
            # The Reporter instruction was already placed in the maintained
            # Remote LLM conversation. Follow-up rounds only send fresh facts.
            include_root=include_reporter_instruction and round_index == 0,
        )
        rounds.append(
            RuntimeRound(
                index=round_index,
                sql=current_sql,
                logic_dag=current_logic_dag,
                physical_plan=physical_plan,
                execution_result=execution_result,
                reporter_result=reporter_result,
            )
        )

        decision = reporter_result.decision
        if not decision.retry:
            return RuntimeLoopResult(
                ok=True,
                answer=decision.answer,
                rounds=rounds,
                final_decision=decision,
            )

        if round_index >= max_retries:
            return RuntimeLoopResult(
                ok=False,
                answer=None,
                rounds=rounds,
                final_decision=decision,
                retry_exhausted=True,
                error={
                    "type": "RetryLimitExceeded",
                    "message": f"Reporter requested retry after {max_retries} retries",
                },
            )

        try:
            current_sql = decision.new_sql["sql"] if decision.new_sql else None
            # Reporter repairs SQL, not internal DAG JSON. This keeps retry
            # outputs aligned with the original Commander contract.
            current_logic_dag = parse_remote_sql_to_logic_dag(
                current_sql or "",
                planner_dsl,
            )
        except (KeyError, TypeError, SqlParseError) as exc:
            return RuntimeLoopResult(
                ok=False,
                answer=None,
                rounds=rounds,
                final_decision=decision,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    raise AssertionError("unreachable retry loop state")


def _planner_dsl(
    *,
    remote_dsl: dict[str, Any] | None,
    local_dsl: dict[str, Any],
    task_dsl: dict[str, Any] | None,
) -> dict[str, Any]:
    source = remote_dsl or local_dsl or task_dsl
    if not isinstance(source, dict):
        raise ValueError("Retry loop requires a DSL for Planner")
    return sanitize_task_dsl_for_prompt(source)
