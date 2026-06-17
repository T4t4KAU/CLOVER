"""Table reasoning NodeAgent with Static Tool Fast Path and local ReAct fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from clover.executor.agents.base import BaseNodeAgent, FastPathDecision
from clover.executor.agents.loop import AgentLoopExecutionError, run_sandbox_agent_loop
from clover.executor.agents.template_tree import (
    render_agent_loop_prompt,
    render_table_empty_filter_repair_prompt,
)
from clover.executor.python_function import (
    PythonFunctionParseError,
    parse_python_function_action,
)
from clover.executor.table_evidence import run_table_evidence_action
from clover.tools import StaticToolError, build_static_tool_call
from clover.tools.table_reasoning import TABLE_REASONING_STATIC_TOOLS
from clover.tools.table_reasoning.pandas_backend import (
    PandasTable,
    PandasTableReasoningExecutor,
)
from clover.task_types import (
    TABLE_REASONING_ANALYZE_TASK_TYPE,
    TABLE_REASONING_QUERY_TASK_TYPE,
)


class TableReasoningNodeAgent(BaseNodeAgent):
    """NodeAgent for table reasoning physical DAG nodes."""

    backend_name = "pandas"
    supported_task_types = frozenset(
        {
            TABLE_REASONING_QUERY_TASK_TYPE,
            TABLE_REASONING_ANALYZE_TASK_TYPE,
        }
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
        if op == "Inspect":
            return FastPathDecision(
                hit=False,
                backend="local_slm",
                miss_reason="requires_open_evidence",
                miss_detail="Open table evidence requests require a local evidence worker.",
            )
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
        if _agent_loop_disabled(self.context.slm_config):
            return False
        if trigger == "fast_path_execution_error" and error is not None:
            if _is_recoverable_local_execution_error(error):
                return True
            return (
                self.context.task_type == TABLE_REASONING_ANALYZE_TASK_TYPE
                and _is_analyze_local_fallback_candidate(error)
            )
        return trigger == "fast_path_miss"

    def should_run_agent_loop_after_fast_path(
        self,
        decision: FastPathDecision,
        output: Any,
    ) -> bool:
        if _agent_loop_disabled(self.context.slm_config):
            return False
        return (
            _supports_empty_filter_repair(self.context.task_type)
            and self.node.get("op") in _EMPTY_OUTPUT_REPAIRABLE_OPS
            and _is_empty_table_output(output)
        )

    def should_keep_fast_path_output_on_agent_loop_failure(
        self,
        decision: FastPathDecision,
        output: Any,
        *,
        trigger: str,
        error: Exception,
    ) -> bool:
        del decision, error
        return (
            trigger == "fast_path_empty_output"
            and _supports_empty_filter_repair(self.context.task_type)
            and self.node.get("op") in _EMPTY_OUTPUT_REPAIRABLE_OPS
            and _is_empty_table_output(output)
        )

    def run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str = "fast_path_miss",
        error: Exception | None = None,
    ) -> Any:
        if self.node.get("op") == "Inspect":
            result = _run_inspect_evidence_node(self)
            self.agent_loop_trace = result.trace
            return result.output
        use_empty_filter_repair = _use_empty_filter_repair_prompt(
            task_type=self.context.task_type,
            node=self.node,
            trigger=trigger,
        )
        prompt_kind = (
            "table_reasoning_empty_filter_repair"
            if use_empty_filter_repair
            else "table_reasoning_agent_loop"
        )
        try:
            result = run_sandbox_agent_loop(
                context=self.context,
                node=self.node,
                sandbox=self.sandbox,
                decision=decision,
                trigger=trigger,
                error=error,
                max_iterations=self.context.agent_loop_max_iterations,
                render_prompt=lambda view, iteration, steps: (
                    render_table_empty_filter_repair_prompt(
                        view=view,
                        iteration=iteration,
                        steps=steps,
                        node=self.node,
                    )
                    if use_empty_filter_repair
                    else render_agent_loop_prompt(
                        task_type=self.context.task_type,
                        view=view,
                        iteration=iteration,
                    )
                ),
                parse_action=_extract_action_json,
                prompt_kind=prompt_kind,
            )
        except AgentLoopExecutionError as exc:
            self.agent_loop_trace = exc.trace
            raise
        self.agent_loop_trace = result.trace
        return result.output


def _run_inspect_evidence_node(agent: TableReasoningNodeAgent) -> Any:
    params = agent.node.get("params")
    if not isinstance(params, dict):
        params = {}
    source_frames = _materialized_frames(
        agent.context.materialize_sources(target="pandas").values()
    )
    view_frames = _materialized_frames(
        agent.context.materialize_dependencies(target="pandas").values()
    )
    if not source_frames:
        raise ValueError("Inspect node requires at least one source table")
    request = _optional_text(params.get("request") or params.get("q"))
    question = _optional_text(
        params.get("question")
        or params.get("q")
        or agent.context.external_params.get("question")
    )
    if not question:
        question = request or "Collect compact table evidence."
    return run_table_evidence_action(
        source_frames=source_frames,
        view_frames=view_frames,
        question=question,
        request=request,
        need=params.get("need"),
        slm_config=agent.context.slm_config,
        slm_dispatcher=agent.context.slm_dispatcher,
        max_iterations=agent.context.agent_loop_max_iterations,
    )


def _materialized_frames(values: Any) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for value in values:
        frame = getattr(value, "frame", None)
        if isinstance(frame, pd.DataFrame):
            frames.append(frame.copy(deep=True))
        elif isinstance(value, pd.DataFrame):
            frames.append(value.copy(deep=True))
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            frames.append(pd.DataFrame(value))
        elif isinstance(value, dict):
            rows = value.get("rows")
            data = value.get("data")
            if isinstance(rows, list):
                frames.append(pd.DataFrame(rows))
            elif isinstance(data, list):
                frames.append(pd.DataFrame(data))
    return frames


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


_EMPTY_OUTPUT_REPAIRABLE_OPS = frozenset({"Filter", "Project", "Derive", "Join"})


def _use_empty_filter_repair_prompt(
    *,
    task_type: str,
    node: dict[str, Any],
    trigger: str,
) -> bool:
    return (
        trigger == "fast_path_empty_output"
        and _supports_empty_filter_repair(task_type)
        and node.get("op") in _EMPTY_OUTPUT_REPAIRABLE_OPS
    )


def _supports_empty_filter_repair(task_type: str) -> bool:
    return task_type in {
        TABLE_REASONING_ANALYZE_TASK_TYPE,
        TABLE_REASONING_QUERY_TASK_TYPE,
    }


def _extract_action_json(text: str) -> dict[str, Any]:
    try:
        return parse_python_function_action(text)
    except PythonFunctionParseError:
        pass

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
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(text, idx=start)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    return obj


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


def _is_analyze_local_fallback_candidate(error: Exception) -> bool:
    message = str(error).lower()
    local_schema_markers = (
        "unknown column",
        "no such column",
        "has unknown",
    )
    return any(marker in message for marker in local_schema_markers)


def _agent_loop_disabled(slm_config: dict[str, Any] | None) -> bool:
    if not isinstance(slm_config, dict):
        return False
    return bool(slm_config.get("disable_agent_loop"))


def _is_empty_table_output(output: Any) -> bool:
    if isinstance(output, PandasTable):
        return output.frame.empty
    if isinstance(output, pd.DataFrame):
        return output.empty
    frame = getattr(output, "frame", None)
    return isinstance(frame, pd.DataFrame) and frame.empty
