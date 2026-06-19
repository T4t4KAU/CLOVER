"""Table reasoning NodeAgent with Static Tool Fast Path and local ReAct fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from clover.config import (
    ENABLE_EDGE_AGENT,
    ENABLE_NODE_REVIEW,
    runtime_feature_enabled,
)
from clover.executor.agents.base import (
    BaseNodeAgent,
    FastPathDecision,
    FastPathReview,
)
from clover.executor.agents.loop import AgentLoopExecutionError, run_sandbox_agent_loop
from clover.executor.agents.mismatch_classifier import analyze_predicate_mismatch
from clover.executor.agents.template_tree import (
    render_agent_loop_prompt,
    render_rewrite_predicate_prompt,
    render_table_empty_filter_repair_prompt,
)
from clover.executor.python_function import (
    PythonFunctionParseError,
    parse_python_function_action,
)
from clover.task_types import (
    TABLE_REASONING_ANALYZE_TASK_TYPE,
    TABLE_REASONING_QUERY_TASK_TYPE,
)
from clover.tools import StaticToolError, build_static_tool_call
from clover.tools.table_reasoning import TABLE_REASONING_STATIC_TOOLS
from clover.tools.table_reasoning.pandas_backend import (
    PandasTable,
    PandasTableReasoningExecutor,
)


class TableReasoningNodeAgent(BaseNodeAgent):
    """NodeAgent for table reasoning physical DAG nodes."""

    backend_name = "pandas"
    supported_task_types = frozenset(
        {
            TABLE_REASONING_ANALYZE_TASK_TYPE,
            TABLE_REASONING_QUERY_TASK_TYPE,
        }
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mismatch_analysis: dict[str, Any] | None = None
        self._dependency_frame_cache: list[pd.DataFrame] | None = None

    def _compute_mismatch_analysis(self) -> dict[str, Any] | None:
        """Compute mismatch analysis for the current Filter node.

        Returns None if the node is not a Filter or the input frame is
        unavailable. Caches the result in ``_mismatch_analysis``.
        """
        if self._mismatch_analysis is not None:
            return self._mismatch_analysis
        if self.node.get("op") != "Filter":
            self._mismatch_analysis = {}
            return self._mismatch_analysis
        params = self.node.get("params")
        if not isinstance(params, dict):
            self._mismatch_analysis = {}
            return self._mismatch_analysis
        predicate = params.get("predicate")
        if not isinstance(predicate, dict):
            self._mismatch_analysis = {}
            return self._mismatch_analysis
        frame = self._primary_input_frame()
        if frame is None:
            self._mismatch_analysis = {}
            return self._mismatch_analysis
        try:
            self._mismatch_analysis = analyze_predicate_mismatch(predicate, frame)
        except Exception:
            self._mismatch_analysis = {}
        return self._mismatch_analysis

    def _primary_input_frame(self) -> pd.DataFrame | None:
        """Get the primary input DataFrame for the current node."""
        frames = self._dependency_frames()
        return frames[0] if frames else None

    def _dependency_frames(self) -> list[pd.DataFrame]:
        """Materialize dependency tables in node order."""
        if self._dependency_frame_cache is not None:
            return self._dependency_frame_cache
        try:
            self._dependency_frame_cache = _materialized_frames(
                self.context.materialize_dependencies(target="pandas").values()
            )
        except Exception:
            self._dependency_frame_cache = []
        return self._dependency_frame_cache

    def _dominant_mismatch(self) -> str | None:
        """Return the most severe mismatch type across all roots.

        Priority: wrong_column > system_bug > not_found > format > quoting.
        wrong_column triggers Cloud escalation; the remaining classes are
        eligible for local repair.
        """
        analysis = self._compute_mismatch_analysis()
        if not analysis or not analysis.get("roots"):
            return None
        priority = {
            "wrong_column": 5,
            "system_bug": 4,
            "not_found": 3,
            "format": 2,
            "quoting": 1,
        }
        mismatches = [
            root.get("mismatch", "not_found")
            for root in analysis["roots"]
            if isinstance(root, dict)
        ]
        if not mismatches:
            return None
        return max(mismatches, key=lambda m: priority.get(m, 0))

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
        del decision
        if _agent_loop_disabled(self.context.slm_config):
            return False
        return self._is_empty_output_review_candidate(output)

    def _is_empty_output_review_candidate(self, output: Any) -> bool:
        if not (
            _supports_empty_filter_repair(self.context.task_type)
            and self.node.get("op") in _EMPTY_OUTPUT_REPAIRABLE_OPS
            and _is_empty_table_output(output)
        ):
            return False
        dependency_frames = self._dependency_frames()
        # Only repair the first node that turns non-empty inputs into an empty
        # result. A downstream node cannot recover rows already removed.
        return not dependency_frames or all(
            not frame.empty for frame in dependency_frames
        )

    def review_fast_path_output(
        self,
        decision: FastPathDecision,
        output: Any,
    ) -> FastPathReview:
        del decision
        if not runtime_feature_enabled(
            self.context.slm_config,
            ENABLE_NODE_REVIEW,
        ):
            return FastPathReview()
        if not self._is_empty_output_review_candidate(output):
            return FastPathReview()
        edge_disabled = _agent_loop_disabled(self.context.slm_config)
        if self.node.get("op") != "Filter":
            if edge_disabled:
                return FastPathReview(
                    route="cloud_replan",
                    action="escalate",
                    reason="edge_repair_disabled",
                    evidence=_empty_output_evidence(self, output),
                )
            return FastPathReview(
                route="edge_repair",
                action="local_repair",
                trigger="fast_path_empty_output",
                reason="empty_local_operation",
                evidence=_empty_output_evidence(self, output),
            )

        mismatch_analysis = self._compute_mismatch_analysis() or {}
        dominant = self._dominant_mismatch()
        evidence = _empty_output_evidence(
            self,
            output,
            mismatch_analysis=mismatch_analysis,
        )
        if dominant == "wrong_column" or mismatch_analysis.get("candidates"):
            return FastPathReview(
                route="cloud_replan",
                action="escalate",
                reason=(
                    "predicate_candidate_column"
                    if mismatch_analysis.get("candidates")
                    else "predicate_wrong_column"
                ),
                evidence=evidence,
            )
        if edge_disabled:
            return FastPathReview(
                route="cloud_replan",
                action="escalate",
                reason="edge_repair_disabled",
                evidence=evidence,
            )
        return FastPathReview(
            route="edge_repair",
            action="local_repair",
            trigger="fast_path_empty_output",
            reason=f"predicate_{dominant or 'unclassified'}",
            evidence=evidence,
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
        use_empty_filter_repair = _use_empty_filter_repair_prompt(
            task_type=self.context.task_type,
            node=self.node,
            trigger=trigger,
        )
        # Plan A: quoting/format mismatch -> lightweight SQL predicate rewrite.
        # Plan B: not_found mismatch -> full Python solve (empty_filter_repair).
        mismatch_analysis = self._compute_mismatch_analysis() or {}
        use_rewrite_predicate = (
            use_empty_filter_repair
            and self.node.get("op") == "Filter"
            and bool(mismatch_analysis)
            and self._dominant_mismatch() in {"quoting", "format"}
            and _predicate_supports_literal_rewrite(
                self.node.get("params", {}).get("predicate")
            )
        )
        def use_rewrite_for_step(steps: list[dict[str, Any]]) -> bool:
            # Use the compact predicate patch once, then fall back to Python
            # repair if the deterministic re-execution rejects it.
            return use_rewrite_predicate and not steps

        def prompt_kind_for_step(
            iteration: int,
            steps: list[dict[str, Any]],
        ) -> str:
            del iteration
            if use_rewrite_for_step(steps):
                return "table_reasoning_rewrite_predicate"
            if use_empty_filter_repair:
                return "table_reasoning_empty_filter_repair"
            return "table_reasoning_agent_loop"
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
                    render_rewrite_predicate_prompt(
                        view=view,
                        iteration=iteration,
                        steps=steps,
                        node=self.node,
                        mismatch_analysis=mismatch_analysis,
                    )
                    if use_rewrite_for_step(steps)
                    else (
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
                    )
                ),
                parse_action=_extract_action_json,
                prompt_kind=prompt_kind_for_step,
            )
        except AgentLoopExecutionError as exc:
            self.agent_loop_trace = exc.trace
            raise
        self.agent_loop_trace = result.trace
        return result.output


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
        # Small models often embed regex escapes (\W, \b, \d, \s) inside JSON
        # string values, which are invalid JSON. Try fixing them before falling
        # back to the slower first-object scan.
        try:
            payload = json.loads(_fix_invalid_json_escapes(stripped))
        except json.JSONDecodeError:
            payload = _load_first_json_object(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Agent action JSON must be an object")
    return payload


# Valid single-character JSON escapes (besides \uXXXX).
_VALID_JSON_ESCAPES = set('"\\/bfnrt')
_INVALID_JSON_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')


def _fix_invalid_json_escapes(text: str) -> str:
    """Double-escape invalid JSON backslash sequences produced by small models.

    Small models frequently write regex patterns (e.g. ``\\b``, ``\\W``, ``\\d``)
    inside JSON string values. In JSON, ``\\b`` is the backspace control char and
    ``\\W``/``\\d`` are invalid escapes. Doubling the backslash turns them into
    literal backslashes so the downstream regex engine still sees the intended
    pattern.
    """
    return _INVALID_JSON_ESCAPE_RE.sub(lambda m: "\\\\" + m.group(1), text)


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
        return True
    return bool(slm_config.get("disable_agent_loop")) or not runtime_feature_enabled(
        slm_config,
        ENABLE_EDGE_AGENT,
    )


def _is_empty_table_output(output: Any) -> bool:
    if isinstance(output, PandasTable):
        return output.frame.empty
    if isinstance(output, pd.DataFrame):
        return output.empty
    frame = getattr(output, "frame", None)
    return isinstance(frame, pd.DataFrame) and frame.empty


def _empty_output_evidence(
    agent: TableReasoningNodeAgent,
    output: Any,
    *,
    mismatch_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_frame = agent._primary_input_frame()
    evidence: dict[str, Any] = {
        "node": {
            "id": agent.node.get("id"),
            "op": agent.node.get("op"),
        },
        "input_rows": len(input_frame.index) if input_frame is not None else None,
        "output_rows": _table_output_row_count(output),
    }
    if mismatch_analysis:
        evidence["mismatch"] = mismatch_analysis
    return {
        key: value
        for key, value in evidence.items()
        if value is not None
    }


def _table_output_row_count(output: Any) -> int | None:
    if isinstance(output, PandasTable):
        return len(output.frame.index)
    if isinstance(output, pd.DataFrame):
        return len(output.index)
    frame = getattr(output, "frame", None)
    return len(frame.index) if isinstance(frame, pd.DataFrame) else None


def _predicate_supports_literal_rewrite(expr: Any) -> bool:
    """Return whether a predicate is safe for a literal-only patch."""

    if not isinstance(expr, dict):
        return False
    expr_type = expr.get("type")
    if expr_type == "logical_op":
        operands = expr.get("operands")
        return (
            expr.get("op") in {"AND", "OR"}
            and isinstance(operands, list)
            and bool(operands)
            and all(_predicate_supports_literal_rewrite(item) for item in operands)
        )
    if expr_type == "binary_op":
        if expr.get("op") != "=":
            return False
        left = expr.get("left")
        right = expr.get("right")
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and {left.get("type"), right.get("type")} == {"column", "literal"}
        )
    if expr_type == "like":
        value = expr.get("expr")
        pattern = expr.get("pattern")
        return (
            isinstance(value, dict)
            and value.get("type") == "column"
            and isinstance(pattern, dict)
            and pattern.get("type") == "literal"
        )
    if expr_type == "in":
        value = expr.get("expr")
        values = expr.get("values")
        return (
            isinstance(value, dict)
            and value.get("type") == "column"
            and isinstance(values, list)
            and bool(values)
            and all(
                isinstance(item, dict) and item.get("type") == "literal"
                for item in values
            )
        )
    return False
