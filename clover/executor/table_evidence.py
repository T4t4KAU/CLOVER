"""Local table evidence collection for Remote-centered analysis workflows."""

from __future__ import annotations

import ast
import io
import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from clover.executor.agents.template_tree import (
    TABLE_EVIDENCE_LEAF_KEY,
    render_table_evidence_prompt,
)
from clover.executor.python_function import (
    PythonFunctionParseError,
    PythonFunctionTask,
    validate_python_function,
)
from clover.executor.result import json_ready
from clover.executor.sandbox.table_reasoning import SAFE_BUILTINS, _safe_import
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    LocalSlmSequenceDispatcher,
    LocalSlmSequenceRequest,
)
from clover.supervisor.client import extract_token_usage


@dataclass(frozen=True)
class TableEvidenceResult:
    """Accepted evidence returned by one local evidence action."""

    output: Any
    trace: dict[str, Any]


class TableEvidenceExecutionError(RuntimeError):
    """Local evidence collection failed but carries compact trace."""

    def __init__(self, message: str, *, trace: dict[str, Any]) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True)
class _CodeRunResult:
    output: Any
    stdout: str


class _ContractError(ValueError):
    """The submitted evidence ran but did not satisfy the evidence contract."""


class _CodeRunError(RuntimeError):
    """Python code failed after producing optional stdout."""

    def __init__(self, message: str, *, stdout: str) -> None:
        super().__init__(message)
        self.stdout = stdout


class TableEvidenceEnv:
    """Bounded table access exposed to local evidence-collection code."""

    def __init__(
        self,
        *,
        tables: dict[str, pd.DataFrame],
        views: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self._tables = {name: frame.copy(deep=True) for name, frame in tables.items()}
        self._views = {
            name: frame.copy(deep=True) for name, frame in (views or {}).items()
        }

    def tables(self) -> list[str]:
        return sorted(self._tables)

    def views(self) -> list[str]:
        return sorted(self._views)

    def table(self, name: str = "table_1") -> pd.DataFrame:
        return self._copy(self._frame(self._tables, name, "table"))

    def view(self, name: str = "view_1") -> pd.DataFrame:
        return self._copy(self._frame(self._views, name, "view"))

    def frame(self, name: str) -> pd.DataFrame:
        if name in self._views:
            return self.view(name)
        return self.table(name)

    def schema(self, name: str = "table_1") -> list[dict[str, Any]]:
        frame = self.frame(name)
        return [
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "non_null": int(frame[column].notna().sum()),
            }
            for column in frame.columns
        ]

    def head(self, name: str = "table_1", n: int = 20) -> list[dict[str, Any]]:
        frame = self.frame(name)
        return json_ready(frame.head(_bounded_int(n, 1, 50)).to_dict(orient="records"))

    def sample(self, name: str = "table_1", n: int = 20) -> list[dict[str, Any]]:
        frame = self.frame(name)
        count = min(len(frame), _bounded_int(n, 1, 50))
        if count <= 0:
            return []
        return json_ready(frame.sample(n=count, random_state=0).to_dict(orient="records"))

    def values(self, name: str, col: str, n: int = 50) -> list[dict[str, Any]]:
        frame = self.frame(name)
        column = _resolve_column(frame, col)
        counts = frame[column].value_counts(dropna=False).head(_bounded_int(n, 1, 100))
        return [
            {"value": json_ready(index), "count": int(count)}
            for index, count in counts.items()
        ]

    def search_values(self, text: str, n: int = 20) -> list[dict[str, Any]]:
        query = _normalize_match_text(text)
        if not query:
            return []
        limit = _bounded_int(n, 1, 50)
        matches: list[dict[str, Any]] = []
        for scope, frames in (("view", self._views), ("table", self._tables)):
            for table_name, frame in frames.items():
                for column in frame.columns:
                    column_text = str(column)
                    if query in _normalize_match_text(column_text):
                        matches.append(
                            {
                                "scope": scope,
                                "table": table_name,
                                "column": column_text,
                                "value": "<column>",
                                "count": int(len(frame)),
                            }
                        )
                        if len(matches) >= limit:
                            return matches
                    series = frame[column].dropna()
                    if series.empty:
                        continue
                    values = series.astype("string").value_counts().head(200)
                    for value, count in values.items():
                        value_text = str(value)
                        if query in _normalize_match_text(value_text):
                            matches.append(
                                {
                                    "scope": scope,
                                    "table": table_name,
                                    "column": column_text,
                                    "value": json_ready(value),
                                    "count": int(count),
                                }
                            )
                            if len(matches) >= limit:
                                return matches
        return matches

    def _frame(
        self,
        frames: dict[str, pd.DataFrame],
        name: str,
        label: str,
    ) -> pd.DataFrame:
        if name in frames:
            return frames[name]
        available = ", ".join(sorted(frames)) or "<none>"
        raise KeyError(f"Unknown {label} {name!r}; available: {available}")

    @staticmethod
    def _copy(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy(deep=True)


def run_table_evidence_action(
    *,
    source_frames: list[pd.DataFrame],
    view_frames: list[pd.DataFrame] | None = None,
    question: str,
    request: str | None = None,
    need: Any = None,
    slm_config: dict[str, Any] | None,
    slm_dispatcher: LocalSlmSequenceDispatcher | None = None,
    max_iterations: int = 3,
) -> TableEvidenceResult:
    """Run a local code-fill loop that returns evidence, not an answer."""

    task = _python_task(
        source_frames=source_frames,
        view_frames=view_frames or [],
        question=question,
        request=request,
        need=need,
    )
    env = TableEvidenceEnv(
        tables=_named_frames("table", source_frames),
        views=_named_frames("view", view_frames or []),
    )
    bounded_slm_config = _bounded_evidence_slm_config(slm_config)
    observations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    iterations = max(1, int(max_iterations or 1))
    last_error = "no valid evidence output"
    owns_dispatcher = slm_dispatcher is None
    dispatcher = slm_dispatcher or LocalSlmSequenceDispatcher(
        slm_config=bounded_slm_config,
        max_parallel_sequences=_positive_int(
            (bounded_slm_config or {}).get("max_parallel_slm_sequences"),
            default=DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
        ),
        max_pending_sequences=_positive_int(
            (bounded_slm_config or {}).get("max_pending_slm_sequences"),
            default=DEFAULT_MAX_PENDING_SLM_SEQUENCES,
        ),
    )

    try:
        for iteration in range(iterations):
            iteration_index = iteration + 1
            last_iteration = iteration_index >= iterations
            prompt = _render_prompt(
                task,
                observations=observations,
                iteration=iteration_index,
                last_iteration=last_iteration,
            )
            try:
                sequence_result = dispatcher.generate(
                    LocalSlmSequenceRequest(
                        prompt=prompt,
                        leaf_key=TABLE_EVIDENCE_LEAF_KEY,
                        prompt_kind="table_evidence",
                        job_id="table_evidence",
                        iteration=iteration_index,
                        slm_config=bounded_slm_config,
                    )
                )
                llm_result = sequence_result.llm_result
                sequence_trace = sequence_result.trace_metadata()
                usage = extract_token_usage(llm_result.response_payload)
            except Exception as exc:  # noqa: BLE001 - bounded local feedback.
                last_error = str(exc)
                observation = _error_observation("slm_error", exc)
                observations.append(observation)
                steps.append(_trace_step(iteration_index, "slm", observation, {}))
                continue

            try:
                action = _parse_evidence_action(llm_result.text)
            except (PythonFunctionParseError, ValueError) as exc:
                last_error = str(exc)
                observation = _error_observation("invalid_action", exc)
                observations.append(observation)
                steps.append(
                    _trace_step(
                        iteration_index,
                        "parse",
                        observation,
                        usage,
                        sequence_trace=sequence_trace,
                    )
                )
                continue

            if action["action"] == "debug":
                try:
                    run_result = _run_debug_code(env, action["code"])
                except Exception as exc:  # noqa: BLE001 - returned as local feedback.
                    last_error = str(exc)
                    observation = _error_observation("debug_error", exc)
                    if isinstance(exc, _CodeRunError):
                        observation["stdout"] = _bounded_stdout(exc.stdout)
                    observations.append(observation)
                    steps.append(
                        _trace_step(
                            iteration_index,
                            "debug",
                            observation,
                            usage,
                            sequence_trace=sequence_trace,
                        )
                    )
                    continue
                last_error = "debug action did not submit evidence"
                observation = {
                    "type": "debug",
                    "ok": True,
                    "stdout": _bounded_stdout(run_result.stdout),
                    "message": "Use this observation, then submit collect(env).",
                }
                observations.append(observation)
                steps.append(
                    _trace_step(
                        iteration_index,
                        "debug",
                        observation,
                        usage,
                        sequence_trace=sequence_trace,
                    )
                )
                continue

            run_result: _CodeRunResult | None = None
            try:
                run_result = _run_collect_code(task, env, action["code"])
                output = _normalize_evidence_output(run_result.output)
            except _ContractError as exc:
                last_error = str(exc)
                observation = {
                    "type": "contract_error",
                    "ok": False,
                    "error": {"message": str(exc)},
                    "stdout": _bounded_stdout(run_result.stdout if run_result else ""),
                }
                observations.append(observation)
                steps.append(
                    _trace_step(
                        iteration_index,
                        "collect",
                        observation,
                        usage,
                        sequence_trace=sequence_trace,
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001 - returned as local feedback.
                last_error = str(exc)
                observation = _error_observation("python_error", exc)
                if run_result is not None:
                    observation["stdout"] = _bounded_stdout(run_result.stdout)
                elif isinstance(exc, _CodeRunError):
                    observation["stdout"] = _bounded_stdout(exc.stdout)
                observations.append(observation)
                steps.append(
                    _trace_step(
                        iteration_index,
                        "collect",
                        observation,
                        usage,
                        sequence_trace=sequence_trace,
                    )
                )
                continue

            return TableEvidenceResult(
                output=output,
                trace={
                    "kind": "table_evidence",
                    "iterations": iteration_index,
                    "steps": steps
                    + [
                        _trace_step(
                            iteration_index,
                            "collect",
                            {"ok": True},
                            usage,
                            accepted=True,
                            sequence_trace=sequence_trace,
                        )
                    ],
                },
            )
    finally:
        if owns_dispatcher:
            dispatcher.close()

    raise TableEvidenceExecutionError(
        f"table evidence reached max_iterations={iterations}: {last_error}",
        trace={"kind": "table_evidence", "iterations": iterations, "steps": steps},
    )


def _python_task(
    *,
    source_frames: list[pd.DataFrame],
    view_frames: list[pd.DataFrame],
    question: str,
    request: str | None,
    need: Any,
) -> PythonFunctionTask:
    inputs = {
        **_named_frames("table", source_frames),
        **_named_frames("view", view_frames),
    }
    return PythonFunctionTask(
        name="collect",
        args=("env",),
        inputs=inputs,
        contract={"kind": "evidence", "non_null": True},
        prompt_code=_prompt_code(
            source_frames=source_frames,
            view_frames=view_frames,
            question=question,
            request=request,
            need=need,
        ),
    )


def _prompt_code(
    *,
    source_frames: list[pd.DataFrame],
    view_frames: list[pd.DataFrame],
    question: str,
    request: str | None,
    need: Any,
) -> str:
    lines = [
        "# EVIDENCE.py",
        "import pandas as pd",
        "import numpy as np",
        "",
        "def collect(env):",
        '    """',
        "    Return compact evidence, not an answer.",
        "    Use env.view() as seed; use env.table() if seed is narrow.",
        "    Include values/counts/cols/support. pandas/numpy only.",
        "",
        f"    q: {question}",
    ]
    if request:
        lines.append(f"    evidence_request: {request}")
    if need is not None:
        lines.append(
            "    need: "
            + json.dumps(json_ready(need), ensure_ascii=False, separators=(",", ":"))
        )
    lines.extend(
        [
            "    tables:",
            *[f"    {line}" for line in _table_doc('table', source_frames).splitlines()],
            "    views:",
            *[f"    {line}" for line in _table_doc('view', view_frames).splitlines()],
            '    """',
            "    pass",
            "",
            "evidence = collect(env)",
            "assert_evidence(evidence)",
            "print(evidence)",
        ]
    )
    return "\n".join(lines)


def _render_prompt(
    task: PythonFunctionTask,
    *,
    observations: list[dict[str, Any]],
    iteration: int,
    last_iteration: bool,
) -> str:
    feedback = _feedback_text(observations[-2:])
    return render_table_evidence_prompt(
        prompt_code=task.prompt_code,
        feedback=feedback,
        iteration=iteration,
        last_iteration=last_iteration,
    )


def _run_collect_code(
    task: PythonFunctionTask,
    env: TableEvidenceEnv,
    code: str,
) -> _CodeRunResult:
    validate_python_function(code, task)
    namespace: dict[str, Any] = {}
    stdout_buffer = io.StringIO()
    globals_dict = _sandbox_globals(env)
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, globals_dict, namespace)  # noqa: S102 - bounded workspace.
            collect = namespace[task.name]
            output = collect(env)
    except Exception as exc:  # noqa: BLE001 - re-raised with stdout feedback.
        raise _CodeRunError(str(exc), stdout=stdout_buffer.getvalue()) from exc
    return _CodeRunResult(output=output, stdout=stdout_buffer.getvalue())


def _run_debug_code(env: TableEvidenceEnv, code: str) -> _CodeRunResult:
    _validate_debug_code(code)
    stdout_buffer = io.StringIO()
    namespace: dict[str, Any] = {}
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, _sandbox_globals(env), namespace)  # noqa: S102 - bounded workspace.
    except Exception as exc:  # noqa: BLE001 - re-raised with stdout feedback.
        raise _CodeRunError(str(exc), stdout=stdout_buffer.getvalue()) from exc
    return _CodeRunResult(output=None, stdout=stdout_buffer.getvalue())


def _sandbox_globals(env: TableEvidenceEnv) -> dict[str, Any]:
    return {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import},
        "pd": pd,
        "np": np,
        "env": env,
        "assert_evidence": assert_evidence,
    }


def assert_evidence(value: Any) -> None:
    _normalize_evidence_output(value)


def _normalize_evidence_output(value: Any) -> Any:
    if value is None:
        raise _ContractError("evidence must be non-null")
    if isinstance(value, pd.DataFrame):
        if value.empty:
            raise _ContractError("evidence table is empty")
        return {
            "rows": int(len(value)),
            "cols": [str(column) for column in value.columns],
            "data": json_ready(value.head(12).to_dict(orient="records")),
        }
    if isinstance(value, pd.Series):
        if value.empty:
            raise _ContractError("evidence series is empty")
        return json_ready(value.tolist())
    if isinstance(value, dict):
        if not value:
            raise _ContractError("evidence object is empty")
        return _compact_json(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise _ContractError("evidence list is empty")
        return _compact_json(list(value))
    if isinstance(value, (str, int, float, bool, np.generic)):
        return {"value": json_ready(value)}
    return json_ready(value)


def _parse_evidence_action(text: str) -> dict[str, str]:
    payload = _extract_json_object(text)
    debug_code = payload.get("d")
    collect_code = payload.get("c", payload.get("s"))
    has_debug = isinstance(debug_code, str) and bool(debug_code.strip())
    has_collect = isinstance(collect_code, str) and bool(collect_code.strip())
    if has_debug == has_collect:
        raise PythonFunctionParseError("Return exactly one of string fields d or c")
    if has_debug:
        return {"action": "debug", "code": debug_code}
    return {"action": "collect", "code": collect_code}


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PythonFunctionParseError("evidence action is empty")
    candidates = _extract_fenced_json_blocks(text)
    candidates.append(text.strip())
    decoder = json.JSONDecoder()
    errors: list[str] = []
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            for index, char in enumerate(stripped):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(stripped[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
            continue
        if isinstance(payload, dict):
            return payload
        errors.append("JSON payload must be an object")
    detail = "; ".join(errors) if errors else "no JSON object found"
    raise PythonFunctionParseError(f"Unable to parse evidence JSON: {detail}")


def _extract_fenced_json_blocks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return []
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].startswith("```"):
        return []
    return ["\n".join(lines[1:-1]).strip()]


def _validate_debug_code(code: str) -> None:
    if len(code) > 5000:
        raise PythonFunctionParseError("debug code is too long")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonFunctionParseError(f"Invalid debug Python syntax: {exc}") from exc
    forbidden_calls = {"open", "eval", "exec", "compile", "input", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PythonFunctionParseError("debug imports are not allowed; pd and np are already available")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PythonFunctionParseError("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise PythonFunctionParseError("dunder names are not allowed")
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in forbidden_calls:
                raise PythonFunctionParseError(f"debug call {function.id} is not allowed")


def _bounded_evidence_slm_config(
    slm_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if slm_config is None:
        return None
    selected = dict(slm_config)
    timeout_limit = _positive_float(
        selected.get("local_evidence_timeout_seconds", selected.get("node_timeout_seconds")),
        default=60.0,
    )
    current_timeout = _positive_float(selected.get("timeout"), default=timeout_limit)
    selected["timeout"] = min(current_timeout, timeout_limit)
    selected["max_retries"] = 0
    return selected


def _positive_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return number


def _positive_int(value: Any, *, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return number


def _named_frames(prefix: str, frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        f"{prefix}_{index}": frame.copy(deep=True)
        for index, frame in enumerate(frames, start=1)
    }


def _table_doc(prefix: str, frames: list[pd.DataFrame]) -> str:
    if not frames:
        return "<none>"
    lines: list[str] = []
    for name, frame in _named_frames(prefix, frames).items():
        columns = [str(column) for column in frame.columns]
        head = json_ready(frame.head(3).to_dict(orient="records"))
        lines.append(f"{name} rows={len(frame)}")
        lines.append(f"cols={json.dumps(columns[:60], ensure_ascii=False)}")
        if len(columns) > 60:
            lines.append(f"+cols={len(columns) - 60}")
        lines.append(f"head={json.dumps(head, ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(lines)


def _feedback_text(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "<none>"
    return json.dumps(json_ready(observations), ensure_ascii=False, separators=(",", ":"))


def _error_observation(kind: str, exc: Exception) -> dict[str, Any]:
    return {"type": kind, "ok": False, "error": {"message": str(exc)}}


def _trace_step(
    iteration: int,
    action: str,
    observation: dict[str, Any],
    usage: dict[str, Any],
    *,
    accepted: bool = False,
    sequence_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "iteration": iteration,
        "action": action,
        "observation": json_ready(observation),
        "token_usage": json_ready(usage),
    }
    if accepted:
        payload["accepted"] = True
    if sequence_trace is not None:
        payload["sequence"] = json_ready(sequence_trace)
    return payload


def _bounded_stdout(stdout: str, limit: int = 1000) -> str:
    text = str(stdout or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))


def _compact_json(value: Any, *, max_list: int = 20, max_keys: int = 20) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(len(value)),
            "cols": [str(column) for column in value.columns[:40]],
            "data": json_ready(value.head(12).to_dict(orient="records")),
        }
    if isinstance(value, pd.Series):
        return _compact_json(value.tolist(), max_list=max_list, max_keys=max_keys)
    if isinstance(value, dict):
        return {
            str(key): _compact_json(item, max_list=max_list, max_keys=max_keys)
            for key, item in list(value.items())[:max_keys]
        }
    if isinstance(value, (list, tuple)):
        return [
            _compact_json(item, max_list=max_list, max_keys=max_keys)
            for item in list(value)[:max_list]
        ]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...<truncated>"
    return json_ready(value)


def _resolve_column(frame: pd.DataFrame, requested: str) -> Any:
    requested_norm = _normalize_match_text(requested)
    for column in frame.columns:
        if _normalize_match_text(str(column)) == requested_norm:
            return column
    for column in frame.columns:
        if requested_norm in _normalize_match_text(str(column)):
            return column
    raise KeyError(f"Column not found: {requested!r}")


def _normalize_match_text(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())
