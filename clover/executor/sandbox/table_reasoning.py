"""Table reasoning sandbox policy for local Agent Loops."""

from __future__ import annotations

import ast
import copy
import io
import importlib
import json
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from clover.executor.context import NodeExecutionContext
from clover.executor.handles import SandboxProjector, TableHandle, ValueHandle
from clover.executor.handles.table import copy_frame as copy_table_frame
from clover.executor.result import error_payload, json_ready, summarize_output
from clover.executor.sandbox.core import SandboxActionResult
from clover.tools.table_reasoning.pandas_backend import (
    PandasExecutionError,
    PandasTable,
    PandasTableReasoningExecutor,
)


TABLE_OUTPUT_OPS = {
    "Scan",
    "Filter",
    "Project",
    "Derive",
    "Aggregate",
    "Group",
    "Sort",
    "Limit",
    "Distinct",
    "Join",
    "SetOp",
    "RepeatUnion",
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "dir": dir,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "globals": globals,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "locals": locals,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "vars": vars,
    "AttributeError": AttributeError,
    "ImportError": ImportError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "NameError": NameError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
    "__import__": None,
}

SAFE_IMPORT_ROOTS = {
    "collections",
    "datetime",
    "decimal",
    "functools",
    "itertools",
    "json",
    "math",
    "numpy",
    "operator",
    "pandas",
    "re",
    "statistics",
}

LOCAL_PATH_PATTERN = re.compile(r"(?<![\w.])/(?:[^\s'\"<>|{}]+/?)+")


@dataclass
class TableReasoningSandboxState:
    """Mutable local workspace state for one table reasoning Agent Loop."""

    task: dict[str, Any]
    inputs: dict[str, Any]
    resources: dict[str, Any]
    feedback: list[dict[str, Any]]
    workspace_globals: dict[str, Any]
    workspace_locals: dict[str, Any]
    max_stdout_chars: int = 4000
    step_index: int = 0
    trace_steps: list[dict[str, Any]] = field(default_factory=list)


class TableReasoningSandboxPolicy:
    """Own one short-lived Python workspace for a table operation."""

    def start(
        self,
        context: NodeExecutionContext,
        *,
        decision: Any,
        trigger: str,
        error: Exception | None,
    ) -> TableReasoningSandboxState:
        dependency_handles = {
            dependency: f"dep_{index}"
            for index, dependency in enumerate(context.node.get("dependency", []))
        }
        resource_handles = {
            resource_id: f"source_{index}"
            for index, resource_id in enumerate(context.node.get("input", []))
        }
        projector = SandboxProjector()
        projected_dependencies = context.resource_view.project_dependencies(projector)
        projected_sources = context.resource_view.project_sources(projector)
        inputs = {
            handle: _workspace_input_value(projected_dependencies[dependency])
            for dependency, handle in dependency_handles.items()
            if dependency in projected_dependencies
        }
        resources = {
            handle: _workspace_input_value(projected_sources[resource_id])
            for resource_id, handle in resource_handles.items()
            if resource_id in projected_sources
        }
        task = {
            "operation": context.node.get("op"),
            "params": copy.deepcopy(context.node.get("params", {})),
            "input_handles": list(inputs),
            "resource_handles": list(resources),
            "operation_note": _operation_note(context.node),
            "output_contract": _output_contract(context.node),
        }
        feedback = [_initial_feedback(decision, trigger, error)]
        helpers = TableReasoningSandboxHelpers()
        tool = TableReasoningToolReference(
            task_type=context.task_type,
            node=context.node,
            tool_name=getattr(decision, "tool", None),
            inputs=inputs,
            resources=resources,
            dependency_handles=dependency_handles,
            resource_handles=resource_handles,
            external_params=context.external_params,
        )
        workspace_globals = {
            "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import},
            "np": np,
            "pd": pd,
            "helpers": helpers,
            "TableHandle": TableHandle,
            "PandasTable": PandasTable,
        }
        workspace_locals = {
            "inputs": inputs,
            "resources": resources,
            "task": copy.deepcopy(task),
            "params": copy.deepcopy(task["params"]),
            "tool": tool,
        }
        workspace_locals.update(inputs)
        workspace_locals.update(resources)
        return TableReasoningSandboxState(
            task=task,
            inputs=inputs,
            resources=resources,
            feedback=feedback,
            workspace_globals=workspace_globals,
            workspace_locals=workspace_locals,
        )

    def view(
        self,
        state: TableReasoningSandboxState,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "task": _json_safe_copy(state.task),
            "inputs": {
                name: _value_summary(value)
                for name, value in state.inputs.items()
            },
            "resources": {
                name: _value_summary(resource)
                for name, resource in state.resources.items()
            },
            "tool": _json_safe_copy(state.workspace_locals["tool"].describe()),
            "feedback": _json_safe_copy(state.feedback),
            "observations": _json_safe_copy(observations),
        }

    def run_action(
        self,
        state: TableReasoningSandboxState,
        action: dict[str, Any],
    ) -> SandboxActionResult:
        state.step_index += 1
        action_name = action.get("action")
        if action_name == "run_python":
            result = _run_python_action(state, action)
        elif action_name == "submit_result":
            result = _submit_result_action(state, action)
        elif action_name == "abort":
            result = _abort_action(action)
        else:
            result = SandboxActionResult(
                ok=False,
                observation={
                    "type": "action_error",
                    "ok": False,
                    "error": {
                        "message": f"Unknown action: {action_name!r}",
                    },
                },
            )
        _append_trace_step(state, action, result)
        return result

    def close(self, state: TableReasoningSandboxState) -> None:
        # Clear all workspace-owned objects. The executor receives a detached
        # result object before this method is called.
        state.inputs.clear()
        state.resources.clear()
        state.feedback.clear()
        state.workspace_locals.clear()
        state.workspace_globals.clear()
        state.trace_steps.clear()


class TableReasoningSandboxHelpers:
    """Small helper surface for table reasoning repair code."""

    def to_number(self, value: Any) -> Any:
        if isinstance(value, pd.Series):
            cleaned = (
                value.astype("string")
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace("%", "", regex=False)
            )
            return pd.to_numeric(cleaned, errors="coerce")
        return pd.to_numeric(value, errors="coerce")

    def to_datetime(self, value: Any) -> Any:
        return pd.to_datetime(value, errors="coerce")

    def parse_list_like(self, value: Any) -> Any:
        if isinstance(value, pd.Series):
            return value.apply(self.parse_list_like)
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
        return parsed if isinstance(parsed, list) else value


class TableReasoningToolReference:
    """Optional local tool reference available inside the Agent workspace."""

    def __init__(
        self,
        *,
        task_type: str,
        node: dict[str, Any],
        tool_name: str | None,
        inputs: dict[str, Any],
        resources: dict[str, Any],
        dependency_handles: dict[str, str],
        resource_handles: dict[str, str],
        external_params: dict[str, Any],
    ) -> None:
        self.name = tool_name
        self.operation = node.get("op")
        self.params = copy.deepcopy(node.get("params", {}))
        self.output = node.get("output")
        self._task_type = task_type
        self._node = {
            "id": node.get("id"),
            "op": node.get("op"),
            "dependency": list(node.get("dependency", [])),
            "input": list(node.get("input", [])),
            "output": node.get("output"),
        }
        self._inputs = inputs
        self._resources = resources
        self._dependency_handles = dict(dependency_handles)
        self._resource_handles = dict(resource_handles)
        self._external_params = copy.deepcopy(external_params)

    def describe(self) -> dict[str, Any]:
        return {
            "available": self.name is not None,
            "name": self.name,
            "operation": self.operation,
            "params": copy.deepcopy(self.params),
            "optional": True,
        }

    def run(self, params: dict[str, Any] | None = None) -> Any:
        """Try the local tool with the current workspace objects."""

        if not self.name:
            raise PandasExecutionError("No local tool is available for this operation")
        operation = self.operation
        selected_params = copy.deepcopy(params if params is not None else self.params)
        if operation == "Scan":
            return self._run_scan()
        call = {
            "task_type": self._task_type,
            "tool": self.name,
            "op": operation,
            "node_id": self._node["id"],
            "input": list(self._node["input"]),
            "dependency": list(self._node["dependency"]),
            "resources": {},
            "upstream_outputs": self._upstream_outputs(),
            "params": selected_params,
            "external_params": copy.deepcopy(self._external_params),
            "output": self.output,
        }
        executor = PandasTableReasoningExecutor(
            resources={},
            external_params=self._external_params,
        )
        return executor.execute_call(call)

    __call__ = run

    def _run_scan(self) -> PandasTable:
        resource_values = [
            self._resources[handle]
            for resource_id in self._node["input"]
            for handle in [self._resource_handles.get(resource_id)]
            if handle in self._resources
        ]
        if len(resource_values) != 1:
            raise PandasExecutionError("Scan expects one available resource")
        value = resource_values[0]
        if isinstance(value, TableHandle):
            return value.to_pandas_table()
        if isinstance(value, pd.DataFrame):
            return PandasTable(value.copy())
        if isinstance(value, PandasTable):
            return value.copy()
        raise PandasExecutionError(
            f"Scan resource is not table-like: {type(value).__name__}"
        )

    def _upstream_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for dependency, handle in self._dependency_handles.items():
            if handle in self._inputs:
                outputs[dependency] = _tool_runtime_value(self._inputs[handle])
        return outputs


def _run_python_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    code = action.get("code")
    if not isinstance(code, str) or not code.strip():
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "action_error",
                "ok": False,
                "error": {"message": "run_python requires non-empty code"},
            },
        )

    stdout_buffer = io.StringIO()
    before_keys = set(state.workspace_locals)
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, state.workspace_globals, state.workspace_locals)  # noqa: S102 - intentional local Agent workspace.
    except Exception as exc:  # noqa: BLE001 - becomes an Agent observation.
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "python_error",
                "ok": False,
                "error": _compact_error(exc),
            },
        )

    stdout = stdout_buffer.getvalue()
    if len(stdout) > state.max_stdout_chars:
        stdout = stdout[: state.max_stdout_chars] + "\n...[truncated]"
    updated_keys = [
        key
        for key in state.workspace_locals
        if not _is_reserved_workspace_name(key)
    ]
    created_keys = sorted(set(updated_keys) - before_keys)
    if "result" in state.workspace_locals:
        candidate = state.workspace_locals["result"]
        candidate, validation_error = _normalize_candidate_output(candidate, state.task)
        if validation_error is None:
            return SandboxActionResult(
                ok=True,
                output=candidate,
                accepted=True,
                terminal=True,
            )
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "repair_needed",
                "ok": False,
                "message": "The code created result, but it does not satisfy the output contract.",
                "error": {"message": validation_error},
                "candidate_summary": _value_summary(state.workspace_locals["result"]),
                "stdout": stdout,
                "created": created_keys,
                "workspace": {
                    key: _value_summary(state.workspace_locals[key])
                    for key in updated_keys
                },
            },
        )
    return SandboxActionResult(
        ok=True,
        observation={
            "type": "repair_needed",
            "ok": False,
            "message": "The code ran, but no variable named result was created.",
            "stdout": stdout,
            "created": created_keys,
            "workspace": {
                key: _value_summary(state.workspace_locals[key])
                for key in updated_keys
            },
        },
    )


def _submit_result_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    name = action.get("name") or action.get("value") or "result"
    if not isinstance(name, str) or not name.strip():
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "action_error",
                "ok": False,
                "error": {"message": "submit_result requires a variable name"},
            },
        )
    if name not in state.workspace_locals:
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "action_error",
                "ok": False,
                "error": {"message": f"Unknown workspace variable: {name}"},
            },
        )
    candidate = state.workspace_locals[name]
    candidate, validation_error = _normalize_candidate_output(candidate, state.task)
    if validation_error is not None:
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "output_validation_error",
                "ok": False,
                "error": {"message": validation_error},
                "candidate_summary": _value_summary(candidate),
            },
        )
    return SandboxActionResult(
        ok=True,
        output=candidate,
        accepted=True,
        terminal=True,
    )


def _abort_action(action: dict[str, Any]) -> SandboxActionResult:
    reason = action.get("reason")
    message = reason if isinstance(reason, str) and reason.strip() else "Agent aborted"
    return SandboxActionResult(
        ok=False,
        terminal=True,
        error={"type": "AgentAbort", "message": message},
    )


def _normalize_candidate_output(
    candidate: Any,
    task: dict[str, Any],
) -> tuple[Any, str | None]:
    operation = task.get("operation")
    if operation in TABLE_OUTPUT_OPS:
        if isinstance(candidate, TableHandle):
            candidate = PandasTable(
                _copy_frame(candidate.frame),
                group_keys=_candidate_group_keys(candidate, task),
            )
        elif isinstance(candidate, pd.DataFrame):
            candidate = PandasTable(
                _copy_frame(candidate),
                group_keys=_candidate_group_keys(candidate, task),
            )
        elif isinstance(candidate, PandasTable):
            candidate = candidate.copy()
            if operation == "Group" and not candidate.group_keys:
                candidate.group_keys = _output_group_keys(task)
        elif _has_dataframe_payload(candidate):
            candidate = PandasTable(
                _copy_frame(candidate.frame),
                group_keys=_candidate_group_keys(candidate, task),
            )
        if not isinstance(candidate, PandasTable):
            return (
                candidate,
                f"Expected a table-like output for {operation}, "
                f"got {type(candidate).__name__}",
            )
        if not hasattr(candidate, "frame"):
            return candidate, f"Expected a table-like output for {operation}"
        return candidate.copy(), None
    if operation == "FormatAnswer":
        if isinstance(candidate, PandasTable):
            return candidate, "Expected an answer value, got a table-like output"
        try:
            normalized = _json_safe_copy(_normalize_format_answer_candidate(candidate, task))
        except Exception as exc:  # noqa: BLE001 - validation report only.
            return candidate, f"FormatAnswer output is not JSON-ready: {exc}"
        return normalized, None
    try:
        normalized = _json_safe_copy(candidate)
    except Exception as exc:  # noqa: BLE001 - validation report only.
        return candidate, f"Output is not JSON-ready: {exc}"
    return normalized, None


def _output_contract(node: dict[str, Any]) -> dict[str, Any]:
    operation = node.get("op")
    if operation in TABLE_OUTPUT_OPS:
        kind = "table"
        required_type = "table_like"
    elif operation == "FormatAnswer":
        kind = "answer_value"
        required_type = "json_ready_value"
    else:
        kind = "value"
        required_type = "json_ready_value"
    return {
        "handle": "result",
        "kind": kind,
        "required_type": required_type,
    }


def _operation_note(node: dict[str, Any]) -> str | None:
    operation = node.get("op")
    params = node.get("params", {})
    if operation == "Group":
        return (
            "Group is metadata-only: preserve all input rows and set group_keys "
            "from params.keys for a later grouped Aggregate."
        )
    if operation == "Aggregate" and params.get("grouped"):
        return (
            "Grouped Aggregate must use the input table handle's group_keys as "
            "the grouping keys and output those key columns plus aggregate aliases."
        )
    return None


def _normalize_format_answer_candidate(candidate: Any, task: dict[str, Any]) -> Any:
    answer = task.get("params", {}).get("answer", {})
    answer_type = str(answer.get("type", "")).lower()
    if (
        answer_type.startswith("list")
        and isinstance(candidate, list)
        and len(candidate) == 1
        and isinstance(candidate[0], list)
    ):
        return candidate[0]
    return candidate


def _initial_feedback(
    decision: Any,
    trigger: str,
    error: Exception | None,
) -> dict[str, Any]:
    feedback = {
        "ok": False,
    }
    if error is not None:
        feedback["message"] = "An earlier automatic attempt did not complete this operation."
        feedback["error"] = _compact_error(error)
    else:
        detail = getattr(decision, "miss_detail", None) or ""
        feedback["message"] = "No automatic result is available for this operation."
        feedback["error"] = {
            "type": getattr(decision, "miss_reason", None) or trigger,
            "message": detail,
        }
    return feedback


def _append_trace_step(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
    result: SandboxActionResult,
) -> None:
    state.trace_steps.append(
        {
            "step": state.step_index,
            "action": action.get("action"),
            "ok": result.ok,
            "accepted": result.accepted,
            "terminal": result.terminal,
            "observation_type": (
                result.observation.get("type")
                if isinstance(result.observation, dict)
                else None
            ),
            "error": copy.deepcopy(result.error),
        }
    )


def _workspace_input_value(value: Any) -> Any:
    if isinstance(value, (TableHandle, ValueHandle)):
        return value.copy()
    if isinstance(value, PandasTable):
        return TableHandle(value.frame, group_keys=value.group_keys)
    if isinstance(value, pd.DataFrame):
        return TableHandle(value)
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - fallback for unusual runtime values.
        return value


def _tool_runtime_value(value: Any) -> Any:
    if isinstance(value, TableHandle):
        return value.to_pandas_table()
    if isinstance(value, ValueHandle):
        return value.unwrap()
    if isinstance(value, PandasTable):
        return value.copy()
    if isinstance(value, pd.DataFrame):
        return PandasTable(_copy_frame(value))
    return copy.deepcopy(value)


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,  # noqa: A002 - matches Python import hook signature.
    locals: dict[str, Any] | None = None,  # noqa: A002 - matches Python import hook signature.
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    root_name = name.split(".", 1)[0]
    if root_name not in SAFE_IMPORT_ROOTS:
        raise ImportError(f"Import is not available in this local workspace: {name}")
    return importlib.import_module(name)


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, ModuleType):
        return {
            "type": "module",
            "name": value.__name__,
        }
    if isinstance(value, type):
        return {
            "type": "type",
            "name": value.__name__,
        }
    if callable(value):
        return {
            "type": "callable",
            "name": getattr(value, "__name__", type(value).__name__),
        }
    if isinstance(value, TableHandle):
        summary = {
            "type": "table_handle",
            "rows": int(len(value.frame)),
            "columns": [str(column) for column in value.frame.columns],
            "preview": json_ready(value.frame.head(3).to_dict(orient="records")),
        }
        if value.group_keys:
            summary["group_keys"] = json_ready(value.group_keys)
        return summary
    if isinstance(value, ValueHandle):
        return _json_safe_copy(summarize_output(value.unwrap()))
    if isinstance(value, pd.DataFrame):
        return {
            "type": type(value).__name__,
            "rows": int(len(value)),
            "columns": [str(column) for column in value.columns],
            "preview": json_ready(value.head(3).to_dict(orient="records")),
        }
    return _json_safe_copy(summarize_output(value))


def _copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return copy_table_frame(frame)


def _candidate_group_keys(candidate: Any, task: dict[str, Any]) -> list[dict[str, Any]]:
    group_keys = _group_keys_from_value(candidate)
    if group_keys:
        return group_keys
    return _output_group_keys(task)


def _output_group_keys(task: dict[str, Any]) -> list[dict[str, Any]]:
    if task.get("operation") != "Group":
        return []
    keys = task.get("params", {}).get("keys", [])
    return copy.deepcopy(keys if isinstance(keys, list) else [])


def _group_keys_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, TableHandle):
        return copy.deepcopy(value.group_keys)
    if isinstance(value, PandasTable):
        return copy.deepcopy(value.group_keys)
    return []


def _compact_error(exc: Exception) -> dict[str, Any]:
    payload = error_payload(exc)
    return {
        "type": payload["type"],
        "message": _redact_local_paths(payload["message"]),
    }


def _reserved_workspace_names() -> set[str]:
    return {
        "inputs",
        "resources",
        "task",
        "params",
        "tool",
        "pd",
        "np",
        "PandasTable",
        "TableHandle",
        "helpers",
    }


def _is_reserved_workspace_name(name: str) -> bool:
    return (
        name in _reserved_workspace_names()
        or name.startswith("dep_")
        or name.startswith("source_")
    )


def _has_dataframe_payload(value: Any) -> bool:
    frame = getattr(value, "frame", None)
    return isinstance(frame, pd.DataFrame)


def _json_safe_copy(value: Any) -> Any:
    payload = json_ready(value)
    try:
        encoded = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return {
            "type": type(value).__name__,
            "preview": str(value),
        }
    return json.loads(encoded)


def _redact_local_paths(text: str) -> str:
    return LOCAL_PATH_PATTERN.sub("<path>", text)
