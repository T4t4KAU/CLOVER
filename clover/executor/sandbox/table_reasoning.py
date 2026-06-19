"""Table reasoning sandbox policy for local Agent Loops."""

from __future__ import annotations

import ast
import copy
import importlib
import io
import json
import re
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from clover.config import ENABLE_CONTRACT_GATE, runtime_feature_enabled
from clover.executor.context import NodeExecutionContext
from clover.executor.handles import SandboxProjector, TableHandle, ValueHandle
from clover.executor.handles.table import copy_frame as copy_table_frame
from clover.executor.node_views import NodeView
from clover.executor.node_views.table import render_table_node_view
from clover.executor.python_function import (
    PythonFunctionParseError,
    PythonFunctionTask,
    validate_python_function,
)
from clover.executor.result import error_payload, json_ready, summarize_output
from clover.executor.sandbox.core import SandboxActionResult
from clover.optimizer.table_reasoning.sql_parser import (
    SqlParseError,
    parse_predicate_fragment,
)
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

    node: dict[str, Any]
    task: dict[str, Any]
    python_task: PythonFunctionTask
    inputs: dict[str, Any]
    resources: dict[str, Any]
    feedback: list[dict[str, Any]]
    workspace_globals: dict[str, Any]
    workspace_locals: dict[str, Any]
    max_stdout_chars: int = 4000
    step_index: int = 0
    trace_steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _ActionTableFrameResult:
    name: str | None = None
    frame: pd.DataFrame | None = None
    error: SandboxActionResult | None = None


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
        tables = _workspace_tables(
            inputs=inputs,
            resources=resources,
            dependency_handles=dependency_handles,
            resource_handles=resource_handles,
        )
        task = {
            "operation": context.node.get("op"),
            "params": copy.deepcopy(context.node.get("params", {})),
            "input_handles": list(inputs),
            "resource_handles": list(resources),
            "table_names": sorted(tables),
            "failure_trigger": (
                trigger
                if trigger in {"fast_path_empty_output", "fast_path_execution_error"}
                else None
            ),
            "reject_empty_output": (
                trigger == "fast_path_empty_output"
                and context.node.get("op") in TABLE_OUTPUT_OPS
            ),
            "operation_note": _operation_note(context.node),
            "output_contract": _output_contract(context.node),
            "_contract_gate_enabled": runtime_feature_enabled(
                context.slm_config,
                ENABLE_CONTRACT_GATE,
            ),
        }
        python_task = _python_function_task(
            node=context.node,
            task=task,
            inputs=inputs,
            resources=resources,
            dependency_handles=dependency_handles,
            resource_handles=resource_handles,
        )
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
            "tables": tables,
            "task": copy.deepcopy(task),
            "params": copy.deepcopy(task["params"]),
            "tool": tool,
        }
        workspace_locals.update(inputs)
        workspace_locals.update(resources)
        return TableReasoningSandboxState(
            node=copy.deepcopy(context.node),
            task=task,
            python_task=python_task,
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
    ) -> NodeView:
        world: dict[str, Any] = {
            "code": state.python_task.prompt_code,
            "inputs": {
                name: _frame_prompt_summary(value)
                for name, value in state.python_task.inputs.items()
            },
            "feedback": _compact_feedback(state.feedback),
            "observations": _compact_observations(observations),
            "output_contract": state.task.get("output_contract"),
        }
        # Inject the original cloud SQL as a hint for the Edge Agent.
        source_sql = state.node.get("source_sql")
        if isinstance(source_sql, str) and source_sql.strip():
            world["source_sql"] = source_sql.strip()
        # Inject a one-line few-shot hint tailored to the node op so the SLM
        # sees a concrete pattern without bloating the static prompt prefix.
        few_shot_hint = _few_shot_hint_for_op(state.node)
        if few_shot_hint:
            world["few_shot_hint"] = few_shot_hint
        diag = _failure_diagnostic(state)
        if diag is not None:
            world["diag"] = diag
        return render_table_node_view(state.node, world=world)

    def run_action(
        self,
        state: TableReasoningSandboxState,
        action: dict[str, Any],
    ) -> SandboxActionResult:
        state.step_index += 1
        action_name = action.get("action")
        if action_name == "solve":
            result = _solve_action(state, action)
        elif action_name == "run_python":
            result = _run_python_action(state, action)
        elif action_name == "rewrite_predicate":
            result = _rewrite_predicate_action(state, action)
        elif action_name == "schema":
            result = _schema_action(state, action)
        elif action_name == "sample":
            result = _sample_action(state, action)
        elif action_name == "values":
            result = _values_action(state, action)
        elif action_name == "describe":
            result = _describe_action(state, action)
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
    """Small helper surface for table reasoning sandbox code."""

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


def _python_function_task(
    *,
    node: dict[str, Any],
    task: dict[str, Any],
    inputs: dict[str, Any],
    resources: dict[str, Any],
    dependency_handles: dict[str, str],
    resource_handles: dict[str, str],
) -> PythonFunctionTask:
    function_inputs = _python_function_inputs(
        node=node,
        inputs=inputs,
        resources=resources,
        dependency_handles=dependency_handles,
        resource_handles=resource_handles,
    )
    contract = _python_function_contract(node, task)
    predicate_columns = _predicate_columns(node.get("params", {}).get("predicate"))
    return PythonFunctionTask(
        name="solve",
        args=tuple(function_inputs),
        inputs=function_inputs,
        contract=contract,
        metadata={
            "op": node.get("op"),
            "predicate_columns": predicate_columns,
        },
        prompt_code=_python_function_prompt_code(
            node=node,
            args=tuple(function_inputs),
            contract=contract,
        ),
    )


def _python_function_inputs(
    *,
    node: dict[str, Any],
    inputs: dict[str, Any],
    resources: dict[str, Any],
    dependency_handles: dict[str, str],
    resource_handles: dict[str, str],
) -> dict[str, pd.DataFrame]:
    values = []
    for dependency in node.get("dependency", []):
        handle = dependency_handles.get(dependency)
        if handle in inputs:
            frame = _table_frame_copy(inputs[handle])
            if frame is not None:
                values.append(frame)
    for resource_id in node.get("input", []):
        handle = resource_handles.get(resource_id)
        if handle in resources:
            frame = _table_frame_copy(resources[handle])
            if frame is not None:
                values.append(frame)

    names = _python_function_arg_names(len(values))
    return {
        name: frame
        for name, frame in zip(names, values, strict=True)
    }


def _python_function_arg_names(count: int) -> tuple[str, ...]:
    if count == 0:
        return ()
    if count == 1:
        return ("df",)
    if count == 2:
        return ("left", "right")
    return tuple(f"df{index}" for index in range(count))


def _table_frame_copy(value: Any) -> pd.DataFrame | None:
    if isinstance(value, TableHandle):
        return _copy_frame(value.frame)
    if isinstance(value, PandasTable):
        return _copy_frame(value.frame)
    if isinstance(value, pd.DataFrame):
        return _copy_frame(value)
    frame = getattr(value, "frame", None)
    if isinstance(frame, pd.DataFrame):
        return _copy_frame(frame)
    return None


def _python_function_contract(
    node: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    operation = node.get("op")
    if operation in TABLE_OUTPUT_OPS:
        contract: dict[str, Any] = {
            "kind": "dataframe",
        }
        if task.get("reject_empty_output"):
            contract["non_empty"] = True
        required_columns = _required_output_columns(node)
        if required_columns:
            contract["columns"] = required_columns
        if operation == "Aggregate" and not node.get("params", {}).get("grouped"):
            contract["single_row"] = True
        return contract
    if operation == "FormatAnswer":
        return {
            "kind": "answer",
            "type": node.get("params", {}).get("answer", {}).get("type"),
        }
    return {"kind": "json_value"}


def _required_output_columns(node: dict[str, Any]) -> list[str]:
    operation = node.get("op")
    params = node.get("params", {})
    if operation == "Aggregate":
        return [
            str(item.get("alias"))
            for item in params.get("aggregations", [])
            if isinstance(item, dict) and item.get("alias")
        ]
    if operation == "Project":
        return [
            str(item.get("alias"))
            for item in params.get("expressions", [])
            if isinstance(item, dict) and item.get("alias")
        ]
    return []


def _python_function_prompt_code(
    *,
    node: dict[str, Any],
    args: tuple[str, ...],
    contract: dict[str, Any],
) -> str:
    signature = f"def solve({', '.join(args)}):"
    instruction = _python_function_instruction(node)
    contract_lines = _contract_assertion_lines(contract)
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        signature,
        '    """',
        f"    {instruction}",
        '    """',
        "    # Fill in the function body.",
        "    pass",
        "",
        f"result = solve({', '.join(args)})",
        *contract_lines,
    ]
    return "\n".join(lines)


def _python_function_instruction(node: dict[str, Any]) -> str:
    operation = node.get("op")
    params = node.get("params", {})
    if operation == "Filter":
        predicate = params.get("predicate")
        return (
            "Return a pandas.DataFrame containing rows matching: "
            f"{_expr_text(predicate)}"
        )
    if operation == "Derive":
        return (
            "Return a pandas.DataFrame. Preserve input row grain and compute "
            f"the requested derived values from params: {_compact_json(params)}"
        )
    if operation == "Aggregate":
        return (
            "Return a pandas.DataFrame for the aggregate operation described by "
            f"params: {_compact_json(params)}"
        )
    if operation == "Project":
        return (
            "Return a pandas.DataFrame with the requested projected columns from "
            f"params: {_compact_json(params)}"
        )
    if operation == "FormatAnswer":
        answer = params.get("answer", {})
        return f"Return the answer value of type {answer.get('type')}."
    return f"Complete operation {operation} using params: {_compact_json(params)}"


def _expr_text(expr: Any) -> str:
    if not isinstance(expr, dict):
        return "the requested condition"
    expr_type = expr.get("type")
    if expr_type == "column":
        return str(expr.get("name", "column"))
    if expr_type == "literal":
        return repr(expr.get("value"))
    if expr_type == "null":
        return "NULL"
    if expr_type == "binary_op":
        return (
            f"{_expr_text(expr.get('left'))} "
            f"{_operator_text(expr.get('op'))} "
            f"{_expr_text(expr.get('right'))}"
        )
    if expr_type == "logical_op":
        op = str(expr.get("op", "AND")).lower()
        operands = [
            _expr_text(item)
            for item in expr.get("operands", [])
            if isinstance(item, dict)
        ]
        separator = f" {op} "
        return separator.join(operands) if operands else "all rows"
    if expr_type == "not":
        return f"not ({_expr_text(expr.get('expr'))})"
    if expr_type == "is_null":
        return f"{_expr_text(expr.get('expr'))} is null"
    if expr_type == "is_not_null":
        return f"{_expr_text(expr.get('expr'))} is not null"
    if expr_type == "in":
        values = ", ".join(
            _expr_text(item)
            for item in expr.get("values", [])
            if isinstance(item, dict)
        )
        return f"{_expr_text(expr.get('expr'))} in [{values}]"
    if expr_type == "like":
        return f"{_expr_text(expr.get('expr'))} like {_expr_text(expr.get('pattern'))}"
    if expr_type == "function_call":
        args = ", ".join(
            _expr_text(item)
            for item in expr.get("args", [])
            if isinstance(item, dict)
        )
        return f"{expr.get('function')}({args})"
    return _compact_json(expr)


def _operator_text(operator: Any) -> str:
    mapping = {
        "=": "equals",
        "==": "equals",
        "!=": "does not equal",
        "<>": "does not equal",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
    }
    return mapping.get(str(operator), str(operator))


def _predicate_columns(expr: Any) -> list[str]:
    columns: list[str] = []

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            return
        if item.get("type") == "column":
            name = item.get("name")
            if isinstance(name, str) and name not in columns:
                columns.append(name)
            return
        for value in item.values():
            if isinstance(value, dict):
                visit(value)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

    visit(expr)
    return columns


def _compact_json(value: Any) -> str:
    return json.dumps(_json_safe_copy(value), ensure_ascii=False, separators=(",", ":"))


def _contract_assertion_lines(contract: dict[str, Any]) -> list[str]:
    kind = contract.get("kind")
    if kind != "dataframe":
        return []
    lines = [
        'assert isinstance(result, pd.DataFrame), "solve must return pandas.DataFrame"',
    ]
    for column in contract.get("columns", []):
        lines.append(
            f'assert {column!r} in result.columns, "missing required column: {column}"'
        )
    if contract.get("single_row"):
        lines.append('assert len(result) == 1, "solve must return one row"')
    if contract.get("non_empty"):
        lines.append('assert len(result) > 0, "solve returned empty DataFrame"')
    return lines


def _solve_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    code = action.get("code")
    if not isinstance(code, str) or not code.strip():
        return _action_error("solve requires Python function code")

    try:
        validate_python_function(code, state.python_task)
    except PythonFunctionParseError as exc:
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "invalid_solve_function",
                "ok": False,
                "error": {"message": str(exc)},
                "available_args": list(state.python_task.args),
            },
        )

    stdout_buffer = io.StringIO()
    namespace: dict[str, Any] = {}
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, state.workspace_globals, namespace)  # noqa: S102 - intentional local Agent workspace.
            solve = namespace[state.python_task.name]
            args = [
                _copy_frame(frame)
                for frame in state.python_task.inputs.values()
            ]
            candidate = solve(*args)
    except Exception as exc:  # noqa: BLE001 - becomes an Agent observation.
        stdout = _bounded_stdout(state, stdout_buffer.getvalue())
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "python_error",
                "ok": False,
                "error": _compact_error(exc),
                "traceback_tail": _traceback_tail(exc),
                "stdout": stdout,
                "feedback": _python_function_error_feedback(state, exc),
            },
        )

    stdout = _bounded_stdout(state, stdout_buffer.getvalue())
    if not _contract_gate_enabled(state):
        return SandboxActionResult(
            ok=True,
            output=candidate,
            accepted=True,
            terminal=True,
        )
    contract_error = _validate_python_function_contract(candidate, state.python_task)
    if contract_error is not None:
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "contract_error",
                "ok": False,
                "error": {"message": contract_error},
                "stdout": stdout,
                "candidate_summary": _value_summary(candidate),
                "feedback": _python_function_contract_feedback(
                    state,
                    contract_error,
                    candidate,
                ),
            },
        )

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
            "type": "contract_error",
            "ok": False,
            "error": {"message": validation_error},
            "stdout": stdout,
            "candidate_summary": _value_summary(candidate),
            "feedback": _python_function_contract_feedback(
                state,
                validation_error,
                candidate,
            ),
        },
    )


def _rewrite_predicate_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    """Parse an SLM-supplied SQL predicate and re-run the Filter with it.

    This is the lightweight repair path (plan A): the SLM outputs a SQL
    predicate fragment instead of a full Python function, and the sandbox
    parses it and re-executes the Filter via the pandas backend.
    """
    predicate_sql = action.get("predicate")
    if not isinstance(predicate_sql, str) or not predicate_sql.strip():
        return _action_error("rewrite_predicate requires a 'predicate' SQL string")

    try:
        new_predicate = parse_predicate_fragment(predicate_sql)
    except SqlParseError as exc:
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "invalid_predicate_sql",
                "ok": False,
                "error": {"message": str(exc)},
            },
        )

    node = state.node
    original_predicate = node.get("params", {}).get("predicate")
    if not _same_predicate_structure(original_predicate, new_predicate):
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "invalid_predicate_patch",
                "ok": False,
                "error": {
                    "message": (
                        "Predicate rewrite may change literals only; "
                        "columns, operators, and condition structure must stay unchanged."
                    )
                },
            },
        )
    node_params = copy.deepcopy(node.get("params", {}))
    node_params["predicate"] = new_predicate
    upstream_outputs: dict[str, Any] = {}
    input_handles = state.task.get("input_handles", [])
    for dependency, handle in zip(
        node.get("dependency", []),
        input_handles,
        strict=False,
    ):
        if handle in state.inputs:
            upstream_outputs[dependency] = _tool_runtime_value(state.inputs[handle])

    call = {
        "task_type": state.task.get("task_type"),
        "tool": state.task.get("tool"),
        "op": "Filter",
        "node_id": node.get("id"),
        "input": list(node.get("input", [])),
        "dependency": list(node.get("dependency", [])),
        "resources": {},
        "upstream_outputs": upstream_outputs,
        "params": node_params,
        "external_params": copy.deepcopy(state.task.get("external_params", {})),
        "output": node.get("output"),
    }

    executor = PandasTableReasoningExecutor(
        resources={},
        external_params=call.get("external_params", {}),
    )
    try:
        result = executor.execute_call(call)
    except PandasExecutionError as exc:
        return SandboxActionResult(
            ok=False,
            observation={
                "type": "python_error",
                "ok": False,
                "error": _compact_error(exc),
                "traceback_tail": _traceback_tail(exc),
            },
        )

    if not _contract_gate_enabled(state):
        return SandboxActionResult(
            ok=True,
            output=result,
            accepted=True,
            terminal=True,
        )

    candidate, validation_error = _normalize_candidate_output(result, state.task)
    if validation_error is None and not _candidate_is_empty(candidate):
        return SandboxActionResult(
            ok=True,
            output=candidate,
            accepted=True,
            terminal=True,
        )

    return SandboxActionResult(
        ok=False,
        observation={
            "type": "empty_output",
            "ok": False,
            "error": {"message": "Rewritten predicate still returned 0 rows."},
            "feedback": _rewrite_predicate_feedback(state, new_predicate),
        },
    )


def _candidate_is_empty(candidate: Any) -> bool:
    if isinstance(candidate, PandasTable):
        return candidate.frame.empty
    if isinstance(candidate, pd.DataFrame):
        return candidate.empty
    if isinstance(candidate, dict):
        # Serialized table handle: {"type": "table", "rows": N, ...}
        if candidate.get("type") == "table":
            return int(candidate.get("rows", 0)) == 0
    return False


def _same_predicate_structure(original: Any, rewritten: Any) -> bool:
    return _predicate_structure(original) == _predicate_structure(rewritten)


def _predicate_structure(value: Any) -> Any:
    if isinstance(value, list):
        return [_predicate_structure(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "literal":
        return {"type": "literal"}
    return {
        key: _predicate_structure(item)
        for key, item in value.items()
        if key not in {"value", "value_type"}
    }


def _rewrite_predicate_feedback(
    state: TableReasoningSandboxState,
    new_predicate: dict[str, Any],
) -> dict[str, Any]:
    """Build column-value feedback for a failed rewrite_predicate attempt."""
    from clover.executor.node_views.table import _expr_sql

    feedback: dict[str, Any] = {
        "rewritten_sql": _expr_sql(new_predicate),
    }
    column_values: dict[str, list[Any]] = {}
    for frame in state.python_task.inputs.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        for column in frame.columns:
            col_str = str(column)
            if col_str not in column_values:
                values = _top_column_values(frame[column], limit=8)
                if values:
                    column_values[col_str] = values
    if column_values:
        feedback["column_values"] = column_values
    return feedback


def _top_column_values(series: pd.Series, *, limit: int = 8) -> list[Any]:
    cleaned = series.dropna()
    if cleaned.empty:
        return []
    value_counts = cleaned.astype(str).value_counts().head(limit)
    return [
        {"value": str(idx), "count": int(count)}
        for idx, count in value_counts.items()
    ]


def _bounded_stdout(state: TableReasoningSandboxState, stdout: str) -> str:
    if len(stdout) > state.max_stdout_chars:
        return stdout[: state.max_stdout_chars] + "\n...[truncated]"
    return stdout


def _validate_python_function_contract(
    candidate: Any,
    task: PythonFunctionTask,
) -> str | None:
    contract = task.contract
    if contract.get("kind") != "dataframe":
        return None
    if not isinstance(candidate, pd.DataFrame):
        return f"solve must return pandas.DataFrame, got {type(candidate).__name__}"
    missing = [
        column
        for column in contract.get("columns", [])
        if column not in candidate.columns
    ]
    if missing:
        return f"solve result is missing required columns: {missing}"
    if contract.get("single_row") and len(candidate) != 1:
        return "solve must return one row"
    if contract.get("non_empty") and candidate.empty:
        return "solve returned empty DataFrame"
    return None


def _python_function_error_feedback(
    state: TableReasoningSandboxState,
    exc: Exception,
) -> dict[str, Any]:
    feedback: dict[str, Any] = {
        "available_args": list(state.python_task.args),
        "available_libs": ["pd", "np", "helpers", "print"],
        "error_type": type(exc).__name__,
    }
    if isinstance(exc, NameError):
        feedback["message"] = (
            "Only solve arguments and provided libraries are visible. "
            "Use the exact argument names listed in available_args."
        )
        feedback["hint"] = "Check for typos in column names or undefined variables."
    elif isinstance(exc, KeyError):
        feedback["columns"] = _input_columns_feedback(state.python_task)
        feedback["message"] = (
            "KeyError: the requested column does not exist. "
            "Use one of the columns listed in 'columns'."
        )
        feedback["hint"] = (
            "Column names are case-sensitive. Compare against 'columns' field."
        )
    elif isinstance(exc, (TypeError, ValueError)):
        feedback["message"] = (
            f"{type(exc).__name__}: check operand types before comparison or arithmetic. "
            "Use pd.to_numeric(series, errors='coerce') for numeric coercion, "
            "or str(series).str.strip() for text normalization."
        )
        feedback["hint"] = (
            "Mixed-type columns often need explicit coercion before comparison."
        )
    elif isinstance(exc, AttributeError):
        feedback["columns"] = _input_columns_feedback(state.python_task)
        feedback["message"] = (
            "AttributeError: the object does not have the requested attribute. "
            "Verify the column exists and is a Series before calling .str/.dt/.apply."
        )
        feedback["hint"] = "Use df['col'] access; check dtype with df['col'].dtype."
    elif isinstance(exc, IndexError):
        feedback["message"] = (
            "IndexError: index out of range. Avoid positional indexing on "
            "DataFrames; use .iloc with explicit bounds or boolean masks."
        )
    else:
        feedback["message"] = (
            f"{type(exc).__name__}: inspect the traceback and fix the root cause."
        )
    return feedback


def _python_function_contract_feedback(
    state: TableReasoningSandboxState,
    message: str,
    candidate: Any,
) -> dict[str, Any]:
    feedback: dict[str, Any] = {
        "contract": state.python_task.contract,
    }
    message_lower = message.lower()
    if "empty" in message_lower:
        feedback["column_values"] = _predicate_value_feedback(state.python_task)
        feedback["hint"] = (
            "Empty output means the predicate matched nothing. "
            "Inspect 'column_values' for the actual values in the data, then "
            "relax the match: use str.contains(pattern, case=False, regex=False) "
            "for substring match, or normalize both sides with "
            ".str.casefold().str.replace(r'[^a-z0-9]', '', regex=True) before "
            "comparing. Do NOT pass casefold= as a keyword argument to str.contains."
        )
    if "missing required columns" in message_lower:
        feedback["expected_columns"] = list(
            state.python_task.contract.get("columns", [])
        )
        feedback["hint"] = (
            "Add the missing columns listed in 'expected_columns' to the result."
        )
    if "one row" in message_lower:
        feedback["hint"] = (
            "Aggregate must produce exactly one row. "
            "Wrap the result in a single-row DataFrame: pd.DataFrame({...})."
        )
    if isinstance(candidate, pd.DataFrame):
        feedback["columns"] = [str(column) for column in candidate.columns]
        feedback["rows"] = int(len(candidate))
    return feedback


def _input_columns_feedback(task: PythonFunctionTask) -> dict[str, list[str]]:
    return {
        name: [str(column) for column in frame.columns]
        for name, frame in task.inputs.items()
        if isinstance(frame, pd.DataFrame)
    }


def _predicate_value_feedback(task: PythonFunctionTask) -> dict[str, Any]:
    columns = task.metadata.get("predicate_columns")
    if not isinstance(columns, list) or not columns:
        return {}
    if not task.inputs:
        return {}
    frame = next(iter(task.inputs.values()))
    if not isinstance(frame, pd.DataFrame):
        return {}
    feedback: dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        counts = frame[column].value_counts(dropna=False).head(12)
        feedback[str(column)] = [
            {
                "value": _to_json_scalar(index),
                "count": int(count),
            }
            for index, count in counts.items()
        ]
    return feedback


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
        if not _contract_gate_enabled(state):
            return SandboxActionResult(
                ok=True,
                output=candidate,
                accepted=True,
                terminal=True,
            )
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
                "type": "contract_error",
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
            "type": "contract_error",
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


def _schema_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    frame_result = _action_table_frame(state, action)
    if frame_result.error is not None:
        return frame_result.error
    frame = frame_result.frame
    assert frame is not None
    observation = {
        "type": "schema",
        "ok": True,
        "table": frame_result.name,
        "rows": int(len(frame)),
        "columns": [
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "non_null": int(frame[column].notna().sum()),
            }
            for column in frame.columns
        ],
    }
    return SandboxActionResult(ok=True, observation=_json_safe_copy(observation))


def _sample_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    frame_result = _action_table_frame(state, action)
    if frame_result.error is not None:
        return frame_result.error
    frame = frame_result.frame
    assert frame is not None
    n = _bounded_action_int(action.get("n"), default=5, minimum=1, maximum=10)
    observation = {
        "type": "sample",
        "ok": True,
        "table": frame_result.name,
        "rows": int(len(frame)),
        "sample": frame.head(n).to_dict(orient="records"),
    }
    return SandboxActionResult(ok=True, observation=_json_safe_copy(observation))


def _values_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    frame_result = _action_table_frame(state, action)
    if frame_result.error is not None:
        return frame_result.error
    frame = frame_result.frame
    assert frame is not None
    column = action.get("column")
    if not isinstance(column, str) or column not in frame.columns:
        return _action_error(
            "values requires an existing column",
            extra={"columns": [str(item) for item in frame.columns]},
        )
    n = _bounded_action_int(action.get("n"), default=20, minimum=1, maximum=30)
    counts = frame[column].value_counts(dropna=False).head(n)
    observation = {
        "type": "values",
        "ok": True,
        "table": frame_result.name,
        "column": column,
        "values": [
            {
                "value": _to_json_scalar(index),
                "count": int(count),
            }
            for index, count in counts.items()
        ],
    }
    return SandboxActionResult(ok=True, observation=_json_safe_copy(observation))


def _describe_action(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> SandboxActionResult:
    frame_result = _action_table_frame(state, action)
    if frame_result.error is not None:
        return frame_result.error
    frame = frame_result.frame
    assert frame is not None
    column = action.get("column")
    if not isinstance(column, str) or column not in frame.columns:
        return _action_error(
            "describe requires an existing column",
            extra={"columns": [str(item) for item in frame.columns]},
        )
    series = frame[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        description = {
            "count": int(numeric.notna().sum()),
            "mean": _to_json_scalar(numeric.mean()),
            "std": _to_json_scalar(numeric.std()),
            "min": _to_json_scalar(numeric.min()),
            "max": _to_json_scalar(numeric.max()),
        }
    else:
        counts = series.value_counts(dropna=False).head(10)
        description = {
            "count": int(series.notna().sum()),
            "top_values": [
                {"value": _to_json_scalar(index), "count": int(count)}
                for index, count in counts.items()
            ],
        }
    observation = {
        "type": "describe",
        "ok": True,
        "table": frame_result.name,
        "column": column,
        "dtype": str(series.dtype),
        "description": description,
    }
    return SandboxActionResult(ok=True, observation=_json_safe_copy(observation))


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
    if not _contract_gate_enabled(state):
        return SandboxActionResult(
            ok=True,
            output=candidate,
            accepted=True,
            terminal=True,
        )
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


def _action_table_frame(
    state: TableReasoningSandboxState,
    action: dict[str, Any],
) -> _ActionTableFrameResult:
    name = action.get("table")
    tables = state.workspace_locals.get("tables", {})
    if not isinstance(tables, dict) or not tables:
        return _ActionTableFrameResult(
            error=_action_error("No tables are available")
        )
    if not isinstance(name, str) or not name.strip():
        if len(tables) == 1:
            name = next(iter(tables))
        else:
            return _ActionTableFrameResult(
                error=_action_error(
                    "A table name is required",
                    extra={"tables": sorted(str(item) for item in tables)},
                )
            )
    if name not in tables:
        return _ActionTableFrameResult(
            error=_action_error(
                f"Unknown table: {name}",
                extra={"tables": sorted(str(item) for item in tables)},
            )
        )
    value = tables[name]
    if isinstance(value, TableHandle):
        return _ActionTableFrameResult(name=name, frame=value.frame)
    if isinstance(value, PandasTable):
        return _ActionTableFrameResult(name=name, frame=value.frame)
    if isinstance(value, pd.DataFrame):
        return _ActionTableFrameResult(name=name, frame=value)
    frame = getattr(value, "frame", None)
    if isinstance(frame, pd.DataFrame):
        return _ActionTableFrameResult(name=name, frame=frame)
    return _ActionTableFrameResult(
        error=_action_error(f"Table {name} is not dataframe-like")
    )


def _action_error(
    message: str,
    *,
    extra: dict[str, Any] | None = None,
) -> SandboxActionResult:
    observation = {
        "type": "action_error",
        "ok": False,
        "error": {"message": message},
    }
    if extra:
        observation.update(extra)
    return SandboxActionResult(ok=False, observation=_json_safe_copy(observation))


def _bounded_action_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _to_json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001 - keep original value for json_ready.
            pass
    return json_ready(value)


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
        if task.get("reject_empty_output") and candidate.frame.empty:
            return candidate, "Expected a useful non-empty table output"
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


def _contract_gate_enabled(state: TableReasoningSandboxState) -> bool:
    return state.task.get("_contract_gate_enabled") is not False


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
    if trigger == "fast_path_empty_output":
        feedback["message"] = "An earlier automatic attempt returned no useful result."
        feedback["error"] = {
            "type": trigger,
            "message": "Inspect the provided tables before deciding.",
        }
        return feedback
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


def _workspace_tables(
    *,
    inputs: dict[str, Any],
    resources: dict[str, Any],
    dependency_handles: dict[str, str],
    resource_handles: dict[str, str],
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for handle, value in {**inputs, **resources}.items():
        if _is_table_like(value):
            tables[handle] = value
    for dependency, handle in dependency_handles.items():
        value = inputs.get(handle)
        if _is_table_like(value):
            tables[dependency] = value
    for resource_id, handle in resource_handles.items():
        value = resources.get(handle)
        if _is_table_like(value):
            tables[resource_id] = value
    return tables


def _is_table_like(value: Any) -> bool:
    if isinstance(value, (TableHandle, PandasTable, pd.DataFrame)):
        return True
    return isinstance(getattr(value, "frame", None), pd.DataFrame)


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


def _frame_prompt_summary(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "cols": _truncate_list(columns, limit=80),
    }
    if len(columns) > 80:
        summary["more_cols"] = len(columns) - 80
    # Inject top distinct values per column so the Edge Agent can see actual
    # data formats (e.g. "$1.2M", "Name (CODE)") without needing a separate
    # observation step.
    sample_values: dict[str, list[Any]] = {}
    for column in columns[:20]:
        series = frame[column]
        top_values = series.value_counts(dropna=True).head(3)
        sample_values[column] = [
            _to_json_scalar(index) for index in top_values.index
        ]
    if sample_values:
        summary["sample_values"] = sample_values
    return summary


def _few_shot_hint_for_op(node: dict[str, Any]) -> str | None:
    """Return a one-line pattern hint tailored to the node op.

    The hint is injected into the dynamic payload (world.few_shot_hint) so it
    stays out of the static prompt prefix and only appears for ops that need
    guidance. Returns None for ops with no tailored hint.
    """
    op = node.get("op")
    params = node.get("params", {}) if isinstance(node.get("params"), dict) else {}
    if op == "Derive":
        return (
            "Pattern: parse text to numbers with regex; "
            "strip currency symbols, multiply by suffix (K=1e3, M=1e6, B=1e9)."
        )
    if op == "Filter":
        predicate = params.get("predicate")
        if isinstance(predicate, dict) and predicate.get("type") in {
            "like",
            "logical_op",
        }:
            return (
                "Pattern: normalize both sides (casefold, keep alnum only), "
                "then use str.contains for fuzzy text match."
            )
        return None
    if op == "FormatAnswer":
        return (
            "Pattern: return pd.DataFrame({\"answer\": [value]}) with exactly one row."
        )
    if op == "Join":
        return (
            "Pattern: use pd.merge(left, right, on=key, how='inner'); "
            "rename conflicting columns with suffixes."
        )
    return None


def _failure_diagnostic(
    state: TableReasoningSandboxState,
) -> dict[str, Any] | None:
    trigger = state.task.get("failure_trigger")
    if trigger not in {"fast_path_empty_output", "fast_path_execution_error"}:
        return None
    inputs = {
        name: _frame_failure_diagnostic(frame, state)
        for name, frame in state.python_task.inputs.items()
        if isinstance(frame, pd.DataFrame)
    }
    if not inputs:
        return None
    if trigger == "fast_path_empty_output":
        hint = "Previous automatic result was empty. Inspect values and implement solve directly."
    else:
        hint = "Previous automatic execution failed. Inspect columns and implement solve directly."
    return {
        "hint": hint,
        "inputs": inputs,
    }


def _frame_failure_diagnostic(
    frame: pd.DataFrame,
    state: TableReasoningSandboxState,
) -> dict[str, Any]:
    referenced_columns = _referenced_columns(state.task.get("params", {}))
    selected_columns = _diagnostic_columns(
        frame,
        referenced_columns=referenced_columns,
        limit=12,
    )
    diag: dict[str, Any] = {
        "head": _frame_head_records(frame, selected_columns, rows=5),
    }
    values = _diagnostic_values(frame, referenced_columns)
    if values:
        diag["values"] = values
    ranges = _diagnostic_ranges(frame, selected_columns)
    if ranges:
        diag["ranges"] = ranges
    return diag


def _referenced_columns(value: Any) -> list[str]:
    columns: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("type") == "column":
                name = item.get("name")
                if isinstance(name, str) and name not in columns:
                    columns.append(name)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return columns


def _diagnostic_columns(
    frame: pd.DataFrame,
    *,
    referenced_columns: list[str],
    limit: int,
) -> list[Any]:
    selected: list[Any] = []
    frame_columns = list(frame.columns)
    by_text = {str(column): column for column in frame_columns}
    for column in referenced_columns:
        actual = by_text.get(column)
        if actual is not None and actual not in selected:
            selected.append(actual)
    for column in frame_columns:
        if column not in selected:
            selected.append(column)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _frame_head_records(
    frame: pd.DataFrame,
    columns: list[Any],
    *,
    rows: int,
) -> list[dict[str, Any]]:
    if not columns:
        return []
    return json_ready(frame.loc[:, columns].head(rows).to_dict(orient="records"))


def _diagnostic_values(
    frame: pd.DataFrame,
    referenced_columns: list[str],
) -> dict[str, Any]:
    if not referenced_columns:
        return {}
    by_text = {str(column): column for column in frame.columns}
    values: dict[str, Any] = {}
    for column in referenced_columns[:8]:
        actual = by_text.get(column)
        if actual is None:
            continue
        counts = frame[actual].value_counts(dropna=False).head(8)
        values[column] = [
            {
                "v": _to_json_scalar(index),
                "n": int(count),
            }
            for index, count in counts.items()
        ]
    return values


def _diagnostic_ranges(
    frame: pd.DataFrame,
    columns: list[Any],
) -> dict[str, Any]:
    ranges: dict[str, Any] = {}
    for column in columns:
        numeric = _numeric_series(frame[column])
        if numeric is None:
            continue
        ranges[str(column)] = {
            "min": _to_json_scalar(numeric.min()),
            "max": _to_json_scalar(numeric.max()),
        }
    return ranges


def _numeric_series(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype("string")
            .str.replace(r"[\$,]", "", regex=True)
            .str.replace("%", "", regex=False)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric = numeric.dropna()
    return numeric if not numeric.empty else None


def _compact_feedback(feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_feedback_item(item) for item in feedback[-2:]]


def _compact_feedback_item(item: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": bool(item.get("ok"))}
    message = item.get("message")
    if isinstance(message, str) and message:
        payload["msg"] = _truncate_text(_redact_local_paths(message), 240)
    error = item.get("error")
    if isinstance(error, dict):
        payload["err"] = _compact_error_payload(error)
    return payload


def _compact_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_observation(item) for item in observations[-3:]]


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in observation.items():
        if key == "workspace":
            continue
        if key == "stdout" and isinstance(value, str):
            payload[key] = _truncate_text(_redact_local_paths(value), 700)
        elif key == "error" and isinstance(value, dict):
            payload[key] = _compact_error_payload(value)
        elif key == "candidate_summary" and isinstance(value, dict):
            payload[key] = _compact_summary(value)
        elif key == "sample" and isinstance(value, list):
            payload[key] = _truncate_list(_json_safe_copy(value), limit=5)
        elif key == "values" and isinstance(value, list):
            payload[key] = _truncate_list(_json_safe_copy(value), limit=15)
        elif key == "columns" and isinstance(value, list):
            payload[key] = _truncate_list(_json_safe_copy(value), limit=80)
        else:
            payload[key] = _json_safe_copy(value)
    return payload


def _compact_error_payload(error: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    error_type = error.get("type")
    if error_type:
        payload["type"] = error_type
    message = error.get("message")
    if isinstance(message, str) and message:
        payload["msg"] = _truncate_text(_redact_local_paths(message), 400)
    return payload


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in summary.items()
        if key not in {"preview"}
    }
    columns = payload.get("columns")
    if isinstance(columns, list):
        payload["columns"] = _truncate_list(columns, limit=80)
    return _json_safe_copy(payload)


def _truncate_list(items: list[Any], *, limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    return items[:limit] + [{"truncated": len(items) - limit}]


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


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


def _traceback_tail(exc: Exception, *, max_lines: int = 8) -> list[str]:
    """Return the last few traceback lines so the Agent can locate the error."""

    frames = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = frames[-max_lines:]
    return [_redact_local_paths(line.rstrip("\n")) for line in tail]


def _reserved_workspace_names() -> set[str]:
    return {
        "inputs",
        "resources",
        "tables",
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
