"""Table reasoning Node View rendering."""

from __future__ import annotations

from typing import Any

from clover.executor.node_views.core import NodeView, NodeViewRenderError


def render_table_node_view(
    node: dict[str, Any],
    *,
    world: dict[str, Any] | None = None,
) -> NodeView:
    """Render one table IR node into a sandbox-local Python code task."""

    selected_world = dict(world or {})
    op = node.get("op")
    if not isinstance(op, str) or not op:
        raise NodeViewRenderError("Table Node View requires node.op")
    inputs = selected_world.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    prompt_code = selected_world.get("code")
    if not isinstance(prompt_code, str) or not prompt_code.strip():
        prompt_code = _default_python_task_code(node, args=tuple(inputs) or ("df",))
    view_world: dict[str, Any] = {
        "inputs": inputs,
        "feedback": _list_or_empty(selected_world.get("feedback")),
        "observations": _list_or_empty(selected_world.get("observations")),
    }
    if isinstance(selected_world.get("diag"), dict):
        view_world["diag"] = dict(selected_world["diag"])
    return NodeView(
        kind=f"table_reasoning.{op.lower()}",
        language="python",
        world=view_world,
        task=prompt_code,
        metadata={
            "op": op,
            "output_contract": selected_world.get("output_contract"),
        },
    )


def _default_python_task_code(node: dict[str, Any], *, args: tuple[str, ...]) -> str:
    signature = f"def solve({', '.join(args)}):"
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        signature,
        '    """',
        f"    {_python_instruction(node)}",
        '    """',
        "    pass",
        "",
        f"result = solve({', '.join(args)})",
    ]
    return "\n".join(lines)


def _python_instruction(node: dict[str, Any]) -> str:
    op = node.get("op")
    params = node.get("params", {})
    if op == "Filter":
        predicate = params.get("predicate")
        return (
            "Return a pandas.DataFrame containing rows matching: "
            f"{_expr_text(predicate)}"
        )
    return (
        "Return the result for this table operation. "
        f"Operation: {op}; params: {params!r}"
    )


def _expr_text(expr: Any) -> str:
    if not isinstance(expr, dict):
        return repr(expr)
    expr_type = expr.get("type")
    if expr_type == "column":
        return str(expr.get("name", ""))
    if expr_type == "literal":
        return repr(expr.get("value"))
    if expr_type == "binary_op":
        return (
            f"({_expr_text(expr.get('left'))} "
            f"{expr.get('op', '')} {_expr_text(expr.get('right'))})"
        )
    if expr_type == "logical_op":
        op = str(expr.get("op", "AND")).upper()
        operands = [
            _expr_text(operand)
            for operand in expr.get("operands", [])
            if isinstance(operand, dict)
        ]
        return f" {op} ".join(operands) if operands else "TRUE"
    return _expr_sql(expr)


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _expr_sql(expr: dict[str, Any]) -> str:
    expr_type = expr.get("type")
    if expr_type == "column":
        return _quote_identifier(str(expr.get("name", "")))
    if expr_type == "literal":
        return _literal_sql(expr.get("value"))
    if expr_type == "null":
        return "NULL"
    if expr_type == "binary_op":
        return (
            f"{_paren_expr(expr.get('left'))} {expr.get('op', '').upper()} "
            f"{_paren_expr(expr.get('right'))}"
        )
    if expr_type == "logical_op":
        op = str(expr.get("op", "AND")).upper()
        operands = [
            _paren_expr(operand)
            for operand in expr.get("operands", [])
            if isinstance(operand, dict)
        ]
        return f" {op} ".join(operands) if operands else "TRUE"
    if expr_type == "not":
        return f"NOT {_paren_expr(expr.get('expr'))}"
    if expr_type == "is_null":
        return f"{_paren_expr(expr.get('expr'))} IS NULL"
    if expr_type == "is_not_null":
        return f"{_paren_expr(expr.get('expr'))} IS NOT NULL"
    if expr_type == "in":
        values = ", ".join(
            _expr_sql(value)
            for value in expr.get("values", [])
            if isinstance(value, dict)
        )
        return f"{_paren_expr(expr.get('expr'))} IN ({values})"
    if expr_type == "like":
        operator = "LIKE" if expr.get("case_sensitive", True) else "ILIKE"
        return f"{_paren_expr(expr.get('expr'))} {operator} {_paren_expr(expr.get('pattern'))}"
    if expr_type == "function_call":
        name = str(expr.get("function", "")).upper()
        args = ", ".join(
            _expr_sql(arg)
            for arg in expr.get("args", [])
            if isinstance(arg, dict)
        )
        return f"{name}({args})"
    raise NodeViewRenderError(f"Unsupported table expression in Node View: {expr_type!r}")


def _paren_expr(expr: Any) -> str:
    if not isinstance(expr, dict):
        raise NodeViewRenderError("Expression must be an object")
    if expr.get("type") in {"column", "literal", "null", "function_call"}:
        return _expr_sql(expr)
    return f"({_expr_sql(expr)})"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"
