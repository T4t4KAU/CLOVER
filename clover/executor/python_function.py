"""Python function-fill tasks for local Agent execution."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any


class PythonFunctionParseError(ValueError):
    """Raised when a local model response is not a valid function fill."""


@dataclass(frozen=True)
class PythonFunctionTask:
    """A sandbox-local code-fill task with explicit inputs and contract."""

    name: str
    args: tuple[str, ...]
    prompt_code: str
    contract: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


def parse_python_function_action(text: str) -> dict[str, Any]:
    """Parse the local SLM JSON response into a sandbox action."""

    payload = _extract_json_object(text)
    code = payload.get("s")
    if not isinstance(code, str) or not code.strip():
        raise PythonFunctionParseError("Python function action must include string field s")
    return {
        "action": "solve",
        "code": code,
    }


def validate_python_function(code: str, task: PythonFunctionTask) -> None:
    """Ensure the model only defines the expected function."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonFunctionParseError(f"Invalid Python syntax: {exc}") from exc

    body = [
        stmt
        for stmt in tree.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        raise PythonFunctionParseError(
            f"Code must define exactly one function named {task.name}"
        )
    function = body[0]
    if function.name != task.name:
        raise PythonFunctionParseError(f"Function must be named {task.name}")
    if function.decorator_list:
        raise PythonFunctionParseError("Function decorators are not allowed")
    args = function.args
    if (
        args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.posonlyargs
    ):
        raise PythonFunctionParseError("solve must use only ordinary positional args")
    arg_names = tuple(arg.arg for arg in args.args)
    if arg_names != task.args:
        expected = ", ".join(task.args)
        actual = ", ".join(arg_names)
        raise PythonFunctionParseError(
            f"solve args must be ({expected}); got ({actual})"
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PythonFunctionParseError("Python function action is empty")
    candidates = _extract_fenced_json_blocks(text)
    candidates.append(text)
    errors: list[str] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.strip())
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(payload, dict):
            errors.append("JSON payload must be an object")
            continue
        return payload
    detail = "; ".join(errors) if errors else "no JSON object found"
    raise PythonFunctionParseError(f"Unable to parse function-fill JSON: {detail}")


def _extract_fenced_json_blocks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return []
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].startswith("```"):
        return []
    return ["\n".join(lines[1:-1]).strip()]
