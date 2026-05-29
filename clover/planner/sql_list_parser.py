"""Parse table_reasoning_v2 SQL arrays into per-question Logic DAGs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from clover.planner.sql_parser import (
    SqlParseError,
    parse_remote_sql_to_logic_dag,
    parse_sql_response,
)


@dataclass(frozen=True)
class ParsedSqlList:
    """Index-aligned SQL list returned by Commander for v2 tasks."""

    sqls: tuple[str, ...]


def parse_sql_list_response(
    remote_response: str,
    remote_dsl: dict[str, Any],
) -> ParsedSqlList:
    """Parse and validate a table_reasoning_v2 Commander SQL array."""

    questions, answers = _question_answer_lists(remote_dsl)
    payload = _extract_json_array(remote_response)
    if len(payload) != len(questions):
        raise SqlParseError(
            "SQL array length must equal questions length: "
            f"{len(payload)} != {len(questions)}"
        )

    sqls: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str) or not item.strip():
            raise SqlParseError(f"SQL array item {index} must be a non-empty string")
        v1_dsl = _v1_remote_dsl(remote_dsl, questions[index], answers[index])
        parsed = parse_sql_response(item, v1_dsl)
        _validate_answer_alias(parsed.sql, answers[index], index)
        sqls.append(parsed.sql)
    return ParsedSqlList(sqls=tuple(sqls))


def parse_remote_sql_list_to_logic_dag(
    remote_response: str,
    remote_dsl: dict[str, Any],
) -> dict[str, Any]:
    """Parse a v2 SQL array into a batch Logic DAG wrapper of v1 DAGs."""

    questions, answers = _question_answer_lists(remote_dsl)
    parsed = parse_sql_list_response(remote_response, remote_dsl)
    subtasks = []
    for index, sql in enumerate(parsed.sqls):
        v1_dsl = _v1_remote_dsl(remote_dsl, questions[index], answers[index])
        subtasks.append(
            {
                "id": f"Q{index}",
                "index": index,
                "question": questions[index],
                "answer": answers[index],
                "sql": sql,
                "logic_dag": parse_remote_sql_to_logic_dag(sql, v1_dsl),
            }
        )
    return {
        "task_type": "table_reasoning_v2",
        "subtasks": subtasks,
    }


def _question_answer_lists(
    remote_dsl: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if remote_dsl.get("task_type") != "table_reasoning_v2":
        raise SqlParseError("SQL list parser requires task_type table_reasoning_v2")
    questions = remote_dsl.get("questions")
    answers = remote_dsl.get("answers")
    if not isinstance(questions, list) or not questions:
        raise SqlParseError("table_reasoning_v2 requires a non-empty questions list")
    if not all(isinstance(item, str) and item.strip() for item in questions):
        raise SqlParseError("table_reasoning_v2 questions must be non-empty strings")
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise SqlParseError("table_reasoning_v2 answers length must match questions")
    if not all(isinstance(item, dict) and item.get("name") for item in answers):
        raise SqlParseError("table_reasoning_v2 answers must define answer names")
    return questions, answers


def _v1_remote_dsl(
    remote_dsl: dict[str, Any],
    question: str,
    answer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning_v1",
        "question": question,
        "sources": remote_dsl.get("sources", []),
        "answer": answer,
    }


def _extract_json_array(remote_response: str) -> list[Any]:
    if not isinstance(remote_response, str) or not remote_response.strip():
        raise SqlParseError("Remote SQL array response is empty")
    text = remote_response.strip()
    fenced = _extract_fenced_json(text)
    if fenced is not None:
        text = fenced
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SqlParseError(f"Unable to parse SQL JSON array: {exc}") from exc
    if not isinstance(payload, list):
        raise SqlParseError("Remote SQL response must be a JSON array")
    return payload


def _extract_fenced_json(text: str) -> str | None:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _validate_answer_alias(sql: str, answer: dict[str, Any], index: int) -> None:
    answer_name = str(answer.get("name", ""))
    if not answer_name:
        raise SqlParseError(f"answers[{index}].name is required")
    quoted = f'"{answer_name}"'
    unquoted_pattern = re.compile(rf"\bAS\s+{re.escape(answer_name)}\b", re.IGNORECASE)
    if quoted not in sql and not unquoted_pattern.search(sql):
        raise SqlParseError(
            f"SQL array item {index} must alias its output as {quoted}"
        )
