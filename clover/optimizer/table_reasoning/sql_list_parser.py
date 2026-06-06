"""Parse batched table reasoning SQL arrays into per-question DAG fragments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from clover.optimizer.table_reasoning.sql_parser import (
    SqlParseError,
    parse_remote_sql_to_logic_dag,
    parse_sql_response,
)
from clover.task_types import is_table_task_type


@dataclass(frozen=True)
class ParsedSqlList:
    """Index-aligned SQL list returned by remote decomposition for batch tasks."""

    sqls: tuple[str, ...]


def parse_sql_list_response(
    remote_response: str,
    remote_dsl: dict[str, Any],
) -> ParsedSqlList:
    """Parse and validate a table query reasoning SQL array."""

    questions, answers = _question_answer_lists(remote_dsl)
    payload = _extract_json_array(remote_response)
    if len(payload) != len(questions):
        raise SqlParseError(
            "SQL array length must equal questions length: "
            f"{len(payload)} != {len(questions)}"
        )

    sqls: list[str] = []
    for index, item in enumerate(payload):
        item_sql = _sql_from_array_item(item, index)
        query_dsl = _query_remote_dsl(
            remote_dsl,
            questions[index],
            answers[index],
        )
        parsed = parse_sql_response(item_sql, query_dsl)
        _validate_answer_alias(parsed.sql, answers[index], index)
        sqls.append(parsed.sql)
    return ParsedSqlList(sqls=tuple(sqls))


def _sql_from_array_item(item: Any, index: int) -> str:
    if isinstance(item, str) and item.strip():
        return item
    if isinstance(item, dict):
        if "final" in item:
            raise SqlParseError(f"SQL array item {index} must not include final")
        sql = item.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql
        if "sqls" in item:
            raise SqlParseError(
                f"SQL array item {index} contains multi-SQL evidence; "
                "this parser expects one SQL per question"
            )
    raise SqlParseError(
        f"SQL array item {index} must be a SQL string or an object with sql"
    )


def parse_remote_sql_list_to_logic_dag(
    remote_response: str,
    remote_dsl: dict[str, Any],
) -> dict[str, Any]:
    """Parse a SQL array into a batch Logic DAG wrapper of query fragments."""

    questions, answers = _question_answer_lists(remote_dsl)
    parsed = parse_sql_list_response(remote_response, remote_dsl)
    query_plans = []
    for index, sql in enumerate(parsed.sqls):
        query_dsl = _query_remote_dsl(
            remote_dsl,
            questions[index],
            answers[index],
        )
        query_plans.append(
            {
                "id": f"Q{index}",
                "index": index,
                "answer": answers[index],
                **_query_fragment(
                    parse_remote_sql_to_logic_dag(sql, query_dsl)
                ),
            }
        )
    return {
        "task_type": remote_dsl["task_type"],
        "resource_processing": [],
        "query_plans": query_plans,
    }


def _question_answer_lists(
    remote_dsl: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not is_table_task_type(remote_dsl.get("task_type")):
        raise SqlParseError("SQL list parser requires a table_reasoning task_type")
    questions = remote_dsl.get("questions")
    answers = remote_dsl.get("answers")
    if not isinstance(questions, list) or not questions:
        raise SqlParseError(
            "table_reasoning.query batch requires a non-empty questions list"
        )
    if not all(isinstance(item, str) and item.strip() for item in questions):
        raise SqlParseError(
            "table_reasoning.query batch questions must be non-empty strings"
        )
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise SqlParseError(
            "table_reasoning.query batch answers length must match questions"
        )
    if not all(isinstance(item, dict) and item.get("name") for item in answers):
        raise SqlParseError(
            "table_reasoning.query batch answers must define answer names"
        )
    return questions, answers


def _query_remote_dsl(
    remote_dsl: dict[str, Any],
    question: str,
    answer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_type": remote_dsl["task_type"],
        "question": question,
        "sources": remote_dsl.get("sources", []),
        "answer": answer,
    }


def _query_fragment(logic_dag: dict[str, Any]) -> dict[str, Any]:
    query_plans = logic_dag.get("query_plans")
    if isinstance(query_plans, list) and len(query_plans) == 1:
        query_plan = query_plans[0]
        if isinstance(query_plan, dict):
            return {
                "nodes": query_plan.get("nodes", []),
                "edges": query_plan.get("edges", []),
            }
    return {
        "nodes": logic_dag.get("nodes", []),
        "edges": logic_dag.get("edges", []),
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
