"""Lightweight raw table question to task DSL builder."""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUILD_TABLE_DSL_TOOL_NAME = "build_table_dsl"
BUILDER_AGENT_MODE = "builder_agent"

ANSWER_TYPES = {
    "boolean",
    "category",
    "date",
    "list",
    "list[category]",
    "list[number]",
    "list[string]",
    "number",
    "string",
    "table",
}
PROMPT_ANSWER_TYPES = {
    "boolean",
    "number",
    "string",
    "list[number]",
    "list[string]",
}
ANSWER_TYPE_ALIASES = {
    "b": "boolean",
    "bool": "boolean",
    "integer": "number",
    "float": "number",
    "double": "number",
    "ln": "list[number]",
    "ls": "list[string]",
    "n": "number",
    "s": "string",
    "str": "string",
    "text": "string",
    "list_number": "list[number]",
    "list_string": "list[string]",
}
INTENTS = {
    "lookup",
    "filter",
    "count",
    "aggregate",
    "compare",
    "rank",
    "arithmetic",
    "time",
    "multi_hop",
    "fact_check",
}
REQUIRES = {
    "exact_match",
    "numeric_calculation",
    "date_time",
    "row_selection",
}
NUMERIC_WORDS = {
    "average",
    "avg",
    "calculate",
    "count",
    "difference",
    "how many",
    "how much",
    "maximum",
    "mean",
    "median",
    "minimum",
    "number of",
    "sum",
    "total",
}
AGGREGATE_WORDS = {
    "average",
    "avg",
    "mean",
    "median",
    "minimum",
    "maximum",
    "sum",
}
BOOLEAN_PREFIXES = (
    "are ",
    "can ",
    "did ",
    "do ",
    "does ",
    "has ",
    "have ",
    "is ",
    "was ",
    "were ",
)
TARGET_COLUMN_SYNONYMS = {
    "country": ("nation",),
    "nation": ("country", "team"),
    "team": ("club", "nation"),
    "year": ("season",),
    "season": ("year",),
}


@dataclass(frozen=True)
class TableDslBuilderResult:
    """Task DSL plus model trace for a lightweight DSL builder call."""

    task_dsl: dict[str, Any]
    prompt: str
    raw_output: str
    parsed_output: dict[str, Any]
    response_payload: dict[str, Any]
    table_profile: dict[str, Any]
    fallback_used: bool
    builder_mode: str = BUILDER_AGENT_MODE
    tool_call: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None


class BuildTableDSLTool:
    """Resource-level static tool for raw table question to CLOVER task DSL."""

    name = BUILD_TABLE_DSL_TOOL_NAME

    def build_call(
        self,
        *,
        question: str,
        source_id: int = 0,
        source_file: str = "table.csv",
    ) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": {
                "question": question,
                "source_id": source_id,
                "source_file": source_file,
            },
        }

    def run(
        self,
        *,
        question: str,
        table_path: str | Path,
        source_file: str = "table.csv",
        answer_type: str | None = None,
        task_type: str = "table_reasoning.analyze",
        source_id: int = 0,
        max_preview_rows: int = 8,
        max_columns: int = 64,
    ) -> TableDslBuilderResult:
        profile = table_profile_for_dsl_builder(
            table_path,
            max_preview_rows=max_preview_rows,
            max_columns=max_columns,
        )
        tool_call = self.build_call(
            question=question,
            source_id=source_id,
            source_file=source_file,
        )
        diagnostics = static_table_dsl_metadata(
            question=question,
            table_path=table_path,
            table_profile=profile,
        )
        selected_answer_type = _normalize_answer_type(
            answer_type,
            fallback=diagnostics["answer_type"],
        )
        task_dsl = _task_dsl_from_parts(
            question=question,
            source_file=source_file,
            source_id=source_id,
            task_type=task_type,
            answer_type=selected_answer_type,
            hints={},
        )
        return TableDslBuilderResult(
            task_dsl=task_dsl,
            prompt="",
            raw_output="",
            parsed_output={},
            response_payload={},
            table_profile=profile,
            fallback_used=False,
            builder_mode="build_table_dsl_tool",
            tool_call=tool_call,
            diagnostics=diagnostics,
        )


class TableDSLBuilderAgentError(ValueError):
    """Raised when the SLM BuilderAgent cannot choose a valid DSL tool."""


def build_table_task_dsl_with_builder_agent(
    *,
    question: str,
    table_path: str | Path,
    source_file: str = "table.csv",
    answer_type: str | None = None,
    task_type: str = "table_reasoning.analyze",
    source_id: int = 0,
    slm_config: dict[str, Any],
    client: Any | None = None,
    max_preview_rows: int = 8,
    max_columns: int = 64,
) -> TableDslBuilderResult:
    """Build a table task DSL through the BuilderAgent-compatible entry point."""

    # The current table builder has exactly one executable tool. Keep the
    # public entry point, but do not ask a model to spell a static tool name.
    del slm_config, client
    return BuildTableDSLTool().run(
        question=question,
        table_path=table_path,
        source_file=source_file,
        answer_type=answer_type,
        task_type=task_type,
        source_id=source_id,
        max_preview_rows=max_preview_rows,
        max_columns=max_columns,
    )


def table_profile_for_dsl_builder(
    table_path: str | Path,
    *,
    max_preview_rows: int = 2,
    max_columns: int = 48,
    max_cell_chars: int = 40,
) -> dict[str, Any]:
    """Return a compact deterministic table profile for the DSL builder."""

    path = Path(table_path).expanduser().resolve()
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        dialect = _detect_dialect(sample)
        reader = csv.DictReader(handle, dialect=dialect, doublequote=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        columns = list(reader.fieldnames)
        shown_columns = columns[:max_columns]
        preview_rows: list[dict[str, str]] = []
        column_samples: dict[str, list[str]] = {column: [] for column in shown_columns}
        row_count = 0
        for row in reader:
            row_count += 1
            if len(preview_rows) < max_preview_rows:
                preview = {
                    column: _truncate_cell(row.get(column, ""), max_cell_chars)
                    for column in shown_columns
                }
                preview_rows.append(preview)
            for column in shown_columns:
                if len(column_samples[column]) >= max_preview_rows:
                    continue
                value = str(row.get(column, "") or "").strip()
                if value:
                    column_samples[column].append(value)

    return {
        "format": "csv",
        "shape": {"rows": row_count, "columns": len(columns)},
        "columns": columns,
        "shown_columns": shown_columns,
        "omitted_columns": max(0, len(columns) - len(shown_columns)),
        "column_kinds": {
            column: _infer_column_kind(values)
            for column, values in column_samples.items()
        },
        "preview_rows": preview_rows,
    }


def render_table_dsl_builder_agent_prompt(
    *,
    question: str,
    sources: list[dict[str, Any]],
) -> str:
    """Render the SLM BuilderAgent prompt for tool selection only."""

    payload = {
        "question": question,
        "sources": sources,
    }
    return (
        "Choose one tool for this task. Return JSON only.\n"
        "Tools:\n"
        '- build_table_dsl: build a table reasoning DSL from the question and one table source.\n'
        'Return exactly {"tool":"build_table_dsl","arguments":{"source_id":0}}.\n'
        "Do not build the DSL. Do not inspect table contents. Do not add keys.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def parse_table_dsl_builder_output(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a builder response."""

    candidate = _extract_json_object(text)
    if not candidate:
        return {}
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_builder_agent_tool_call(text: str) -> dict[str, Any]:
    """Parse a BuilderAgent tool call JSON object."""

    parsed = parse_table_dsl_builder_output(text)
    if not parsed:
        raise TableDSLBuilderAgentError("BuilderAgent did not return one JSON object")
    return parsed


def static_table_dsl_metadata(
    *,
    question: str,
    table_path: str | Path,
    table_profile: dict[str, Any] | None = None,
    max_value_hit_rows: int = 1000,
) -> dict[str, Any]:
    """Return deterministic routing metadata for BuildTableDSLTool."""

    profile = table_profile or table_profile_for_dsl_builder(table_path)
    columns = list(profile["columns"])
    value_hits = _value_hit_columns(
        question=question,
        table_path=table_path,
        columns=columns,
        max_rows=max_value_hit_rows,
    )
    target_column = _infer_target_column(
        question=question,
        columns=columns,
        column_kinds=profile["column_kinds"],
    )
    selected_columns = _static_selected_columns(
        question=question,
        columns=columns,
        preview_rows=profile["preview_rows"],
        value_hits=value_hits,
        target_column=target_column,
    )
    intent = _static_intents(
        question=question,
        value_hits=value_hits,
        target_column=target_column,
    )
    requires = _static_requires(
        question=question,
        intent=intent,
        value_hits=value_hits,
    )
    answer_type = _static_answer_type(
        question=question,
        target_column=target_column,
        column_kinds=profile["column_kinds"],
        selected_columns=selected_columns,
    )
    hints: dict[str, Any] = {}
    if intent:
        hints["intent"] = intent
    if selected_columns:
        hints["columns"] = selected_columns
    if requires:
        hints["requires"] = requires
    return {
        "answer_type": answer_type,
        "hints": hints,
        "target_column": target_column,
        "value_hits": value_hits,
        "tool": BUILD_TABLE_DSL_TOOL_NAME,
    }


def _task_dsl_from_parts(
    *,
    question: str,
    source_file: str,
    source_id: int,
    task_type: str,
    answer_type: str,
    hints: dict[str, Any],
) -> dict[str, Any]:
    task_dsl = {
        "task_type": task_type,
        "question": question,
        "sources": [
            {
                "id": source_id,
                "type": "table",
                "file": source_file,
            }
        ],
        "answer": {
            "name": "answer",
            "type": answer_type,
        },
    }
    if hints:
        task_dsl["hints"] = hints
    return task_dsl


def _prompt_column_score(
    question: str,
    column: str,
    preview_rows: list[dict[str, str]],
) -> int:
    text = _question_text(question)
    column_text = _question_text(column)
    score = 0
    if column_text and column_text in text:
        score += 6
    for token in _meaningful_tokens(column_text):
        if token in text:
            score += 2
    for row in preview_rows:
        value_text = _question_text(row.get(column, ""))
        if value_text and value_text in text:
            score += 3
            continue
        for token in _meaningful_tokens(value_text):
            if token in text:
                score += 1
    return score


def _meaningful_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    ]


def _value_hit_columns(
    *,
    question: str,
    table_path: str | Path,
    columns: list[str],
    max_rows: int,
) -> list[dict[str, Any]]:
    question_tokens = set(_question_tokens(question))
    question_text = _question_text(question)
    hits: dict[str, set[str]] = {}
    path = Path(table_path).expanduser().resolve()
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        dialect = _detect_dialect(sample)
        reader = csv.DictReader(handle, dialect=dialect, doublequote=True)
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break
            for column in columns:
                value = _normalize_cell_value(row.get(column, ""))
                if not value or len(value) > 60:
                    continue
                if _cell_value_in_question(value, question_text, question_tokens):
                    hits.setdefault(column, set()).add(value)
    return [
        {"column": column, "values": sorted(values)[:5]}
        for column, values in hits.items()
    ]


def _static_selected_columns(
    *,
    question: str,
    columns: list[str],
    preview_rows: list[dict[str, str]],
    value_hits: list[dict[str, Any]],
    target_column: str | None,
    max_selected: int = 8,
) -> list[str]:
    hit_columns = {str(hit.get("column")) for hit in value_hits}
    scores: list[tuple[int, int, str]] = []
    for index, column in enumerate(columns):
        score = _prompt_column_score(question, column, preview_rows)
        if column == target_column:
            score += 10
        if column in hit_columns:
            score += 5
        if _column_name_has_numeric_role(column) and _question_mentions_number(question):
            score += 1
        scores.append((score, index, column))
    selected = [
        column
        for score, _index, column in sorted(scores, key=lambda item: (-item[0], item[1]))
        if score > 0
    ][:max_selected]
    if target_column and target_column not in selected:
        selected.insert(0, target_column)
    return _dedupe(selected[:max_selected])


def _static_intents(
    *,
    question: str,
    value_hits: list[dict[str, Any]],
    target_column: str | None,
) -> list[str]:
    text = _question_text(question)
    intents: list[str] = []
    if _static_answer_type(
        question=question,
        target_column=target_column,
        column_kinds={},
        selected_columns=[],
    ) == "boolean":
        intents.append("fact_check")
    if _asks_for_count(text):
        intents.append("count")
    if _asks_for_aggregate(text):
        intents.append("aggregate")
    if _asks_for_arithmetic(text):
        intents.append("arithmetic")
    if _asks_for_comparison(text):
        intents.append("compare")
    if _asks_for_rank(text):
        intents.append("rank")
    if _mentions_time(text):
        intents.append("time")
    if value_hits or _has_filter_language(text):
        intents.append("filter")
    if target_column or text.startswith(("which ", "who ", "what ", "in which ")):
        intents.append("lookup")
    if _has_multi_part_question(text):
        intents.append("multi_hop")
    if not intents:
        intents.append("lookup")
    return _dedupe(intents)


def _static_requires(
    *,
    question: str,
    intent: list[str],
    value_hits: list[dict[str, Any]],
) -> list[str]:
    text = _question_text(question)
    requires: list[str] = []
    if value_hits or any(token in text for token in ("'", '"', " equals ", " equal to ")):
        requires.append("exact_match")
    if any(item in intent for item in ("count", "aggregate", "arithmetic", "compare")):
        requires.append("numeric_calculation")
    if "time" in intent:
        requires.append("date_time")
    if any(item in intent for item in ("lookup", "filter", "rank", "multi_hop", "fact_check")):
        requires.append("row_selection")
    return _dedupe(requires)


def _static_answer_type(
    *,
    question: str,
    target_column: str | None,
    column_kinds: dict[str, str],
    selected_columns: list[str],
) -> str:
    text = _question_text(question)
    if text.startswith(BOOLEAN_PREFIXES) or "true or false" in text:
        return "boolean"
    if _has_compound_output_question(text):
        return "string"
    if _asks_for_list(text):
        return "list[string]"
    if _asks_for_entity(text) and target_column is None:
        return "string"
    if target_column and text.startswith(("what is ", "what was ", "what were ")):
        kind = column_kinds.get(target_column, "string")
        if kind == "number" or _column_name_looks_numeric_answer(target_column):
            return "number"
        return "string"
    if target_column and text.startswith(("which ", "who ", "in which ")):
        kind = column_kinds.get(target_column, "string")
        if (
            kind == "number"
            or _column_name_is_temporal_numeric(target_column)
            or _column_name_looks_numeric_answer(target_column)
        ):
            return "number"
        return "string"
    if text.startswith("what percentage "):
        return "number"
    if _asks_for_count(text) or _asks_for_aggregate(text) or _asks_for_arithmetic(text):
        return "number"
    if text.startswith(("what is ", "what was ", "what were ", "according to the table what ")):
        answer_column = _first_numeric_selected_column(
            selected_columns=selected_columns,
            column_kinds=column_kinds,
        )
        if answer_column is not None:
            return "number"
    if target_column:
        kind = column_kinds.get(target_column, "string")
        return "number" if kind == "number" else "string"
    return _heuristic_answer_type(question)


def _infer_target_column(
    *,
    question: str,
    columns: list[str],
    column_kinds: dict[str, str],
) -> str | None:
    del column_kinds
    terms = _target_terms_from_question(question)
    if not terms:
        return None
    candidates: list[tuple[int, int, str]] = []
    for index, column in enumerate(columns):
        score = 0
        for term in terms:
            score = max(score, _target_column_score(term, column))
        if score:
            candidates.append((score, index, column))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][2]


def _target_terms_from_question(question: str) -> list[str]:
    text = _question_text(question)
    patterns = (
        r"\bwhat (?:is|was|were) the ([a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:of|for|in|achieved|when|,|\?)",
        r"\bin which ([a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:did|does|do|was|were|is|has|had|will|ranked|with|,|\?)",
        r"\bwhich ([a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:has|had|won|ranked|finished|drove|did|does|do|was|were|is|with|in|of|among|according|,|\?)",
        r"\bwhat ([a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:has|had|won|ranked|finished|drove|did|does|do|was|were|is|with|in|of|among|,|\?)",
    )
    terms = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            terms.append(match.group(1).strip())
    if text.startswith("who "):
        terms.extend(["person", "player", "name", "driver", "winner"])
    return _dedupe([term for term in terms if term])


def _target_column_score(term: str, column: str) -> int:
    term_tokens = _expand_target_tokens(_meaningful_tokens(term))
    column_tokens = set(_expand_token_forms(_meaningful_tokens(column)))
    if not term_tokens or not column_tokens:
        return 0
    term_text = _question_text(term)
    column_text = _question_text(column)
    if term_text == column_text:
        return 10
    if term_text and term_text in column_text:
        return 8
    overlap = sum(1 for token in term_tokens if token in column_tokens)
    return overlap * 3


def _expand_target_tokens(tokens: list[str]) -> list[str]:
    expanded = _expand_token_forms(tokens)
    for token in tokens:
        expanded.extend(TARGET_COLUMN_SYNONYMS.get(token, ()))
    return _dedupe(expanded)


def _expand_token_forms(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        singular = _singular_token(token)
        if singular != token:
            expanded.append(singular)
    return _dedupe(expanded)


def _singular_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _cell_value_in_question(
    value: str,
    question_text: str,
    question_tokens: set[str],
) -> bool:
    value_text = _question_text(value)
    value_tokens = _question_tokens(value)
    if not value_text or not value_tokens:
        return False
    if len(value_text) >= 3 and value_text in question_text:
        return True
    if len(value_tokens) == 1:
        token = value_tokens[0]
        return len(token) >= 2 and token in question_tokens
    return all(token in question_tokens for token in value_tokens)


def _question_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    normalized = []
    for token in tokens:
        ordinal = re.fullmatch(r"(\d+)(st|nd|rd|th)", token)
        normalized.append(ordinal.group(1) if ordinal else token)
    return normalized


def _normalize_cell_value(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip().lower()


def _column_name_has_numeric_role(column: str) -> bool:
    text = _question_text(column)
    return any(
        token in text
        for token in (
            "amount",
            "count",
            "difference",
            "gold",
            "losses",
            "points",
            "rank",
            "score",
            "silver",
            "total",
            "wins",
        )
    )


def _question_mentions_number(question: str) -> bool:
    return any(token.isdigit() for token in _question_tokens(question))


def _asks_for_count(text: str) -> bool:
    return text.startswith("how many ") or " number of " in f" {text} " or " count " in f" {text} "


def _asks_for_aggregate(text: str) -> bool:
    if any(word in text for word in AGGREGATE_WORDS):
        return True
    return text.startswith(("what is the total ", "what was the total ", "what are the total "))


def _asks_for_arithmetic(text: str) -> bool:
    return any(
        word in text
        for word in (
            "calculate",
            "difference",
            "how much more",
            "percentage increase",
            "ratio",
            "required",
        )
    )


def _asks_for_comparison(text: str) -> bool:
    return any(word in text for word in ("greater", "less", "more than", "compare", "surpass"))


def _asks_for_rank(text: str) -> bool:
    return any(word in text for word in ("top", "rank", "ranked", "highest", "lowest", "largest", "smallest"))


def _mentions_time(text: str) -> bool:
    return any(word in text for word in ("date", "year", "month", "season", "before", "after", "between"))


def _has_filter_language(text: str) -> bool:
    return any(
        word in text
        for word in (
            " according to ",
            " drove ",
            " finished ",
            " has ",
            " have ",
            " in the ",
            " with ",
            " won ",
        )
    )


def _has_multi_part_question(text: str) -> bool:
    return any(word in text for word in (" and which ", " and what ", " and how ", " then "))


def _has_compound_output_question(text: str) -> bool:
    return any(
        word in text
        for word in (
            " and in which ",
            " and which ",
            " and what ",
            " and how many ",
            " and how much ",
        )
    )


def _asks_for_list(text: str) -> bool:
    return text.startswith("list ") or text.startswith("what are ") or text.startswith("who are ")


def _asks_for_entity(text: str) -> bool:
    return text.startswith("which ") or text.startswith("who ")


def _column_name_is_temporal_numeric(column: str) -> bool:
    text = _question_text(column)
    return text in {"rank", "ranking", "season", "year"} or text.endswith(" year")


def _column_name_looks_numeric_answer(column: str) -> bool:
    text = _question_text(column)
    return any(
        token in text
        for token in (
            "amount",
            "cost",
            "difference",
            "margin",
            "mintage",
            "percentage",
            "points",
            "population",
            "price",
            "rank",
            "ranking",
            "rate",
            "score",
            "total",
            "winnings",
        )
    )


def _first_numeric_selected_column(
    *,
    selected_columns: list[str],
    column_kinds: dict[str, str],
) -> str | None:
    for column in selected_columns:
        if column_kinds.get(column) == "number":
            return column
    return None


def _builder_agent_slm_config(slm_config: dict[str, Any]) -> dict[str, Any]:
    selected = dict(slm_config)
    selected["temperature"] = 0
    selected["top_p"] = selected.get("top_p", 1.0)
    max_tokens = selected.get("max_tokens", selected.get("max_output_tokens", 96))
    try:
        selected["max_tokens"] = min(int(max_tokens), 96)
    except (TypeError, ValueError):
        selected["max_tokens"] = 96
    if "max_output_tokens" in selected:
        try:
            selected["max_output_tokens"] = min(int(selected["max_output_tokens"]), 96)
        except (TypeError, ValueError):
            selected["max_output_tokens"] = 96
    return selected


def _generate_builder_slm_text(
    prompt: str,
    *,
    slm_config: dict[str, Any],
    client: Any | None,
) -> Any:
    from clover.executor.local_slm import generate_slm_text

    return generate_slm_text(prompt, slm_config=slm_config, client=client)


def _normalize_builder_agent_tool_call(
    parsed: dict[str, Any],
    *,
    question: str,
    source_id: int,
    source_file: str,
) -> dict[str, Any]:
    tool = parsed.get("tool")
    if tool != BUILD_TABLE_DSL_TOOL_NAME:
        raise TableDSLBuilderAgentError(
            f"BuilderAgent selected unsupported tool: {tool!r}"
        )
    arguments = parsed.get("arguments")
    if not isinstance(arguments, dict):
        raise TableDSLBuilderAgentError("BuilderAgent tool call missing arguments")
    selected_source_id = arguments.get("source_id", source_id)
    try:
        selected_source_id = int(selected_source_id)
    except (TypeError, ValueError) as exc:
        raise TableDSLBuilderAgentError(
            f"BuilderAgent selected invalid source_id: {selected_source_id!r}"
        ) from exc
    if selected_source_id != source_id:
        raise TableDSLBuilderAgentError(
            f"BuilderAgent selected source_id {selected_source_id}, expected {source_id}"
        )
    return BuildTableDSLTool().build_call(
        question=question,
        source_id=selected_source_id,
        source_file=source_file,
    )


def _normalize_answer_type(value: Any, *, fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        normalized = ANSWER_TYPE_ALIASES.get(normalized, normalized)
        if normalized in PROMPT_ANSWER_TYPES:
            return normalized
        if normalized in ANSWER_TYPES:
            return normalized
    fallback_normalized = str(fallback or "string").strip().lower()
    return fallback_normalized if fallback_normalized in ANSWER_TYPES else "string"


def _heuristic_answer_type(question: str) -> str:
    text = _question_text(question)
    if text.startswith(BOOLEAN_PREFIXES) or "true or false" in text:
        return "boolean"
    if any(word in text for word in NUMERIC_WORDS):
        return "number"
    if text.startswith(("list ", "which ", "what are ", "who are ")):
        return "list[string]"
    return "string"


def _extract_json_object(text: str) -> str | None:
    stripped = str(text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return None


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.get_dialect("excel")
    if dialect.delimiter not in {",", "\t", ";", "|"}:
        return csv.get_dialect("excel")
    return dialect


def _infer_column_kind(values: list[str]) -> str:
    if not values:
        return "empty"
    numeric = sum(1 for value in values if _is_number(value))
    if numeric == len(values):
        return "number"
    if numeric:
        return "mixed"
    return "string"


def _is_number(value: str) -> bool:
    text = value.strip().replace(",", "")
    if not text:
        return False
    try:
        float(text.rstrip("%"))
    except ValueError:
        return False
    return True


def _truncate_cell(value: Any, max_chars: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "..."


def _question_text(question: str) -> str:
    return " ".join(str(question or "").lower().split())


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = [
    "BUILD_TABLE_DSL_TOOL_NAME",
    "BUILDER_AGENT_MODE",
    "BuildTableDSLTool",
    "TableDSLBuilderAgentError",
    "TableDslBuilderResult",
    "build_table_task_dsl_with_builder_agent",
    "parse_builder_agent_tool_call",
    "parse_table_dsl_builder_output",
    "render_table_dsl_builder_agent_prompt",
    "static_table_dsl_metadata",
    "table_profile_for_dsl_builder",
]
