"""Rule-based mismatch classification for empty Filter repair.

Classifies why a Filter predicate returned 0 rows by comparing SQL literals
against actual column values. Used to route repair strategy:

- quoting / format  -> SLM rewrites SQL predicate (lightweight, plan A)
- not_found         -> SLM writes Python (flexible, plan B)
- wrong_column or candidate column -> global replanning (plan C)
- system_bug        -> SLM writes Python against the local dataframe
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def analyze_predicate_mismatch(
    predicate: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Analyze a failed Filter predicate against the actual frame.

    Returns a dict with:
        sql: rendered SQL fragment of the predicate
        roots: per-column mismatch analysis (sql_lit, actual, mismatch)
        candidates: columns whose values partially match a literal (max 5)
    """
    from clover.executor.node_views.table import _expr_sql

    sql = _expr_sql(predicate)
    literals = _extract_predicate_literals(predicate)
    columns = _predicate_columns(predicate)

    by_text = {str(col): col for col in frame.columns}
    roots: list[dict[str, Any]] = []
    for col_name in columns:
        actual_col = by_text.get(col_name)
        if actual_col is None:
            roots.append({
                "col": col_name,
                "sql_lit": literals.get(col_name, []),
                "actual": [],
                "mismatch": "wrong_column",
            })
            continue
        col_literals = literals.get(col_name, [])
        actual_values = _top_values(frame[actual_col])
        mismatch = classify_mismatch(col_literals, actual_values)
        roots.append({
            "col": col_name,
            "sql_lit": col_literals,
            "actual": actual_values,
            "mismatch": mismatch,
        })

    exclude_cols = set(columns)
    candidates = find_candidate_columns(frame, literals, exclude_cols)

    return {
        "sql": sql,
        "roots": roots,
        "candidates": candidates,
    }


def classify_mismatch(
    sql_literals: list[str],
    actual_values: list[str],
) -> str:
    """Classify why SQL literals don't match actual values.

    Returns one of: quoting, format, not_found, wrong_column, system_bug.
    """
    if not actual_values:
        return "wrong_column"

    # Check quoting/format: actual value matches literal after stripping quotes.
    # These are the most common repairable cases, so check first.
    for lit in sql_literals:
        lit_content = _strip_quotes(lit).lower()
        if not lit_content:
            continue
        for val in actual_values:
            val_content = _strip_quotes(str(val)).lower()
            if lit_content == val_content:
                return "quoting"
            if lit_content in val_content or val_content in lit_content:
                return "format"

    # Check format: both sides look like dates in different representations.
    if _looks_like_date_literals(sql_literals):
        for val in actual_values:
            if _looks_like_date_value(str(val)):
                return "format"

    # Check system_bug: actual value contains the literal substring (LIKE should
    # have matched but failed due to backend bug, e.g. newline in cell).
    for lit in sql_literals:
        lit_clean = _strip_sql_wildcards(lit).lower()
        if not lit_clean:
            continue
        for val in actual_values:
            if lit_clean in str(val).lower():
                return "system_bug"

    # Check wrong_column: literal looks like a date but values are numeric.
    if _looks_like_date_literals(sql_literals) and _all_numeric(actual_values):
        return "wrong_column"

    return "not_found"


def find_candidate_columns(
    frame: pd.DataFrame,
    literals: dict[str, list[str]],
    exclude_cols: set[str],
) -> list[dict[str, Any]]:
    """Find columns whose values partially match any SQL literal.

    Used to suggest alternative columns when the predicate column is wrong.
    Returns at most 5 candidates with top-5 sample values.
    """
    all_literals: list[str] = []
    for col_lits in literals.values():
        all_literals.extend(col_lits)

    candidates: list[dict[str, Any]] = []
    by_text = {str(col): col for col in frame.columns}
    for col_name in frame.columns:
        col_str = str(col_name)
        if col_str in exclude_cols:
            continue
        actual_col = by_text.get(col_str)
        if actual_col is None:
            continue
        col_values = frame[actual_col].dropna().astype(str)
        if col_values.empty:
            continue
        matched_literal: str | None = None
        matched_values: list[str] = []
        for lit in all_literals:
            lit_clean = _strip_sql_wildcards(lit).lower()
            if not lit_clean or len(lit_clean) < 3:
                continue
            mask = col_values.str.contains(lit_clean, case=False, regex=False)
            if mask.any():
                matched_literal = lit
                matched_values = col_values[mask].drop_duplicates().head(5).tolist()
                break
        if matched_literal is not None:
            candidates.append({
                "col": col_str,
                "literal": matched_literal,
                "matches": matched_values,
                "sample": _top_values(frame[actual_col], limit=5),
            })
        if len(candidates) >= 5:
            break
    return candidates


def _extract_predicate_literals(expr: Any) -> dict[str, list[str]]:
    """Extract literal values grouped by the column they're compared against.

    Walks the predicate AST. For binary_op (column op literal), records the
    literal under the column name. For IN (column IN (...)), records all values.
    """
    result: dict[str, list[str]] = {}

    def visit(item: Any, current_col: str | None = None) -> None:
        if not isinstance(item, dict):
            return
        expr_type = item.get("type")
        if expr_type == "column":
            name = item.get("name")
            if isinstance(name, str):
                current_col = name
        elif expr_type == "literal":
            value = item.get("value")
            if current_col is not None and value is not None:
                result.setdefault(current_col, []).append(str(value))
            return
        elif expr_type == "binary_op":
            left = item.get("left")
            right = item.get("right")
            if isinstance(left, dict) and left.get("type") == "column":
                visit(right, current_col=left.get("name"))
            elif isinstance(right, dict) and right.get("type") == "column":
                visit(left, current_col=right.get("name"))
            else:
                visit(left, current_col=current_col)
                visit(right, current_col=current_col)
            return
        elif expr_type == "in":
            col_expr = item.get("expr")
            if isinstance(col_expr, dict) and col_expr.get("type") == "column":
                col_name = col_expr.get("name")
                for value_item in item.get("values", []):
                    if isinstance(value_item, dict) and value_item.get("type") == "literal":
                        val = value_item.get("value")
                        if col_name is not None and val is not None:
                            result.setdefault(str(col_name), []).append(str(val))
            return
        elif expr_type == "like":
            col_expr = item.get("expr")
            pattern_expr = item.get("pattern")
            if isinstance(col_expr, dict) and col_expr.get("type") == "column":
                col_name = col_expr.get("name")
                if isinstance(pattern_expr, dict) and pattern_expr.get("type") == "literal":
                    pattern = pattern_expr.get("value")
                    if col_name is not None and pattern is not None:
                        result.setdefault(str(col_name), []).append(str(pattern))
            return
        for value in item.values():
            if isinstance(value, dict):
                visit(value, current_col=current_col)
            elif isinstance(value, list):
                for child in value:
                    visit(child, current_col=current_col)

    visit(expr)
    return result


def _predicate_columns(expr: Any) -> list[str]:
    """Extract column names referenced in a predicate AST."""
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


def _top_values(series: pd.Series, *, limit: int = 8) -> list[str]:
    """Return top-N most frequent non-null values as strings."""
    cleaned = series.dropna()
    if cleaned.empty:
        return []
    value_counts = cleaned.astype(str).value_counts().head(limit)
    return [str(idx) for idx in value_counts.index]


def _strip_sql_wildcards(literal: str) -> str:
    """Strip SQL LIKE wildcards and surrounding quotes from a literal."""
    text = str(literal).strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.replace("%", "").replace("_", " ").strip()


def _strip_quotes(literal: str) -> str:
    """Strip surrounding quotes (single or double) from a literal."""
    text = str(literal).strip()
    if len(text) >= 2:
        if (text.startswith("'") and text.endswith("'")) or (
            text.startswith('"') and text.endswith('"')
        ):
            return text[1:-1]
    return text


def _looks_like_date_literals(literals: list[str]) -> bool:
    """Heuristic: do the literals look like date references?"""
    date_markers = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "/", "-",
    )
    for lit in literals:
        lit_lower = str(lit).lower()
        if any(marker in lit_lower for marker in date_markers):
            return True
    return False


def _looks_like_date_value(value: str) -> bool:
    """Heuristic: does a value look like a date in textual form?"""
    text = str(value).strip().lower()
    if not text:
        return False
    month_names = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    )
    if any(month in text for month in month_names):
        return True
    # ISO-like: YYYY-MM-DD or DD/MM/YYYY
    if re.match(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return True
    if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        return True
    return False


def _all_numeric(values: list[str]) -> bool:
    """Heuristic: are all values numeric-looking?"""
    if not values:
        return False
    for val in values:
        text = str(val).strip()
        if not text:
            continue
        try:
            float(text)
        except ValueError:
            return False
    return True
