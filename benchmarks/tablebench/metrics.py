"""Standard non-visual TableBench scoring utilities."""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from typing import Any


NUMERIC_TOLERANCE_SUBTYPES = frozenset(
    {
        "CorrelationAnalysis",
        "TrendForecasting",
        "StatisticalAnalysis",
    }
)


@dataclass(frozen=True)
class TableBenchScore:
    metric: str
    score: float
    correct: bool
    expected: str
    actual: str


def tablebench_metric_name(qtype: str | None, qsubtype: str | None) -> str:
    """Return the official-style metric used by a non-visual TableBench case."""

    qtype_text = str(qtype or "").strip()
    qsubtype_text = str(qsubtype or "").strip()
    if qtype_text == "Visualization":
        return "Pass@1"
    if qtype_text in {"FactChecking", "NumericalReasoning"}:
        return "EM"
    if qtype_text == "DataAnalysis":
        if qsubtype_text in NUMERIC_TOLERANCE_SUBTYPES:
            return "EM_with_error_10"
        if qsubtype_text == "ImpactAnalysis":
            return "EM"
        return "ROUGE-L"
    return "EM"


def score_tablebench_answer(
    *,
    expected: Any,
    actual: Any,
    qtype: str | None,
    qsubtype: str | None,
) -> TableBenchScore:
    metric = tablebench_metric_name(qtype, qsubtype)
    expected_text = answer_text(expected)
    actual_text = answer_text(actual)
    if metric == "EM_with_error_10":
        score = 1.0 if exact_match_with_error_10(expected_text, actual_text) else 0.0
    elif metric == "ROUGE-L":
        score = rouge_l_f1(expected_text, actual_text)
    elif metric == "Pass@1":
        score = 1.0 if normalized_text(expected_text) == normalized_text(actual_text) else 0.0
    else:
        score = 1.0 if exact_match(expected_text, actual_text, qsubtype=qsubtype) else 0.0
    return TableBenchScore(
        metric=metric,
        score=score,
        correct=score >= 1.0 - 1e-12,
        expected=expected_text,
        actual=actual_text,
    )


def answer_text(value: Any) -> str:
    if isinstance(value, dict) and "answer" in value:
        value = value["answer"]
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(answer_text(item) for item in value)
    return str(value).strip()


def exact_match(expected: str, actual: str, *, qsubtype: str | None = None) -> bool:
    if boolean_value(expected) is not None and boolean_value(actual) is not None:
        return boolean_value(expected) == boolean_value(actual)
    if numeric_match(expected, actual):
        return True
    if list_match(expected, actual, order_sensitive=_is_order_sensitive(qsubtype)):
        return True
    return normalized_text(expected) == normalized_text(actual)


def exact_match_with_error_10(expected: str, actual: str) -> bool:
    expected_mentions = numeric_mentions(expected)
    actual_mentions = numeric_mentions(actual)
    if not expected_mentions or not actual_mentions:
        return exact_match(expected, actual)
    for expected_value in expected_mentions[0].candidates:
        for actual_value in actual_mentions[0].candidates:
            if math.isclose(expected_value, 0.0, abs_tol=1e-12):
                if abs(actual_value - expected_value) <= 0.1:
                    return True
                continue
            if abs(actual_value - expected_value) / abs(expected_value) <= 0.10:
                return True
    return False


def numeric_match(expected: str, actual: str) -> bool:
    expected_mentions = numeric_mentions(expected)
    actual_mentions = numeric_mentions(actual)
    if not expected_mentions or not actual_mentions:
        return False
    expected_number = expected_mentions[0]
    actual_number = actual_mentions[0]
    for left in expected_number.candidates:
        for right in actual_number.candidates:
            if math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
                return True
            places = numeric_decimal_places(expected_number.text)
            if places > 0:
                tolerance = 0.5 * (10 ** -places) + 1e-12
                if abs(right - left) <= tolerance:
                    return True
    return False


@dataclass(frozen=True)
class NumericMention:
    value: float
    text: str
    percent: bool
    candidates: tuple[float, ...]


def numeric_mentions(value: Any) -> list[NumericMention]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return []
        return [NumericMention(number, str(value), False, (number,))]
    text = str(value).strip()
    if not text:
        return []
    mentions: list[NumericMention] = []
    for match in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", text):
        token = match.group(0).strip()
        percent = token.endswith("%")
        raw_text = token.rstrip("%").strip()
        try:
            number = float(raw_text.replace(",", ""))
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        candidates = {number}
        if percent:
            candidates.add(number / 100.0)
        mentions.append(
            NumericMention(
                value=number,
                text=token,
                percent=percent,
                candidates=tuple(sorted(candidates)),
            )
        )
    return mentions


def parse_number(value: Any) -> float | None:
    mentions = numeric_mentions(value)
    if not mentions:
        return None
    return mentions[0].value


def numeric_decimal_places(value: Any) -> int:
    """Return the decimal precision expressed by the first numeric literal."""

    if isinstance(value, bool) or value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    match = re.search(r"[-+]?\d[\d,]*(?:\.(\d+))?", text)
    if match is None:
        return 0
    decimals = match.group(1)
    return len(decimals) if decimals is not None else 0


def normalized_text(value: Any) -> str:
    text = answer_text(value).lower().strip()
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    table = str.maketrans({char: " " for char in string.punctuation})
    text = text.translate(table)
    return " ".join(text.split())


def boolean_value(value: Any) -> bool | None:
    text = normalized_text(value)
    truthy = {
        "true",
        "yes",
        "y",
        "support",
        "supports",
        "supported",
        "correct",
    }
    falsy = {
        "false",
        "no",
        "n",
        "refute",
        "refutes",
        "refuted",
        "incorrect",
    }
    if text in truthy:
        return True
    if text in falsy:
        return False
    return None


def list_match(expected: str, actual: str, *, order_sensitive: bool) -> bool:
    expected_items = normalized_list_items(expected)
    actual_items = normalized_list_items(actual)
    if len(expected_items) <= 1 or len(actual_items) <= 1:
        return False
    if order_sensitive:
        return expected_items == actual_items
    return sorted(expected_items) == sorted(actual_items)


def normalized_list_items(value: Any) -> list[str]:
    text = answer_text(value).strip()
    if not text:
        return []
    if _is_single_thousands_number(text):
        return []
    stripped = text.strip("[](){}")
    if "," not in stripped:
        return []
    return [
        normalized_text(part.strip().strip("\"'"))
        for part in stripped.split(",")
        if normalized_text(part.strip().strip("\"'"))
    ]


def _is_single_thousands_number(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", text.strip()))


def _is_order_sensitive(qsubtype: str | None) -> bool:
    return "ranking" in str(qsubtype or "").lower()


def rouge_l_f1(expected: str, actual: str) -> float:
    reference_tokens = normalized_text(expected).split()
    prediction_tokens = normalized_text(actual).split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_len(reference_tokens, prediction_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def _lcs_len(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]
