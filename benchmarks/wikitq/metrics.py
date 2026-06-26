"""WikiTableQuestions denotation scoring utilities."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WikiTQScore:
    metric: str
    score: float
    correct: bool
    expected: str
    actual: str


@dataclass(frozen=True)
class StringValue:
    normalized: str

    def match(self, other: "Value") -> bool:
        left_bool = boolean_alias(self.normalized)
        right_bool = boolean_alias(other.normalized)
        if left_bool is not None and right_bool is not None:
            return left_bool == right_bool
        return self.normalized == other.normalized


@dataclass(frozen=True)
class NumberValue:
    amount: int | float
    normalized: str

    def match(self, other: "Value") -> bool:
        if self.normalized == other.normalized:
            return True
        return isinstance(other, NumberValue) and abs(self.amount - other.amount) < 1e-6


@dataclass(frozen=True)
class DateValue:
    year: int
    month: int
    day: int
    normalized: str

    def match(self, other: "Value") -> bool:
        if self.normalized == other.normalized:
            return True
        return (
            isinstance(other, DateValue)
            and (self.year, self.month, self.day)
            == (other.year, other.month, other.day)
        )


Value = StringValue | NumberValue | DateValue


def score_wikitq_answer(
    *,
    expected: Any,
    actual: Any,
    expected_canon: Any | None = None,
) -> WikiTQScore:
    """Score a WikiTQ prediction with official-style denotation exact match."""

    expected_items = answer_items(expected)
    canon_items = answer_items(expected_canon) if expected_canon is not None else None
    target_values = to_value_list(expected_items, canon_items)
    predicted_candidates = prediction_value_candidates(
        actual,
        expected_count=len(target_values),
    )
    correct = any(check_denotation(target_values, candidate) for candidate in predicted_candidates)
    actual_items = answer_items(actual)
    return WikiTQScore(
        metric="denotation_em",
        score=1.0 if correct else 0.0,
        correct=correct,
        expected=answer_text(expected_items),
        actual=answer_text(actual_items),
    )


def normalize_text(value: Any) -> str:
    text = str(value if value is not None else "")
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub("[\u2018\u2019\u00b4`]", "'", text)
    text = re.sub("[\u201c\u201d]", '"', text)
    text = re.sub("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    while True:
        old_text = text
        text = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[\u2022\u2666\u2020\u2021*#+])*$", "", text.strip())
        text = re.sub(r"(?<!^)( \([^)]*\))*$", "", text.strip())
        text = re.sub(r'^"([^"]*)"$', r"\1", text.strip())
        if text == old_text:
            break
    if text.endswith("."):
        text = text[:-1]
    return re.sub(r"\s+", " ", text).lower().strip()


def to_value(original_string: Any, canon_string: Any | None = None) -> Value:
    if isinstance(original_string, (StringValue, NumberValue, DateValue)):
        return original_string
    original_text = "" if original_string is None else str(original_string)
    canon_text = original_text if canon_string is None else str(canon_string)
    number = parse_number(canon_text)
    if number is not None:
        return NumberValue(_integer_if_close(number), normalize_text(original_text))
    date = parse_date(canon_text)
    if date is not None:
        year, month, day = date
        if month == -1 and day == -1:
            return NumberValue(year, normalize_text(original_text))
        return DateValue(year, month, day, normalize_text(original_text))
    return StringValue(normalize_text(original_text))


def to_value_list(
    original_strings: list[Any],
    canon_strings: list[Any] | None = None,
) -> list[Value]:
    if canon_strings is not None and len(original_strings) == len(canon_strings):
        values = [to_value(original, canon) for original, canon in zip(original_strings, canon_strings)]
    else:
        values = [to_value(original) for original in original_strings]
    return _dedupe_values(values)


def check_denotation(target_values: list[Value], predicted_values: list[Value]) -> bool:
    if len(target_values) != len(predicted_values):
        return False
    return all(any(target.match(predicted) for predicted in predicted_values) for target in target_values)


def parse_number(text: Any) -> int | float | None:
    if isinstance(text, bool) or text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
        except ValueError:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number


def parse_date(text: Any) -> tuple[int, int, int] | None:
    value = str(text or "").strip().lower()
    parts = value.split("-")
    if len(parts) != 3:
        return None
    try:
        year = -1 if parts[0] in {"xx", "xxxx"} else int(parts[0])
        month = -1 if parts[1] == "xx" else int(parts[1])
        day = -1 if parts[2] == "xx" else int(parts[2])
    except ValueError:
        return None
    if year == month == day == -1:
        return None
    if month != -1 and not 1 <= month <= 12:
        return None
    if day != -1 and not 1 <= day <= 31:
        return None
    return year, month, day


def answer_items(value: Any) -> list[str]:
    if isinstance(value, dict) and "answer" in value:
        value = value["answer"]
    elif isinstance(value, dict):
        items: list[str] = []
        for item in value.values():
            items.extend(answer_items(item))
        return [item for item in items if item != ""]
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(_integer_if_close(value))]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(answer_items(item))
        return [item for item in items if item != ""]
    text = str(value).strip()
    if not text:
        return []
    parsed = _json_value_or_none(text)
    if isinstance(parsed, (list, tuple)):
        return answer_items(parsed)
    return [text]


def boolean_alias(value: Any) -> bool | None:
    text = normalize_text(value)
    if text in {"yes", "true", "y"}:
        return True
    if text in {"no", "false", "n"}:
        return False
    return None


def prediction_value_candidates(value: Any, *, expected_count: int) -> list[list[Value]]:
    base_items = answer_items(value)
    item_candidates = [base_items]
    if len(base_items) == 1 and expected_count > 1:
        text = base_items[0]
        item_candidates.extend(_split_prediction_text(text))
    candidates = [to_value_list(items) for items in item_candidates if items]
    if not candidates:
        candidates = [[]]
    unique: list[list[Value]] = []
    seen = set()
    for candidate in candidates:
        key = tuple(sorted(repr(value_item) for value_item in candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def tsv_unescape(text: str) -> str:
    return text.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


def tsv_unescape_list(text: str) -> list[str]:
    return [tsv_unescape(item) for item in str(text or "").split("|")]


def answer_text(value: Any) -> str:
    return ", ".join(answer_items(value))


def _split_prediction_text(text: str) -> list[list[str]]:
    candidates: list[list[str]] = []
    if "|" in text:
        candidates.append([part.strip() for part in text.split("|") if part.strip()])
    if ";" in text:
        candidates.append([part.strip() for part in text.split(";") if part.strip()])
    if ", " in text:
        candidates.append([part.strip() for part in text.split(",") if part.strip()])
    if " and " in text.lower():
        candidates.append(
            [
                part.strip()
                for part in re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
                if part.strip()
            ]
        )
    return candidates


def _json_value_or_none(text: str) -> Any | None:
    if not text.startswith(("[", "{")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _dedupe_values(values: list[Value]) -> list[Value]:
    unique = []
    seen = set()
    for value in values:
        key = repr(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _integer_if_close(value: int | float) -> int | float:
    if isinstance(value, float) and abs(value - round(value)) < 1e-6:
        return int(round(value))
    return value
