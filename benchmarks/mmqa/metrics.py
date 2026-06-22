"""MMQA multi-table answer scoring utilities.

MMQA answers come in two shapes:

* ``str`` -- a plain string answer. Multi-value answers are usually
  comma-separated (e.g. ``"Treasury, 115897"``), but converted SQL-style
  outputs may use semicolons when individual values themselves contain commas.
* ``dict`` -- a DataFrame-like object with ``columns`` / ``index`` / ``data``
  keys, produced when the gold SQL returns a small result table.

The helpers here flatten both shapes into a canonical ``list[str]`` for
denotation exact-match scoring, reusing WikiTQ's value-matching logic so that
``3`` vs ``3.0`` and date variants are treated as equal.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import product
from typing import Any

from benchmarks.wikitq.metrics import (
    NumberValue,
    StringValue,
    Value,
    check_denotation,
    normalize_text,
    parse_number,
    to_value,
)


@dataclass(frozen=True)
class MMQAScore:
    metric: str
    score: float
    correct: bool
    expected: str
    actual: str


def flatten_mmqa_answer(value: Any) -> list[str]:
    """Flatten an MMQA gold answer into a canonical list of string items.

    * ``dict`` answers (DataFrame-like ``{columns, index, data}``) are flattened
      by reading every cell in ``data`` row by row.
    * ``str`` answers are split on semicolons when present. Commas are kept
      because many table values are names such as ``"Last, First"``.
    * ``list`` / ``tuple`` answers are flattened recursively, treating string
      list items as atomic values.
    * Numbers and booleans are stringified.
    """

    if value is None:
        return []
    if isinstance(value, dict):
        return _flatten_dict_answer(value)
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)):
        return [_number_to_str(value)]
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    items.append(stripped)
                continue
            items.extend(flatten_mmqa_answer(item))
        return [item for item in items if item != ""]
    text = str(value).strip()
    if not text:
        return []
    parsed = _json_value_or_none(text)
    if isinstance(parsed, (list, tuple)):
        return flatten_mmqa_answer(parsed)
    if isinstance(parsed, dict):
        return flatten_mmqa_answer(parsed)
    return _split_text_answer(text)


def _flatten_dict_answer(value: dict[str, Any]) -> list[str]:
    # DataFrame-like: {columns, index, data}
    if "data" in value and isinstance(value["data"], list):
        items: list[str] = []
        for row in value["data"]:
            if isinstance(row, (list, tuple)):
                for cell in row:
                    items.extend(flatten_mmqa_answer(cell))
            else:
                items.extend(flatten_mmqa_answer(row))
        return [item for item in items if item != ""]
    # Fallback: {answer: ...} shape used by some CLOVER result records.
    if "answer" in value:
        return flatten_mmqa_answer(value["answer"])
    # Generic dict: flatten values in stable key order.
    items = []
    for key in sorted(value.keys()):
        items.extend(flatten_mmqa_answer(value[key]))
    return items


def _split_text_answer(text: str) -> list[str]:
    # Prefer semicolons when present because converted SQL-style answers use
    # them to separate rows whose values may themselves contain commas, e.g.
    # "Schmidt, Kertzmann and Lubowitz; Schmitt-Lang".
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
        return [part for part in parts if part]
    return [text]


def _number_to_str(value: int | float) -> str:
    if isinstance(value, float) and abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return str(value)


def score_mmqa_answer(
    *,
    expected: Any,
    actual: Any,
    expected_answer_type: str | None = None,
) -> MMQAScore:
    """Score an MMQA prediction with denotation exact match.

    The expected answer is flattened canonically; the prediction is allowed to
    match either as a single string or as a split list of the same length.
    """

    expected_items = flatten_mmqa_answer(expected)
    target_candidates = _value_candidates_from_items(
        expected_items,
        expected_answer_type=expected_answer_type,
    )
    expected_count = len(target_candidates[0]) if target_candidates else 0
    predicted_candidates = _prediction_value_candidates(
        actual,
        expected_count=expected_count,
    )
    correct = any(
        check_denotation(target_values, candidate)
        for target_values in target_candidates
        for candidate in predicted_candidates
    )
    actual_items = flatten_mmqa_answer(actual)
    return MMQAScore(
        metric="denotation_em",
        score=1.0 if correct else 0.0,
        correct=correct,
        expected=", ".join(expected_items),
        actual=", ".join(actual_items),
    )


def _prediction_value_candidates(
    value: Any,
    *,
    expected_count: int,
) -> list[list[Value]]:
    base_items = flatten_mmqa_answer(value)
    item_candidates: list[list[str]] = _item_set_variants(base_items)
    if len(base_items) == 1 and expected_count > 1:
        text = base_items[0]
        for split_items in _split_prediction_text(text):
            item_candidates.extend(_item_set_variants(split_items))
    if len(base_items) > 1 and expected_count > len(base_items):
        for split_items in _split_prediction_rows(base_items, expected_count):
            item_candidates.extend(_item_set_variants(split_items))
    candidates = [_dedupe_values([to_value(item) for item in items]) for items in item_candidates if items]
    if not candidates:
        candidates = [[]]
    unique: list[list[Value]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(sorted(repr(item) for item in candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _value_candidates_from_items(
    items: list[str],
    *,
    expected_answer_type: str | None = None,
) -> list[list[Value]]:
    item_variants = _item_set_variants(items)
    if len(items) == 1:
        for split_items in _split_expected_text(
            items[0],
            expected_answer_type=expected_answer_type,
        ):
            item_variants.extend(_item_set_variants(split_items))
    candidates = [
        _dedupe_values([to_value(item) for item in variant])
        for variant in item_variants
    ]
    return candidates or [[]]


def _item_set_variants(items: list[str]) -> list[list[str]]:
    if not items:
        return [[]]
    per_item = [_item_variants(item) for item in items]
    variants: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for combo in product(*per_item):
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            variants.append(list(combo))
        if len(variants) >= 64:
            break
    return variants


def _item_variants(item: str) -> list[str]:
    variants = [item]
    name_variant = _comma_name_variant(item)
    if name_variant and name_variant not in variants:
        variants.append(name_variant)
    mojibake_variant = _mojibake_variant(item)
    if mojibake_variant and mojibake_variant not in variants:
        variants.append(mojibake_variant)
    return variants


def _mojibake_variant(text: str) -> str | None:
    """Repair common UTF-8 decoded as Latin-1 artifacts, e.g. SÃ£o -> São."""

    if not any(marker in text for marker in ("Ã", "Â", "â")):
        return None
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return repaired if repaired != text else None


def _comma_name_variant(text: str) -> str | None:
    match = re.fullmatch(r"\s*([^,;|]+),\s*([^,;|]+)\s*", text)
    if not match:
        return None
    last, first = match.group(1).strip(), match.group(2).strip()
    if not (_looks_like_name_part(last) and _looks_like_name_part(first)):
        return None
    return f"{first} {last}"


def _looks_like_name_part(text: str) -> bool:
    if any(char.isdigit() for char in text):
        return False
    tokens = [token for token in re.split(r"[\s.-]+", text) if token]
    return bool(tokens) and all(any(char.isalpha() for char in token) for token in tokens)


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


def _split_prediction_rows(items: list[str], expected_count: int) -> list[list[str]]:
    candidates: list[list[str]] = []
    row_splits: list[list[str]] = []
    for item in items:
        split = _split_row_text(item)
        if len(split) <= 1:
            return candidates
        row_splits.append(split)
    flattened = [cell for row in row_splits for cell in row]
    if len(flattened) == expected_count:
        candidates.append(flattened)
    return candidates


def _split_row_text(text: str) -> list[str]:
    for separator in ("|", "\t"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    if ", " in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text.strip()] if text.strip() else []


def _split_expected_text(
    text: str,
    *,
    expected_answer_type: str | None,
) -> list[list[str]]:
    if not str(expected_answer_type or "").startswith("list"):
        return []
    if ";" in text:
        return []
    parts = [
        part.strip()
        for part in re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(parts) <= 1:
        return []
    if all(_looks_like_date_text(part) for part in parts):
        return [parts]
    return []


def _looks_like_date_text(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    month_pattern = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?"
    )
    if re.search(month_pattern, value, flags=re.IGNORECASE) and re.search(r"\d", value):
        return True
    if re.fullmatch(r"\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?", value):
        return True
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?", value):
        return True
    return False


def _dedupe_values(values: list[Value]) -> list[Value]:
    unique: list[Value] = []
    seen: set[str] = set()
    for value in values:
        key = repr(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _json_value_or_none(text: str) -> Any | None:
    if not text.startswith(("[", "{")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


__all__ = [
    "MMQAScore",
    "flatten_mmqa_answer",
    "score_mmqa_answer",
]
