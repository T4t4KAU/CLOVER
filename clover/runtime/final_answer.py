"""Final answer normalization at runtime boundaries."""

from __future__ import annotations

import re
from typing import Any

from clover.task_types import DOCUMENT_REASONING_TASK_TYPE


_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?")


def finalize_answer(
    *,
    task_type: str,
    question: str | None,
    answer: Any,
    explanation: str | None = None,
    observation: dict[str, Any] | None = None,
) -> Any:
    """Return the user-facing answer for a completed task."""

    if task_type != DOCUMENT_REASONING_TASK_TYPE:
        return answer
    return finalize_document_numerical_answer(
        question=question,
        answer=answer,
        explanation=explanation,
        observation=observation,
    )


def finalize_document_numerical_answer(
    *,
    question: str | None,
    answer: Any,
    explanation: str | None = None,
    observation: dict[str, Any] | None = None,
) -> Any:
    """Conservatively make document numerical answers self-contained.

    This is not a reasoning step. It only appends numbers already present in the
    Supervisor explanation or compact worker evidence when the answer itself is
    too terse for downstream evaluation.
    """

    if answer is None or not isinstance(answer, str):
        return answer
    answer_text = answer.strip()
    if not answer_text or _contains_number(answer_text):
        return answer
    if not _looks_numerical_question(question):
        return answer

    detail = _best_numeric_detail(
        explanation=explanation,
        observation=observation,
    )
    if not detail:
        return answer
    if _same_text(answer_text, detail):
        return answer
    return _join_answer_detail(answer_text, detail)


def _looks_numerical_question(question: str | None) -> bool:
    if not isinstance(question, str):
        return True
    text = question.lower()
    cues = (
        "amount",
        "average",
        "between",
        "cash flow",
        "changed",
        "debt",
        "decreased",
        "divided",
        "ebitda",
        "financial metric",
        "increased",
        "least",
        "margin",
        "most",
        "ratio",
        "round",
        "usd",
        "%",
    )
    return any(cue in text for cue in cues) or _contains_number(text)


def _best_numeric_detail(
    *,
    explanation: str | None,
    observation: dict[str, Any] | None,
) -> str | None:
    selected: list[str] = []
    seen_numbers: set[str] = set()
    for text in _candidate_texts(explanation=explanation, observation=observation):
        candidate = _first_numeric_sentence(text)
        if not candidate:
            continue
        numbers = _number_tokens(candidate)
        if not selected:
            selected.append(candidate)
            seen_numbers.update(numbers)
            continue
        if numbers and not numbers <= seen_numbers:
            selected.append(candidate)
            seen_numbers.update(numbers)
            break
    if not selected:
        return None
    return " ".join(selected)


def _candidate_texts(
    *,
    explanation: str | None,
    observation: dict[str, Any] | None,
) -> list[str]:
    texts: list[str] = []
    if isinstance(explanation, str) and explanation.strip():
        texts.append(explanation.strip())
    if isinstance(observation, dict):
        for key in ("evidence_summary", "prior_evidence_summary"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    return texts


def _first_numeric_sentence(text: str) -> str | None:
    for sentence in _split_sentences(text):
        cleaned = _clean_detail(sentence)
        if cleaned and _contains_number(cleaned):
            return cleaned
    cleaned = _clean_detail(text)
    return cleaned if cleaned and _contains_number(cleaned) else None


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part for part in parts if part.strip()]


def _contains_number(text: str) -> bool:
    return _NUMBER_RE.search(text) is not None


def _number_tokens(text: str) -> set[str]:
    return {_normalize_number_token(match.group(0)) for match in _NUMBER_RE.finditer(text)}


def _normalize_number_token(token: str) -> str:
    return token.strip("$()%").replace(",", "")


def _clean_detail(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    cleaned = cleaned.strip(" -")
    if len(cleaned) > 360:
        cleaned = cleaned[:357].rstrip() + "..."
    return cleaned


def _same_text(left: str, right: str) -> bool:
    return left.strip().lower().rstrip(".") == right.strip().lower().rstrip(".")


def _join_answer_detail(answer: str, detail: str) -> str:
    separator = "" if answer.endswith((".", "!", "?")) else "."
    return f"{answer}{separator} {detail}"
