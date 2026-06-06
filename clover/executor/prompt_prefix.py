"""Prompt prefix metadata used for cache-locality tracing."""

from __future__ import annotations

import hashlib
from typing import Any

from clover.executor.token_count import rough_token_count


CACHE_PREFIX_PATH_METADATA_KEY = "cache_prefix_path"
CACHE_PREFIX_SEGMENT_TOKEN_ESTIMATES_METADATA_KEY = (
    "cache_prefix_segment_token_estimates"
)
CACHE_PREFIX_TOKEN_ESTIMATE_METADATA_KEY = "cache_prefix_token_estimate"
SCHEDULING_KIND_METADATA_KEY = "scheduling_kind"
SCHEDULING_KIND_LOCAL_SLM_PREFIX = "local_slm_prefix"

DOCUMENT_WORKER_TEMPLATE_ID = "document_worker"
DOCUMENT_WORKER_OUTPUT_CONTRACT_ID = "json_worker_output"
DOCUMENT_WORKER_RULES_ID = "chunk_local_rules"
DOCUMENT_WORKER_EXAMPLES_ID = "chunk_local_examples"


def document_worker_prefix_metadata(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return cache-locality metadata for one document worker prompt."""

    selected = params if isinstance(params, dict) else {}
    instruction = _first_text(selected, ("local_instruction", "task"))
    if not instruction:
        return {}
    advice = _first_text(selected, ("local_guidance", "advice"))
    fixed_opening_tokens = rough_token_count(
        "You are a local document worker. Use only the document excerpt to "
        "complete the task."
    )
    task_tokens = rough_token_count(f"Task:\n{instruction}")
    advice_tokens = rough_token_count(f"Advice:\n{advice}") if advice else 0
    output_contract_tokens = rough_token_count(
        'Return one JSON object only: {"answer": null, "citation": null, '
        '"explanation": ""}'
    )
    rules_tokens = rough_token_count(
        "Use only this excerpt. Do not combine evidence across chunks. "
        "Preserve fiscal periods, units, line item names, and signs exactly "
        "when present. Do not do cross-chunk calculations. citation should be "
        "a short quote or page-local reference from this excerpt."
    )
    examples_tokens = rough_token_count(
        "Examples: Relevant excerpt -> answer with stated value, citation, and "
        "explanation. Component excerpt -> answer with present component values. "
        "Irrelevant excerpt -> null answer and citation."
    )
    path = (
        "agent:document_worker",
        f"template:{DOCUMENT_WORKER_TEMPLATE_ID}",
        f"output:{DOCUMENT_WORKER_OUTPUT_CONTRACT_ID}",
        f"rules:{DOCUMENT_WORKER_RULES_ID}",
        f"examples:{DOCUMENT_WORKER_EXAMPLES_ID}",
        f"task:{_stable_hash(instruction)}",
        f"advice:{_stable_hash(advice) if advice else 'empty'}",
    )
    segment_estimates = (
        fixed_opening_tokens,
        1,
        output_contract_tokens,
        rules_tokens,
        examples_tokens,
        task_tokens,
        advice_tokens,
    )
    return {
        SCHEDULING_KIND_METADATA_KEY: SCHEDULING_KIND_LOCAL_SLM_PREFIX,
        CACHE_PREFIX_PATH_METADATA_KEY: path,
        CACHE_PREFIX_SEGMENT_TOKEN_ESTIMATES_METADATA_KEY: segment_estimates,
        CACHE_PREFIX_TOKEN_ESTIMATE_METADATA_KEY: sum(segment_estimates),
    }


def _first_text(params: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
