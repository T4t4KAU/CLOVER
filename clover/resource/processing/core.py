"""Prepare physical-plan resources before Executor submission."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clover.resource.cache import CacheEntry, ResourceCache
from clover.resource.preprocess.pdf_schema import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_CHUNK_SIZE_CHARS,
    load_cached_chunks,
    materialize_page_chunks,
    materialize_pdf_text,
    materialize_text_chunks,
)


class ResourceProcessingError(ValueError):
    """Raised when physical-plan resource processing cannot be completed."""


@dataclass
class PhysicalPlanResourceBuilder:
    """Materialize physical-plan resource_processing steps."""

    cache: ResourceCache | None = None

    def build(self, physical_plan: dict[str, Any]) -> dict[str, Any]:
        return _prepare_physical_plan_resources(physical_plan, cache=self.cache)


def prepare_physical_plan_resources(
    physical_plan: dict[str, Any],
    *,
    cache: ResourceCache | None = None,
) -> dict[str, Any]:
    """Materialize declared resource views and return an Executor-ready plan."""

    return PhysicalPlanResourceBuilder(cache=cache).build(physical_plan)


def _prepare_physical_plan_resources(
    physical_plan: dict[str, Any],
    *,
    cache: ResourceCache | None = None,
) -> dict[str, Any]:
    """Materialize declared resource views and return an Executor-ready plan."""

    plan = copy.deepcopy(physical_plan)
    steps = plan.pop("resource_processing", [])
    if not steps:
        return plan

    resource_cache = cache or ResourceCache()
    resources_by_id = _resource_map(plan.get("resources", []))
    views: dict[str, dict[str, Any]] = {}
    processed_source_ids: set[str] = set()

    for step in steps:
        op = step.get("op")
        if op == "extract_text":
            view = _process_extract_text(
                step=step,
                resources_by_id=resources_by_id,
                cache=resource_cache,
            )
        elif op == "chunk_text":
            view = _process_chunk_text(
                step=step,
                views=views,
                cache=resource_cache,
            )
        else:
            raise ResourceProcessingError(f"Unsupported resource processing op: {op}")
        views[step["output"]] = view
        if view.get("root_source"):
            processed_source_ids.add(str(view["root_source"]))

    chunk_resources: dict[str, dict[str, Any]] = {}
    plan_question = str(plan.get("question") or "")
    for group in plan.get("map_groups", []):
        group_input = group.get("input", {})
        resource_view = group_input.get("resource_view")
        if resource_view is None:
            continue
        view = views.get(resource_view)
        if view is None or view.get("kind") != "document_chunks":
            raise ResourceProcessingError(f"Unknown document chunk view: {resource_view}")
        selected_chunks = _select_chunks(
            view["chunks"],
            group_input.get("chunks", "all"),
            rank_hint=_group_rank_hint(group, question=plan_question),
            max_selected_chunks=_document_max_selected_chunks(),
        )
        selected_resource_ids = []
        for chunk in selected_chunks:
            resource = _chunk_resource_spec(view, chunk)
            existing = chunk_resources.get(resource["id"])
            if existing is not None and existing != resource:
                raise ResourceProcessingError(
                    f"Conflicting chunk resource binding: {resource['id']}"
                )
            chunk_resources[resource["id"]] = resource
            selected_resource_ids.append(resource["id"])
        group["input"] = {"chunks": selected_resource_ids}

    retained_resources = [
        resource
        for resource_id, resource in resources_by_id.items()
        if resource_id not in processed_source_ids
    ]
    plan["resources"] = sorted(
        retained_resources + list(chunk_resources.values()),
        key=lambda resource: resource["id"],
    )
    return plan


def _process_extract_text(
    *,
    step: dict[str, Any],
    resources_by_id: dict[str, dict[str, Any]],
    cache: ResourceCache,
) -> dict[str, Any]:
    source_id = step.get("source")
    source = resources_by_id.get(source_id)
    if source is None:
        raise ResourceProcessingError(f"Unknown extract_text source: {source_id}")
    if source.get("type") != "document" or source.get("source_type") != "pdf":
        raise ResourceProcessingError(
            f"extract_text requires a PDF document source: {source_id}"
        )
    params = step.get("params", {})
    entry = materialize_pdf_text(
        source["path"],
        cache=cache,
        extractor=params.get("extractor", "pymupdf"),
    )
    return {
        "kind": "document_text",
        "root_source": source_id,
        "source": copy.deepcopy(source),
        "entry": entry,
        "metadata": entry.metadata,
    }


def _process_chunk_text(
    *,
    step: dict[str, Any],
    views: dict[str, dict[str, Any]],
    cache: ResourceCache,
) -> dict[str, Any]:
    source_view = views.get(step.get("source"))
    if source_view is None or source_view.get("kind") != "document_text":
        raise ResourceProcessingError(f"Unknown chunk_text source: {step.get('source')}")
    params = step.get("params", {})
    text_entry = _require_cache_entry(source_view.get("entry"))
    strategy = params.get("strategy", "sliding_window")
    unit = params.get("unit", "char")
    if _document_force_page_chunks():
        chunks_entry = materialize_page_chunks(text_entry, cache=cache)
    elif strategy == "sliding_window" and unit == "char":
        chunks_entry = materialize_text_chunks(
            text_entry,
            cache=cache,
            chunk_size_chars=int(params.get("size", DEFAULT_CHUNK_SIZE_CHARS)),
            chunk_overlap_chars=int(params.get("overlap", DEFAULT_CHUNK_OVERLAP_CHARS)),
        )
    elif strategy == "page" and unit == "page":
        chunks_entry = materialize_page_chunks(text_entry, cache=cache)
    else:
        raise ResourceProcessingError(
            f"Unsupported chunk_text strategy/unit: {strategy}/{unit}"
        )
    return {
        "kind": "document_chunks",
        "root_source": source_view["root_source"],
        "source": copy.deepcopy(source_view["source"]),
        "entry": chunks_entry,
        "chunks": load_cached_chunks(chunks_entry),
        "metadata": chunks_entry.metadata,
    }


def _require_cache_entry(value: Any) -> CacheEntry:
    if not isinstance(value, CacheEntry):
        raise ResourceProcessingError("Expected cached document text entry")
    return value


def _select_chunks(
    chunks: list[dict[str, Any]],
    selector: Any,
    *,
    rank_hint: str = "",
    max_selected_chunks: int | None = None,
) -> list[dict[str, Any]]:
    if selector == "all":
        selected = list(chunks)
        if max_selected_chunks is not None and len(selected) > max_selected_chunks:
            selected = _ranked_chunk_subset(
                selected,
                rank_hint=rank_hint,
                max_chunks=max_selected_chunks,
            )
        return selected
    if not isinstance(selector, list):
        raise ResourceProcessingError("chunk selector must be all or a list")
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    selected = []
    for chunk_id in selector:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ResourceProcessingError(f"Unknown chunk selector: {chunk_id}")
        selected.append(chunk)
    return selected


def _document_max_selected_chunks() -> int | None:
    raw = os.getenv("CLOVER_DOCUMENT_MAX_SELECTED_CHUNKS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ResourceProcessingError(
            "CLOVER_DOCUMENT_MAX_SELECTED_CHUNKS must be an integer"
        ) from exc
    if value <= 0:
        return None
    return value


def _document_force_page_chunks() -> bool:
    raw = os.getenv("CLOVER_DOCUMENT_FORCE_PAGE_CHUNKS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _group_rank_hint(group: dict[str, Any], *, question: str = "") -> str:
    params = group.get("params", {})
    parts = [question.strip()] if question.strip() else []
    if not isinstance(params, dict):
        return "\n".join(parts)
    for key in ("local_instruction", "local_guidance", "advice"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _ranked_chunk_subset(
    chunks: list[dict[str, Any]],
    *,
    rank_hint: str,
    max_chunks: int,
) -> list[dict[str, Any]]:
    if max_chunks >= len(chunks):
        return list(chunks)
    tokens = _query_tokens(rank_hint)
    if not tokens:
        return _even_chunk_subset(chunks, max_chunks=max_chunks)
    scored = [
        (_chunk_relevance_score(chunk, tokens=tokens), index, chunk)
        for index, chunk in enumerate(chunks)
    ]
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_chunks]
    if top and top[0][0] <= 0:
        return _even_chunk_subset(chunks, max_chunks=max_chunks)
    selected_indexes = {index for _, index, _ in top}
    return [
        chunk
        for index, chunk in enumerate(chunks)
        if index in selected_indexes
    ]


def _even_chunk_subset(chunks: list[dict[str, Any]], *, max_chunks: int) -> list[dict[str, Any]]:
    if max_chunks >= len(chunks):
        return list(chunks)
    if max_chunks == 1:
        return [chunks[0]]
    indexes = {
        round(index * (len(chunks) - 1) / (max_chunks - 1))
        for index in range(max_chunks)
    }
    return [chunk for index, chunk in enumerate(chunks) if index in indexes]


def _query_tokens(text: str) -> dict[str, int]:
    tokens: dict[str, int] = {}
    for token in _tokenize(text):
        if token in _STOPWORDS:
            continue
        tokens[token] = tokens.get(token, 0) + 1
    lower = text.lower()
    if "income statement" in lower or "statement of income" in lower:
        _boost_tokens(tokens, "operations revenue revenues income expense expenses")
    if "balance sheet" in lower or "financial position" in lower:
        _boost_tokens(
            tokens,
            "assets liabilities equity property equipment inventory receivable payable",
        )
    if "cash flow" in lower:
        _boost_tokens(
            tokens,
            "cash flows operating investing financing capital expenditures capex",
        )
    return tokens


def _boost_tokens(tokens: dict[str, int], text: str) -> None:
    for token in _tokenize(text):
        tokens[token] = tokens.get(token, 0) + 2


def _chunk_relevance_score(chunk: dict[str, Any], *, tokens: dict[str, int]) -> int:
    text = str(chunk.get("text", "")).lower()
    if not text:
        return 0
    chunk_tokens = set(_tokenize(text))
    score = sum(weight for token, weight in tokens.items() if token in chunk_tokens)
    for phrase in _FINANCIAL_PHRASES:
        if phrase in text:
            score += 3
    return score


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9&%-]{2,}", text.lower())
        if len(token) >= 3
    ]


_STOPWORDS = {
    "about",
    "above",
    "after",
    "answer",
    "based",
    "between",
    "calculate",
    "chunk",
    "company",
    "defined",
    "document",
    "extract",
    "facts",
    "financial",
    "following",
    "from",
    "information",
    "line",
    "local",
    "metric",
    "numerical",
    "only",
    "provided",
    "question",
    "ratio",
    "relevant",
    "return",
    "round",
    "statement",
    "statements",
    "task",
    "this",
    "using",
    "value",
    "what",
    "when",
    "where",
    "with",
    "within",
    "year",
}


_FINANCIAL_PHRASES = (
    "consolidated balance sheets",
    "consolidated statement of income",
    "consolidated statements of income",
    "consolidated statements of operations",
    "consolidated statements of cash flows",
    "statement of financial position",
    "total net revenues",
    "property and equipment",
    "cash provided by operating activities",
    "capital expenditures",
)


def _chunk_resource_spec(
    view: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, Any]:
    entry = _require_cache_entry(view.get("entry"))
    source = view["source"]
    source_id = str(view["root_source"])
    chunk_id = str(chunk["chunk_id"])
    return {
        "id": f"{source_id}:{chunk_id}",
        "type": "document_chunk",
        "source": source_id,
        "source_type": source.get("source_type", "pdf"),
        "path": str(Path(entry.artifact_path("chunks")).resolve()),
        "format": "text",
        "content_ref": entry.item_ref("chunks", chunk_id),
        "artifact": "chunks",
        "item_id": chunk_id,
        "chunk_id": chunk_id,
        "page_indexing": chunk.get("page_indexing"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "page_count": chunk.get("page_count"),
        "char_start": chunk.get("char_start"),
        "char_end": chunk.get("char_end"),
        "char_count": chunk.get("char_count"),
        "schema": {
            "content_format": "text",
            "strategy": chunk.get("strategy", "sliding_window"),
            "unit": chunk.get("unit", "char"),
        },
    }


def _resource_map(resources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for resource in resources:
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            raise ResourceProcessingError(f"Resource missing id: {resource}")
        mapped[resource_id] = resource
    return mapped
