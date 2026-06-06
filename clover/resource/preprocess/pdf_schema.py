"""Static PDF text extraction and document chunk indexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from clover.resource.cache import (
    CacheBuildResult,
    CacheEntry,
    CacheSpec,
    ResourceCache,
)


DEFAULT_CHUNK_SIZE_CHARS = 3000
DEFAULT_CHUNK_OVERLAP_CHARS = 20
PAGE_INDEXING = "zero_based"
PDF_TEXT_CACHE_VERSION = 1
TEXT_CHUNK_CACHE_VERSION = 1
PAGE_CHUNK_CACHE_VERSION = 1


def extract_pdf_schema(
    path: str | Path,
    *,
    cache: ResourceCache | None = None,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> dict[str, Any]:
    """Extract PDF metadata and index cached text chunks."""

    pdf_path = Path(path)
    _validate_chunking(chunk_size_chars, chunk_overlap_chars)
    resource_cache = cache or ResourceCache()

    text_entry = materialize_pdf_text(
        pdf_path,
        cache=resource_cache,
    )
    chunks_entry = materialize_text_chunks(
        text_entry,
        cache=resource_cache,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    text_metadata = text_entry.metadata
    chunk_metadata = chunks_entry.metadata
    chunks = load_cached_chunks(chunks_entry)

    schema = _schema_from_metadata(text_metadata, chunk_metadata)
    chunk_map = {
        "page_indexing": PAGE_INDEXING,
        "default_strategy": "sliding_window",
        "strategy": "sliding_window",
        "unit": "char",
        "size": chunk_size_chars,
        "overlap": chunk_overlap_chars,
        "preserve_page_spans": True,
        "content_format": "text",
        "chunks": [
            _chunk_reference(entry=chunks_entry, chunk=chunk)
            for chunk in chunks
        ],
    }
    return {
        "schema": schema,
        "chunk_map": chunk_map,
        "resource_cache": {
            "text": text_entry.to_context(),
            "chunks": chunks_entry.to_context(),
        },
    }


def materialize_pdf_text(
    path: str | Path,
    *,
    cache: ResourceCache | None = None,
    extractor: str = "pymupdf",
) -> CacheEntry:
    """Return a cached text view for one PDF."""

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if extractor != "pymupdf":
        raise ValueError(f"Unsupported PDF text extractor: {extractor}")

    resource_cache = cache or ResourceCache()
    content_hash = _hash_file(pdf_path)
    spec = CacheSpec(
        namespace="document_text",
        input_key=f"sha256:{content_hash}",
        producer_key=_pdf_text_producer_key(extractor=extractor),
    )

    def build(workspace: Path) -> CacheBuildResult:
        raw_pages = _extract_pages(pdf_path)
        document_text, pages = _build_document_text(raw_pages)
        pages_path = workspace / "pages.jsonl"
        text_path = workspace / "document.txt"
        _write_jsonl(pages_path, pages)
        text_path.write_text(document_text, encoding="utf-8")
        return CacheBuildResult(
            artifacts={
                "pages": pages_path,
                "text": text_path,
            },
            metadata=_build_text_metadata(
                pdf_path=pdf_path,
                content_hash=content_hash,
                extractor=extractor,
                pages=pages,
                document_text=document_text,
            ),
            artifact_metadata={
                "pages": {"format": "jsonl", "role": "document_pages"},
                "text": {"format": "text", "role": "document_text"},
            },
        )

    return resource_cache.get_or_build(
        spec=spec,
        builder=build,
        required_artifacts=("pages", "text"),
    )


def materialize_text_chunks(
    text_entry: CacheEntry,
    *,
    cache: ResourceCache | None = None,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> CacheEntry:
    """Return cached character-window chunks for a text view."""

    _validate_chunking(chunk_size_chars, chunk_overlap_chars)
    resource_cache = cache or ResourceCache()
    text_metadata = text_entry.metadata
    spec = CacheSpec(
        namespace="document_text_chunks",
        input_key=f"text:{text_entry.cache_key}",
        producer_key=_chunk_text_producer_key(
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        ),
    )

    def build(workspace: Path) -> CacheBuildResult:
        text = text_entry.artifact_path("text").read_text(encoding="utf-8")
        pages = load_jsonl_records(text_entry.artifact_path("pages"))
        chunks = _build_char_chunks(
            text,
            pages,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )
        chunks_path = workspace / _chunk_store_name(
            chunk_size_chars,
            chunk_overlap_chars,
        )
        _write_jsonl(chunks_path, chunks)
        return CacheBuildResult(
            artifacts={"chunks": chunks_path},
            metadata=_build_chunk_metadata(
                text_metadata=text_metadata,
                chunks=chunks,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
                chunk_store_name=chunks_path.name,
            ),
            artifact_metadata={
                "chunks": {"format": "jsonl", "role": "document_chunks"},
            },
        )

    return resource_cache.get_or_build(
        spec=spec,
        builder=build,
        required_artifacts=("chunks",),
    )


def materialize_page_chunks(
    text_entry: CacheEntry,
    *,
    cache: ResourceCache | None = None,
) -> CacheEntry:
    """Return cached one-page chunks for a text view."""

    resource_cache = cache or ResourceCache()
    text_metadata = text_entry.metadata
    spec = CacheSpec(
        namespace="document_page_chunks",
        input_key=f"text:{text_entry.cache_key}",
        producer_key=_page_chunk_producer_key(),
    )

    def build(workspace: Path) -> CacheBuildResult:
        document_text = text_entry.artifact_path("text").read_text(encoding="utf-8")
        pages = load_jsonl_records(text_entry.artifact_path("pages"))
        chunks = _build_page_chunks(document_text, pages)
        chunks_path = workspace / "chunks_page.jsonl"
        _write_jsonl(chunks_path, chunks)
        return CacheBuildResult(
            artifacts={"chunks": chunks_path},
            metadata=_build_page_chunk_metadata(
                text_metadata=text_metadata,
                chunks=chunks,
                chunk_store_name=chunks_path.name,
            ),
            artifact_metadata={
                "chunks": {"format": "jsonl", "role": "document_chunks"},
            },
        )

    return resource_cache.get_or_build(
        spec=spec,
        builder=build,
        required_artifacts=("chunks",),
    )


def load_cached_chunks(entry: CacheEntry) -> list[dict[str, Any]]:
    """Load cached chunk records from a chunk cache entry."""

    return load_jsonl_records(entry.artifact_path("chunks"))


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL artifact produced by document preprocessing."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "PDF preprocessing requires PyMuPDF. Install package 'PyMuPDF' "
            "in the active Python environment."
        ) from exc

    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text") or ""
            pages.append(
                {
                    "page_index": page_index,
                    "page_indexing": PAGE_INDEXING,
                    "raw_char_count": len(text),
                    "text": text,
                }
            )
    return pages


def _build_document_text(
    raw_pages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    segments: list[str] = []
    pages: list[dict[str, Any]] = []
    cursor = 0
    for raw_page in raw_pages:
        segment = f"[Page {raw_page['page_index']}]\n{raw_page['text']}"
        char_start = cursor
        char_end = char_start + len(segment)
        pages.append(
            {
                **raw_page,
                "char_start": char_start,
                "char_end": char_end,
                "char_count": len(segment),
            }
        )
        segments.append(segment)
        cursor = char_end + 2
    return "\n\n".join(segments), pages


def _build_char_chunks(
    text: str,
    pages: list[dict[str, Any]],
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = chunk_size_chars - chunk_overlap_chars
    if not text:
        page_start, page_end = _page_span_for_range(pages, 0, 0)
        return [
            {
                "chunk_id": "chunk_0",
                "strategy": "sliding_window",
                "unit": "char",
                "char_start": 0,
                "char_end": 0,
                "char_count": 0,
                "page_indexing": PAGE_INDEXING,
                "page_start": page_start,
                "page_end": page_end,
                "page_count": _page_count(page_start, page_end),
                "text": "",
            }
        ]

    for chunk_index, char_start in enumerate(range(0, len(text), step)):
        char_end = min(char_start + chunk_size_chars, len(text))
        chunk_text = text[char_start:char_end]
        page_start, page_end = _page_span_for_range(pages, char_start, char_end)
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_index}",
                "strategy": "sliding_window",
                "unit": "char",
                "char_start": char_start,
                "char_end": char_end,
                "char_count": len(chunk_text),
                "page_indexing": PAGE_INDEXING,
                "page_start": page_start,
                "page_end": page_end,
                "page_count": _page_count(page_start, page_end),
                "text": chunk_text,
            }
        )
        if char_end == len(text):
            break
    return chunks


def _build_page_chunks(
    document_text: str,
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not pages:
        return [
            {
                "chunk_id": "chunk_0",
                "strategy": "page",
                "unit": "page",
                "char_start": 0,
                "char_end": 0,
                "char_count": 0,
                "page_indexing": PAGE_INDEXING,
                "page_start": None,
                "page_end": None,
                "page_count": 0,
                "text": "",
            }
        ]

    for chunk_index, page in enumerate(pages):
        char_start = int(page.get("char_start", 0) or 0)
        char_end = int(page.get("char_end", char_start) or char_start)
        chunk_text = document_text[char_start:char_end]
        page_index = page.get("page_index")
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_index}",
                "strategy": "page",
                "unit": "page",
                "char_start": char_start,
                "char_end": char_end,
                "char_count": len(chunk_text),
                "page_indexing": PAGE_INDEXING,
                "page_start": page_index,
                "page_end": page_index,
                "page_count": 1,
                "text": chunk_text,
            }
        )
    return chunks


def _page_span_for_range(
    pages: list[dict[str, Any]],
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    matching = [
        page["page_index"]
        for page in pages
        if page["char_start"] < char_end and page["char_end"] > char_start
    ]
    if not matching and pages:
        matching = [pages[0]["page_index"]]
    if not matching:
        return None, None
    return min(matching), max(matching)


def _page_count(page_start: int | None, page_end: int | None) -> int:
    if page_start is None or page_end is None:
        return 0
    return page_end - page_start + 1


def _build_text_metadata(
    *,
    pdf_path: Path,
    content_hash: str,
    extractor: str,
    pages: list[dict[str, Any]],
    document_text: str,
) -> dict[str, Any]:
    stat = pdf_path.stat()
    raw_char_count = sum(page["raw_char_count"] for page in pages)
    empty_pages = sum(1 for page in pages if page["raw_char_count"] == 0)

    return {
        "version": PDF_TEXT_CACHE_VERSION,
        "format": "pdf",
        "extractor": extractor,
        "source_path": str(pdf_path.resolve()),
        "file_name": pdf_path.name,
        "content_hash": content_hash,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "page_indexing": PAGE_INDEXING,
        "page_count": len(pages),
        "text_layer_available": raw_char_count > 0,
        "ocr_required": raw_char_count == 0,
        "empty_pages": empty_pages,
        "raw_char_count": raw_char_count,
        "char_count": len(document_text),
    }


def _build_chunk_metadata(
    *,
    text_metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    chunk_store_name: str,
) -> dict[str, Any]:
    return {
        "version": TEXT_CHUNK_CACHE_VERSION,
        "format": "text",
        "source_path": text_metadata.get("source_path"),
        "source_text_cache_key": text_metadata.get("cache_key"),
        "page_indexing": PAGE_INDEXING,
        "page_count": text_metadata.get("page_count", 0),
        "char_count": text_metadata.get("char_count", 0),
        "chunking": {
            "default_strategy": "sliding_window",
            "strategy": "sliding_window",
            "unit": "char",
            "size": chunk_size_chars,
            "overlap": chunk_overlap_chars,
            "preserve_page_spans": True,
            "chunk_count": len(chunks),
            "chunk_store_name": chunk_store_name,
        },
    }


def _build_page_chunk_metadata(
    *,
    text_metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
    chunk_store_name: str,
) -> dict[str, Any]:
    return {
        "version": PAGE_CHUNK_CACHE_VERSION,
        "format": "text",
        "source_path": text_metadata.get("source_path"),
        "source_text_cache_key": text_metadata.get("cache_key"),
        "page_indexing": PAGE_INDEXING,
        "page_count": text_metadata.get("page_count", 0),
        "char_count": text_metadata.get("char_count", 0),
        "chunking": {
            "default_strategy": "page",
            "strategy": "page",
            "unit": "page",
            "size": 1,
            "overlap": 0,
            "preserve_page_spans": True,
            "chunk_count": len(chunks),
            "chunk_store_name": chunk_store_name,
        },
    }


def _schema_from_metadata(
    text_metadata: dict[str, Any],
    chunk_metadata: dict[str, Any],
) -> dict[str, Any]:
    chunking = chunk_metadata["chunking"]
    return {
        "format": "pdf",
        "page_count": text_metadata["page_count"],
        "page_indexing": text_metadata["page_indexing"],
        "text_extraction": {
            "extractor": text_metadata["extractor"],
            "text_layer_available": text_metadata["text_layer_available"],
            "ocr_required": text_metadata["ocr_required"],
            "empty_pages": text_metadata["empty_pages"],
            "raw_char_count": text_metadata["raw_char_count"],
            "char_count": text_metadata["char_count"],
        },
        "chunking": {
            "default_strategy": chunking["default_strategy"],
            "strategy": chunking["strategy"],
            "unit": chunking["unit"],
            "size": chunking["size"],
            "overlap": chunking["overlap"],
            "preserve_page_spans": chunking["preserve_page_spans"],
            "chunk_count": chunking["chunk_count"],
            "content": "resource_cache",
        },
    }


def _chunk_reference(
    *,
    entry: CacheEntry,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    chunk_id = chunk["chunk_id"]
    return {
        "chunk_id": chunk_id,
        "content_ref": entry.item_ref("chunks", chunk_id),
        "page_indexing": chunk["page_indexing"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "page_count": chunk["page_count"],
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "char_count": chunk["char_count"],
    }


def _pdf_text_producer_key(*, extractor: str) -> str:
    payload = {
        "recipe": "pdf_extract_text",
        "recipe_version": PDF_TEXT_CACHE_VERSION,
        "extractor": extractor,
        "page_indexing": PAGE_INDEXING,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"pdf_extract_text:{digest[:16]}"


def _chunk_text_producer_key(
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> str:
    payload = {
        "recipe": "chunk_text",
        "recipe_version": TEXT_CHUNK_CACHE_VERSION,
        "strategy": "sliding_window",
        "unit": "char",
        "size": chunk_size_chars,
        "overlap": chunk_overlap_chars,
        "preserve_page_spans": True,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"chunk_text:{digest[:16]}"


def _page_chunk_producer_key() -> str:
    payload = {
        "recipe": "chunk_text",
        "recipe_version": PAGE_CHUNK_CACHE_VERSION,
        "strategy": "page",
        "unit": "page",
        "size": 1,
        "overlap": 0,
        "preserve_page_spans": True,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"chunk_text_page:{digest[:16]}"


def _chunk_store_name(chunk_size_chars: int, chunk_overlap_chars: int) -> str:
    if chunk_overlap_chars == 0:
        return f"chunks_char_{chunk_size_chars}.jsonl"
    return f"chunks_char_{chunk_size_chars}_overlap_{chunk_overlap_chars}.jsonl"


def _validate_chunking(chunk_size_chars: int, chunk_overlap_chars: int) -> None:
    if chunk_size_chars < 1:
        raise ValueError("chunk_size_chars must be at least 1")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars cannot be negative")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    temp_path.replace(path)
