"""Resource preprocessing and source-view description."""

from clover.resource.preprocess.core import preprocess_task_dsl
from clover.resource.preprocess.csv_schema import extract_csv_schema
from clover.resource.preprocess.pdf_schema import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_CHUNK_SIZE_CHARS,
    PAGE_INDEXING,
    extract_pdf_schema,
    load_cached_chunks,
    load_jsonl_records,
    materialize_page_chunks,
    materialize_pdf_text,
    materialize_text_chunks,
)

__all__ = [
    "DEFAULT_CHUNK_OVERLAP_CHARS",
    "DEFAULT_CHUNK_SIZE_CHARS",
    "PAGE_INDEXING",
    "extract_csv_schema",
    "extract_pdf_schema",
    "load_cached_chunks",
    "load_jsonl_records",
    "materialize_page_chunks",
    "materialize_pdf_text",
    "materialize_text_chunks",
    "preprocess_task_dsl",
]
