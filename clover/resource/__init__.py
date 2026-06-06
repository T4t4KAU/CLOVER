"""Unified resource APIs for preprocessing, cache, and plan preparation."""

from clover.resource.cache import (
    CacheBuildResult,
    CacheEntry,
    CacheSpec,
    ResourceCache,
    ResourceCacheConfig,
    ResourceCacheError,
)
from clover.resource.preprocess import (
    extract_csv_schema,
    extract_pdf_schema,
    load_cached_chunks,
    load_jsonl_records,
    materialize_page_chunks,
    materialize_pdf_text,
    materialize_text_chunks,
    preprocess_task_dsl,
)
from clover.resource.processing import (
    PhysicalPlanResourceBuilder,
    ResourceProcessingError,
    prepare_physical_plan_resources,
)

__all__ = [
    "CacheBuildResult",
    "CacheEntry",
    "CacheSpec",
    "ResourceCache",
    "ResourceCacheConfig",
    "ResourceCacheError",
    "ResourceProcessingError",
    "PhysicalPlanResourceBuilder",
    "extract_csv_schema",
    "extract_pdf_schema",
    "load_cached_chunks",
    "load_jsonl_records",
    "materialize_page_chunks",
    "materialize_pdf_text",
    "materialize_text_chunks",
    "prepare_physical_plan_resources",
    "preprocess_task_dsl",
]
