"""Unified resource APIs for preprocessing, cache, and plan preparation."""

from clover.resource.cache import (
    CacheBuildResult,
    CacheEntry,
    CacheSpec,
    ResourceCache,
    ResourceCacheConfig,
    ResourceCacheError,
)
from clover.resource.dsl_builder import (
    BUILD_TABLE_DSL_TOOL_NAME,
    BUILDER_AGENT_MODE,
    BuildTableDSLTool,
    TableDSLBuilderAgentError,
    TableDslBuilderResult,
    build_table_task_dsl_with_builder_agent,
    parse_builder_agent_tool_call,
    parse_table_dsl_builder_output,
    render_table_dsl_builder_agent_prompt,
    static_table_dsl_metadata,
    table_profile_for_dsl_builder,
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
    "BUILD_TABLE_DSL_TOOL_NAME",
    "BUILDER_AGENT_MODE",
    "BuildTableDSLTool",
    "ResourceCache",
    "ResourceCacheConfig",
    "ResourceCacheError",
    "ResourceProcessingError",
    "TableDSLBuilderAgentError",
    "TableDslBuilderResult",
    "PhysicalPlanResourceBuilder",
    "build_table_task_dsl_with_builder_agent",
    "extract_csv_schema",
    "extract_pdf_schema",
    "load_cached_chunks",
    "load_jsonl_records",
    "materialize_page_chunks",
    "materialize_pdf_text",
    "materialize_text_chunks",
    "prepare_physical_plan_resources",
    "parse_builder_agent_tool_call",
    "parse_table_dsl_builder_output",
    "preprocess_task_dsl",
    "render_table_dsl_builder_agent_prompt",
    "static_table_dsl_metadata",
    "table_profile_for_dsl_builder",
]
