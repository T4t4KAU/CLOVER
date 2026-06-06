"""External resource materialization registry."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class MaterializationError(RuntimeError):
    """Raised when an external resource cannot be materialized."""


@dataclass
class ResourceMaterializationContext:
    """Shared caches used while materializing executor resources."""

    caches: dict[str, Any] = field(default_factory=dict)

    def cache(self, namespace: str) -> dict[str, Any]:
        value = self.caches.setdefault(namespace, {})
        if not isinstance(value, dict):
            raise MaterializationError(
                f"Materialization cache namespace is not a dict: {namespace}"
            )
        return value


class ExternalResourceMaterializer(Protocol):
    """Convert one external resource to one target representation."""

    def supports(self, resource: Any, target: str) -> bool:
        ...

    def materialize(
        self,
        resource: Any,
        *,
        target: str,
        context: ResourceMaterializationContext,
    ) -> Any:
        ...


@dataclass
class ExternalMaterializerRegistry:
    """Ordered registry for external resource materializers."""

    materializers: list[ExternalResourceMaterializer] = field(default_factory=list)

    def register(self, materializer: ExternalResourceMaterializer) -> None:
        self.materializers.append(materializer)

    def materialize(
        self,
        resource: Any,
        *,
        target: str,
        context: ResourceMaterializationContext,
    ) -> Any:
        for materializer in self.materializers:
            if materializer.supports(resource, target):
                return materializer.materialize(
                    resource,
                    target=target,
                    context=context,
                )
        kind = getattr(resource, "kind", None)
        resource_id = getattr(resource, "id", None)
        raise MaterializationError(
            f"No materializer for external resource {resource_id!r} "
            f"of kind {kind!r} to target {target!r}"
        )


class ResourceSpecMaterializer:
    """Return the optimizer-visible resource specification."""

    def supports(self, resource: Any, target: str) -> bool:
        return target == "resource_spec"

    def materialize(
        self,
        resource: Any,
        *,
        target: str,
        context: ResourceMaterializationContext,
    ) -> dict[str, Any]:
        return copy.deepcopy(getattr(resource, "spec", {}))


class TablePandasMaterializer:
    """Materialize table resources through the table backend."""

    targets = {"python", "pandas", "table"}

    def supports(self, resource: Any, target: str) -> bool:
        return getattr(resource, "kind", None) == "table" and target in self.targets

    def materialize(
        self,
        resource: Any,
        *,
        target: str,
        context: ResourceMaterializationContext,
    ) -> Any:
        from clover.tools.table_reasoning.pandas_backend import (
            PandasTable,
            _read_resource_frame,
        )

        frame = _read_resource_frame(resource.spec, context.cache("table"))
        return PandasTable(frame)


class DocumentChunkTextMaterializer:
    """Materialize cached document chunk resources."""

    targets = {"text", "chunk_record"}

    def supports(self, resource: Any, target: str) -> bool:
        return (
            getattr(resource, "kind", None) == "document_chunk"
            and target in self.targets
        )

    def materialize(
        self,
        resource: Any,
        *,
        target: str,
        context: ResourceMaterializationContext,
    ) -> Any:
        record = _load_chunk_record(resource.spec, context=context)
        if target == "text":
            return str(record.get("text", ""))
        return copy.deepcopy(record)


class ExternalSpecFallbackMaterializer:
    """Keep generic external resources lightweight by returning their spec."""

    targets = {"python", "pandas", "sandbox"}

    def supports(self, resource: Any, target: str) -> bool:
        return target in self.targets

    def materialize(
        self,
        resource: Any,
        *,
        target: str,
        context: ResourceMaterializationContext,
    ) -> dict[str, Any]:
        return copy.deepcopy(getattr(resource, "spec", {}))


DEFAULT_EXTERNAL_MATERIALIZERS = ExternalMaterializerRegistry(
    [
        ResourceSpecMaterializer(),
        TablePandasMaterializer(),
        DocumentChunkTextMaterializer(),
        ExternalSpecFallbackMaterializer(),
    ]
)


def materialize_external_resource(
    resource: Any,
    *,
    target: str,
    context: ResourceMaterializationContext,
) -> Any:
    return DEFAULT_EXTERNAL_MATERIALIZERS.materialize(
        resource,
        target=target,
        context=context,
    )


def external_materialization_cache_key(resource: Any, target: str) -> str | None:
    """Return the per-resource cache key for an external materialization."""

    kind = getattr(resource, "kind", None)
    if kind == "table" and target in TablePandasMaterializer.targets:
        return "table:pandas"
    if kind == "document_chunk" and target in DocumentChunkTextMaterializer.targets:
        return f"document_chunk:{target}"
    return None


def copy_materialized_value(value: Any) -> Any:
    """Return an isolated value when a cached materialization is reused."""

    copy_method = getattr(value, "copy", None)
    if callable(copy_method):
        try:
            return copy_method()
        except TypeError:
            pass
    return copy.deepcopy(value)


def _load_chunk_record(
    spec: dict[str, Any],
    *,
    context: ResourceMaterializationContext,
) -> dict[str, Any]:
    path_value = spec.get("path")
    if not path_value:
        raise MaterializationError("document_chunk resource is missing path")
    path = str(Path(path_value).expanduser().resolve())
    item_id = spec.get("item_id") or spec.get("chunk_id")
    if not item_id:
        raise MaterializationError("document_chunk resource is missing item_id")

    chunk_store_cache = context.cache("document_chunk_store")
    records = chunk_store_cache.get(path)
    if records is None:
        records = _index_jsonl_chunks(Path(path))
        chunk_store_cache[path] = records
    try:
        return records[str(item_id)]
    except KeyError as exc:
        raise MaterializationError(
            f"document_chunk item {item_id!r} not found in {path}"
        ) from exc


def _index_jsonl_chunks(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise MaterializationError(f"document_chunk store not found: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = record.get("chunk_id")
            if chunk_id is not None:
                records[str(chunk_id)] = record
    return records
