"""Resource object primitives for executor-owned data lifecycle."""

from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pyarrow.feather as _feather

    _HAS_FEATHER = True
except ImportError:
    _HAS_FEATHER = False

from clover.executor.result import summarize_output
from clover.executor.resources.materializers import (
    ResourceMaterializationContext,
    copy_materialized_value,
    external_materialization_cache_key,
    materialize_external_resource,
)


class ResourceError(RuntimeError):
    """Raised when a resource cannot be materialized or stored."""


@dataclass
class ResourceObject:
    """Base executor resource with unified access and lifecycle hooks."""

    id: str
    kind: str
    storage: str
    owner: str
    metadata: dict[str, Any] = field(default_factory=dict)
    producer_node: str | None = None
    summary_cache: dict[str, Any] | None = None
    size_bytes: int = 0
    pinned: int = 0
    closed: bool = False
    retained: bool = False
    last_access: float = field(default_factory=time.monotonic)

    def summary(self) -> dict[str, Any]:
        if self.summary_cache is not None:
            return copy.deepcopy(self.summary_cache)
        value = self.materialize()
        self.summary_cache = summarize_output(value)
        return copy.deepcopy(self.summary_cache)

    def estimate_size(self) -> int:
        return int(self.size_bytes or 0)

    def pin(self) -> None:
        if self.closed:
            raise ResourceError(f"Cannot pin closed resource: {self.id}")
        self.pinned += 1
        self.last_access = time.monotonic()

    def unpin(self) -> None:
        if self.pinned > 0:
            self.pinned -= 1
        self.last_access = time.monotonic()

    def materialize(self, *, target: str = "python") -> Any:
        raise NotImplementedError

    def release_materialized(self) -> None:
        self.last_access = time.monotonic()

    def close(self) -> None:
        self.closed = True
        self.release_materialized()


@dataclass
class MemoryResourceObject(ResourceObject):
    """Resource whose live value is stored in memory."""

    value: Any = None

    def materialize(self, *, target: str = "python") -> Any:
        if self.closed:
            raise ResourceError(f"Resource is not materialized: {self.id}")
        self.last_access = time.monotonic()
        return self.value

    def release_materialized(self) -> None:
        self.value = None
        self.size_bytes = 0
        self.last_access = time.monotonic()

    def close(self) -> None:
        self.release_materialized()
        self.closed = True


@dataclass
class FileExternalResourceObject(ResourceObject):
    """External file resource. Closing never deletes the source file."""

    spec: dict[str, Any] = field(default_factory=dict)
    materialization_context: ResourceMaterializationContext = field(
        default_factory=ResourceMaterializationContext
    )
    cached_values: dict[str, Any] = field(default_factory=dict)
    cached_size_bytes: int = 0

    def materialize(self, *, target: str = "python") -> Any:
        if self.closed:
            raise ResourceError(f"Cannot materialize closed resource: {self.id}")
        self.last_access = time.monotonic()
        cache_key = external_materialization_cache_key(self, target)
        if cache_key is None:
            return materialize_external_resource(
                self,
                target=target,
                context=self.materialization_context,
            )
        if cache_key not in self.cached_values:
            self.cached_values[cache_key] = materialize_external_resource(
                self,
                target=target,
                context=self.materialization_context,
            )
            self.cached_size_bytes = sum(
                estimate_value_size(value)
                for value in self.cached_values.values()
            )
        return copy_materialized_value(self.cached_values[cache_key])

    def release_materialized(self) -> None:
        self.cached_values.clear()
        self.cached_size_bytes = 0
        self.last_access = time.monotonic()

    def close(self) -> None:
        self.release_materialized()
        self.closed = True


@dataclass
class FileSpilledResourceObject(ResourceObject):
    """Executor-owned resource spilled to a temporary file."""

    path: Path | None = None
    metadata_path: Path | None = None
    cached_value: Any | None = None
    cached_size_bytes: int = 0
    serializer: str = "pickle"

    def materialize(self, *, target: str = "python") -> Any:
        if self.closed:
            raise ResourceError(f"Cannot materialize closed resource: {self.id}")
        if self.cached_value is None:
            if self.path is None or not self.path.is_file():
                raise ResourceError(f"Missing spilled resource file: {self.id}")
            if self.serializer == "feather" and _HAS_FEATHER:
                self.cached_value = _feather.read_feather(self.path)
            else:
                with self.path.open("rb") as handle:
                    self.cached_value = pickle.load(handle)
            self.cached_size_bytes = estimate_value_size(self.cached_value)
        self.last_access = time.monotonic()
        return self.cached_value

    def release_materialized(self) -> None:
        self.cached_value = None
        self.cached_size_bytes = 0
        self.last_access = time.monotonic()

    def close(self) -> None:
        self.release_materialized()
        for path in (self.path, self.metadata_path):
            if path is None:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.closed = True


def memory_resource(
    resource_id: str,
    value: Any,
    *,
    producer_node: str | None = None,
    retained: bool = False,
) -> MemoryResourceObject:
    return MemoryResourceObject(
        id=resource_id,
        kind=infer_resource_kind(value),
        storage="memory",
        owner="executor",
        producer_node=producer_node,
        metadata=_metadata_from_value(value),
        summary_cache=summarize_output(value),
        size_bytes=estimate_value_size(value),
        value=value,
        retained=retained,
    )


def external_file_resource(
    spec: dict[str, Any],
    *,
    materialization_context: ResourceMaterializationContext | None = None,
    table_cache: dict[str, Any] | None = None,
) -> FileExternalResourceObject:
    path = Path(spec["path"]).expanduser()
    size = path.stat().st_size if path.is_file() else 0
    context = materialization_context or ResourceMaterializationContext()
    if table_cache is not None:
        context.caches["table"] = table_cache
    return FileExternalResourceObject(
        id=str(spec["id"]),
        kind=str(spec.get("type") or "file"),
        storage="file_external",
        owner="external",
        metadata=_metadata_from_spec(spec),
        size_bytes=size,
        spec=copy.deepcopy(spec),
        materialization_context=context,
    )


def spilled_file_resource(
    resource_id: str,
    value: Any,
    *,
    path: Path,
    metadata_path: Path,
    producer_node: str | None = None,
    retained: bool = False,
) -> FileSpilledResourceObject:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    is_dataframe = _is_dataframe_like(value)
    if is_dataframe and _HAS_FEATHER:
        _feather.write_feather(value, path)
        serializer = "feather"
    else:
        with path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        serializer = "pickle"

    size = path.stat().st_size
    metadata = _metadata_from_value(value)
    summary = summarize_output(value)
    payload = {
        "id": resource_id,
        "kind": infer_resource_kind(value),
        "storage": "file_spilled",
        "serializer": serializer,
        "path": str(path),
        "size_bytes": size,
        "producer_node": producer_node,
        "summary": summary,
        "metadata": metadata,
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return FileSpilledResourceObject(
        id=resource_id,
        kind=payload["kind"],
        storage="file_spilled",
        owner="executor",
        metadata=metadata,
        producer_node=producer_node,
        summary_cache=summary,
        size_bytes=size,
        path=path,
        metadata_path=metadata_path,
        retained=retained,
        serializer=serializer,
    )


def infer_resource_kind(value: Any) -> str:
    if _is_dataframe_like(value):
        return "table"
    if _is_dataframe_like(getattr(value, "frame", None)):
        return "table"
    return "value"


def estimate_value_size(value: Any) -> int:
    frame = getattr(value, "frame", None)
    frame_size = _dataframe_size_bytes(frame)
    if frame_size is not None:
        return frame_size
    value_size = _dataframe_size_bytes(value)
    if value_size is not None:
        return value_size
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:  # noqa: BLE001 - approximate unusual objects.
        return sys.getsizeof(value)


def _is_dataframe_like(value: Any) -> bool:
    return hasattr(value, "columns") and callable(getattr(value, "memory_usage", None))


def _dataframe_size_bytes(value: Any) -> int | None:
    if not _is_dataframe_like(value):
        return None
    try:
        usage = value.memory_usage(index=True, deep=True)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    except Exception:  # noqa: BLE001 - fall back to pickle/sys sizing.
        return None


def _metadata_from_value(value: Any) -> dict[str, Any]:
    group_keys = getattr(value, "group_keys", None)
    if group_keys:
        return {"group_keys": copy.deepcopy(group_keys)}
    return {}


def _metadata_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: copy.deepcopy(value)
        for key, value in spec.items()
        if key not in {"path", "file", "url", "uri"}
    }
    path = spec.get("path")
    if path:
        metadata["filename"] = os.path.basename(str(path))
    return metadata
