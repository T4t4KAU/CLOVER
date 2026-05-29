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

import pandas as pd

from clover.executor.result import summarize_output
from clover.tools.table_reasoning.pandas_backend import (
    PandasTable,
    _read_resource_frame,
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
        if self.closed or self.value is None:
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
    table_cache: dict[str, pd.DataFrame] | None = None
    cached_value: Any | None = None
    cached_size_bytes: int = 0

    def materialize(self, *, target: str = "python") -> Any:
        if self.closed:
            raise ResourceError(f"Cannot materialize closed resource: {self.id}")
        self.last_access = time.monotonic()
        if target == "resource_spec":
            return copy.deepcopy(self.spec)
        if self.kind == "table":
            if self.cached_value is None:
                frame = _read_resource_frame(self.spec, self.table_cache)
                self.cached_value = PandasTable(frame)
                self.cached_size_bytes = estimate_value_size(self.cached_value)
            return self.cached_value.copy()
        return copy.deepcopy(self.spec)

    def release_materialized(self) -> None:
        self.cached_value = None
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
    table_cache: dict[str, pd.DataFrame] | None = None,
) -> FileExternalResourceObject:
    path = Path(spec["path"]).expanduser()
    size = path.stat().st_size if path.is_file() else 0
    return FileExternalResourceObject(
        id=str(spec["id"]),
        kind=str(spec.get("type") or "file"),
        storage="file_external",
        owner="external",
        metadata=_metadata_from_spec(spec),
        size_bytes=size,
        spec=copy.deepcopy(spec),
        table_cache=table_cache,
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
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    size = path.stat().st_size
    metadata = _metadata_from_value(value)
    summary = summarize_output(value)
    payload = {
        "id": resource_id,
        "kind": infer_resource_kind(value),
        "storage": "file_spilled",
        "serializer": "pickle",
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
    )


def infer_resource_kind(value: Any) -> str:
    if isinstance(value, PandasTable) or isinstance(value, pd.DataFrame):
        return "table"
    if hasattr(value, "frame") and isinstance(getattr(value, "frame"), pd.DataFrame):
        return "table"
    return "value"


def estimate_value_size(value: Any) -> int:
    frame = getattr(value, "frame", None)
    if isinstance(frame, pd.DataFrame):
        return int(frame.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:  # noqa: BLE001 - approximate unusual objects.
        return sys.getsizeof(value)


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
