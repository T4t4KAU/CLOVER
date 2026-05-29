"""Resource store with memory budget and spill-to-/tmp support."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clover.executor.resources.objects import (
    FileExternalResourceObject,
    MemoryResourceObject,
    ResourceObject,
    external_file_resource,
    memory_resource,
    spilled_file_resource,
)


DEFAULT_SPILL_ROOT = Path(tempfile.gettempdir()) / "clover_spill"
SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ResourceLimits:
    """Capacity policy for one executor run."""

    memory_budget_bytes: int = 1024 * 1024 * 1024
    spill_threshold_bytes: int = 256 * 1024 * 1024
    spill_root: Path = DEFAULT_SPILL_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "spill_root", Path(self.spill_root))


class ResourceStore:
    """Own external resources, node outputs, summaries, and spilled files."""

    def __init__(
        self,
        *,
        external_resources: dict[str, dict[str, Any]] | None = None,
        table_cache: dict[str, Any] | None = None,
        limits: ResourceLimits | None = None,
    ) -> None:
        self.limits = limits or ResourceLimits()
        self.table_cache = table_cache
        self.run_id = f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.spill_dir = self.limits.spill_root / self.run_id
        self._sources: dict[str, ResourceObject] = {}
        self._outputs: dict[str, ResourceObject] = {}
        self._summaries: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        if external_resources:
            for spec in external_resources.values():
                self.register_external(spec)

    def register_external(self, spec: dict[str, Any]) -> ResourceObject:
        with self._lock:
            resource = external_file_resource(spec, table_cache=self.table_cache)
            self._sources[resource.id] = resource
            return resource

    def source_specs(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                resource_id: resource.materialize(target="resource_spec")
                for resource_id, resource in self._sources.items()
                if isinstance(resource, FileExternalResourceObject)
            }

    def put_output(
        self,
        name: str,
        value: Any,
        *,
        producer_node: str | None = None,
        retained: bool = False,
    ) -> ResourceObject:
        with self._lock:
            resource = self._resource_for_output(
                name,
                value,
                producer_node=producer_node,
                retained=retained,
            )
            self._outputs[name] = resource
            self._summaries[name] = resource.summary()
            self._ensure_capacity(0)
            return self._outputs[name]

    def get_output(self, name: str) -> ResourceObject:
        with self._lock:
            return self._outputs[name]

    def get_source(self, name: str) -> ResourceObject:
        with self._lock:
            return self._sources[name]

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._outputs

    def has_output(self, name: str) -> bool:
        return self.has(name)

    def has_source(self, name: str) -> bool:
        with self._lock:
            return name in self._sources

    def missing(self, names: list[str]) -> list[str]:
        with self._lock:
            return [name for name in names if name not in self._outputs]

    def node_view(
        self,
        *,
        node_id: str,
        dependencies: list[str],
        sources: list[str],
    ) -> "NodeResourceView":
        return NodeResourceView(
            store=self,
            node_id=node_id,
            dependency_names=list(dependencies),
            source_names=list(sources),
        )

    def materialize_outputs(
        self,
        names: list[str],
        *,
        target: str = "pandas",
    ) -> dict[str, Any]:
        with self._lock:
            return {
                name: self._outputs[name].materialize(target=target)
                for name in names
                if name in self._outputs
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                name: resource.materialize(target="python")
                for name, resource in self._outputs.items()
            }

    def summaries(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._summaries)

    def release(self, name: str) -> None:
        with self._lock:
            resource = self._outputs.pop(name, None)
            if resource is not None:
                self._summaries.setdefault(name, resource.summary())
                resource.close()

    def close_all(self) -> None:
        with self._lock:
            for resource in list(self._outputs.values()) + list(self._sources.values()):
                resource.close()
            self._outputs.clear()
            self._sources.clear()
            try:
                shutil.rmtree(self.spill_dir)
            except FileNotFoundError:
                pass

    def current_memory_bytes(self) -> int:
        total = 0
        for resource in list(self._outputs.values()) + list(self._sources.values()):
            if resource.storage == "memory":
                total += resource.estimate_size()
            cached = getattr(resource, "cached_size_bytes", 0)
            total += int(cached or 0)
        return total

    def _resource_for_output(
        self,
        name: str,
        value: Any,
        *,
        producer_node: str | None,
        retained: bool,
    ) -> ResourceObject:
        estimated_size = memory_resource(
            name,
            value,
            producer_node=producer_node,
            retained=retained,
        ).estimate_size()
        if estimated_size > self.limits.spill_threshold_bytes:
            return self._spill_value(
                name,
                value,
                producer_node=producer_node,
                retained=retained,
            )
        self._ensure_capacity(estimated_size)
        return memory_resource(
            name,
            value,
            producer_node=producer_node,
            retained=retained,
        )

    def _ensure_capacity(self, required_bytes: int) -> None:
        budget = self.limits.memory_budget_bytes
        while self.current_memory_bytes() + required_bytes > budget:
            candidate = self._spill_candidate()
            if candidate is None:
                return
            value = candidate.materialize(target="python")
            spilled = self._spill_value(
                candidate.id,
                value,
                producer_node=candidate.producer_node,
                retained=candidate.retained,
            )
            self._outputs[candidate.id] = spilled
            candidate.close()

    def _spill_candidate(self) -> MemoryResourceObject | None:
        candidates = [
            resource
            for resource in self._outputs.values()
            if isinstance(resource, MemoryResourceObject)
            and resource.pinned <= 0
            and not resource.closed
            and resource.value is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda resource: resource.last_access)

    def _spill_value(
        self,
        name: str,
        value: Any,
        *,
        producer_node: str | None,
        retained: bool,
    ) -> ResourceObject:
        safe_name = _safe_resource_name(name)
        path = self.spill_dir / "resources" / f"{safe_name}.pkl"
        metadata_path = self.spill_dir / "metadata" / f"{safe_name}.json"
        resource = spilled_file_resource(
            name,
            value,
            path=path,
            metadata_path=metadata_path,
            producer_node=producer_node,
            retained=retained,
        )
        self._manifest_event(
            {
                "event": "create",
                "id": name,
                "path": str(path),
                "size_bytes": resource.size_bytes,
                "time": time.time(),
            }
        )
        return resource

    def _manifest_event(self, payload: dict[str, Any]) -> None:
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.spill_dir / "manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class NodeResourceView:
    """Current-node view over dependency and source resources."""

    def __init__(
        self,
        *,
        store: ResourceStore,
        node_id: str,
        dependency_names: list[str],
        source_names: list[str],
    ) -> None:
        self.store = store
        self.node_id = node_id
        self.dependency_names = dependency_names
        self.source_names = source_names
        self._pinned = False

    def pin(self) -> None:
        if self._pinned:
            return
        for resource in self.dependencies().values():
            resource.pin()
        for resource in self.sources().values():
            resource.pin()
        self._pinned = True

    def unpin(self) -> None:
        if not self._pinned:
            return
        for resource in self.dependencies().values():
            resource.unpin()
        for resource in self.sources().values():
            resource.unpin()
        self._pinned = False

    def dependencies(self) -> dict[str, ResourceObject]:
        return {
            name: self.store.get_output(name)
            for name in self.dependency_names
            if self.store.has(name)
        }

    def sources(self) -> dict[str, ResourceObject]:
        return {
            name: self.store.get_source(name)
            for name in self.source_names
            if self.store.has_source(name)
        }

    def materialize_dependencies(self, *, target: str = "pandas") -> dict[str, Any]:
        return {
            name: resource.materialize(target=target)
            for name, resource in self.dependencies().items()
        }

    def materialize_sources(self, *, target: str = "resource_spec") -> dict[str, Any]:
        return {
            name: resource.materialize(target=target)
            for name, resource in self.sources().items()
        }

    def project_dependencies(self, projector: Any) -> dict[str, Any]:
        return {
            name: projector.project(resource, role="dependency")
            for name, resource in self.dependencies().items()
        }

    def project_sources(self, projector: Any) -> dict[str, Any]:
        return {
            name: projector.project(resource, role="source")
            for name, resource in self.sources().items()
        }

    def summary(self) -> dict[str, Any]:
        return {
            "dependencies": {
                name: resource.summary()
                for name, resource in self.dependencies().items()
            },
            "sources": {
                name: resource.summary()
                for name, resource in self.sources().items()
            },
        }


def normalize_resource_specs(
    resources: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if resources is None:
        return {}
    if isinstance(resources, dict):
        return {str(key): dict(value) for key, value in resources.items()}
    return {str(resource["id"]): dict(resource) for resource in resources}


def _safe_resource_name(name: str) -> str:
    safe = SAFE_ID_PATTERN.sub("_", name).strip("._")
    return safe or uuid.uuid4().hex
