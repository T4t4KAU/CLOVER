"""Persistent derived-resource cache with LRU eviction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


DEFAULT_MAX_TOTAL_BYTES = 5 * 1024**3
DEFAULT_TARGET_RATIO = 0.8
DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0
DEFAULT_LOCK_STALE_SECONDS = 600.0
RESOURCE_CACHE_SCHEME = "resource_cache"
SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


class ResourceCacheError(RuntimeError):
    """Raised when a derived-resource cache entry cannot be used."""


@dataclass(frozen=True)
class CacheSpec:
    """Opaque identity for one downstream-derived cache entry."""

    namespace: str
    input_key: str
    producer_key: str

    def payload(self) -> dict[str, str]:
        return {
            "namespace": self.namespace,
            "input_key": self.input_key,
            "producer_key": self.producer_key,
        }


@dataclass(frozen=True)
class CacheBuildResult:
    """Artifacts produced by a downstream cache builder."""

    artifacts: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceCacheConfig:
    """Capacity and root settings for the derived-resource cache."""

    root: str | Path | None = None
    max_total_bytes: int | None = None
    target_ratio: float = DEFAULT_TARGET_RATIO
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS
    lock_stale_seconds: float = DEFAULT_LOCK_STALE_SECONDS


@dataclass(frozen=True)
class CacheEntry:
    """A complete cache entry returned to downstream code."""

    cache_key: str
    root: Path
    path: Path
    manifest: dict[str, Any]
    hit: bool

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def access_path(self) -> Path:
        return self.path / "access.json"

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.manifest.get("metadata", {}))

    def artifact_path(self, name: str) -> Path:
        artifact = self._artifact(name)
        return self.path / artifact["path"]

    def artifact_ref(self, name: str) -> str:
        self._artifact(name)
        return f"{RESOURCE_CACHE_SCHEME}://{self.cache_key}/{name}"

    def item_ref(self, artifact_name: str, item_id: str) -> str:
        self._artifact(artifact_name)
        return f"{RESOURCE_CACHE_SCHEME}://{self.cache_key}/{artifact_name}/{item_id}"

    def to_context(self) -> dict[str, Any]:
        artifacts = {}
        for name, artifact in self.manifest.get("artifacts", {}).items():
            artifacts[name] = {
                "ref": self.artifact_ref(name),
                "path": str(self.artifact_path(name)),
                "size_bytes": artifact.get("size_bytes", 0),
                **dict(artifact.get("metadata", {})),
            }
        return {
            "cache_key": self.cache_key,
            "root": str(self.root),
            "entry_dir": str(self.path),
            "manifest_path": str(self.manifest_path),
            "hit": self.hit,
            "artifacts": artifacts,
        }

    def _artifact(self, name: str) -> dict[str, Any]:
        artifacts = self.manifest.get("artifacts", {})
        if name not in artifacts:
            raise ResourceCacheError(
                f"Cache artifact not found: {self.cache_key}/{name}"
            )
        return artifacts[name]


class ResourceCache:
    """Filesystem-backed cache for downstream-derived resources.

    The cache does not interpret artifacts. Downstream code owns the producer
    identity and artifact semantics; this class handles reuse, atomic commits,
    references, and LRU capacity control.
    """

    def __init__(self, config: ResourceCacheConfig | None = None) -> None:
        self.config = config or ResourceCacheConfig()
        self.root = _configured_root(self.config)
        self.max_total_bytes = _configured_max_total_bytes(self.config)
        self.target_ratio = float(self.config.target_ratio)
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _best_effort_chmod_private(self.root)

    def get_or_build(
        self,
        *,
        spec: CacheSpec,
        builder: Callable[[Path], CacheBuildResult],
        required_artifacts: tuple[str, ...] = (),
    ) -> CacheEntry:
        """Return a complete entry, building it if no valid entry exists."""

        cache_key = self.cache_key(spec)
        entry = self._complete_entry(cache_key, spec, required_artifacts, hit=True)
        if entry is not None:
            self._touch(entry.path)
            self.gc_lru(protected_keys={cache_key})
            return entry

        entry_dir = self._entry_dir(cache_key)
        with self._entry_build_lock(entry_dir):
            entry = self._complete_entry(cache_key, spec, required_artifacts, hit=True)
            if entry is not None:
                self._touch(entry.path)
            else:
                build_dir = entry_dir / f".build_{uuid.uuid4().hex}"
                build_dir.mkdir(parents=True)
                try:
                    result = builder(build_dir)
                    entry = self._commit_build(
                        cache_key=cache_key,
                        spec=spec,
                        build_result=result,
                        entry_dir=entry_dir,
                    )
                finally:
                    shutil.rmtree(build_dir, ignore_errors=True)

        self.gc_lru(protected_keys={cache_key})
        return entry

    def cache_key(self, spec: CacheSpec) -> str:
        payload = json.dumps(spec.payload(), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"rc_{digest[:32]}"

    def resolve_ref(self, ref: str) -> dict[str, Any]:
        prefix = f"{RESOURCE_CACHE_SCHEME}://"
        if not ref.startswith(prefix):
            raise ResourceCacheError(f"Unsupported cache ref: {ref}")
        rest = ref[len(prefix) :]
        cache_key, separator, path = rest.partition("/")
        parts = [part for part in path.split("/") if part]
        if not cache_key or not separator or not parts:
            raise ResourceCacheError(f"Invalid cache ref: {ref}")

        entry = self._entry_from_key(cache_key)
        artifact_name = parts[0]
        return {
            "cache_key": cache_key,
            "artifact": artifact_name,
            "item_id": "/".join(parts[1:]) or None,
            "path": str(entry.artifact_path(artifact_name)),
        }

    def gc_lru(self, *, protected_keys: set[str] | None = None) -> None:
        """Evict least-recently-used entries until capacity is below target."""

        protected_keys = protected_keys or set()
        with _BuildLock(
            self.root / ".gc.lock",
            timeout_seconds=self.config.lock_timeout_seconds,
            stale_seconds=self.config.lock_stale_seconds,
        ):
            total_size = self._total_cache_size()
            if total_size <= self.max_total_bytes:
                return

            target_size = int(self.max_total_bytes * self.target_ratio)
            candidates = [
                path
                for path in self._entry_dirs()
                if path.name not in protected_keys and not _has_active_lock(path)
            ]
            candidates.sort(key=self._last_accessed_at)

            for entry_dir in candidates:
                if total_size <= target_size:
                    break
                entry_size = _path_size(entry_dir)
                shutil.rmtree(entry_dir, ignore_errors=True)
                total_size -= entry_size

    def delete(self, cache_key: str) -> None:
        shutil.rmtree(self._entry_dir(cache_key), ignore_errors=True)

    def clear(self) -> None:
        for entry_dir in self._entry_dirs():
            if not _has_active_lock(entry_dir):
                shutil.rmtree(entry_dir, ignore_errors=True)

    def _complete_entry(
        self,
        cache_key: str,
        spec: CacheSpec,
        required_artifacts: tuple[str, ...],
        *,
        hit: bool,
    ) -> CacheEntry | None:
        entry_dir = self._entry_dir(cache_key)
        manifest_path = entry_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("cache_key") != cache_key:
            return None
        if manifest.get("spec") != spec.payload():
            return None
        if not _artifacts_complete(entry_dir, manifest, required_artifacts):
            return None
        return CacheEntry(
            cache_key=cache_key,
            root=self.root,
            path=entry_dir,
            manifest=manifest,
            hit=hit,
        )

    def _commit_build(
        self,
        *,
        cache_key: str,
        spec: CacheSpec,
        build_result: CacheBuildResult,
        entry_dir: Path,
    ) -> CacheEntry:
        if not build_result.artifacts:
            raise ResourceCacheError("Cache builder returned no artifacts")

        staged_dir = entry_dir / f".artifacts_{uuid.uuid4().hex}"
        staged_dir.mkdir(parents=True)
        artifacts: dict[str, dict[str, Any]] = {}
        used_names: set[str] = set()
        total_size = 0

        for artifact_name, source_path in build_result.artifacts.items():
            source_path = Path(source_path)
            if not source_path.is_file():
                raise ResourceCacheError(f"Missing cache artifact: {source_path}")
            destination_name = _artifact_file_name(
                artifact_name,
                source_path,
                used_names,
            )
            used_names.add(destination_name)
            destination = staged_dir / destination_name
            shutil.copy2(source_path, destination)
            size = destination.stat().st_size
            total_size += size
            artifacts[artifact_name] = {
                "path": f"artifacts/{destination_name}",
                "size_bytes": size,
                "metadata": dict(
                    build_result.artifact_metadata.get(artifact_name, {})
                ),
            }

        artifact_dir = entry_dir / "artifacts"
        old_artifact_dir = entry_dir / f".old_artifacts_{uuid.uuid4().hex}"
        if artifact_dir.exists():
            artifact_dir.replace(old_artifact_dir)
        staged_dir.replace(artifact_dir)
        shutil.rmtree(old_artifact_dir, ignore_errors=True)

        now = time.time()
        manifest = {
            "cache_key": cache_key,
            "spec": spec.payload(),
            "metadata": dict(build_result.metadata),
            "artifacts": artifacts,
            "created_at": now,
            "size_bytes": total_size,
        }
        _write_json(entry_dir / "manifest.json", manifest)
        self._touch(entry_dir)
        return CacheEntry(
            cache_key=cache_key,
            root=self.root,
            path=entry_dir,
            manifest=manifest,
            hit=False,
        )

    def _entry_from_key(self, cache_key: str) -> CacheEntry:
        entry_dir = self._entry_dir(cache_key)
        manifest = _read_json(entry_dir / "manifest.json")
        return CacheEntry(
            cache_key=cache_key,
            root=self.root,
            path=entry_dir,
            manifest=manifest,
            hit=True,
        )

    def _entry_dir(self, cache_key: str) -> Path:
        return self.root / cache_key

    @contextmanager
    def _entry_build_lock(self, entry_dir: Path):
        build_lock: _BuildLock | None = None
        with _BuildLock(
            self.root / ".gc.lock",
            timeout_seconds=self.config.lock_timeout_seconds,
            stale_seconds=self.config.lock_stale_seconds,
        ):
            entry_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            build_lock = _BuildLock(
                entry_dir / "build.lock",
                timeout_seconds=self.config.lock_timeout_seconds,
                stale_seconds=self.config.lock_stale_seconds,
            )
            build_lock.__enter__()
        try:
            yield
        finally:
            if build_lock is not None:
                build_lock.__exit__(None, None, None)

    def _entry_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [path for path in self.root.iterdir() if path.is_dir()]

    def _touch(self, entry_dir: Path) -> None:
        access_path = entry_dir / "access.json"
        access_count = 0
        if access_path.is_file():
            try:
                access_count = int(_read_json(access_path).get("access_count", 0))
            except (OSError, json.JSONDecodeError, ValueError):
                access_count = 0
        _write_json(
            access_path,
            {
                "last_accessed_at": time.time(),
                "access_count": access_count + 1,
            },
        )

    def _last_accessed_at(self, entry_dir: Path) -> float:
        access_path = entry_dir / "access.json"
        if access_path.is_file():
            try:
                return float(_read_json(access_path).get("last_accessed_at", 0.0))
            except (OSError, json.JSONDecodeError, ValueError):
                return 0.0
        return 0.0

    def _total_cache_size(self) -> int:
        return sum(
            _path_size(entry_dir)
            for entry_dir in self._entry_dirs()
            if not _has_active_lock(entry_dir)
        )


class _BuildLock:
    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        stale_seconds: float,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds

    def __enter__(self) -> "_BuildLock":
        started = time.monotonic()
        # Use a temp-file + os.link atomic rename to avoid the TOCTOU race
        # between _is_stale() and _unlink().  os.link raises FileExistsError
        # atomically if the target already exists, so only one process wins.
        temp_path = self.path.with_suffix(f".tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
        while True:
            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "pid": os.getpid(),
                            "created_at": time.time(),
                        },
                        handle,
                        ensure_ascii=False,
                    )
                os.link(str(temp_path), str(self.path))
                # We won the lock — link succeeded because self.path did not exist.
                temp_path.unlink()
                return self
            except FileExistsError:
                temp_path.unlink(missing_ok=True)
                if self._is_stale():
                    # Stale lock: try a contested unlink + immediate link to
                    # reduce the window where another process also unlinks.
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() - started >= self.timeout_seconds:
                    raise ResourceCacheError(
                        f"Timed out waiting for cache build lock: {self.path}"
                    )
                time.sleep(0.1)
                continue

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._unlink()

    def _is_stale(self) -> bool:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        return time.time() - mtime > self.stale_seconds

    def _unlink(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _configured_root(config: ResourceCacheConfig) -> Path:
    root = os.environ.get("CLOVER_RESOURCE_CACHE_ROOT") or config.root
    if root is not None:
        return Path(root).expanduser().resolve()
    uid = getattr(os, "getuid", lambda: "user")()
    return Path(tempfile.gettempdir()) / f"clover-{uid}" / "resource_cache"


def _configured_max_total_bytes(config: ResourceCacheConfig) -> int:
    env_value = os.environ.get("CLOVER_RESOURCE_CACHE_MAX_BYTES")
    if env_value:
        return int(env_value)
    if config.max_total_bytes is not None:
        return int(config.max_total_bytes)
    return DEFAULT_MAX_TOTAL_BYTES


def _artifacts_complete(
    entry_dir: Path,
    manifest: dict[str, Any],
    required_artifacts: tuple[str, ...],
) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for artifact_name in required_artifacts:
        if artifact_name not in artifacts:
            return False
    for artifact in artifacts.values():
        relative_path = artifact.get("path")
        if not relative_path:
            return False
        path = entry_dir / relative_path
        if not path.is_file():
            return False
        expected_size = artifact.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            return False
    return True


def _artifact_file_name(
    artifact_name: str,
    source_path: Path,
    used_names: set[str],
) -> str:
    candidate = _safe_component(source_path.name or artifact_name)
    if not candidate or candidate in {".", ".."}:
        candidate = _safe_component(artifact_name) or uuid.uuid4().hex
    if candidate not in used_names:
        return candidate

    stem = Path(candidate).stem or _safe_component(artifact_name) or "artifact"
    suffix = Path(candidate).suffix
    index = 2
    while True:
        indexed = f"{stem}_{index}{suffix}"
        if indexed not in used_names:
            return indexed
        index += 1


def _safe_component(value: str) -> str:
    return SAFE_COMPONENT_PATTERN.sub("_", value).strip("._")


def _has_active_lock(entry_dir: Path) -> bool:
    return (entry_dir / "build.lock").exists()


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    try:
        children = list(path.rglob("*"))
    except FileNotFoundError:
        return 0
    for child in children:
        try:
            if child.is_file():
                total += child.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def _best_effort_chmod_private(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass
