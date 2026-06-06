from __future__ import annotations

import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clover.resource import (
    CacheBuildResult,
    CacheSpec,
    ResourceCache,
    ResourceCacheConfig,
)


class ResourceCacheTest(unittest.TestCase):
    def test_get_or_build_reuses_complete_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResourceCache(ResourceCacheConfig(root=tmpdir))
            spec = CacheSpec(
                namespace="test",
                input_key="sha256:source",
                producer_key="producer",
            )
            build_count = 0

            def build(workspace: Path) -> CacheBuildResult:
                nonlocal build_count
                build_count += 1
                artifact = workspace / "artifact.txt"
                artifact.write_text("cached payload", encoding="utf-8")
                return CacheBuildResult(
                    artifacts={"payload": artifact},
                    metadata={"rows": 1},
                    artifact_metadata={"payload": {"format": "text"}},
                )

            first = cache.get_or_build(
                spec=spec,
                builder=build,
                required_artifacts=("payload",),
            )
            second = cache.get_or_build(
                spec=spec,
                builder=build,
                required_artifacts=("payload",),
            )
            resolved = cache.resolve_ref(second.artifact_ref("payload"))

        self.assertFalse(first.hit)
        self.assertTrue(second.hit)
        self.assertEqual(build_count, 1)
        self.assertEqual(second.metadata["rows"], 1)
        self.assertEqual(Path(resolved["path"]).name, "artifact.txt")

    def test_gc_lru_evicts_oldest_entry_when_over_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResourceCache(
                ResourceCacheConfig(
                    root=tmpdir,
                    max_total_bytes=6000,
                    target_ratio=0.5,
                )
            )
            first = cache.get_or_build(
                spec=CacheSpec(
                    namespace="test",
                    input_key="sha256:first",
                    producer_key="producer",
                ),
                builder=_payload_builder("a" * 3000),
                required_artifacts=("payload",),
            )
            time.sleep(0.01)
            second = cache.get_or_build(
                spec=CacheSpec(
                    namespace="test",
                    input_key="sha256:second",
                    producer_key="producer",
                ),
                builder=_payload_builder("b" * 3000),
                required_artifacts=("payload",),
            )

            first_exists = first.path.exists()
            second_exists = second.path.exists()

        self.assertFalse(first_exists)
        self.assertTrue(second_exists)

    def test_concurrent_builds_and_gc_do_not_remove_active_entry_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ResourceCacheConfig(
                root=tmpdir,
                max_total_bytes=1200,
                target_ratio=0.5,
            )

            def build_one(index: int) -> str:
                cache = ResourceCache(config)
                entry = cache.get_or_build(
                    spec=CacheSpec(
                        namespace="test",
                        input_key=f"sha256:item-{index}",
                        producer_key="producer",
                    ),
                    builder=_payload_builder(str(index) * 800),
                    required_artifacts=("payload",),
                )
                return entry.cache_key

            def gc_one() -> None:
                ResourceCache(config).gc_lru()

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = []
                for index in range(24):
                    futures.append(executor.submit(build_one, index))
                    futures.append(executor.submit(gc_one))
                for future in futures:
                    future.result()


def _payload_builder(payload: str):
    def build(workspace: Path) -> CacheBuildResult:
        artifact = workspace / "payload.txt"
        artifact.write_text(payload, encoding="utf-8")
        return CacheBuildResult(artifacts={"payload": artifact})

    return build
