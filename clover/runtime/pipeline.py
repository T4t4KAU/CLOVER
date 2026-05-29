"""Generic synchronous pipeline helpers used by task-specific runtimes."""

from __future__ import annotations

import heapq
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, TypeVar


T = TypeVar("T")


@dataclass
class StageProfile:
    """Aggregated timing and throughput for one pipeline stage."""

    calls: int = 0
    items: int = 0
    total_seconds: float = 0.0

    @property
    def average_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "items": self.items,
            "total_seconds": self.total_seconds,
            "average_seconds": self.average_seconds,
        }


@dataclass
class PipelineProfiler:
    """Small profiler for synchronous pipeline stages."""

    stages: dict[str, StageProfile] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage_name: str, *, items: int = 0) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            stage = self.stages.setdefault(stage_name, StageProfile())
            stage.calls += 1
            stage.items += items
            stage.total_seconds += elapsed

    def increment(self, counter_name: str, amount: int = 1) -> None:
        self.counters[counter_name] = self.counters.get(counter_name, 0) + amount

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": {
                name: profile.to_dict()
                for name, profile in sorted(self.stages.items())
            },
            "counters": dict(sorted(self.counters.items())),
        }


class GroupedPriorityQueue(Generic[T]):
    """Priority queue partitioned by group key.

    Items are popped from the group whose next item has the lowest priority.
    Within the same priority, insertion order is preserved.
    """

    def __init__(self) -> None:
        self._groups: dict[str, list[tuple[int, int, T]]] = {}
        self._sequence = 0
        self._size = 0

    def push(self, group_key: str, item: T, *, priority: int = 0) -> None:
        heap = self._groups.setdefault(group_key, [])
        heapq.heappush(heap, (priority, self._sequence, item))
        self._sequence += 1
        self._size += 1

    def pop_best_group(self, max_items: int) -> tuple[str, list[T]] | None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        group_key = self._best_group_key()
        if group_key is None:
            return None
        priority = self._groups[group_key][0][0]
        return group_key, self.pop_many(group_key, max_items, priority=priority)

    def pop_many(
        self,
        group_key: str,
        max_items: int,
        *,
        priority: int | None = None,
    ) -> list[T]:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        heap = self._groups.get(group_key)
        if not heap:
            return []
        items = []
        while heap and len(items) < max_items:
            if priority is not None and heap[0][0] != priority:
                break
            _, _, item = heapq.heappop(heap)
            items.append(item)
            self._size -= 1
        if not heap:
            self._groups.pop(group_key, None)
        return items

    def __bool__(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size

    def _best_group_key(self) -> str | None:
        best_key: str | None = None
        best_head: tuple[int, int] | None = None
        for group_key, heap in self._groups.items():
            if not heap:
                continue
            head = (heap[0][0], heap[0][1])
            if best_head is None or head < best_head:
                best_key = group_key
                best_head = head
        return best_key


@dataclass(frozen=True)
class CaseResult:
    """Final per-case output emitted by a CLOVER runtime."""

    case_id: str
    answer_key: str
    status: str
    answer: Any = None
    error: dict[str, Any] | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "answer_key": self.answer_key,
            "status": self.status,
            "ok": self.ok,
            "answer": self.answer,
            "error": self.error,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }
