"""Generic synchronous pipeline helpers used by task-specific runtimes."""

from __future__ import annotations

import heapq
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, TypeVar

P = TypeVar("P")
R = TypeVar("R")
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

    def record(self, stage_name: str, *, items: int = 0, elapsed: float = 0.0) -> None:
        stage = self.stages.setdefault(stage_name, StageProfile())
        stage.calls += 1
        stage.items += items
        stage.total_seconds += elapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": {
                name: profile.to_dict()
                for name, profile in sorted(self.stages.items())
            },
            "counters": dict(sorted(self.counters.items())),
        }


@dataclass(frozen=True)
class InflightCallResult(Generic[R]):
    """Completed asynchronous pipeline call."""

    value: R | None
    elapsed: float
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class InflightJob(Generic[P, R]):
    """One asynchronous stage call plus task-specific payload."""

    payload: P
    items: int
    future: Future[InflightCallResult[R]]


class InflightStage(Generic[P, R]):
    """Small generic inflight queue for overlapped runtime pipeline stages.

    The stage owns only scheduling mechanics. Task-specific code decides what a
    payload means, how to build the callable, and how to consume the result.
    """

    def __init__(
        self,
        *,
        stage_name: str,
        max_workers: int,
        profiler: PipelineProfiler | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.stage_name = stage_name
        self.max_workers = max_workers
        self.profiler = profiler
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: list[InflightJob[P, R]] = []

    @property
    def has_capacity(self) -> bool:
        return len(self._jobs) < self.max_workers

    def submit(
        self,
        payload: P,
        call: Callable[[], R],
        *,
        items: int = 0,
    ) -> None:
        if not self.has_capacity:
            raise ValueError(f"{self.stage_name} inflight stage is full")
        future = self._executor.submit(_timed_call, call)
        self._jobs.append(InflightJob(payload=payload, items=items, future=future))

    def drain_ready(
        self,
        handler: Callable[[P, InflightCallResult[R]], None],
        *,
        wait_for_one: bool = False,
    ) -> int:
        if not self._jobs:
            return 0
        done_futures = {job.future for job in self._jobs if job.future.done()}
        if not done_futures and wait_for_one:
            done_futures, _ = wait(
                [job.future for job in self._jobs],
                return_when=FIRST_COMPLETED,
            )
        if not done_futures:
            return 0

        ready: list[InflightJob[P, R]] = []
        remaining: list[InflightJob[P, R]] = []
        for job in self._jobs:
            if job.future in done_futures:
                ready.append(job)
            else:
                remaining.append(job)
        self._jobs = remaining

        for job in ready:
            try:
                result = job.future.result()
            except BaseException as exc:  # noqa: BLE001 - normalize stage failure.
                result = InflightCallResult(value=None, elapsed=0.0, error=exc)
            if self.profiler is not None:
                self.profiler.record(
                    self.stage_name,
                    items=job.items,
                    elapsed=result.elapsed,
                )
            handler(job.payload, result)
        return len(ready)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "InflightStage[P, R]":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __bool__(self) -> bool:
        return bool(self._jobs)

    def __len__(self) -> int:
        return len(self._jobs)


def _timed_call(call: Callable[[], R]) -> InflightCallResult[R]:
    started = time.perf_counter()
    try:
        value = call()
    except BaseException as exc:  # noqa: BLE001 - propagated as stage result.
        return InflightCallResult(
            value=None,
            elapsed=time.perf_counter() - started,
            error=exc,
        )
    return InflightCallResult(
        value=value,
        elapsed=time.perf_counter() - started,
    )


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
