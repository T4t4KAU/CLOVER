"""Prompt-template scheduling primitives for local SLM jobs.

This module is intentionally independent from the executor's physical DAG
scheduler. It only indexes ready SLM jobs by their stable prompt-template leaf
and returns jobs in a prefix-friendly leaf order.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field
from typing import Any, Iterable


TemplateLeafKey = tuple[str, ...]


@dataclass(frozen=True)
class TemplateLeafSpec:
    """One stable prompt-template leaf used by SLM scheduling."""

    key: TemplateLeafKey
    template_paths: tuple[str, ...] = ()
    description: str = ""
    static_token_count: int = 0


@dataclass(frozen=True)
class SlmJob:
    """One ready SLM job visible to the prompt-template scheduler."""

    job_id: str
    leaf_key: TemplateLeafKey
    prompt_len: int = 0
    payload_len: int = 0
    prefix_signature: str = ""
    prefix_token_count: int = 0
    payload: Any = None

    @property
    def locality_key(self) -> tuple[str, int, int, str]:
        """Sort key for jobs that share one template leaf."""

        return (
            self.prefix_signature,
            max(0, self.prompt_len),
            max(0, self.payload_len),
            self.job_id,
        )


@dataclass
class _TemplateNode:
    name: str
    children: dict[str, "_TemplateNode"] = field(default_factory=dict)
    leaf_spec: TemplateLeafSpec | None = None
    prev_leaf: int | None = None
    next_leaf: int | None = None


@dataclass
class _LeafQueue:
    rank: int
    spec: TemplateLeafSpec
    items: list[tuple[tuple[str, int, int, str], SlmJob]] = field(default_factory=list)

    def push(self, job: SlmJob) -> None:
        insort(self.items, (job.locality_key, job))

    def pop(self) -> SlmJob:
        return self.items.pop(0)[1]

    def __bool__(self) -> bool:
        return bool(self.items)


class ThreadedPrefixTemplateQueue:
    """Dynamic per-epoch leaf queues over one shared static template tree."""

    def __init__(
        self,
        leaf_specs: tuple[TemplateLeafSpec, ...],
        leaf_ranks_by_key: dict[TemplateLeafKey, int],
    ) -> None:
        self._leaf_specs = leaf_specs
        self._leaf_ranks_by_key = leaf_ranks_by_key
        self._queues = [
            _LeafQueue(rank=rank, spec=spec)
            for rank, spec in enumerate(self._leaf_specs)
        ]
        self._non_empty_ranks: list[int] = []
        self._cursor: int | None = None
        self._direction = 1

    @property
    def leaf_count(self) -> int:
        return len(self._leaf_specs)

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def cursor(self) -> int | None:
        return self._cursor

    def leaf_rank(self, leaf_key: TemplateLeafKey) -> int:
        try:
            return self._leaf_ranks_by_key[tuple(leaf_key)]
        except KeyError as exc:
            raise KeyError(f"Unknown SLM template leaf: {leaf_key!r}") from exc

    def leaf_key(self, rank: int) -> TemplateLeafKey:
        return self._leaf_specs[rank].key

    def leaf_keys(self) -> tuple[TemplateLeafKey, ...]:
        return tuple(spec.key for spec in self._leaf_specs)

    def thread_order(self) -> tuple[TemplateLeafKey, ...]:
        """Return leaf keys in DFS thread order."""

        return self.leaf_keys()

    def submit(self, job: SlmJob) -> None:
        """Insert one ready SLM job into its leaf-local queue."""

        rank = self.leaf_rank(job.leaf_key)
        was_empty = not self._queues[rank]
        self._queues[rank].push(job)
        if was_empty:
            insort(self._non_empty_ranks, rank)
            if self._cursor is None:
                self._cursor = rank

    def pop_initial(self) -> SlmJob | None:
        """Pop from the first non-empty leaf in thread order."""

        if not self._non_empty_ranks:
            return None
        return self._pop_leaf(self._non_empty_ranks[0])

    def pop_near(
        self,
        *,
        anchor_leaf_key: TemplateLeafKey | None = None,
        anchor_rank: int | None = None,
    ) -> SlmJob | None:
        """Pop a job near the anchor leaf using elevator-style refill."""

        if not self._non_empty_ranks:
            return None
        rank = self._resolve_anchor(anchor_leaf_key, anchor_rank)
        if rank is not None and self._queues[rank]:
            return self._pop_leaf(rank)

        if rank is None:
            return self.pop_initial()

        if self._direction >= 0:
            next_rank = self._next_non_empty_after(rank)
            if next_rank is not None:
                return self._pop_leaf(next_rank)
            self._direction = -1
            prev_rank = self._prev_non_empty_before(rank)
            if prev_rank is not None:
                return self._pop_leaf(prev_rank)
        else:
            prev_rank = self._prev_non_empty_before(rank)
            if prev_rank is not None:
                return self._pop_leaf(prev_rank)
            self._direction = 1
            next_rank = self._next_non_empty_after(rank)
            if next_rank is not None:
                return self._pop_leaf(next_rank)
        return None

    def fill_initial(self, max_jobs: int) -> list[SlmJob]:
        """Fill an initial micro-batch from the leftmost ready leaves."""

        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        jobs: list[SlmJob] = []
        while len(jobs) < max_jobs:
            job = self.pop_initial() if not jobs else self.pop_near()
            if job is None:
                break
            jobs.append(job)
        return jobs

    def refill_after(self, done_job: SlmJob, *, max_jobs: int = 1) -> list[SlmJob]:
        """Fill freed SLM slots anchored at the just-completed job."""

        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        anchor_rank = self.leaf_rank(done_job.leaf_key)
        jobs: list[SlmJob] = []
        for _ in range(max_jobs):
            job = self.pop_near(anchor_rank=anchor_rank if not jobs else None)
            if job is None:
                break
            jobs.append(job)
            anchor_rank = self.leaf_rank(job.leaf_key)
        return jobs

    def _resolve_anchor(
        self,
        anchor_leaf_key: TemplateLeafKey | None,
        anchor_rank: int | None,
    ) -> int | None:
        if anchor_leaf_key is not None and anchor_rank is not None:
            raise ValueError("Use either anchor_leaf_key or anchor_rank, not both")
        if anchor_leaf_key is not None:
            return self.leaf_rank(anchor_leaf_key)
        if anchor_rank is not None:
            if anchor_rank < 0 or anchor_rank >= self.leaf_count:
                raise IndexError(f"Anchor rank out of range: {anchor_rank}")
            return anchor_rank
        return self._cursor

    def _pop_leaf(self, rank: int) -> SlmJob:
        job = self._queues[rank].pop()
        if not self._queues[rank]:
            index = bisect_left(self._non_empty_ranks, rank)
            if index < len(self._non_empty_ranks) and self._non_empty_ranks[index] == rank:
                self._non_empty_ranks.pop(index)
        self._cursor = rank
        return job

    def _next_non_empty_after(self, rank: int) -> int | None:
        index = bisect_right(self._non_empty_ranks, rank)
        if index >= len(self._non_empty_ranks):
            return None
        return self._non_empty_ranks[index]

    def _prev_non_empty_before(self, rank: int) -> int | None:
        index = bisect_left(self._non_empty_ranks, rank) - 1
        if index < 0:
            return None
        return self._non_empty_ranks[index]


class ThreadedPrefixTemplateTree:
    """Static prompt-template tree with DFS-threaded leaves.

    Internal nodes encode stable prompt-template layers. Leaf nodes store ready
    SLM jobs that share the same stable template path. The leaf thread is the
    left-to-right DFS order over registered leaves.
    """

    def __init__(self, leaf_specs: Iterable[TemplateLeafSpec]) -> None:
        specs = list(leaf_specs)
        if not specs:
            raise ValueError("ThreadedPrefixTemplateTree requires at least one leaf")
        self._root = _TemplateNode("root")
        self._leaf_specs: list[TemplateLeafSpec] = []
        self._leaf_ranks_by_key: dict[TemplateLeafKey, int] = {}
        for spec in specs:
            self._insert_leaf_spec(spec)
        self._thread_leaves()
        self._queues = [
            _LeafQueue(rank=rank, spec=spec)
            for rank, spec in enumerate(self._leaf_specs)
        ]
        self._non_empty_ranks: list[int] = []
        self._cursor: int | None = None
        self._direction = 1

    @property
    def leaf_count(self) -> int:
        return len(self._leaf_specs)

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def cursor(self) -> int | None:
        return self._cursor

    def new_queue(self) -> ThreadedPrefixTemplateQueue:
        """Return an empty dynamic queue over this tree's static leaf thread."""

        return ThreadedPrefixTemplateQueue(
            tuple(self._leaf_specs),
            self._leaf_ranks_by_key,
        )

    def leaf_rank(self, leaf_key: TemplateLeafKey) -> int:
        try:
            return self._leaf_ranks_by_key[tuple(leaf_key)]
        except KeyError as exc:
            raise KeyError(f"Unknown SLM template leaf: {leaf_key!r}") from exc

    def leaf_key(self, rank: int) -> TemplateLeafKey:
        return self._leaf_specs[rank].key

    def leaf_keys(self) -> tuple[TemplateLeafKey, ...]:
        return tuple(spec.key for spec in self._leaf_specs)

    def thread_order(self) -> tuple[TemplateLeafKey, ...]:
        """Return leaf keys in DFS thread order."""

        return self.leaf_keys()

    def submit(self, job: SlmJob) -> None:
        """Insert one ready SLM job into its leaf-local queue."""

        rank = self.leaf_rank(job.leaf_key)
        was_empty = not self._queues[rank]
        self._queues[rank].push(job)
        if was_empty:
            insort(self._non_empty_ranks, rank)
            if self._cursor is None:
                self._cursor = rank

    def pop_initial(self) -> SlmJob | None:
        """Pop from the first non-empty leaf in thread order."""

        if not self._non_empty_ranks:
            return None
        return self._pop_leaf(self._non_empty_ranks[0])

    def pop_near(
        self,
        *,
        anchor_leaf_key: TemplateLeafKey | None = None,
        anchor_rank: int | None = None,
    ) -> SlmJob | None:
        """Pop a job near the anchor leaf using elevator-style refill."""

        if not self._non_empty_ranks:
            return None
        rank = self._resolve_anchor(anchor_leaf_key, anchor_rank)
        if rank is not None and self._queues[rank]:
            return self._pop_leaf(rank)

        if rank is None:
            return self.pop_initial()

        if self._direction >= 0:
            next_rank = self._next_non_empty_after(rank)
            if next_rank is not None:
                return self._pop_leaf(next_rank)
            self._direction = -1
            prev_rank = self._prev_non_empty_before(rank)
            if prev_rank is not None:
                return self._pop_leaf(prev_rank)
        else:
            prev_rank = self._prev_non_empty_before(rank)
            if prev_rank is not None:
                return self._pop_leaf(prev_rank)
            self._direction = 1
            next_rank = self._next_non_empty_after(rank)
            if next_rank is not None:
                return self._pop_leaf(next_rank)
        return None

    def fill_initial(self, max_jobs: int) -> list[SlmJob]:
        """Fill an initial micro-batch from the leftmost ready leaves."""

        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        jobs: list[SlmJob] = []
        while len(jobs) < max_jobs:
            job = self.pop_initial() if not jobs else self.pop_near()
            if job is None:
                break
            jobs.append(job)
        return jobs

    def refill_after(self, done_job: SlmJob, *, max_jobs: int = 1) -> list[SlmJob]:
        """Fill freed SLM slots anchored at the just-completed job."""

        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        anchor_rank = self.leaf_rank(done_job.leaf_key)
        jobs: list[SlmJob] = []
        for _ in range(max_jobs):
            job = self.pop_near(anchor_rank=anchor_rank if not jobs else None)
            if job is None:
                break
            jobs.append(job)
            anchor_rank = self.leaf_rank(job.leaf_key)
        return jobs

    def _insert_leaf_spec(self, spec: TemplateLeafSpec) -> None:
        key = tuple(spec.key)
        if not key:
            raise ValueError("Template leaf key cannot be empty")
        if key in self._leaf_ranks_by_key:
            raise ValueError(f"Duplicate template leaf key: {key!r}")
        node = self._root
        for layer in key:
            if not layer:
                raise ValueError(f"Template leaf key contains empty layer: {key!r}")
            node = node.children.setdefault(layer, _TemplateNode(layer))
        if node.leaf_spec is not None:
            raise ValueError(f"Duplicate template leaf key: {key!r}")
        node.leaf_spec = TemplateLeafSpec(
            key=key,
            template_paths=tuple(spec.template_paths),
            description=spec.description,
            static_token_count=spec.static_token_count,
        )

    def _thread_leaves(self) -> None:
        leaves: list[tuple[TemplateLeafKey, TemplateLeafSpec, _TemplateNode]] = []
        self._collect_leaves(self._root, (), leaves)
        previous_node: _TemplateNode | None = None
        for rank, (key, spec, node) in enumerate(leaves):
            if previous_node is not None:
                previous_node.next_leaf = rank
                node.prev_leaf = rank - 1
            self._leaf_ranks_by_key[key] = rank
            self._leaf_specs.append(spec)
            previous_node = node

    def _collect_leaves(
        self,
        node: _TemplateNode,
        path: TemplateLeafKey,
        out: list[tuple[TemplateLeafKey, TemplateLeafSpec, _TemplateNode]],
    ) -> None:
        if node.leaf_spec is not None:
            out.append((path, node.leaf_spec, node))
        for name, child in node.children.items():
            self._collect_leaves(child, path + (name,), out)

    def _resolve_anchor(
        self,
        anchor_leaf_key: TemplateLeafKey | None,
        anchor_rank: int | None,
    ) -> int | None:
        if anchor_leaf_key is not None and anchor_rank is not None:
            raise ValueError("Use either anchor_leaf_key or anchor_rank, not both")
        if anchor_leaf_key is not None:
            return self.leaf_rank(anchor_leaf_key)
        if anchor_rank is not None:
            if anchor_rank < 0 or anchor_rank >= self.leaf_count:
                raise IndexError(f"Anchor rank out of range: {anchor_rank}")
            return anchor_rank
        return self._cursor

    def _pop_leaf(self, rank: int) -> SlmJob:
        job = self._queues[rank].pop()
        if not self._queues[rank]:
            index = bisect_left(self._non_empty_ranks, rank)
            if index < len(self._non_empty_ranks) and self._non_empty_ranks[index] == rank:
                self._non_empty_ranks.pop(index)
        self._cursor = rank
        return job

    def _next_non_empty_after(self, rank: int) -> int | None:
        index = bisect_right(self._non_empty_ranks, rank)
        if index >= len(self._non_empty_ranks):
            return None
        return self._non_empty_ranks[index]

    def _prev_non_empty_before(self, rank: int) -> int | None:
        index = bisect_left(self._non_empty_ranks, rank) - 1
        if index < 0:
            return None
        return self._non_empty_ranks[index]
