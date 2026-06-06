"""Sequence-level dispatcher for local SLM requests."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import itertools
import threading
import time
from typing import Any

from clover.executor.agents.template_tree import build_slm_template_scheduler_tree
from clover.executor.local_slm import generate_slm_text
from clover.executor.slm_scheduler import SlmJob, TemplateLeafKey
from clover.executor.token_count import (
    configured_tokenizer_name,
    count_tokens,
    prefix_signature,
)
from clover.supervisor.client import RemoteLLMResult


SLM_SCHEDULER_TPTT = "tptt"
SLM_SCHEDULER_FIFO = "fifo"
SLM_SCHEDULER_CHOICES = frozenset({SLM_SCHEDULER_TPTT, SLM_SCHEDULER_FIFO})
DEFAULT_MAX_PARALLEL_SLM_SEQUENCES = 8
DEFAULT_MAX_PENDING_SLM_SEQUENCES = 1024
DEFAULT_MAX_TPTT_LEAF_SEQUENCES_PER_TREE = 64
DEFAULT_TPTT_COALESCE_MS = 5.0
DEFAULT_TPTT_PREFIX_TOKENS = 64


@dataclass(frozen=True)
class LocalSlmSequenceRequest:
    """One local SLM sequence produced by an Agent Loop iteration."""

    prompt: str
    leaf_key: TemplateLeafKey
    prompt_kind: str
    sequence_id: str | None = None
    node_id: str | None = None
    job_id: str | None = None
    iteration: int | None = None
    payload_len: int = 0
    prompt_len: int = 0
    prefix_signature: str = ""
    prefix_token_count: int = 0
    slm_config: dict[str, Any] | None = None
    client: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalSlmSequenceResult:
    """Completed local SLM sequence plus scheduling trace metadata."""

    sequence_id: str
    request: LocalSlmSequenceRequest
    llm_result: RemoteLLMResult
    queue_wait_ms: float
    inference_ms: float

    @property
    def text(self) -> str:
        return self.llm_result.text

    @property
    def response_payload(self) -> dict[str, Any]:
        return self.llm_result.response_payload

    @property
    def response_id(self) -> str | None:
        return self.llm_result.response_id

    def trace_metadata(self) -> dict[str, Any]:
        payload = {
            "sequence_id": self.sequence_id,
            "leaf_key": list(self.request.leaf_key),
            "prompt_kind": self.request.prompt_kind,
            "queue_wait_ms": self.queue_wait_ms,
            "inference_ms": self.inference_ms,
            "prompt_len": self.request.prompt_len,
            "payload_len": self.request.payload_len,
            "prefix_signature": self.request.prefix_signature,
            "prefix_token_count": self.request.prefix_token_count,
        }
        scheduler = self.request.metadata.get("slm_scheduler")
        if scheduler:
            payload["slm_scheduler"] = scheduler
        tptt_epoch = self.request.metadata.get("tptt_epoch")
        if tptt_epoch is not None:
            payload["tptt_epoch"] = tptt_epoch
        if self.request.node_id:
            payload["node_id"] = self.request.node_id
        if self.request.job_id:
            payload["job_id"] = self.request.job_id
        if self.request.iteration is not None:
            payload["iteration"] = self.request.iteration
        return payload


@dataclass
class _QueuedSequence:
    request: LocalSlmSequenceRequest
    sequence_id: str
    future: Future[LocalSlmSequenceResult]
    submitted_at: float


@dataclass
class _TpttEpoch:
    epoch_id: int
    tree: Any
    sequence_count: int = 0
    pending_count: int = 0
    sealed: bool = False
    first_submitted_at: float | None = None


class LocalSlmSequenceDispatcher:
    """Schedule local SLM sequences with TPTT or FIFO policy."""

    def __init__(
        self,
        *,
        slm_config: dict[str, Any] | None = None,
        client: Any | None = None,
        max_parallel_sequences: int = DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
        max_pending_sequences: int = DEFAULT_MAX_PENDING_SLM_SEQUENCES,
        tokenizer_name: str | None = None,
        slm_scheduler: str | None = None,
        max_tptt_leaf_sequences_per_tree: int | None = None,
        tptt_coalesce_ms: float | None = None,
        tptt_prefix_tokens: int | None = None,
    ) -> None:
        if max_parallel_sequences <= 0:
            raise ValueError("max_parallel_slm_sequences must be positive")
        if max_pending_sequences <= 0:
            raise ValueError("max_pending_slm_sequences must be positive")
        self._slm_config = slm_config
        self._client = client
        self._max_parallel_sequences = int(max_parallel_sequences)
        self._max_pending_sequences = int(max_pending_sequences)
        self._slm_scheduler = _resolve_slm_scheduler(
            explicit=slm_scheduler,
            slm_config=slm_config,
        )
        self._uses_tptt = self._slm_scheduler == SLM_SCHEDULER_TPTT
        self._tokenizer_name = tokenizer_name or configured_tokenizer_name(slm_config)
        self._leaf_validator = build_slm_template_scheduler_tree(
            tokenizer_name=self._tokenizer_name
        )
        self._max_tptt_leaf_sequences_per_tree = _resolve_positive_int(
            explicit=max_tptt_leaf_sequences_per_tree,
            slm_config=slm_config,
            key="max_tptt_leaf_sequences_per_tree",
            default=DEFAULT_MAX_TPTT_LEAF_SEQUENCES_PER_TREE,
        )
        self._tptt_coalesce_seconds = _resolve_nonnegative_float(
            explicit=tptt_coalesce_ms,
            slm_config=slm_config,
            key="tptt_coalesce_ms",
            default=DEFAULT_TPTT_COALESCE_MS,
        ) / 1000.0
        self._tptt_prefix_tokens = _resolve_positive_int(
            explicit=tptt_prefix_tokens,
            slm_config=slm_config,
            key="tptt_prefix_tokens",
            default=DEFAULT_TPTT_PREFIX_TOKENS,
        )
        self._tptt_epochs: deque[_TpttEpoch] = deque()
        self._tptt_epoch_counter = itertools.count(1)
        self._active_tptt_epoch = self._new_tptt_epoch()
        self._fifo: deque[SlmJob] = deque()
        self._pool = ThreadPoolExecutor(max_workers=self._max_parallel_sequences)
        self._condition = threading.Condition()
        self._pending_count = 0
        self._inflight_count = 0
        self._closed = False
        self._sequence_counter = itertools.count(1)

    def generate(self, request: LocalSlmSequenceRequest) -> LocalSlmSequenceResult:
        """Submit a sequence and wait synchronously for its local SLM result."""

        queued = self._enqueue(request)
        self._dispatch_available()
        return queued.future.result()

    def close(self, *, wait: bool = True) -> None:
        cancelled: list[SlmJob] = []
        with self._condition:
            self._closed = True
            if self._uses_tptt:
                while self._tptt_epochs:
                    epoch = self._tptt_epochs.popleft()
                    while True:
                        job = epoch.tree.pop_initial()
                        if job is None:
                            break
                        self._pending_count -= 1
                        epoch.pending_count -= 1
                        cancelled.append(job)
            else:
                while self._fifo:
                    cancelled.append(self._fifo.popleft())
                    self._pending_count -= 1
            self._condition.notify_all()
        for job in cancelled:
            queued: _QueuedSequence = job.payload
            queued.future.set_exception(
                RuntimeError("Local SLM sequence dispatcher is closed")
            )
        self._pool.shutdown(wait=wait, cancel_futures=not wait)

    def _enqueue(self, request: LocalSlmSequenceRequest) -> _QueuedSequence:
        leaf_key = tuple(request.leaf_key)
        # Reject unregistered leaves before occupying pending capacity.
        self._leaf_validator.leaf_rank(leaf_key)
        request = self._normalize_request(request, leaf_key=leaf_key)
        sequence_id = request.sequence_id or self._next_sequence_id(request)
        queued = _QueuedSequence(
            request=request,
            sequence_id=sequence_id,
            future=Future(),
            submitted_at=time.perf_counter(),
        )
        job = SlmJob(
            job_id=sequence_id,
            leaf_key=leaf_key,
            prompt_len=request.prompt_len,
            payload_len=request.payload_len,
            prefix_signature=request.prefix_signature,
            prefix_token_count=request.prefix_token_count,
            payload=queued,
        )
        with self._condition:
            while (
                not self._closed
                and self._pending_count >= self._max_pending_sequences
            ):
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Local SLM sequence dispatcher is closed")
            if self._uses_tptt:
                epoch = self._tptt_epoch_for_submit()
                request.metadata["slm_scheduler"] = SLM_SCHEDULER_TPTT
                request.metadata["tptt_epoch"] = epoch.epoch_id
                epoch.tree.submit(job)
                if epoch.pending_count == 0:
                    epoch.first_submitted_at = time.perf_counter()
                    self._tptt_epochs.append(epoch)
                epoch.pending_count += 1
                epoch.sequence_count += 1
                if epoch.sequence_count >= self._max_tptt_leaf_sequences_per_tree:
                    epoch.sealed = True
            else:
                request.metadata["slm_scheduler"] = SLM_SCHEDULER_FIFO
                self._fifo.append(job)
            self._pending_count += 1
            self._condition.notify_all()
        return queued

    def _normalize_request(
        self,
        request: LocalSlmSequenceRequest,
        *,
        leaf_key: TemplateLeafKey,
    ) -> LocalSlmSequenceRequest:
        prompt_len = request.prompt_len
        if prompt_len <= 0:
            prompt_len = count_tokens(
                request.prompt,
                tokenizer_name=self._tokenizer_name,
            )
        payload_len = request.payload_len if request.payload_len > 0 else prompt_len
        prefix_id = request.prefix_signature
        prefix_token_count = request.prefix_token_count
        if not prefix_id:
            prefix_id, prefix_token_count = prefix_signature(
                request.prompt,
                tokenizer_name=self._tokenizer_name,
                prefix_tokens=self._tptt_prefix_tokens,
            )
        return LocalSlmSequenceRequest(
            prompt=request.prompt,
            leaf_key=leaf_key,
            prompt_kind=request.prompt_kind,
            sequence_id=request.sequence_id,
            node_id=request.node_id,
            job_id=request.job_id,
            iteration=request.iteration,
            payload_len=payload_len,
            prompt_len=prompt_len,
            prefix_signature=prefix_id,
            prefix_token_count=prefix_token_count,
            slm_config=request.slm_config,
            client=request.client,
            metadata=dict(request.metadata),
        )

    def _dispatch_available(self, *, anchor_job: SlmJob | None = None) -> None:
        jobs: list[SlmJob] = []
        with self._condition:
            while (
                not self._closed
                and self._inflight_count < self._max_parallel_sequences
            ):
                if anchor_job is not None and not jobs:
                    if self._uses_tptt:
                        job = self._pop_tptt_job(anchor_job=anchor_job)
                    else:
                        job = self._pop_fifo()
                else:
                    if self._uses_tptt:
                        job = self._pop_tptt_job(pop_near=bool(jobs))
                    else:
                        job = self._pop_fifo()
                if job is None:
                    wait_seconds = (
                        self._tptt_coalesce_wait_seconds()
                        if self._uses_tptt and not jobs
                        else 0.0
                    )
                    if wait_seconds > 0.0:
                        self._condition.wait(timeout=wait_seconds)
                        continue
                    break
                self._pending_count -= 1
                self._inflight_count += 1
                jobs.append(job)
        for job in jobs:
            self._pool.submit(self._run_job, job)

    def _run_job(self, job: SlmJob) -> None:
        queued: _QueuedSequence = job.payload
        started = time.perf_counter()
        queue_wait_ms = (started - queued.submitted_at) * 1000.0
        try:
            llm_started = time.perf_counter()
            llm_result = generate_slm_text(
                queued.request.prompt,
                slm_config=queued.request.slm_config or self._slm_config,
                client=queued.request.client if queued.request.client is not None else self._client,
            )
            inference_ms = (time.perf_counter() - llm_started) * 1000.0
            queued.future.set_result(
                LocalSlmSequenceResult(
                    sequence_id=queued.sequence_id,
                    request=queued.request,
                    llm_result=llm_result,
                    queue_wait_ms=queue_wait_ms,
                    inference_ms=inference_ms,
                )
            )
        except Exception as exc:  # noqa: BLE001 - propagate through caller's future.
            queued.future.set_exception(exc)
        finally:
            with self._condition:
                self._inflight_count -= 1
                self._condition.notify_all()
            self._dispatch_available(anchor_job=job)

    def _next_sequence_id(self, request: LocalSlmSequenceRequest) -> str:
        if request.job_id:
            prefix = request.job_id
        elif request.node_id:
            prefix = request.node_id
        else:
            prefix = "local_slm"
        iteration = f"__i{request.iteration}" if request.iteration is not None else ""
        return f"{prefix}{iteration}__seq{next(self._sequence_counter)}"

    def _pop_fifo(self) -> SlmJob | None:
        if not self._fifo:
            return None
        return self._fifo.popleft()

    def _new_tptt_epoch(self) -> _TpttEpoch:
        return _TpttEpoch(
            epoch_id=next(self._tptt_epoch_counter),
            tree=self._leaf_validator.new_queue(),
        )

    def _tptt_epoch_for_submit(self) -> _TpttEpoch:
        if self._active_tptt_epoch.sequence_count >= self._max_tptt_leaf_sequences_per_tree:
            self._active_tptt_epoch.sealed = True
            self._active_tptt_epoch = self._new_tptt_epoch()
        return self._active_tptt_epoch

    def _pop_tptt_job(
        self,
        *,
        anchor_job: SlmJob | None = None,
        pop_near: bool = False,
    ) -> SlmJob | None:
        epoch = self._ready_tptt_epoch()
        if epoch is None:
            return None
        anchor_epoch = _job_tptt_epoch(anchor_job) if anchor_job is not None else None
        if anchor_job is not None and anchor_epoch == epoch.epoch_id:
            refill = epoch.tree.refill_after(anchor_job, max_jobs=1)
            job = refill[0] if refill else None
        elif pop_near:
            job = epoch.tree.pop_near()
        else:
            job = epoch.tree.pop_initial()
        if job is None:
            return None
        epoch.pending_count -= 1
        if epoch.pending_count <= 0:
            epoch.first_submitted_at = None
            self._discard_empty_tptt_epochs()
        return job

    def _ready_tptt_epoch(self) -> _TpttEpoch | None:
        self._discard_empty_tptt_epochs()
        if not self._tptt_epochs:
            return None
        epoch = self._tptt_epochs[0]
        if self._tptt_epoch_ready(epoch):
            return epoch
        return None

    def _tptt_epoch_ready(self, epoch: _TpttEpoch) -> bool:
        if epoch.sealed:
            return True
        if self._tptt_coalesce_seconds <= 0.0:
            return True
        if epoch.first_submitted_at is None:
            return True
        return (time.perf_counter() - epoch.first_submitted_at) >= self._tptt_coalesce_seconds

    def _tptt_coalesce_wait_seconds(self) -> float:
        self._discard_empty_tptt_epochs()
        if not self._tptt_epochs:
            return 0.0
        epoch = self._tptt_epochs[0]
        if self._tptt_epoch_ready(epoch):
            return 0.0
        if epoch.first_submitted_at is None:
            return 0.0
        deadline = epoch.first_submitted_at + self._tptt_coalesce_seconds
        return max(0.0, deadline - time.perf_counter())

    def _discard_empty_tptt_epochs(self) -> None:
        while self._tptt_epochs and self._tptt_epochs[0].pending_count <= 0:
            self._tptt_epochs.popleft()


def _resolve_slm_scheduler(
    *,
    explicit: str | None,
    slm_config: dict[str, Any] | None,
) -> str:
    if explicit is not None:
        return _normalize_slm_scheduler(explicit)
    if isinstance(slm_config, dict) and slm_config.get("slm_scheduler") is not None:
        return _normalize_slm_scheduler(str(slm_config["slm_scheduler"]))
    return SLM_SCHEDULER_TPTT


def _job_tptt_epoch(job: SlmJob | None) -> int | None:
    if job is None:
        return None
    queued = job.payload
    if isinstance(queued, _QueuedSequence):
        value = queued.request.metadata.get("tptt_epoch")
        if isinstance(value, int):
            return value
    return None


def _resolve_positive_int(
    *,
    explicit: int | None,
    slm_config: dict[str, Any] | None,
    key: str,
    default: int,
) -> int:
    value: Any = explicit
    if value is None and isinstance(slm_config, dict):
        value = slm_config.get(key)
    try:
        selected = int(value if value is not None else default)
    except (TypeError, ValueError):
        selected = default
    if selected <= 0:
        selected = default
    return selected


def _resolve_nonnegative_float(
    *,
    explicit: float | None,
    slm_config: dict[str, Any] | None,
    key: str,
    default: float,
) -> float:
    value: Any = explicit
    if value is None and isinstance(slm_config, dict):
        value = slm_config.get(key)
    try:
        selected = float(value if value is not None else default)
    except (TypeError, ValueError):
        selected = default
    if selected < 0.0:
        selected = default
    return selected


def _normalize_slm_scheduler(value: str) -> str:
    scheduler = str(value).strip().lower()
    if scheduler not in SLM_SCHEDULER_CHOICES:
        choices = ", ".join(sorted(SLM_SCHEDULER_CHOICES))
        raise ValueError(f"Unknown Local SLM scheduler {value!r}; choose one of: {choices}")
    return scheduler
