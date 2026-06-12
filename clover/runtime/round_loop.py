"""Generic supervised runtime loop records and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RoundLoopState:
    """State carried from one supervised round to the next."""

    round_index: int = 0
    feedback: str | None = None
    scratchpad: str | None = None
    previous_observation: dict[str, Any] | None = None
    next_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundLoopStep:
    """Artifacts and decisions produced by one supervised round."""

    index: int
    command_output: str
    logic_dag: dict[str, Any]
    physical_plan: dict[str, Any]
    execution_result: Any
    supervisor_result: Any


@dataclass(frozen=True)
class RoundLoopResult:
    """Final output of a supervised multi-round loop."""

    ok: bool
    answer: Any
    rounds: list[RoundLoopStep]
    final_decision: Any | None = None
    retry_exhausted: bool = False
    error: dict[str, Any] | None = None


class RuntimeLoopAdapter(Protocol):
    """Adapter required by the task-neutral runtime loop.

    The loop owns only queue progression and backpressure. Task adapters decide
    whether a task enters through Remote planning, local commands, or another
    task-specific entry policy. Runtime barriers are task-level concurrency
    gates: they release downstream work once their dependencies have completed.
    Result collection stays in the Executor collector layer.
    """

    def submit_remote_prefetch(self) -> None:
        """Submit remote jobs while the remote window has capacity."""

    def drain_remote(self, *, wait_for_one: bool) -> int:
        """Move completed remote jobs into command or local queues."""

    def parse_commands(self) -> None:
        """Lower ready commands into local work."""

    def has_ready_barriers(self) -> bool:
        """Return whether completed runtime barriers can release downstream work."""

    def advance_barriers(self) -> bool:
        """Release ready runtime barriers and return whether queues changed."""

    def execute_local_once(self) -> bool:
        """Execute one local work item or batch."""

    def has_pending_remote(self) -> bool:
        """Return whether more remote jobs can be submitted."""

    def has_remote_inflight(self) -> bool:
        """Return whether remote jobs are currently running."""

    def has_commands(self) -> bool:
        """Return whether commands are ready to lower."""

    def has_local_work(self) -> bool:
        """Return whether local work is ready to execute."""


@dataclass
class RuntimeLoop:
    """Task-neutral Remote/Local command-observation loop."""

    adapter: RuntimeLoopAdapter

    def run(self) -> None:
        while self.has_work:
            # Submit cloud work opportunistically; local queues may continue to
            # drain while the next cloud endpoint is already in flight.
            self.adapter.submit_remote_prefetch()
            self.adapter.drain_remote(wait_for_one=False)

            # A remote result, parsed command, or local completion can satisfy
            # a runtime barrier, so check barriers after every queue transition.
            if self._advance_barriers():
                continue

            if self.adapter.has_commands():
                self.adapter.parse_commands()

            if self._advance_barriers():
                continue

            if self.adapter.has_local_work():
                if self.adapter.execute_local_once():
                    # Immediately drain remote results that arrived during
                    # the (blocking) local execution so their commands can
                    # be parsed without waiting for the next loop iteration.
                    self.adapter.drain_remote(wait_for_one=False)
                    if self.adapter.has_commands():
                        self.adapter.parse_commands()
                    if self._advance_barriers():
                        continue

            if self._advance_barriers():
                continue

            if self.adapter.has_remote_inflight():
                # If no local progress is possible, block for one cloud result
                # instead of spinning on empty queues.
                self.adapter.drain_remote(wait_for_one=True)

    @property
    def has_work(self) -> bool:
        return (
            self.adapter.has_pending_remote()
            or self.adapter.has_remote_inflight()
            or self.adapter.has_commands()
            or self.adapter.has_ready_barriers()
            or self.adapter.has_local_work()
        )

    def _advance_barriers(self) -> bool:
        if not self.adapter.has_ready_barriers():
            return False
        return self.adapter.advance_barriers()


def run_runtime_loop(adapter: RuntimeLoopAdapter) -> None:
    """Run the generic supervised runtime loop."""

    RuntimeLoop(adapter).run()
