from __future__ import annotations

import unittest

from clover.runtime.pipeline import (
    GroupedPriorityQueue,
    InflightStage,
    PipelineProfiler,
)
from clover.runtime.round_loop import RuntimeLoop
from clover.runtime.table_reasoning.pipeline import _merge_batch_hints


class GroupedPriorityQueueTest(unittest.TestCase):
    def test_pops_lowest_retry_priority_before_higher_retry_items(self) -> None:
        queue: GroupedPriorityQueue[str] = GroupedPriorityQueue()
        queue.push("table_a", "a_retry_1", priority=1)
        queue.push("table_b", "b_retry_0", priority=0)
        queue.push("table_a", "a_retry_0", priority=0)
        queue.push("table_a", "a_retry_1_later", priority=1)

        self.assertEqual(queue.pop_best_group(4), ("table_b", ["b_retry_0"]))
        self.assertEqual(queue.pop_best_group(4), ("table_a", ["a_retry_0"]))
        self.assertEqual(
            queue.pop_best_group(4),
            ("table_a", ["a_retry_1", "a_retry_1_later"]),
        )

    def test_preserves_fifo_within_same_priority(self) -> None:
        queue: GroupedPriorityQueue[str] = GroupedPriorityQueue()
        queue.push("table_a", "first", priority=0)
        queue.push("table_a", "second", priority=0)
        queue.push("table_a", "third", priority=0)

        self.assertEqual(queue.pop_best_group(2), ("table_a", ["first", "second"]))
        self.assertEqual(queue.pop_best_group(2), ("table_a", ["third"]))


class InflightStageTest(unittest.TestCase):
    def test_drains_completed_stage_calls_with_payloads(self) -> None:
        profiler = PipelineProfiler()
        observed = []

        with InflightStage[str, int](
            stage_name="remote",
            max_workers=2,
            profiler=profiler,
        ) as stage:
            stage.submit("first", lambda: 1, items=1)
            stage.submit("second", lambda: 2, items=1)
            self.assertFalse(stage.has_capacity)

            while stage:
                stage.drain_ready(
                    lambda payload, result: observed.append(
                        (payload, result.value, result.ok)
                    ),
                    wait_for_one=True,
                )

        self.assertEqual(
            sorted(observed),
            [("first", 1, True), ("second", 2, True)],
        )
        self.assertEqual(profiler.stages["remote"].calls, 2)
        self.assertEqual(profiler.stages["remote"].items, 2)

    def test_normalizes_stage_call_errors(self) -> None:
        def fail() -> int:
            raise RuntimeError("boom")

        observed = []
        with InflightStage[str, int](stage_name="remote", max_workers=1) as stage:
            stage.submit("bad", fail)
            stage.drain_ready(
                lambda payload, result: observed.append((payload, result.error)),
                wait_for_one=True,
            )

        self.assertEqual(observed[0][0], "bad")
        self.assertIsInstance(observed[0][1], RuntimeError)


class RuntimeLoopTest(unittest.TestCase):
    def test_overlaps_remote_prefetch_with_local_work(self) -> None:
        hooks = _FakePipelineHooks()

        RuntimeLoop(hooks).run()

        self.assertEqual(hooks.events[:2], ["submit:1", "submit:2"])
        self.assertIn("execute:1", hooks.events)
        self.assertIn("execute:2", hooks.events)
        self.assertEqual(hooks.completed, [1, 2])

    def test_advances_ready_barrier_without_other_work(self) -> None:
        hooks = _BarrierPipelineHooks()

        RuntimeLoop(hooks).run()

        self.assertEqual(
            hooks.events,
            ["barrier:release", "parse:1", "execute:1"],
        )
        self.assertEqual(hooks.completed, [1])


class TableReasoningBatchHintsTest(unittest.TestCase):
    def test_preserves_shared_hints_and_groups_question_hints_by_answer(self) -> None:
        hints = [
            {
                "join_candidates": [
                    {"left": {"table": "party"}, "right": {"table": "party_host"}}
                ],
                "question_value_matches": [
                    {"table": "party", "column": "Location", "matches": ["Amsterdam"]}
                ],
                "question_column_matches": [
                    {"table": "host", "column": "Age", "matches": ["age"]}
                ],
            },
            {
                "join_candidates": [
                    {"left": {"table": "party"}, "right": {"table": "party_host"}}
                ],
                "question_value_matches": [
                    {"table": "host", "column": "Name", "matches": ["Lloyd Daniels"]}
                ],
                "question_column_matches": [],
            },
        ]

        merged = _merge_batch_hints(
            hints,
            answer_keys=["answer_1", "answer_2"],
            questions=["Where was the party?", "Who is Lloyd Daniels?"],
        )

        self.assertEqual(merged["join_candidates"], hints[0]["join_candidates"])
        self.assertNotIn("question_value_matches", merged)
        self.assertEqual(
            [entry["answer"] for entry in merged["question_value_matches_by_answer"]],
            ["answer_1", "answer_2"],
        )
        self.assertEqual(
            merged["question_column_matches_by_answer"],
            [
                {
                    "answer": "answer_1",
                    "matches": hints[0]["question_column_matches"],
                    "question": "Where was the party?",
                }
            ],
        )


class _FakePipelineHooks:
    def __init__(self) -> None:
        self.pending = [1, 2]
        self.inflight: list[int] = []
        self.commands: list[int] = []
        self.local_work: list[int] = []
        self.completed: list[int] = []
        self.events: list[str] = []

    def submit_remote_prefetch(self) -> None:
        while self.pending and len(self.inflight) < 2:
            item = self.pending.pop(0)
            self.inflight.append(item)
            self.events.append(f"submit:{item}")

    def drain_remote(self, *, wait_for_one: bool) -> int:
        if not self.inflight:
            return 0
        item = self.inflight.pop(0)
        self.commands.append(item)
        self.events.append(f"drain:{item}")
        return 1

    def parse_commands(self) -> None:
        while self.commands:
            item = self.commands.pop(0)
            self.local_work.append(item)
            self.events.append(f"parse:{item}")

    def has_ready_barriers(self) -> bool:
        return False

    def advance_barriers(self) -> bool:
        return False

    def execute_local_once(self) -> bool:
        if not self.local_work:
            return False
        item = self.local_work.pop(0)
        self.completed.append(item)
        self.events.append(f"execute:{item}")
        return True

    def has_pending_remote(self) -> bool:
        return bool(self.pending)

    def has_remote_inflight(self) -> bool:
        return bool(self.inflight)

    def has_commands(self) -> bool:
        return bool(self.commands)

    def has_local_work(self) -> bool:
        return bool(self.local_work)


class _BarrierPipelineHooks(_FakePipelineHooks):
    def __init__(self) -> None:
        super().__init__()
        self.pending.clear()
        self.barrier_ready = True

    def has_ready_barriers(self) -> bool:
        return self.barrier_ready

    def advance_barriers(self) -> bool:
        if not self.barrier_ready:
            return False
        self.barrier_ready = False
        self.commands.append(1)
        self.events.append("barrier:release")
        return True


if __name__ == "__main__":
    unittest.main()
