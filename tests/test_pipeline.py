from __future__ import annotations

import unittest

from clover.runtime.pipeline import GroupedPriorityQueue


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


if __name__ == "__main__":
    unittest.main()
