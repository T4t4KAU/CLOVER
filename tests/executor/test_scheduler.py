from __future__ import annotations

import unittest

from clover.executor import (
    ExecutionPlan,
    ExecutionPlanBuilder,
    ExecutionUnit,
    Scheduler,
)
from clover.executor.prompt_prefix import (
    CACHE_PREFIX_PATH_METADATA_KEY,
    CACHE_PREFIX_SEGMENT_TOKEN_ESTIMATES_METADATA_KEY,
    DOCUMENT_WORKER_EXAMPLES_ID,
    DOCUMENT_WORKER_OUTPUT_CONTRACT_ID,
    DOCUMENT_WORKER_RULES_ID,
    DOCUMENT_WORKER_TEMPLATE_ID,
    SCHEDULING_KIND_METADATA_KEY,
)


class SchedulerTest(unittest.TestCase):
    def test_schedules_ready_units_in_plan_order(self) -> None:
        state = _ResourceState(sources={"table_1", "table_2"})
        scheduler = _scheduler_from_plan(
            _join_plan(),
            resource_state=state,
        )

        self.assertEqual(scheduler.consumers_by_artifact, {"T0": 1, "T1": 1})
        self.assertEqual(scheduler.retained_artifacts, {"answer"})
        self.assertEqual(len(scheduler.collectors), 1)
        self.assertEqual(scheduler.collectors[0].kind, "final_answer")
        self.assertEqual(scheduler.collectors[0].inputs, ("answer",))

        unit = scheduler.next_ready()
        self.assertIsNotNone(unit)
        assert unit is not None
        self.assertEqual(unit.id, "N0")
        state.artifacts.add(unit.output)
        scheduler.mark_succeeded(unit.id)

        unit = scheduler.next_ready()
        self.assertIsNotNone(unit)
        assert unit is not None
        self.assertEqual(unit.id, "N1")
        state.artifacts.add(unit.output)
        scheduler.mark_succeeded(unit.id)

        unit = scheduler.next_ready()
        self.assertIsNotNone(unit)
        assert unit is not None
        self.assertEqual(unit.id, "N2")
        state.artifacts.add(unit.output)
        scheduler.mark_succeeded(unit.id)

        self.assertTrue(scheduler.done)
        self.assertIsNone(scheduler.next_ready())

    def test_reports_blocked_units_when_resources_are_missing(self) -> None:
        scheduler = _scheduler_from_plan(
            _join_plan(),
            resource_state=_ResourceState(),
        )

        self.assertIsNone(scheduler.next_ready())
        self.assertEqual(scheduler.blocked_unit_ids(), ["N0", "N1", "N2"])

    def test_schedules_ready_units_in_batches(self) -> None:
        state = _ResourceState(sources={"table_1", "table_2"})
        scheduler = _scheduler_from_plan(
            _join_plan(),
            resource_state=state,
        )

        batch = scheduler.next_ready_batch(max_units=4)
        self.assertEqual([unit.id for unit in batch], ["N0", "N1"])
        self.assertEqual(scheduler.next_ready_batch(max_units=4), [])

        for unit in batch:
            state.artifacts.add(unit.output)
            scheduler.mark_succeeded(unit.id)

        batch = scheduler.next_ready_batch(max_units=4)
        self.assertEqual([unit.id for unit in batch], ["N2"])

    def test_builds_document_map_group_units_and_collector(self) -> None:
        state = _ResourceState(
            sources={
                "document_1:chunk_0",
                "document_1:chunk_1",
                "document_1:chunk_2",
            }
        )
        scheduler = _scheduler_from_plan(
            _document_map_plan(),
            resource_state=state,
        )

        self.assertEqual(scheduler.consumers_by_artifact, {})
        self.assertEqual(
            scheduler.retained_artifacts,
            {
                "G0__0__document_1_chunk_0",
                "G0__1__document_1_chunk_1",
                "G0__2__document_1_chunk_2",
                "G0",
            },
        )
        self.assertEqual(len(scheduler.collectors), 1)
        collector = scheduler.collectors[0]
        self.assertEqual(collector.id, "G0")
        self.assertEqual(collector.kind, "map_group_evidence")
        self.assertEqual(
            collector.inputs,
            (
                "G0__0__document_1_chunk_0",
                "G0__1__document_1_chunk_1",
                "G0__2__document_1_chunk_2",
            ),
        )
        self.assertEqual(collector.output, "G0")
        self.assertEqual(collector.params["items"][0]["output"], "G0__0__document_1_chunk_0")
        self.assertEqual(
            collector.params["items"][0]["chunk_resource_id"],
            "document_1:chunk_0",
        )
        self.assertEqual(collector.params["items"][0]["chunk_index"], 0)
        self.assertEqual(collector.params["items"][0]["task_id"], 0)

        batch = scheduler.next_ready_batch(max_units=2)
        self.assertEqual(
            [unit.id for unit in batch],
            [
                "G0__0__document_1_chunk_0",
                "G0__1__document_1_chunk_1",
            ],
        )
        self.assertEqual(batch[0].node["op"], "map")
        self.assertEqual(batch[0].node["input"], ["document_1:chunk_0"])
        self.assertEqual(
            batch[0].node["metadata"]["chunk_resource_id"],
            "document_1:chunk_0",
        )
        self.assertEqual(batch[0].metadata["group_output"], "G0")
        self.assertEqual(scheduler.next_ready_batch(max_units=2), [])

        for unit in batch:
            state.artifacts.add(unit.output)
            scheduler.mark_succeeded(unit.id)

        batch = scheduler.next_ready_batch(max_units=2)
        self.assertEqual([unit.id for unit in batch], ["G0__2__document_1_chunk_2"])

    def test_builds_static_minions_collector_for_document_plan(self) -> None:
        state = _ResourceState(
            sources={
                "document_1:chunk_0",
                "document_1:chunk_1",
            }
        )
        plan = _document_map_plan()
        plan["static_collectors"] = [
            {
                "id": "document_evidence",
                "kind": "minions_transform_outputs",
                "function_name": "transform_outputs",
                "source": "def transform_outputs(jobs):\n    return ''",
                "output": "document_evidence",
            }
        ]
        plan["map_groups"][0]["input"]["chunks"] = [
            "document_1:chunk_0",
            "document_1:chunk_1",
        ]
        scheduler = _scheduler_from_plan(plan, resource_state=state)

        self.assertEqual(len(scheduler.collectors), 1)
        collector = scheduler.collectors[0]
        self.assertEqual(collector.kind, "minions_transform_outputs")
        self.assertEqual(collector.output, "document_evidence")
        self.assertEqual(
            collector.inputs,
            ("G0__0__document_1_chunk_0", "G0__1__document_1_chunk_1"),
        )
        self.assertEqual(collector.params["items"][0]["task"], "Extract revenue values.")

    def test_expands_document_map_group_replicas(self) -> None:
        state = _ResourceState(sources={"document_1:chunk_0"})
        plan = _document_map_plan()
        plan["map_groups"][0]["input"]["chunks"] = ["document_1:chunk_0"]
        plan["map_groups"][0]["replicas"] = 2

        scheduler = _scheduler_from_plan(plan, resource_state=state)
        batch = scheduler.next_ready_batch(max_units=4)

        self.assertEqual(
            [unit.id for unit in batch],
            [
                "G0__0__document_1_chunk_0__sample_0",
                "G0__0__document_1_chunk_0__sample_1",
            ],
        )
        self.assertEqual(batch[1].metadata["replica_index"], 1)
        self.assertEqual(
            scheduler.collectors[0].inputs,
            (
                "G0__0__document_1_chunk_0__sample_0",
                "G0__0__document_1_chunk_0__sample_1",
            ),
        )

    def test_workflow_scheduler_does_not_group_shared_prefix_units(self) -> None:
        scheduler = Scheduler.from_execution_plan(
            ExecutionPlan(
                units=(
                    _unit("A0", 0, ("worker", "task_a", "rules")),
                    _unit("B0", 1, ("worker", "task_b", "rules")),
                    _unit("A1", 2, ("worker", "task_a", "rules")),
                    _unit("C0", 3, ()),
                )
            ),
            resource_state=_ResourceState(),
        )

        batch = scheduler.next_ready_batch(max_units=2)

        self.assertEqual([unit.id for unit in batch], ["A0", "B0"])

    def test_document_map_units_have_prompt_prefix_metadata(self) -> None:
        state = _ResourceState(
            sources={
                "document_1:chunk_0",
                "document_1:chunk_1",
                "document_1:chunk_2",
            }
        )
        scheduler = _scheduler_from_plan(
            _document_map_plan(),
            resource_state=state,
        )

        batch = scheduler.next_ready_batch(max_units=1)

        self.assertIn(CACHE_PREFIX_PATH_METADATA_KEY, batch[0].metadata)
        self.assertIn(CACHE_PREFIX_SEGMENT_TOKEN_ESTIMATES_METADATA_KEY, batch[0].metadata)
        self.assertEqual(batch[0].metadata[SCHEDULING_KIND_METADATA_KEY], "local_slm_prefix")
        cache_prefix_path = batch[0].metadata[CACHE_PREFIX_PATH_METADATA_KEY]
        self.assertEqual(
            cache_prefix_path[:5],
            (
                "agent:document_worker",
                f"template:{DOCUMENT_WORKER_TEMPLATE_ID}",
                f"output:{DOCUMENT_WORKER_OUTPUT_CONTRACT_ID}",
                f"rules:{DOCUMENT_WORKER_RULES_ID}",
                f"examples:{DOCUMENT_WORKER_EXAMPLES_ID}",
            ),
        )
        self.assertTrue(cache_prefix_path[5].startswith("task:"))
        self.assertTrue(cache_prefix_path[6].startswith("advice:"))
        self.assertEqual(
            len(batch[0].metadata[CACHE_PREFIX_SEGMENT_TOKEN_ESTIMATES_METADATA_KEY]),
            len(cache_prefix_path),
        )


class _ResourceState:
    def __init__(
        self,
        *,
        sources: set[str] | None = None,
        artifacts: set[str] | None = None,
    ) -> None:
        self.sources = set(sources or set())
        self.artifacts = set(artifacts or set())

    def has_artifact(self, name: str) -> bool:
        return name in self.artifacts

    def has_source(self, name: str) -> bool:
        return name in self.sources


def _unit(
    unit_id: str,
    index: int,
    cache_prefix_path: tuple[str, ...],
) -> ExecutionUnit:
    metadata = {}
    if cache_prefix_path:
        metadata = {
            "cache_prefix_path": cache_prefix_path,
            "cache_prefix_segment_token_estimates": tuple(
                10 for _ in cache_prefix_path
            ),
        }
    return ExecutionUnit(
        id=unit_id,
        task_type="document_reasoning",
        op="map",
        node={
            "id": unit_id,
            "op": "map",
            "dependency": [],
            "input": [],
            "output": f"{unit_id}_out",
            "params": {},
        },
        dependencies=(),
        resources=(),
        output=f"{unit_id}_out",
        index=index,
        metadata=metadata,
    )


def _scheduler_from_plan(plan: dict, *, resource_state: _ResourceState) -> Scheduler:
    return Scheduler.from_execution_plan(
        ExecutionPlanBuilder.default().build(plan),
        resource_state=resource_state,
    )


def _join_plan() -> dict:
    return {
        "task_type": "table_reasoning.query",
        "resources": [
            {"id": "table_1", "type": "table", "path": "/tmp/left.csv"},
            {"id": "table_2", "type": "table", "path": "/tmp/right.csv"},
        ],
        "nodes": [
            {
                "id": "N0",
                "op": "Scan",
                "dependency": [],
                "input": ["table_1"],
                "params": {"source": "table_1"},
                "output": "T0",
            },
            {
                "id": "N1",
                "op": "Scan",
                "dependency": [],
                "input": ["table_2"],
                "params": {"source": "table_2"},
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "Join",
                "dependency": ["T0", "T1"],
                "input": [],
                "params": {},
                "output": "answer",
            },
        ],
        "edges": [],
    }


def _document_map_plan() -> dict:
    return {
        "task_type": "document_reasoning",
        "resources": [
            {
                "id": "document_1:chunk_0",
                "type": "document_chunk",
                "path": "/tmp/chunks.jsonl",
                "format": "text",
            },
            {
                "id": "document_1:chunk_1",
                "type": "document_chunk",
                "path": "/tmp/chunks.jsonl",
                "format": "text",
            },
            {
                "id": "document_1:chunk_2",
                "type": "document_chunk",
                "path": "/tmp/chunks.jsonl",
                "format": "text",
            },
        ],
        "map_groups": [
            {
                "id": "G0",
                "op": "map",
                "input": {
                    "chunks": [
                        "document_1:chunk_0",
                        "document_1:chunk_1",
                        "document_1:chunk_2",
                    ],
                },
                "params": {"local_instruction": "Extract revenue values."},
                "output": "G0",
                "output_type": "jsonl",
            }
        ],
        "edges": [],
    }


if __name__ == "__main__":
    unittest.main()
