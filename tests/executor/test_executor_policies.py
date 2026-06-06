from __future__ import annotations

import unittest

from clover.executor import DefaultNodeFailurePolicy, ExecutionUnit, NodeExecutionRecord


class ExecutorPolicyTest(unittest.TestCase):
    def test_default_policy_soft_fails_map_group_units_by_semantics(self) -> None:
        unit = ExecutionUnit(
            id="G0__0__chunk_0",
            task_type="future_task_type",
            op="map",
            node={"id": "G0__0__chunk_0", "op": "map", "output": "G0__0__chunk_0"},
            dependencies=(),
            resources=("doc:chunk_0",),
            output="G0__0__chunk_0",
            index=0,
            metadata={
                "source_kind": "map_group",
                "chunk_resource_id": "doc:chunk_0",
                "chunk_index": 0,
            },
        )
        record = NodeExecutionRecord(
            ok=False,
            node_id=unit.id,
            op="map",
            output_name=unit.output,
            error={"type": "WorkerError", "message": "bad json"},
        )

        policy = DefaultNodeFailurePolicy()

        self.assertTrue(policy.should_soft_fail(unit, record))
        output = policy.soft_failure_output(unit, record)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["chunk"]["chunk_resource_id"], "doc:chunk_0")
        self.assertIn("bad json", output["explanation"])

    def test_default_policy_hard_fails_regular_nodes(self) -> None:
        unit = ExecutionUnit(
            id="N0",
            task_type="future_task_type",
            op="Scan",
            node={"id": "N0", "op": "Scan", "output": "T0"},
            dependencies=(),
            resources=("source_1",),
            output="T0",
            index=0,
            metadata={"source_kind": "node"},
        )
        record = NodeExecutionRecord(
            ok=False,
            node_id="N0",
            op="Scan",
            output_name="T0",
            error={"type": "NodeError", "message": "missing source"},
        )

        self.assertFalse(DefaultNodeFailurePolicy().should_soft_fail(unit, record))

    def test_explicit_failure_mode_overrides_source_kind_default(self) -> None:
        record = NodeExecutionRecord(
            ok=False,
            node_id="N0",
            op="map",
            output_name="T0",
            error={"type": "NodeError", "message": "failed"},
        )
        map_unit = ExecutionUnit(
            id="G0__0__chunk_0",
            task_type="future_task_type",
            op="map",
            node={"id": "G0__0__chunk_0", "op": "map", "output": "T0"},
            dependencies=(),
            resources=("doc:chunk_0",),
            output="T0",
            index=0,
            metadata={"source_kind": "map_group", "failure_mode": "hard"},
        )
        node_unit = ExecutionUnit(
            id="N0",
            task_type="future_task_type",
            op="Custom",
            node={"id": "N0", "op": "Custom", "output": "T0", "failure_mode": "soft"},
            dependencies=(),
            resources=(),
            output="T0",
            index=0,
            metadata={"source_kind": "node"},
        )

        policy = DefaultNodeFailurePolicy()

        self.assertFalse(policy.should_soft_fail(map_unit, record))
        self.assertTrue(policy.should_soft_fail(node_unit, record))


if __name__ == "__main__":
    unittest.main()
