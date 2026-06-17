from __future__ import annotations

import unittest

from clover.executor.slm_scheduler import (
    SlmJob,
    TemplateLeafSpec,
    ThreadedPrefixTemplateTree,
)
from clover.executor.agents.template_tree import (
    DOCUMENT_WORKER_LEAF_KEY,
    NODE_AGENT_SLM_TEMPLATE_LEAVES,
    TABLE_BOOLEAN_EMPTY_FILTER_REPAIR_LEAF_KEY,
    TABLE_BOOLEAN_LEAF_KEY,
    TABLE_EVIDENCE_LEAF_KEY,
    TABLE_NUMBER_EMPTY_FILTER_REPAIR_LEAF_KEY,
    TABLE_NUMBER_LEAF_KEY,
    TABLE_STRING_EMPTY_FILTER_REPAIR_LEAF_KEY,
    TABLE_STRING_LEAF_KEY,
    TemplateNode,
    build_slm_template_scheduler_tree,
    collect_slm_template_leaf_specs,
    render_table_empty_filter_repair_prompt,
    render_table_evidence_prompt,
    slm_template_leaf_specs,
    template_paths_for_leaf_key,
    template_paths_for_task_type,
)
from clover.executor.node_views import NodeView


def _spec(name: str) -> TemplateLeafSpec:
    return TemplateLeafSpec(key=("root", name))


def _job(job_id: str, leaf: str, *, prompt_len: int = 10) -> SlmJob:
    return SlmJob(job_id=job_id, leaf_key=("root", leaf), prompt_len=prompt_len)


class ThreadedPrefixTemplateTreeTest(unittest.TestCase):
    def test_thread_order_follows_registered_template_leaf_order(self) -> None:
        tree = ThreadedPrefixTemplateTree(
            [
                TemplateLeafSpec(key=("N", "L", "G", "A")),
                TemplateLeafSpec(key=("N", "L", "G", "B")),
                TemplateLeafSpec(key=("N", "L", "H")),
                TemplateLeafSpec(key=("N", "L", "I", "C")),
                TemplateLeafSpec(key=("N", "L", "I", "D")),
                TemplateLeafSpec(key=("N", "L", "J")),
                TemplateLeafSpec(key=("N", "M", "K", "E")),
                TemplateLeafSpec(key=("N", "M", "K", "F")),
            ]
        )

        self.assertEqual(
            tree.thread_order(),
            (
                ("N", "L", "G", "A"),
                ("N", "L", "G", "B"),
                ("N", "L", "H"),
                ("N", "L", "I", "C"),
                ("N", "L", "I", "D"),
                ("N", "L", "J"),
                ("N", "M", "K", "E"),
                ("N", "M", "K", "F"),
            ),
        )

    def test_same_leaf_is_preferred_before_moving_to_next_leaf(self) -> None:
        tree = ThreadedPrefixTemplateTree([_spec("A"), _spec("B")])
        tree.submit(_job("a-long", "A", prompt_len=20))
        tree.submit(_job("a-short", "A", prompt_len=5))
        tree.submit(_job("b", "B", prompt_len=1))

        first = tree.pop_initial()
        self.assertIsNotNone(first)
        self.assertEqual(first.job_id, "a-short")

        refill = tree.refill_after(first)[0]
        self.assertEqual(refill.job_id, "a-long")

        refill = tree.refill_after(refill)[0]
        self.assertEqual(refill.job_id, "b")

    def test_elevator_refill_reverses_at_right_edge(self) -> None:
        tree = ThreadedPrefixTemplateTree(
            [_spec("A"), _spec("B"), _spec("C"), _spec("D")]
        )
        tree.submit(_job("a", "A"))
        tree.submit(_job("c", "C"))
        done_at_right_edge = _job("done-d", "D")

        refill = tree.refill_after(done_at_right_edge)[0]
        self.assertEqual(refill.job_id, "c")
        self.assertEqual(tree.direction, -1)

        refill = tree.refill_after(refill)[0]
        self.assertEqual(refill.job_id, "a")
        self.assertEqual(tree.direction, -1)

    def test_new_job_becomes_visible_to_next_refill(self) -> None:
        tree = ThreadedPrefixTemplateTree([_spec("A"), _spec("B"), _spec("C")])
        done = _job("done-a", "A")

        self.assertEqual(tree.refill_after(done), [])

        tree.submit(_job("c", "C"))
        refill = tree.refill_after(done)
        self.assertEqual([job.job_id for job in refill], ["c"])

    def test_new_queue_reuses_static_leaf_thread_with_independent_dynamic_items(self) -> None:
        tree = ThreadedPrefixTemplateTree([_spec("A"), _spec("B")])
        first_queue = tree.new_queue()
        second_queue = tree.new_queue()

        first_queue.submit(_job("a", "A"))
        second_queue.submit(_job("b", "B"))

        self.assertEqual(first_queue.thread_order(), tree.thread_order())
        self.assertEqual(second_queue.thread_order(), tree.thread_order())
        self.assertEqual(first_queue.pop_initial().job_id, "a")
        self.assertEqual(first_queue.pop_initial(), None)
        self.assertEqual(second_queue.pop_initial().job_id, "b")

    def test_unknown_leaf_is_rejected(self) -> None:
        tree = ThreadedPrefixTemplateTree([_spec("A")])

        with self.assertRaises(KeyError):
            tree.submit(_job("x", "X"))

    def test_default_template_tree_exposes_registered_layers(self) -> None:
        tree = build_slm_template_scheduler_tree()

        self.assertEqual(tree.leaf_count, len(NODE_AGENT_SLM_TEMPLATE_LEAVES))
        self.assertIn(
            (
                "agent:data",
                "family:table_reasoning",
                "interface:solve_python",
                "tool:pandas_env",
                "contract:number",
                "mode:initial",
            ),
            tree.leaf_keys(),
        )
        self.assertIn(
            (
                "agent:data",
                "family:table_reasoning",
                "interface:solve_python",
                "tool:pandas_env",
                "contract:number",
                "mode:empty_filter_repair",
                "op:filter",
            ),
            tree.leaf_keys(),
        )
        self.assertIn(
            (
                "agent:data",
                "family:document_reasoning",
                "interface:chunk_worker",
                "tool:text_excerpt",
                "contract:evidence_json",
                "mode:initial",
            ),
            tree.leaf_keys(),
        )
        self.assertIn(
            (
                "agent:data",
                "family:table_reasoning",
                "interface:evidence_python",
                "tool:pandas_env",
                "contract:evidence_json",
                "mode:initial",
            ),
            tree.leaf_keys(),
        )

    def test_default_leaf_specs_are_derived_from_static_template_tree(self) -> None:
        _shared_prefix = (
            "common/root.md",
            "table_reasoning/feedback_decoding.md",
        )
        _repair_ops = ("filter", "project", "derive", "join")

        def _repair_key(contract: str, op: str) -> tuple[str, ...]:
            return (
                "agent:data",
                "family:table_reasoning",
                "interface:solve_python",
                "tool:pandas_env",
                f"contract:{contract}",
                "mode:empty_filter_repair",
                f"op:{op}",
            )

        expected_paths: dict[tuple[str, ...], tuple[str, ...]] = {
            TABLE_NUMBER_LEAF_KEY: _shared_prefix + ("table_reasoning/agent_loop.md",),
            TABLE_STRING_LEAF_KEY: _shared_prefix + ("table_reasoning/agent_loop.md",),
            TABLE_BOOLEAN_LEAF_KEY: _shared_prefix + ("table_reasoning/agent_loop.md",),
            TABLE_EVIDENCE_LEAF_KEY: ("table_reasoning/evidence.md",),
            DOCUMENT_WORKER_LEAF_KEY: ("document_reasoning/worker.md",),
        }
        for contract in ("number", "string", "boolean"):
            for op in _repair_ops:
                key = _repair_key(contract, op)
                expected_paths[key] = _shared_prefix + (
                    "table_reasoning/empty_filter_repair.md",
                    f"table_reasoning/repair_hints/{op}.md",
                )

        self.assertEqual(slm_template_leaf_specs(), NODE_AGENT_SLM_TEMPLATE_LEAVES)
        for spec in slm_template_leaf_specs():
            self.assertEqual(spec.template_paths, template_paths_for_leaf_key(spec.key))
            self.assertEqual(spec.template_paths, expected_paths[spec.key])

        self.assertEqual(
            template_paths_for_task_type("table_reasoning.query"),
            expected_paths[TABLE_NUMBER_LEAF_KEY],
        )
        self.assertEqual(
            template_paths_for_task_type("document_reasoning"),
            expected_paths[DOCUMENT_WORKER_LEAF_KEY],
        )

    def test_table_evidence_prompt_keeps_dynamic_payload_after_static_prefix(self) -> None:
        prompt = render_table_evidence_prompt(
            prompt_code="# EVIDENCE.py\npass",
            feedback="none",
            iteration=1,
            last_iteration=False,
        )

        self.assertTrue(prompt.startswith("```python\n# DEBUG.py\n"))
        self.assertLess(prompt.index("# DEBUG.py"), prompt.index("JSON only."))
        self.assertLess(prompt.index("JSON only."), prompt.index("# EVIDENCE.py"))
        self.assertLess(prompt.index("# EVIDENCE.py"), prompt.index("# FEEDBACK"))

    def test_empty_filter_repair_prompt_keeps_case_payload_at_tail(self) -> None:
        view = NodeView(
            kind="table_reasoning.filter",
            language="python",
            task='def solve(df):\n    """Return matching rows."""\n    pass',
            world={
                "inputs": {"df": {"rows": 2, "cols": ["name"]}},
                "diag": {
                    "inputs": {
                        "df": {
                            "values": {
                                "name": [
                                    {"v": "Formula Renault", "n": 1},
                                ],
                            },
                        },
                    },
                },
            },
        )

        prompt = render_table_empty_filter_repair_prompt(
            view=view,
            iteration=1,
            steps=[],
            node={"op": "Filter"},
        )

        self.assertLess(prompt.index("Repair with the smallest change."), prompt.index("Case:"))
        self.assertIn('"sig":"def solve(df):"', prompt)
        self.assertIn('"evidence":{"name":["Formula Renault"]}', prompt)

    def test_static_template_tree_orders_siblings_by_static_delta_tokens(self) -> None:
        tree = TemplateNode(
            name="root",
            children=(
                TemplateNode(
                    name="long",
                    template="document_reasoning/worker.md",
                    children=(TemplateNode(name="leaf"),),
                ),
                TemplateNode(
                    name="short",
                    template="common/root.md",
                    children=(TemplateNode(name="leaf"),),
                ),
            ),
        )

        specs = collect_slm_template_leaf_specs(tree)

        self.assertEqual([spec.key for spec in specs], [("short", "leaf"), ("long", "leaf")])
        self.assertLess(specs[0].static_token_count, specs[1].static_token_count)


if __name__ == "__main__":
    unittest.main()
