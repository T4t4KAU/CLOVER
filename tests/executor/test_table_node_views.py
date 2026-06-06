from __future__ import annotations

import unittest

from clover.executor.node_views import render_node_view


class TableNodeViewTest(unittest.TestCase):
    def test_renders_query_and_analyze_filter_with_same_python_view(self) -> None:
        node = {
            "id": "N1",
            "op": "Filter",
            "params": {
                "predicate": {
                    "type": "logical_op",
                    "op": "AND",
                    "operands": [
                        {
                            "type": "binary_op",
                            "op": "=",
                            "left": {"type": "column", "name": "city"},
                            "right": {"type": "literal", "value": "Winnipeg"},
                        },
                        {
                            "type": "binary_op",
                            "op": ">",
                            "left": {"type": "column", "name": "floors"},
                            "right": {"type": "literal", "value": 10},
                        },
                    ],
                }
            },
            "output": "T1",
        }
        view = render_node_view(
            "table_reasoning.analyze",
            node,
            world={"inputs": {"df": {"columns": ["city", "floors"]}}},
        )
        query_view = render_node_view(
            "table_reasoning.query",
            node,
            world={"inputs": {"df": {"columns": ["city", "floors"]}}},
        )

        self.assertEqual(view.language, "python")
        self.assertEqual(view.kind, "table_reasoning.filter")
        self.assertEqual(view.world["inputs"]["df"]["columns"], ["city", "floors"])
        self.assertIn("def solve(df):", view.task)
        self.assertIn("city = 'Winnipeg'", view.task)
        self.assertIn("floors > 10", view.task)
        self.assertEqual(query_view.task, view.task)

    def test_renders_document_map_node_as_worker_view(self) -> None:
        view = render_node_view(
            "document_reasoning",
            {
                "id": "D1",
                "op": "map",
                "params": {"local_instruction": "Extract revenue evidence."},
                "output": "chunk_result",
            },
            world={
                "chunk_text": "Revenue was $10 million.",
                "advice": "Preserve units.",
            },
        )

        self.assertEqual(view.kind, "document_reasoning.map")
        self.assertEqual(view.language, "json")
        self.assertEqual(view.task, "Extract revenue evidence.")
        self.assertEqual(view.world["chunk_text"], "Revenue was $10 million.")
        self.assertEqual(view.metadata["advice"], "Preserve units.")


if __name__ == "__main__":
    unittest.main()
