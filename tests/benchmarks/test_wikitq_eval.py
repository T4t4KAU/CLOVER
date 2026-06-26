from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.wikitq.eval import _wikitq_adjudicator_prompt


class WikiTQEvalTest(unittest.TestCase):
    def test_adjudicator_prompt_includes_base_sql_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "wikitq_1_1"
            dataset.mkdir()
            table = dataset / "table.csv"
            table.write_text("Team,Points\nAlpha,10\nBeta,8\n", encoding="utf-8")
            prompt = _wikitq_adjudicator_prompt(
                sampled_case={
                    "dataset_id": "wikitq_1_1",
                    "question": "Which team has the most points?",
                },
                base_answer="Alpha",
                direct_answer="Beta",
                base_sql='SELECT Team FROM table_1 ORDER BY Points DESC LIMIT 1',
                base_source="action_static",
                wikitq_root=root,
                table_char_limit=20_000,
            )

        self.assertIn("Candidate A source: \"action_static\"", prompt)
        self.assertIn("ORDER BY Points DESC LIMIT 1", prompt)
        self.assertIn("Prefer evidence-backed A", prompt)


if __name__ == "__main__":
    unittest.main()
