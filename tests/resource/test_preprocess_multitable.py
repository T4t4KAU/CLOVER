from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clover.resource import preprocess_task_dsl


class MultiTablePreprocessTest(unittest.TestCase):
    def test_preprocess_multitable_task_adds_join_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "county.csv").write_text(
                "County_Id,County_name\n"
                "1,Howard\n"
                "2,Mansfield\n"
                "3,Colony\n",
                encoding="utf-8",
            )
            (root / "election.csv").write_text(
                "District,Committee\n"
                "2,Appropriations\n"
                "3,Ways and Means\n",
                encoding="utf-8",
            )
            task_dsl = {
                "task_type": "table_reasoning.query",
                "question": "Which county has an appropriations delegate?",
                "sources": [
                    {"id": "county", "type": "table", "file": "county.csv"},
                    {"id": "election", "type": "table", "file": "election.csv"},
                ],
                "answer": {"name": "answer", "type": "string"},
                "hints": {
                    "primary_keys": ["county id"],
                    "foreign_keys": ["district"],
                },
            }

            result = preprocess_task_dsl(task_dsl, base_dir=root)

        candidates = result["remote_dsl"]["hints"]["join_candidates"]
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(top["left_table"], "county")
        self.assertEqual(top["left_column"], "County_Id")
        self.assertEqual(top["right_table"], "election")
        self.assertEqual(top["right_column"], "District")
        self.assertIn("value_overlap", top["evidence"])
        self.assertEqual(top["sample_matches"], ["2", "3"])
        self.assertEqual(
            result["local_dsl"]["hints"]["join_candidates"],
            candidates,
        )
        paths = result["remote_dsl"]["hints"]["join_paths"]
        self.assertEqual(paths[0]["tables"], ["county", "election"])
        self.assertEqual(paths[0]["length"], 1)

    def test_preprocess_multitable_task_adds_bridge_join_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "party.csv").write_text(
                "Party_ID,Location\n"
                "1,Amsterdam\n"
                "2,Rotterdam\n",
                encoding="utf-8",
            )
            (root / "host.csv").write_text(
                "Host_ID,Age\n"
                "10,34\n"
                "11,41\n",
                encoding="utf-8",
            )
            (root / "party_host.csv").write_text(
                "Party_ID,Host_ID\n"
                "1,10\n"
                "2,11\n",
                encoding="utf-8",
            )
            task_dsl = {
                "task_type": "table_reasoning.query",
                "question": "What is the average age of hosts for Amsterdam parties?",
                "sources": [
                    {"id": "party", "type": "table", "file": "party.csv"},
                    {"id": "host", "type": "table", "file": "host.csv"},
                    {"id": "party_host", "type": "table", "file": "party_host.csv"},
                ],
                "answer": {"name": "answer", "type": "number"},
                "hints": {
                    "primary_keys": ["party id", "host id"],
                    "foreign_keys": ["party id", "host id"],
                },
            }

            result = preprocess_task_dsl(task_dsl, base_dir=root)

        paths = result["remote_dsl"]["hints"]["join_paths"]
        host_party_paths = [
            path
            for path in paths
            if path["tables"][0] == "host" and path["tables"][-1] == "party"
        ]
        self.assertTrue(host_party_paths)
        self.assertEqual(
            host_party_paths[0]["tables"],
            ["host", "party_host", "party"],
        )
        self.assertEqual(host_party_paths[0]["length"], 2)
        value_matches = result["remote_dsl"]["hints"]["question_value_matches"]
        self.assertTrue(
            any(
                item["table"] == "party"
                and item["column"] == "Location"
                and item["matches"] == ["Amsterdam"]
                for item in value_matches
            )
        )
        column_matches = result["remote_dsl"]["hints"]["question_column_matches"]
        self.assertTrue(
            any(
                item["table"] == "host"
                and item["column"] == "Age"
                for item in column_matches
            )
        )


if __name__ == "__main__":
    unittest.main()
