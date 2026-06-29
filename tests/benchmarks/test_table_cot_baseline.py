from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.table_cot_baseline import (
    parse_pure_cot_prediction,
    render_pure_cot_prompt,
    run_table_cot_baseline,
    score_pure_cot_answer,
    select_cot_cases,
)
from clover.supervisor.client import RemoteLLMResult


class PureTableCotBaselineTest(unittest.TestCase):
    def test_prompt_is_explicitly_single_model_and_tool_free(self) -> None:
        prompt = render_pure_cot_prompt(
            dataset="tablefact",
            table={"columns": ["team", "wins"], "data": [["A", 3]]},
            question="Is team A listed with 3 wins?",
            answer_type="boolean",
            context="league season",
        )

        self.assertIn("Reason step by step", prompt)
        self.assertIn("Do not write or execute code, SQL, Python, or use tools", prompt)
        self.assertIn("Final Answer: true", prompt)
        self.assertIn("league season", prompt)

    def test_parser_uses_last_explicit_final_answer(self) -> None:
        prediction = (
            "The first check suggests false.\n"
            "Final Answer: false\n"
            "After checking the matching row, it is entailed.\n"
            "**Final Answer:** true\n"
        )

        self.assertEqual(parse_pure_cot_prediction(prediction), "true")
        self.assertEqual(
            parse_pure_cot_prediction("This matches. Final Answer: true."),
            "true.",
        )
        self.assertEqual(parse_pure_cot_prediction("answer: true"), "")

    def test_dataset_native_scoring(self) -> None:
        tablefact = score_pure_cot_answer(
            dataset="tablefact",
            case_payload={
                "answer": True,
                "qtype": "FactChecking",
                "qsubtype": "simple",
            },
            actual="entailed",
        )
        wikitq = score_pure_cot_answer(
            dataset="wikitq",
            case_payload={
                "answer": ["red", "blue"],
                "answer_canon": ["red", "blue"],
            },
            actual="blue, red",
        )
        tablebench = score_pure_cot_answer(
            dataset="tablebench",
            case_payload={
                "answer": "7",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
            },
            actual="7",
        )

        self.assertTrue(tablefact.correct)
        self.assertEqual(tablefact.metric, "accuracy")
        self.assertTrue(wikitq.correct)
        self.assertEqual(wikitq.metric, "denotation_em")
        self.assertTrue(tablebench.correct)
        self.assertEqual(tablebench.metric, "EM")

    def test_tablefact_subset_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_dataset(
                root,
                cases=[
                    {
                        "case_id": "simple",
                        "question": "simple",
                        "statement": "simple",
                        "answer": True,
                        "type": "boolean",
                        "qtype": "FactChecking",
                        "qsubtype": "simple",
                        "split": "test",
                        "is_small_test": True,
                    },
                    {
                        "case_id": "complex",
                        "question": "complex",
                        "statement": "complex",
                        "answer": False,
                        "type": "boolean",
                        "qtype": "FactChecking",
                        "qsubtype": "complex",
                        "split": "test",
                        "is_small_test": False,
                    },
                ],
            )

            simple = select_cot_cases(
                dataset="tablefact",
                dataset_root=root,
                max_cases=None,
                case_ids=set(),
                dataset_id=None,
                split="test",
                subset="simple",
                qtypes=set(),
                qsubtypes=set(),
                include_visualization=False,
                sample_size=None,
                seed=1,
            )
            small = select_cot_cases(
                dataset="tablefact",
                dataset_root=root,
                max_cases=None,
                case_ids=set(),
                dataset_id=None,
                split="test",
                subset="small",
                qtypes=set(),
                qsubtypes=set(),
                include_visualization=False,
                sample_size=None,
                seed=1,
            )

        self.assertEqual([case["case_id"] for case in simple], ["simple"])
        self.assertEqual([case["case_id"] for case in small], ["simple"])

    def test_full_tablefact_baseline_pipeline_with_mock_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "run"
            self._write_dataset(
                root,
                cases=[
                    {
                        "case_id": "case-1",
                        "question": "Determine whether A has 3 wins.",
                        "statement": "A has 3 wins.",
                        "answer": True,
                        "type": "boolean",
                        "qtype": "FactChecking",
                        "qsubtype": "simple",
                        "split": "test",
                        "caption": "league",
                    }
                ],
            )
            response = RemoteLLMResult(
                text="The row for A shows 3 wins.\nFinal Answer: true",
                response_payload={
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    }
                },
                response_id="mock",
                response_status="completed",
                api_type="chat_completions",
            )

            with patch(
                "benchmarks.table_cot_baseline.generate_remote_text",
                return_value=response,
            ):
                summary = run_table_cot_baseline(
                    dataset="tablefact",
                    dataset_root=root,
                    output_dir=output,
                    remote_config={
                        "provider": "local",
                        "api_type": "chat_completions",
                        "base_url": "http://localhost/v1",
                        "model": "mock",
                    },
                    split="test",
                    max_workers=1,
                )

            record = json.loads(
                (output / "cases_index.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["stage"], "tablefact_pure_cot_baseline")
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["tool_calls"], 0)
        self.assertFalse(summary["baseline_contract"]["uses_tools"])
        self.assertEqual(record["metric"], "accuracy")
        self.assertTrue(record["answer_correct"])

    def test_mmqa_multitable_baseline_pipeline_with_mock_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "mmqa"
            dataset_dir = root / "two_table" / "mmqa_tt_mock"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "table_1.csv").write_text(
                "album_id,album,artist_id\na1,Blue,p1\n",
                encoding="utf-8",
            )
            (dataset_dir / "table_2.csv").write_text(
                "artist_id,artist\np1,Ada\n",
                encoding="utf-8",
            )
            with (dataset_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "case_id": "mmqa_tt_000001",
                            "dataset_id": "mmqa_tt_mock",
                            "question": "Which artist has the album Blue?",
                            "answer": ["Ada"],
                            "answer_raw": "Ada",
                            "type": "string",
                            "split": "two_table",
                            "table_names": ["albums", "artists"],
                            "source_files": ["table_1.csv", "table_2.csv"],
                            "table_count": 2,
                            "foreign_keys": [["albums.artist_id", "artists.artist_id"]],
                            "primary_keys": ["albums.album_id", "artists.artist_id"],
                        }
                    )
                    + "\n"
                )
            output = Path(tmpdir) / "run"
            response = RemoteLLMResult(
                text="Join albums.artist_id to artists.artist_id.\nFinal Answer: Ada",
                response_payload={
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 8,
                        "total_tokens": 48,
                    }
                },
                response_id="mock",
                response_status="completed",
                api_type="chat_completions",
            )

            with patch(
                "benchmarks.table_cot_baseline.generate_remote_text",
                return_value=response,
            ):
                summary = run_table_cot_baseline(
                    dataset="mmqa",
                    dataset_root=root,
                    output_dir=output,
                    remote_config={
                        "provider": "local",
                        "api_type": "chat_completions",
                        "base_url": "http://localhost/v1",
                        "model": "mock",
                    },
                    split="two_table",
                    max_workers=1,
                )

            record = json.loads(
                (output / "cases_index.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["stage"], "mmqa_pure_cot_baseline")
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["input_tokens"], 40)
        self.assertEqual(summary["output_tokens"], 8)
        self.assertEqual(record["metric"], "denotation_em")
        self.assertTrue(record["answer_correct"])

    @staticmethod
    def _write_dataset(root: Path, *, cases: list[dict[str, object]]) -> None:
        dataset_dir = root / "table_1"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "table.csv").write_text(
            "team,wins\nA,3\nB,2\n",
            encoding="utf-8",
        )
        with (dataset_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case) + "\n")


if __name__ == "__main__":
    unittest.main()
