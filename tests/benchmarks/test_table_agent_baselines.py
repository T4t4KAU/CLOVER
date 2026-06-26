from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.baselines.table_agent_baselines import (
    CaseRunStats,
    ModelCallRecord,
    OrchestraRunner,
    build_summary,
    clean_answer,
    model_stats_record,
    parse_line_limit,
    parse_step,
    score_prediction,
)


class FakeEdgeModel:
    def __init__(self) -> None:
        self.roles: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        stats: CaseRunStats,
        role: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> str:
        del prompt, temperature, max_tokens, system_prompt
        self.roles.append(role)
        stats.add(
            ModelCallRecord(
                role=role,
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                elapsed_seconds=0.01,
                prompt_chars=40,
                response_chars=10,
                prompt_preview="prompt",
                response_preview="response",
            )
        )
        if role == "orchestra_decision":
            return "Answer: no"
        return "Answer: yes"


class TableAgentBaselinesTest(unittest.TestCase):
    def test_parse_reactable_sql_and_answer_outputs(self) -> None:
        sql = parse_step('SQL: ```SELECT name FROM DF WHERE score > 3;```.')
        answer = parse_step("Reasoning...\nFinal Answer: ```Australian Open```")

        self.assertEqual(sql.kind, "SQL")
        self.assertEqual(sql.content, "SELECT name FROM DF WHERE score > 3;")
        self.assertEqual(answer.kind, "Answer")
        self.assertEqual(answer.content, "Australian Open")

    def test_clean_answer_handles_short_boolean_period(self) -> None:
        self.assertEqual(clean_answer("Answer: true."), "true")
        self.assertEqual(clean_answer("```no```."), "no")

    def test_native_scoring_for_tablefact_and_wikitq(self) -> None:
        tablefact = score_prediction(
            "tablefact",
            {"expected_answer": True, "qtype": "FactChecking", "qsubtype": "simple"},
            "yes",
        )
        wikitq = score_prediction(
            "wikitq",
            {"expected_answer": ["red", "blue"], "expected_canon": ["red", "blue"]},
            "blue, red",
        )

        self.assertTrue(tablefact.correct)
        self.assertEqual(tablefact.metric, "accuracy")
        self.assertTrue(wikitq.correct)
        self.assertEqual(wikitq.metric, "denotation_em")

    def test_model_stats_and_summary_requested_metrics(self) -> None:
        stats = CaseRunStats()
        stats.add(
            ModelCallRecord(
                role="test",
                prompt_tokens=12,
                completion_tokens=5,
                total_tokens=17,
                elapsed_seconds=0.1,
                prompt_chars=48,
                response_chars=20,
                prompt_preview="prompt",
                response_preview="response",
            )
        )
        record = {
            "sample_index": 0,
            "runtime_ok": True,
            "answer_correct": True,
            "score": 1.0,
            "metric": "accuracy",
            "qtype": "FactChecking",
            "qsubtype": "simple",
            "elapsed_seconds": 2.0,
            **model_stats_record(stats),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = build_summary(
                baseline="reactable",
                dataset="tablefact",
                records=[record],
                selected_cases=[{"case_id": "1"}],
                output_dir=root,
                model_config={"provider": "local", "model": "edge"},
                elapsed_seconds=2.5,
                max_workers=1,
                repeat_times=1,
                max_iters=5,
                max_demo=5,
                line_limit=parse_line_limit("inf"),
                temperature=0.6,
                max_tokens=1024,
                seed=1,
                sample_size=None,
                split="test",
                subset="small",
                cases_index=root / "cases_index.jsonl",
                mismatch_cases=root / "answer_mismatch_cases.jsonl",
                failure_cases=root / "failure_cases.jsonl",
            )

        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["avg_max_context_tokens_per_query"], 12)
        self.assertEqual(summary["avg_generated_tokens_per_query"], 5)
        self.assertEqual(summary["accuracy_on_all_cases"], 1.0)
        self.assertEqual(summary["average_case_seconds"], 2.0)
        self.assertEqual(summary["line_limit"], "inf")

    def test_orchestra_summary_reports_mode_specific_acc(self) -> None:
        records = [
            {
                "sample_index": 0,
                "runtime_ok": True,
                "answer_correct": True,
                "score": 1.0,
                "metric": "accuracy",
                "qtype": "FactChecking",
                "qsubtype": "simple",
                "elapsed_seconds": 3.0,
                "model_calls": 2,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "max_context_tokens": 80,
                "two_agent_answer_correct": True,
                "two_agent_score": 1.0,
                "three_agent_answer_correct": None,
                "three_agent_score": None,
            },
            {
                "sample_index": 1,
                "runtime_ok": True,
                "answer_correct": False,
                "score": 0.0,
                "metric": "accuracy",
                "qtype": "FactChecking",
                "qsubtype": "simple",
                "elapsed_seconds": 1.0,
                "model_calls": 2,
                "input_tokens": 60,
                "output_tokens": 10,
                "total_tokens": 70,
                "max_context_tokens": 40,
                "two_agent_answer_correct": False,
                "two_agent_score": 0.0,
                "three_agent_answer_correct": None,
                "three_agent_score": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = build_summary(
                baseline="orchestra",
                orchestra_mode="2agent",
                dataset="tablefact",
                records=records,
                selected_cases=[{"case_id": "1"}, {"case_id": "2"}],
                output_dir=root,
                model_config={"provider": "local", "model": "edge"},
                elapsed_seconds=3.2,
                max_workers=2,
                repeat_times=1,
                max_iters=5,
                max_demo=None,
                line_limit=10,
                temperature=0.7,
                max_tokens=1024,
                seed=1,
                sample_size=None,
                split="test",
                subset="small",
                cases_index=root / "cases_index.jsonl",
                mismatch_cases=root / "answer_mismatch_cases.jsonl",
                failure_cases=root / "failure_cases.jsonl",
            )

        self.assertEqual(summary["orchestra_mode"], "2agent")
        self.assertEqual(summary["brief_metrics"]["ACC"], 0.5)
        self.assertEqual(summary["brief_metrics"]["ACC_2agent"], 0.5)
        self.assertIsNone(summary["brief_metrics"]["ACC_3agent"])
        self.assertEqual(summary["orchestra_2agent_evaluated"], 2)
        self.assertEqual(summary["orchestra_3agent_evaluated"], 0)

    def test_orchestra_2agent_skips_decision_agent(self) -> None:
        model = FakeEdgeModel()
        runner = OrchestraRunner(
            model=model,  # type: ignore[arg-type]
            mode="2agent",
            repeat_times=1,
            max_iters=5,
            line_limit=10,
            temperature=0.7,
            max_tokens=128,
        )
        stats = CaseRunStats()
        with patch(
            "benchmarks.baselines.table_agent_baselines.read_normalized_table",
            return_value=object(),
        ), patch(
            "benchmarks.baselines.table_agent_baselines.table_formatter",
            return_value="fake table",
        ):
            output = runner.run_case(
                {"dataset": "tablefact", "question": "Is it true?"},
                Path("dummy.csv"),
                stats,
            )

        self.assertEqual(output["prediction"], "yes")
        self.assertEqual(output["two_agent_prediction"], "yes")
        self.assertIsNone(output["three_agent_prediction"])
        self.assertNotIn("orchestra_decision", model.roles)
        self.assertEqual(len(stats.calls), 1)

    def test_orchestra_both_runs_decision_and_keeps_two_agent_answer(self) -> None:
        model = FakeEdgeModel()
        runner = OrchestraRunner(
            model=model,  # type: ignore[arg-type]
            mode="both",
            repeat_times=1,
            max_iters=5,
            line_limit=10,
            temperature=0.7,
            max_tokens=128,
        )
        stats = CaseRunStats()
        with patch(
            "benchmarks.baselines.table_agent_baselines.read_normalized_table",
            return_value=object(),
        ), patch(
            "benchmarks.baselines.table_agent_baselines.table_formatter",
            return_value="fake table",
        ):
            output = runner.run_case(
                {"dataset": "tablefact", "question": "Is it true?"},
                Path("dummy.csv"),
                stats,
            )

        self.assertEqual(output["prediction"], "no")
        self.assertEqual(output["two_agent_prediction"], "yes")
        self.assertEqual(output["three_agent_prediction"], "no")
        self.assertIn("orchestra_decision", model.roles)
        self.assertEqual(len(stats.calls), 2)


if __name__ == "__main__":
    unittest.main()
