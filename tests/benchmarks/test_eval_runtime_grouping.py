from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.databench import eval as databench_eval
from benchmarks.tablebench import eval as tablebench_eval
from clover.runtime import TableReasoningCaseSpec


class EvalRuntimeGroupingTest(unittest.TestCase):
    def test_tablebench_runs_source_groups_in_one_system_instance(self) -> None:
        calls = []

        def fake_run_table_reasoning_system(**kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(task_items={}, profile={}, case_results=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "benchmarks.tablebench.eval.run_table_reasoning_system",
                side_effect=fake_run_table_reasoning_system,
            ):
                results = tablebench_eval._run_system_groups(
                    spec_groups=[[_spec("case_1"), _spec("case_2")], [_spec("case_3")]],
                    remote_config={"model": "fake"},
                    local_slm_config=None,
                    remote_batch_size=4,
                    remote_concurrency=2,
                    max_parallel_execution_units=8,
                    max_parallel_slm_node_jobs=1,
                    max_parallel_slm_sequences=3,
                    max_pending_slm_sequences=16,
                    max_retries=1,
                    validation_mode="none",
                    max_workers=4,
                    case_result_callback=lambda result: None,
                    profile_baseline=False,
                    records_by_case={},
                    completed_records=[],
                    started_by_case={},
                    output_dir=Path(tmpdir),
                    progress_bar=None,
                    progress_lock=Lock(),
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [spec.case_id for spec in calls[0]["case_specs"]],
            ["case_1", "case_2", "case_3"],
        )

    def test_tablebench_eval_passes_builder_specs_without_prebuilding_dsl(self) -> None:
        calls = []

        def fake_run_table_reasoning_system(**kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(task_items={}, profile={}, case_results=[])

        selected_cases = [
            {
                "dataset_id": "dataset_a",
                "case_id": "case-a",
                "case_index": 0,
                "question": "What is the sum of sales?",
                "expected_answer": "15",
                "answer_type": "number",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablebench"
            output_dir = Path(tmpdir) / "run"
            with patch(
                "benchmarks.tablebench.eval.run_table_reasoning_system",
                side_effect=fake_run_table_reasoning_system,
            ), patch("benchmarks.tablebench.adapter.load_tablebench_task") as load_task:
                records, profile = tablebench_eval._run_tablebench_cases(
                    tablebench_root=root,
                    output_dir=output_dir,
                    selected_cases=selected_cases,
                    remote_config={"model": "fake"},
                    local_slm_config={"model": "fake-slm"},
                    max_workers=None,
                    max_retries=1,
                    validation_mode="none",
                    remote_batch_size=4,
                    remote_concurrency=2,
                    max_parallel_execution_units=8,
                    max_parallel_slm_node_jobs=1,
                    max_parallel_slm_sequences=3,
                    max_pending_slm_sequences=16,
                    profile_baseline=False,
                    progress_bar=None,
                )

        load_task.assert_not_called()
        self.assertEqual(len(records), 1)
        self.assertEqual(profile["eval_startup"]["loaded_cases"], 1)
        self.assertEqual(len(calls), 1)
        spec = calls[0]["case_specs"][0]
        self.assertEqual(spec.task_dsl, {})
        self.assertEqual(spec.builder["question"], "What is the sum of sales?")
        self.assertEqual(spec.builder["hints"]["category"], "NumericalReasoning")
        self.assertEqual(spec.metadata["expected_answer"], "15")

    def test_databench_runs_source_groups_in_one_system_instance(self) -> None:
        calls = []

        def fake_run_table_reasoning_system(**kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(task_items={}, profile={})

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "benchmarks.databench.eval.run_table_reasoning_system",
                side_effect=fake_run_table_reasoning_system,
            ):
                results = databench_eval._run_table_system_groups(
                    spec_groups=[[_spec("case_1")], [_spec("case_2"), _spec("case_3")]],
                    remote_config={"model": "fake"},
                    local_slm_config=None,
                    remote_batch_size=4,
                    remote_concurrency=2,
                    max_parallel_execution_units=8,
                    max_parallel_slm_node_jobs=1,
                    max_parallel_slm_sequences=3,
                    max_pending_slm_sequences=16,
                    max_retries=1,
                    validation_mode="none",
                    system_worker_count=4,
                    case_result_callback=lambda result: None,
                    profile_baseline=False,
                    records_by_case={},
                    final_records=[],
                    started_by_case={},
                    output_dir=Path(tmpdir),
                    progress_bar=None,
                    progress_lock=Lock(),
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [spec.case_id for spec in calls[0]["case_specs"]],
            ["case_1", "case_2", "case_3"],
        )


def _spec(case_id: str) -> TableReasoningCaseSpec:
    return TableReasoningCaseSpec(
        case_id=case_id,
        task_dsl={},
        base_dir=Path("."),
        answer_key=f"answer_{case_id}",
    )


if __name__ == "__main__":
    unittest.main()
