from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clover.executor import ExecutionResult
from clover.runtime.mixed_reasoning import pipeline as mixed_pipeline
from clover.runtime.pipeline import InflightStage, PipelineProfiler
from clover.runtime.round_loop import RoundLoopState
from clover.runtime.task import DocumentTaskItem, TableTaskItem, TASK_SUPERVISOR_REVIEW
from clover.runtime.table_reasoning import pipeline as table_pipeline
from clover.supervisor import SupervisorAgent


class MixedReasoningRuntimeTest(unittest.TestCase):
    def test_table_and_document_local_plans_execute_in_one_mixed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            table_item = _table_logic_item(table_path)
            document_item = _document_work_item()
            profiler = PipelineProfiler()
            executor_calls = []

            def fake_execute_execution_plan(execution_plan: object, **kwargs: object) -> ExecutionResult:
                executor_calls.append((execution_plan, dict(kwargs)))
                table_ns = mixed_pipeline._table_plan_namespace([table_item])
                document_ns = mixed_pipeline.document_pipeline._document_plan_namespace(
                    document_item
                )
                document_evidence = {
                    "kind": "map_group_evidence",
                    "worker_count": 1,
                    "included_count": 1,
                    "evidence_summary": "revenue evidence",
                }
                return ExecutionResult(
                    ok=True,
                    answer={
                        f"{table_ns}__answer_1": "France",
                        f"{document_ns}__document_evidence": document_evidence,
                    },
                    outputs={
                        f"{table_ns}__answer_1": "France",
                        f"{document_ns}__document_evidence": document_evidence,
                    },
                    collector_outputs={
                        f"{table_ns}__answer_1": "France",
                        f"{document_ns}__document_evidence": document_evidence,
                    },
                    traces=[
                        {
                            "node_id": f"{table_ns}__N0",
                            "output": f"{table_ns}__answer_1",
                            "status": "succeeded",
                            "fast_path_hit": True,
                        },
                        {
                            "node_id": f"{document_ns}__N0",
                            "output": f"{document_ns}__document_evidence",
                            "status": "succeeded",
                            "fast_path_hit": False,
                        },
                    ],
                    output_summaries={},
                )

            with InflightStage(
                stage_name="test_remote",
                max_workers=1,
                profiler=profiler,
            ) as remote_stage:
                adapter = mixed_pipeline._MixedRuntimeAdapter(
                    pending_table_remote=[],
                    pending_document_remote=[],
                    remote_stage=remote_stage,
                    table_sql_items=[],
                    document_code_items=[],
                    local_items=[
                        mixed_pipeline._MixedLocalWorkItem(
                            kind=mixed_pipeline.TABLE_WORK_DAG,
                            payload=table_item,
                            group_key=table_pipeline._dag_group_key(table_item.task),
                        ),
                        mixed_pipeline._MixedLocalWorkItem(
                            kind=mixed_pipeline.DOCUMENT_WORK_PLAN,
                            payload=document_item,
                            group_key="document::/tmp/report.pdf",
                        ),
                    ],
                    case_results=[],
                    finalized=set(),
                    document_round_steps={},
                    document_round_results={},
                    document_compact_observations={},
                    supervisor=SupervisorAgent(
                        remote_config={
                            "api_type": "chat_completions",
                            "model": "fake-model",
                        }
                    ),
                    remote_batch_size=8,
                    max_parallel_execution_units=4,
                    max_parallel_slm_node_jobs=2,
                    max_parallel_slm_sequences=3,
                    max_pending_slm_sequences=16,
                    max_retries=1,
                    validation_mode=table_pipeline.VALIDATION_NONE,
                    table_cache=None,
                    local_slm_config=None,
                    slm_client=None,
                    local_slm_dispatcher=None,
                    node_timeout_seconds=None,
                    profile_baseline=False,
                    profiler=profiler,
                )
                with patch(
                    "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                    side_effect=fake_execute_execution_plan,
                ):
                    progressed = mixed_pipeline._execute_mixed_local_once(adapter)

        self.assertTrue(progressed)
        self.assertEqual(len(executor_calls), 1)
        self.assertEqual(
            executor_calls[0][1]["collector_context"]["source"],
            "mixed_runtime",
        )
        self.assertEqual(adapter.case_results[0].answer, "France")
        self.assertEqual(document_item.task.status, TASK_SUPERVISOR_REVIEW)
        self.assertEqual(len(adapter.pending_document_remote), 1)
        self.assertIn("revenue evidence", document_item.compact_observation["evidence_summary"])

def _write_people_table(tmpdir: Path) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    table_path = tmpdir / "table.csv"
    table_path.write_text(
        "country,finalWorth\n"
        "France,100\n"
        "United States,200\n",
        encoding="utf-8",
    )
    return table_path


def _table_logic_item(table_path: Path) -> table_pipeline.LogicDagItem:
    sql = 'SELECT "country" AS "answer_1" FROM "table_1" LIMIT 1;'
    local_dsl = {
        "task_type": "table_reasoning.query",
        "question": "Which country appears first?",
        "sources": [
            {
                "id": "table_1",
                "type": "table",
                "path": str(table_path),
                "format": "csv",
            }
        ],
        "answer": {"name": "answer_1", "type": "string"},
    }
    task = TableTaskItem(
        case_id="table_case",
        answer_key="answer_1",
        task_type="table_reasoning.query",
        question=local_dsl["question"],
        answer_type="string",
        source_file=str(table_path),
        source_id="table_1",
        task_dsl=local_dsl,
        local_dsl=local_dsl,
        remote_dsl=local_dsl,
        context={},
    )
    logic_dag = table_pipeline.parse_remote_sql_to_logic_dag(sql, local_dsl)
    return table_pipeline.LogicDagItem(
        task=task,
        command_output=sql,
        output_type="sql",
        logic_dag=logic_dag,
        statements=(sql,),
    )


def _document_work_item() -> mixed_pipeline.document_pipeline._DocumentRoundWorkItem:
    task = DocumentTaskItem(
        case_id="document_case",
        answer_key="answer_2",
        task_type="document_reasoning",
        question="What is revenue?",
        answer_type="string",
        source_file="/tmp/report.pdf",
        source_id="document_1",
        task_dsl={},
        local_dsl={},
        remote_dsl={},
        context={},
    )
    return mixed_pipeline.document_pipeline._DocumentRoundWorkItem(
        task=task,
        state=RoundLoopState(),
        command="def prepare_jobs(context): pass",
        logic_dag={"task_type": "document_reasoning"},
        physical_plan={
            "task_type": "document_reasoning",
            "resources": [],
            "nodes": [
                {
                    "id": "N0",
                    "op": "map",
                    "dependency": [],
                    "input": [],
                    "params": {"local_instruction": "Extract evidence."},
                    "output": "document_evidence",
                    "output_type": "json",
                }
            ],
            "edges": [],
        },
    )
