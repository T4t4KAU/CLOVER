from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.executor import ExecutionResult
from clover.runtime import DocumentReasoningCaseSpec, run_document_reasoning_system
from clover.runtime.document_reasoning import pipeline as document_pipeline
from clover.runtime.round_loop import RoundLoopState
from clover.runtime.task import DocumentTaskItem, TASK_FAILED, TASK_SUPERVISOR_REVIEW


class DocumentReasoningRuntimeTest(unittest.TestCase):
    def test_executes_ready_document_plans_as_one_namespaced_batch(self) -> None:
        first = _document_work_item("case_1", "answer_1", "What is revenue?")
        second = _document_work_item("case_2", "answer_2", "What is profit?")
        pending_remote = []
        case_results = []
        finalized = set()
        round_steps = {}
        round_results = {}
        compact_observations = {}
        executor_calls = []

        def fake_execute_execution_plan(execution_plan: object, **kwargs: object) -> ExecutionResult:
            executor_calls.append((execution_plan, dict(kwargs)))
            units = getattr(execution_plan, "units")
            self.assertEqual(
                [unit.node["params"]["question_context"] for unit in units],
                ["What is revenue?", "What is profit?"],
            )
            return ExecutionResult(
                ok=True,
                answer={
                    "answer_1__r0__document_evidence": {
                        "kind": "map_group_evidence",
                        "worker_count": 1,
                        "included_count": 1,
                        "evidence_summary": "revenue evidence",
                    },
                    "answer_2__r0__document_evidence": {
                        "kind": "map_group_evidence",
                        "worker_count": 1,
                        "included_count": 1,
                        "evidence_summary": "profit evidence",
                    },
                },
                outputs={},
                collector_outputs={
                    "answer_1__r0__document_evidence": {
                        "kind": "map_group_evidence",
                        "worker_count": 1,
                        "included_count": 1,
                        "evidence_summary": "revenue evidence",
                    },
                    "answer_2__r0__document_evidence": {
                        "kind": "map_group_evidence",
                        "worker_count": 1,
                        "included_count": 1,
                        "evidence_summary": "profit evidence",
                    },
                },
                traces=[
                    {
                        "node_id": "answer_1__r0__N0",
                        "output": "answer_1__r0__document_evidence",
                        "status": "succeeded",
                        "fast_path_hit": False,
                    },
                    {
                        "node_id": "answer_2__r0__N0",
                        "output": "answer_2__r0__document_evidence",
                        "status": "succeeded",
                        "fast_path_hit": False,
                    },
                ],
                output_summaries={},
            )

        with patch(
            "clover.runtime.document_reasoning.pipeline.execute_execution_plan",
            side_effect=fake_execute_execution_plan,
        ):
            progressed = document_pipeline._execute_document_plan_batch(
                local_items=[first, second],
                pending_remote=pending_remote,
                case_results=case_results,
                finalized=finalized,
                round_steps=round_steps,
                round_results=round_results,
                compact_observations=compact_observations,
                local_slm_config=None,
                slm_client=None,
                local_slm_dispatcher=None,
                max_parallel_execution_units=8,
                max_parallel_slm_node_jobs=2,
                max_parallel_slm_sequences=4,
                max_pending_slm_sequences=32,
                node_timeout_seconds=None,
                max_retries=1,
                profiler=document_pipeline.PipelineProfiler(),
            )

        self.assertTrue(progressed)
        self.assertEqual(len(executor_calls), 1)
        self.assertEqual(len(pending_remote), 2)
        self.assertEqual(first.task.status, TASK_SUPERVISOR_REVIEW)
        self.assertEqual(second.task.status, TASK_SUPERVISOR_REVIEW)
        self.assertIn("revenue evidence", first.compact_observation["evidence_summary"])
        self.assertIn("profit evidence", second.compact_observation["evidence_summary"])
        self.assertIn("answer_1", compact_observations)
        self.assertIn("answer_2", compact_observations)

    def test_document_executor_failure_finalizes_case_without_synthesis(self) -> None:
        work_item = _document_work_item("case_1", "answer_1", "What is revenue?")
        pending_remote = []
        case_results = []
        finalized = set()
        round_steps = {}
        round_results = {}
        error = {"type": "RuntimeError", "message": "worker failed"}

        def fake_execute_execution_plan(execution_plan: object, **kwargs: object) -> ExecutionResult:
            del execution_plan, kwargs
            return ExecutionResult(
                ok=False,
                answer=None,
                outputs={},
                collector_outputs={},
                traces=[
                    {
                        "node_id": "answer_1__r0__N0",
                        "output": "answer_1__r0__document_evidence",
                        "status": "failed",
                        "error": error,
                    },
                ],
                output_summaries={},
                error=error,
            )

        with patch(
            "clover.runtime.document_reasoning.pipeline.execute_execution_plan",
            side_effect=fake_execute_execution_plan,
        ):
            progressed = document_pipeline._execute_document_plan_batch(
                local_items=[work_item],
                pending_remote=pending_remote,
                case_results=case_results,
                finalized=finalized,
                round_steps=round_steps,
                round_results=round_results,
                compact_observations={},
                local_slm_config=None,
                slm_client=None,
                local_slm_dispatcher=None,
                max_parallel_execution_units=8,
                max_parallel_slm_node_jobs=2,
                max_parallel_slm_sequences=4,
                max_pending_slm_sequences=32,
                node_timeout_seconds=None,
                max_retries=1,
                profiler=document_pipeline.PipelineProfiler(),
            )

        self.assertTrue(progressed)
        self.assertEqual(pending_remote, [])
        self.assertEqual(work_item.task.status, TASK_FAILED)
        self.assertIn("answer_1", finalized)
        self.assertEqual(len(case_results), 1)
        self.assertFalse(case_results[0].ok)
        self.assertEqual(case_results[0].error, error)
        self.assertFalse(round_results["answer_1"].ok)
        self.assertEqual(round_results["answer_1"].error, error)
        self.assertEqual(len(round_steps["answer_1"]), 1)

    @unittest.skipUnless(
        importlib.util.find_spec("fitz") is not None,
        "PyMuPDF is required for document runtime tests",
    )
    def test_runs_minions_style_supervisor_rounds_until_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_path = root / "report.pdf"
            _write_pdf(
                document_path,
                [
                    "The annual report introduces the business.",
                    "Revenue was $140 million in fiscal 2023.",
                ],
            )
            remote_client = _StatefulChatClient(
                [
                    _supervisor_decompose_code("Extract fiscal 2023 revenue evidence."),
                    json.dumps(
                        {
                            "answer": None,
                            "sufficient": False,
                            "explanation": "No usable evidence was returned.",
                            "feedback": "Focus on fiscal 2023 revenue.",
                            "scratchpad": "Need the revenue value and citation.",
                            "next_python_code": _python_code(
                                "Focus on fiscal 2023 revenue and include the citation."
                            ),
                        }
                    ),
                    json.dumps(
                        {
                            "answer": "$140 million",
                            "sufficient": True,
                            "explanation": "Worker evidence states the value.",
                        }
                    ),
                ]
            )
            slm_client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "answer": None,
                            "citation": None,
                            "explanation": "No relevant evidence found.",
                        }
                    ),
                    json.dumps(
                        {
                            "answer": "$140 million",
                            "citation": "Revenue was $140 million in fiscal 2023.",
                            "explanation": "The excerpt states the value directly.",
                        }
                    ),
                ]
            )

            with patch.dict(
                "os.environ",
                {"CLOVER_RESOURCE_CACHE_ROOT": str(root / "resource_cache")},
            ):
                result = run_document_reasoning_system(
                    case_specs=[
                        DocumentReasoningCaseSpec(
                            case_id="case_1",
                            base_dir=root,
                            task_dsl={
                                "task_type": "document_reasoning",
                                "question": "What was revenue in fiscal 2023?",
                                "sources": [
                                    {
                                        "type": "pdf",
                                        "file": document_path.name,
                                    }
                                ],
                                "answer": {"name": "answer", "type": "string"},
                            },
                        )
                    ],
                    remote_config={
                        "api_type": "chat_completions",
                        "model": "fake-remote",
                    },
                    local_slm_config={
                        "api_type": "chat_completions",
                        "model": "fake-slm",
                    },
                    client=remote_client,
                    slm_client=slm_client,
                    max_retries=1,
                    max_parallel_execution_units=1,
                )

        self.assertEqual(len(result.case_results), 1)
        self.assertTrue(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].answer, "$140 million")
        self.assertEqual(result.case_results[0].retry_count, 1)

        round_result = result.round_results["answer_1"]
        self.assertTrue(round_result.ok)
        self.assertEqual(len(round_result.rounds), 2)
        self.assertFalse(round_result.rounds[0].supervisor_result.decision.sufficient)
        self.assertTrue(round_result.rounds[1].supervisor_result.decision.sufficient)
        self.assertEqual(
            result.profile["counters"]["supervisor_decompose_calls"],
            1,
        )
        self.assertEqual(
            result.profile["counters"]["supervisor_synthesis_calls"],
            2,
        )

        first_synthesis_prompt = remote_client.chat.completions.requests[1]["messages"][
            -1
        ]["content"]
        final_synthesis_prompt = remote_client.chat.completions.requests[2]["messages"][
            -1
        ]["content"]
        self.assertIn('"next_python_code"', first_synthesis_prompt)
        self.assertIn("- workers: 1", first_synthesis_prompt)
        self.assertNotIn("job_outputs", first_synthesis_prompt)
        self.assertNotIn("job_manifests", first_synthesis_prompt)
        self.assertNotIn("No relevant evidence found.", first_synthesis_prompt)
        self.assertIn("This is the last supervisor pass", final_synthesis_prompt)
        self.assertNotIn('"next_python_code"', final_synthesis_prompt)
        self.assertNotIn("job_outputs", final_synthesis_prompt)
        self.assertNotIn("job_manifests", final_synthesis_prompt)
        self.assertIn(
            "Focus on fiscal 2023 revenue and include the citation.",
            round_result.rounds[1].command_output,
        )

        first_collector = round_result.rounds[0].execution_result.collector_outputs[
            "document_evidence"
        ]
        second_collector = round_result.rounds[1].execution_result.collector_outputs[
            "document_evidence"
        ]
        self.assertEqual(first_collector["included_count"], 0)
        self.assertEqual(second_collector["included_count"], 1)
        self.assertIn("$140 million", second_collector["evidence_summary"])

    @unittest.skipUnless(
        importlib.util.find_spec("fitz") is not None,
        "PyMuPDF is required for document runtime tests",
    )
    def test_reports_unparseable_document_python_before_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_path = root / "report.pdf"
            _write_pdf(document_path, ["Revenue was $140 million in fiscal 2023."])
            remote_client = _StatefulChatClient(
                [
                    (
                        "def prepare_jobs(context):\n"
                        "    return [JobManifest(chunk='chunk_0', task='Inspect.', advice='')]"
                    ),
                    json.dumps(
                        {
                            "answer": None,
                            "sufficient": False,
                            "explanation": "The worker code did not match the local contract.",
                            "feedback": "Generate contract-compatible worker code.",
                            "next_python_code": _python_code(
                                "Extract fiscal 2023 revenue evidence."
                            ),
                        }
                    ),
                    json.dumps(
                        {
                            "answer": "$140 million",
                            "sufficient": True,
                            "explanation": "Worker evidence states the value.",
                        }
                    ),
                ]
            )
            slm_client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "answer": "$140 million",
                            "citation": "Revenue was $140 million in fiscal 2023.",
                            "explanation": "The excerpt states the value directly.",
                        }
                    )
                ]
            )

            with patch.dict(
                "os.environ",
                {"CLOVER_RESOURCE_CACHE_ROOT": str(root / "resource_cache")},
            ):
                result = run_document_reasoning_system(
                    case_specs=[
                        DocumentReasoningCaseSpec(
                            case_id="case_1",
                            base_dir=root,
                            task_dsl={
                                "task_type": "document_reasoning",
                                "question": "What was revenue in fiscal 2023?",
                                "sources": [{"type": "pdf", "file": document_path.name}],
                                "answer": {"name": "answer", "type": "string"},
                            },
                        )
                    ],
                    remote_config={
                        "api_type": "chat_completions",
                        "model": "fake-remote",
                    },
                    local_slm_config={
                        "api_type": "chat_completions",
                        "model": "fake-slm",
                    },
                    client=remote_client,
                    slm_client=slm_client,
                    max_retries=1,
                    max_parallel_execution_units=1,
                )

        self.assertTrue(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].answer, "$140 million")
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 2)
        self.assertEqual(result.profile["summary"]["remote_calls"], 3)
        report_prompt = remote_client.chat.completions.requests[1]["messages"][-1][
            "content"
        ]
        self.assertNotIn("Repair mode", report_prompt)
        self.assertIn("DocumentPlanParseError", report_prompt)


def _supervisor_decompose_code(task: str) -> str:
    return f"""```python
{_python_code(task)}
```"""


def _document_work_item(
    case_id: str,
    answer_key: str,
    question: str,
) -> document_pipeline._DocumentRoundWorkItem:
    task = DocumentTaskItem(
        case_id=case_id,
        answer_key=answer_key,
        task_type="document_reasoning",
        question=question,
        answer_type="string",
        source_file="/tmp/document.txt",
        source_id="document_1",
        task_dsl={},
        local_dsl={},
        remote_dsl={},
        context={},
    )
    return document_pipeline._DocumentRoundWorkItem(
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


def _python_code(task: str) -> str:
    return f"""
def prepare_jobs(context, prev_job_manifests=None, prev_job_outputs=None):
    job_manifests = []
    for document in context:
        chunks = chunk_by_section(document, max_chunk_size=3000, overlap=20)
        advice = "Return only explicitly stated values with citations."
        for chunk in chunks:
            job_manifests.append(JobManifest(chunk=chunk, task={task!r}, advice=advice))
    return job_manifests

def transform_outputs(jobs):
    evidence = []
    for job in jobs:
        output = job.output
        if output.answer is not None or output.citation is not None:
            evidence.append(
                "answer: " + str(output.answer) + "\\n"
                "citation: " + str(output.citation) + "\\n"
                "explanation: " + str(output.explanation)
            )
    return "\\n\\n".join(evidence)
""".strip()


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    import fitz

    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(path)
    finally:
        document.close()


class _StatefulChatClient:
    def __init__(self, outputs: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_StatefulChatCompletions(outputs))


class _StatefulChatCompletions:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
            }
        )
        output = self.outputs.pop(0)
        return _FakeChatResponse(output)


class _FakeChatResponse:
    id = "document_runtime_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
