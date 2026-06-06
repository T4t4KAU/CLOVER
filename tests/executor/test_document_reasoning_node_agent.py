from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from clover.executor import ExecutionPlanBuilder, execute_execution_plan


class DocumentReasoningNodeAgentTest(unittest.TestCase):
    def test_runs_one_chunk_worker_through_single_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_store = Path(tmpdir) / "chunks.jsonl"
            _write_chunks(
                chunk_store,
                [
                    {
                        "chunk_id": "chunk_0",
                        "text": "Revenue was $140 million in fiscal 2023.",
                        "page_start": 4,
                        "page_end": 4,
                        "page_indexing": "zero_based",
                        "char_start": 0,
                        "char_end": 42,
                    }
                ],
            )

            client = _FakeChatClient(
                json.dumps(
                    {
                        "answer": "$140 million",
                        "citation": "Revenue was $140 million in fiscal 2023.",
                        "explanation": "The excerpt states the revenue directly.",
                    }
                )
            )
            result = _execute_plan(
                _document_map_plan(chunk_store),
                slm_config=_slm_config(),
                slm_client=client,
            )

        output_name = "G0__0__document_1_chunk_0"
        self.assertTrue(result.ok)
        self.assertIsNone(result.answer)
        self.assertEqual(result.fast_path_hits, 0)
        self.assertEqual(result.fast_path_misses, 1)
        self.assertEqual(result.traces[0]["execution_path"], "agent_loop")
        self.assertEqual(result.traces[0]["agent_loop"]["iterations"], 1)
        sequence_trace = result.traces[0]["agent_loop"]["steps"][0]["sequence"]
        self.assertEqual(sequence_trace["prompt_kind"], "document_worker")
        self.assertEqual(
            sequence_trace["leaf_key"],
            [
                "agent:data",
                "family:document_reasoning",
                "interface:chunk_worker",
                "tool:text_excerpt",
                "contract:evidence_json",
                "mode:initial",
            ],
        )
        self.assertEqual(result.outputs[output_name]["answer"], "$140 million")
        self.assertEqual(result.outputs[output_name]["chunk"]["chunk_id"], "chunk_0")
        self.assertEqual(result.outputs[output_name]["chunk"]["page_start"], 4)
        self.assertNotIn("resource_id", result.outputs[output_name]["chunk"])

        collected = result.collector_outputs["G0"]
        self.assertEqual(collected["included_count"], 1)
        self.assertIn("Chunk chunk_0 (pages 4-4)", collected["evidence_summary"])
        self.assertIn("$140 million", collected["evidence_summary"])
        self.assertNotIn("document_1:chunk_0", collected["evidence_summary"])
        prompt = client.chat.completions.last_request["messages"][-1]["content"]
        self.assertIn("Return only explicitly stated values.", prompt)
        self.assertIn("Do not combine evidence across chunks.", prompt)
        self.assertIn("Preserve fiscal periods, units, line item names", prompt)
        self.assertIn("Irrelevant excerpt", prompt)
        self.assertNotIn("document_1:chunk_0", prompt)
        self.assertLess(
            prompt.index("Return one JSON object only:"),
            prompt.index("Task:"),
        )
        self.assertLess(prompt.index("Rules:"), prompt.index("Task:"))
        self.assertLess(prompt.index("Examples:"), prompt.index("Task:"))
        self.assertLess(prompt.index("Task:"), prompt.index("Document excerpt:"))
        self.assertLess(prompt.index("Advice:"), prompt.index("Document excerpt:"))
        self.assertLess(
            prompt.index("Return one JSON object only:"),
            prompt.index("Document excerpt:"),
        )

    def test_invalid_worker_json_becomes_soft_worker_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_store = Path(tmpdir) / "chunks.jsonl"
            _write_chunks(
                chunk_store,
                [
                    {
                        "chunk_id": "chunk_0",
                        "text": "No relevant information here.",
                    }
                ],
            )

            result = _execute_plan(
                _document_map_plan(chunk_store),
                slm_config=_slm_config(),
                slm_client=_FakeChatClient("not json"),
            )

        self.assertTrue(result.ok)
        self.assertIsNone(result.failing_node)
        output = result.outputs["G0__0__document_1_chunk_0"]
        self.assertIsNone(output["answer"])
        self.assertIn("worker failed", output["explanation"])
        self.assertEqual(result.collector_outputs["G0"]["included_count"], 0)
        self.assertEqual(result.traces[0]["execution_path"], "agent_loop")
        self.assertTrue(result.traces[0]["soft_failure"])
        self.assertEqual(result.traces[0]["agent_loop"]["iterations"], 1)
        self.assertEqual(len(result.traces[0]["agent_loop"]["steps"]), 1)

    def test_static_transform_outputs_collector_is_used_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_store = Path(tmpdir) / "chunks.jsonl"
            _write_chunks(
                chunk_store,
                [
                    {
                        "chunk_id": "chunk_0",
                        "text": "Revenue was $140 million in fiscal 2023.",
                    }
                ],
            )
            plan = _document_map_plan(chunk_store)
            plan["static_collectors"] = [
                {
                    "id": "document_evidence",
                    "kind": "minions_transform_outputs",
                    "function_name": "transform_outputs",
                    "source": (
                        "def transform_outputs(jobs):\n"
                        "    return 'TRANSFORMED: ' + jobs[0].output.answer\n"
                    ),
                    "output": "document_evidence",
                }
            ]

            result = _execute_plan(
                plan,
                slm_config=_slm_config(),
                slm_client=_FakeChatClient(
                    json.dumps(
                        {
                            "answer": "$140 million",
                            "citation": "Revenue was $140 million.",
                            "explanation": "The excerpt states it.",
                        }
                    )
                ),
            )

        self.assertTrue(result.ok)
        collected = result.collector_outputs["document_evidence"]
        self.assertEqual(collected["kind"], "minions_transform_outputs")
        self.assertEqual(collected["evidence_summary"], "TRANSFORMED: $140 million")

    def test_node_timeout_is_reported_as_node_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_store = Path(tmpdir) / "chunks.jsonl"
            _write_chunks(
                chunk_store,
                [
                    {
                        "chunk_id": "chunk_0",
                        "text": "Revenue was $140 million in fiscal 2023.",
                    }
                ],
            )

            client = _FakeChatClient(
                json.dumps(
                    {
                        "answer": "$140 million",
                        "citation": "Revenue was $140 million in fiscal 2023.",
                        "explanation": "The excerpt states the revenue directly.",
                    }
                ),
                sleep_seconds=0.5,
            )
            started = time.perf_counter()
            result = _execute_plan(
                _document_map_plan(chunk_store),
                slm_config=_slm_config(),
                slm_client=client,
                node_timeout_seconds=0.01,
            )
            elapsed = time.perf_counter() - started

        self.assertTrue(result.ok)
        self.assertLess(elapsed, 0.3)
        self.assertIsNone(result.failing_node)
        self.assertEqual(result.traces[0]["execution_path"], "timeout")
        self.assertEqual(result.traces[0]["status"], "failed")
        self.assertTrue(result.traces[0]["soft_failure"])


def _write_chunks(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _execute_plan(plan: dict, **kwargs):
    return execute_execution_plan(
        ExecutionPlanBuilder.default().build(plan),
        collector_context=plan,
        **kwargs,
    )


def _document_map_plan(chunk_store: Path) -> dict:
    return {
        "task_type": "document_reasoning",
        "question": "What was revenue in fiscal 2023?",
        "resources": [
            {
                "id": "document_1:chunk_0",
                "type": "document_chunk",
                "source": "document_1",
                "source_type": "pdf",
                "path": str(chunk_store),
                "format": "text",
                "item_id": "chunk_0",
                "chunk_id": "chunk_0",
            }
        ],
        "map_groups": [
            {
                "id": "G0",
                "op": "map",
                "input": {"chunks": ["document_1:chunk_0"]},
                "params": {
                    "local_instruction": "Extract the fiscal 2023 revenue.",
                    "local_guidance": "Return only explicitly stated values.",
                },
                "output": "G0",
                "output_type": "jsonl",
            }
        ],
        "edges": [],
    }


def _slm_config() -> dict:
    return {
        "api_type": "chat_completions",
        "model": "fake-slm",
        "temperature": 0,
    }


class _FakeChatClient:
    def __init__(self, output_text: str, *, sleep_seconds: float = 0.0) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(output_text, sleep_seconds=sleep_seconds),
        )


class _FakeChatCompletions:
    def __init__(self, output_text: str, *, sleep_seconds: float = 0.0) -> None:
        self.output_text = output_text
        self.sleep_seconds = sleep_seconds
        self.last_request: dict[str, object] = {}

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.last_request = kwargs
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return _FakeChatResponse(self.output_text)


class _FakeChatResponse:
    id = "document_worker_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
