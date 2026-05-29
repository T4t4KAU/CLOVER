from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from clover.runtime import TableReasoningCaseSpec, run_table_reasoning_v2_system


class TableReasoningV2RuntimeTest(unittest.TestCase):
    def test_batches_same_table_questions_and_merges_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            _self_made_sql("answer_1"),
                            _country_sql("answer_2"),
                        ]
                    ),
                    json.dumps(
                        {
                            "answer": {
                                "answer_1": True,
                                "answer_2": "United States",
                            },
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                ]
            )

            result = run_table_reasoning_v2_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "Is the person with the highest net worth self-made?",
                        "boolean",
                    ),
                    _case_spec(
                        "case_2",
                        Path(tmpdir),
                        table_path,
                        "What is the country of the person with the highest net worth?",
                        "string",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                local_batch_size=2,
                client=client,
                profile_baseline=True,
            )

        self.assertEqual(
            {item.case_id: item.answer for item in result.case_results},
            {"case_1": True, "case_2": "United States"},
        )
        self.assertEqual(result.profile["counters"]["commander_calls"], 1)
        self.assertEqual(result.profile["counters"]["remote_reporter_calls"], 1)
        self.assertGreaterEqual(result.profile["counters"]["reused_nodes"], 3)
        self.assertIn("baseline_executor", result.profile["stages"])
        self.assertIn("local_executor_speedup", result.profile["summary"])
        self.assertIn("provided shared table schema", client.chat.completions.requests[0]["messages"][0]["content"])
        self.assertIn(
            "Multiple answers may be reviewed together.",
            client.chat.completions.requests[1]["messages"][-1]["content"],
        )

    def test_partial_retry_only_replans_rejected_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            _self_made_sql("answer_1"),
                            'SELECT "country" AS "answer_2" FROM "table_1" '
                            'ORDER BY "finalWorth" ASC LIMIT 1;',
                        ]
                    ),
                    json.dumps(
                        {
                            "answer": {
                                "answer_1": True,
                                "answer_2": None,
                            },
                            "retry": True,
                            "new_sql": {"answer_2": _country_sql("answer_2")},
                        }
                    ),
                    json.dumps(
                        {
                            "answer": {"answer_2": "United States"},
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                ]
            )

            result = run_table_reasoning_v2_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "Self-made?", "boolean"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=2,
                local_batch_size=2,
                max_retries=1,
                client=client,
            )

        results = {item.case_id: item for item in result.case_results}
        self.assertTrue(results["case_1"].ok)
        self.assertEqual(results["case_1"].retry_count, 0)
        self.assertTrue(results["case_2"].ok)
        self.assertEqual(results["case_2"].retry_count, 1)
        self.assertEqual(results["case_2"].answer, "United States")
        self.assertEqual(result.profile["counters"]["remote_reporter_calls"], 2)

    def test_streams_finished_answers_before_all_commander_batches_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_table = _write_people_table(root / "first")
            second_table = _write_people_table(root / "second")
            client = _StatefulChatClient(
                [
                    json.dumps([_country_sql("answer_1")]),
                    json.dumps(
                        {
                            "answer": {"answer_1": "United States"},
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                    json.dumps([_self_made_sql("answer_2")]),
                    json.dumps(
                        {
                            "answer": {"answer_2": True},
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                ]
            )
            callback_events = []

            result = run_table_reasoning_v2_system(
                case_specs=[
                    _case_spec("case_1", root, first_table, "Country?", "string"),
                    _case_spec("case_2", root, second_table, "Self-made?", "boolean"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                local_batch_size=1,
                client=client,
                case_result_callback=lambda item: callback_events.append(
                    (item.case_id, len(client.chat.completions.requests))
                ),
            )

        self.assertEqual([item.case_id for item in result.case_results], ["case_1", "case_2"])
        self.assertEqual(callback_events[0], ("case_1", 2))
        self.assertEqual(callback_events[1], ("case_2", 4))

    def test_execution_error_uses_sql_repair_then_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT "missing" AS "answer_1" FROM "table_1";',
                        ]
                    ),
                    json.dumps(
                        {
                            "answer": {"answer_1": None},
                            "retry": True,
                            "new_sql": {
                                "answer_1": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";'
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "answer": {"answer_1": 2},
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                ]
            )

            result = run_table_reasoning_v2_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                local_batch_size=1,
                max_retries=1,
                client=client,
            )

        self.assertEqual(len(result.case_results), 1)
        self.assertTrue(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].answer, 2)
        self.assertEqual(result.case_results[0].retry_count, 1)
        self.assertEqual(result.profile["counters"]["remote_reporter_sql_repair_calls"], 1)
        self.assertIn("SQL repair for the failed answers only", client.chat.completions.requests[1]["messages"][-1]["content"])

    def test_fail_fast_interrupted_independent_answers_are_requeued(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT "missing" AS "answer_1" FROM "table_1";',
                            _country_sql("answer_2"),
                        ]
                    ),
                    json.dumps(
                        {
                            "answer": {"answer_1": None},
                            "retry": True,
                            "new_sql": {
                                "answer_1": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";'
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "answer": {"answer_2": "United States"},
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                    json.dumps(
                        {
                            "answer": {"answer_1": 2},
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                ]
            )

            result = run_table_reasoning_v2_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=2,
                local_batch_size=2,
                max_retries=1,
                client=client,
            )

        results = {item.case_id: item for item in result.case_results}
        self.assertEqual(results["case_1"].answer, 2)
        self.assertEqual(results["case_1"].retry_count, 1)
        self.assertEqual(results["case_2"].answer, "United States")
        self.assertEqual(results["case_2"].retry_count, 0)
        self.assertEqual(result.profile["counters"]["merged_plan_count"], 3)


def _write_people_table(tmpdir: Path) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    table_path = tmpdir / "table.csv"
    table_path.write_text(
        "selfMade,country,finalWorth\n"
        "False,France,100\n"
        "True,United States,200\n",
        encoding="utf-8",
    )
    return table_path


def _case_spec(
    case_id: str,
    base_dir: Path,
    table_path: Path,
    question: str,
    answer_type: str,
) -> TableReasoningCaseSpec:
    return TableReasoningCaseSpec(
        case_id=case_id,
        base_dir=base_dir,
        task_dsl={
            "task_type": "table_reasoning_v1",
            "question": question,
            "sources": [{"type": "table", "file": str(table_path)}],
            "answer": {"name": "answer", "type": answer_type},
        },
    )


def _self_made_sql(answer_key: str) -> str:
    return (
        f'SELECT "selfMade" AS "{answer_key}" FROM "table_1" '
        'ORDER BY "finalWorth" DESC LIMIT 1;'
    )


def _country_sql(answer_key: str) -> str:
    return (
        f'SELECT "country" AS "{answer_key}" FROM "table_1" '
        'ORDER BY "finalWorth" DESC LIMIT 1;'
    )


class _StatefulChatClient:
    def __init__(self, output_texts: list[str]) -> None:
        self.chat = SimpleNamespace(
            completions=_StatefulChatCompletions(output_texts),
        )


class _StatefulChatCompletions:
    def __init__(self, output_texts: list[str]) -> None:
        self._output_texts = output_texts
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
            }
        )
        return _FakeChatResponse(self._output_texts[len(self.requests) - 1])


class _FakeChatResponse:
    id = "v2_runtime_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
