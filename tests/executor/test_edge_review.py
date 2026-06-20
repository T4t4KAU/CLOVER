from __future__ import annotations

import unittest
from types import SimpleNamespace

from clover.executor.edge_review import (
    detect_edge_review_opportunity,
    run_edge_local_review,
)
from clover.executor.slm_dispatcher import LocalSlmSequenceDispatcher


class EdgeLocalReviewTest(unittest.TestCase):
    def test_normalizes_one_local_field_with_grounded_support(self) -> None:
        client = _FakeChatClient(
            '{"route":"normalize","a":"France","support":["e0"],'
            '"operation":"identity","reason":"country field"}'
        )
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config=_safe_config(),
            client=client,
            max_parallel_sequences=1,
            max_pending_sequences=4,
            slm_scheduler="fifo",
        )
        try:
            result = run_edge_local_review(
                question="Which country appears first?",
                answer_type="string",
                evidence=_one_row_evidence(),
                scope="action_group_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="answer_1",
            )
        finally:
            dispatcher.close()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "France")
        self.assertEqual(result.route, "normalize")
        self.assertIn('"path":"/obs/0/res/rows/0/country"', result.prompt)

    def test_rejects_uncited_hallucinated_value(self) -> None:
        client = _FakeChatClient(
            '{"route":"normalize","a":"Spain","support":["e0"],' '"operation":"identity"}'
        )
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config=_safe_config(),
            client=client,
            max_parallel_sequences=1,
            max_pending_sequences=4,
            slm_scheduler="fifo",
        )
        try:
            result = run_edge_local_review(
                question="Which country appears first?",
                answer_type="string",
                evidence=_one_row_evidence(),
                scope="action_group_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="answer_1",
            )
        finally:
            dispatcher.close()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.accepted)
        self.assertEqual(result.route, "escalate")
        self.assertIn("not present", result.validation_error or "")

    def test_detects_proactive_field_selection_opportunity(self) -> None:
        opportunity = detect_edge_review_opportunity(
            question="Which country appears first?",
            answer_type="string",
            evidence=_one_row_evidence(),
            slm_config=_safe_config(),
        )

        self.assertIsNotNone(opportunity)
        assert opportunity is not None
        self.assertEqual(opportunity.kind, "field_selection")
        self.assertTrue(opportunity.proactive)
        self.assertEqual(opportunity.row_count, 1)
        self.assertEqual(opportunity.column_count, 2)

    def test_does_not_delegate_multirow_argmax_to_proactive_edge_review(self) -> None:
        opportunity = detect_edge_review_opportunity(
            question="Which country has the highest score?",
            answer_type="string",
            evidence={
                "n": 3,
                "cols": ["country", "score"],
                "rows": [
                    {"country": "France", "score": 5},
                    {"country": "Spain", "score": 8},
                    {"country": "Italy", "score": 6},
                ],
            },
            slm_config=_safe_config(),
        )

        self.assertIsNotNone(opportunity)
        assert opportunity is not None
        self.assertEqual(opportunity.kind, "deterministic_result_review")
        self.assertFalse(opportunity.proactive)

    def test_extracts_one_numeric_value_from_percent_or_unit_text(self) -> None:
        client = _FakeChatClient(
            '{"route":"normalize","a":47.5,"support":["e0"],' '"operation":"percent_value"}'
        )
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config=_safe_config(),
            client=client,
            max_parallel_sequences=1,
            max_pending_sequences=4,
            slm_scheduler="fifo",
        )
        try:
            result = run_edge_local_review(
                question="What percentage was reported?",
                answer_type="number",
                evidence={"rows": [{"rate": "47.5%"}], "cols": ["rate"], "n": 1},
                scope="format_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="answer_1",
            )
        finally:
            dispatcher.close()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, 47.5)
        self.assertEqual(result.opportunity.kind, "value_normalization")

    def test_validates_bounded_text_normalization(self) -> None:
        client = _FakeChatClient(
            '{"route":"normalize","a":"France","support":["e0"],'
            '"operation":"strip_parenthetical"}'
        )
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config=_safe_config(),
            client=client,
            max_parallel_sequences=1,
            max_pending_sequences=4,
            slm_scheduler="fifo",
        )
        try:
            result = run_edge_local_review(
                question="Which country?",
                answer_type="string",
                evidence={
                    "rows": [{"country": "France (FRA)"}],
                    "cols": ["country"],
                    "n": 1,
                },
                scope="format_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="answer_1",
            )
        finally:
            dispatcher.close()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "France")

    def test_validates_simple_boolean_review_operation(self) -> None:
        client = _FakeChatClient(
            '{"route":"normalize","a":false,"support":["e0","e1"],' '"operation":"and"}'
        )
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config=_safe_config(),
            client=client,
            max_parallel_sequences=1,
            max_pending_sequences=4,
            slm_scheduler="fifo",
        )
        try:
            result = run_edge_local_review(
                question="Are both local conditions true?",
                answer_type="boolean",
                evidence={
                    "ok": True,
                    "res": {
                        "n": 1,
                        "cols": ["left", "right"],
                        "rows": [{"left": True, "right": False}],
                    },
                },
                scope="format_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="answer_1",
            )
        finally:
            dispatcher.close()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertIs(result.answer, False)

    def test_skips_failed_or_truncated_evidence(self) -> None:
        client = _FakeChatClient('{"route":"accept","a":"France","support":["e0"]}')
        dispatcher = LocalSlmSequenceDispatcher(
            slm_config=_safe_config(),
            client=client,
            max_parallel_sequences=1,
            max_pending_sequences=4,
            slm_scheduler="fifo",
        )
        try:
            failed = run_edge_local_review(
                question="Which country?",
                answer_type="string",
                evidence={"ok": False, "err": {"type": "ExecutionError"}},
                scope="action_group_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="failed",
            )
            truncated = run_edge_local_review(
                question="Which country?",
                answer_type="string",
                evidence={
                    "ok": True,
                    "res": {
                        "n": 10,
                        "cols": ["country"],
                        "rows": [{"country": "France"}],
                    },
                },
                scope="action_group_answer",
                slm_config=_safe_config(),
                dispatcher=dispatcher,
                job_id="truncated",
            )
        finally:
            dispatcher.close()

        self.assertIsNone(failed)
        self.assertIsNone(truncated)
        self.assertEqual(client.chat.completions.requests, [])


def _safe_config() -> dict[str, object]:
    return {
        "api_type": "chat_completions",
        "model": "edge-model",
        "edge_review_mode": "safe",
        "edge_review_max_actions": 4,
        "edge_review_max_rows": 5,
        "edge_review_max_columns": 5,
        "edge_review_max_facts": 40,
    }


def _one_row_evidence() -> dict[str, object]:
    return {
        "ok": True,
        "obs": [
            {
                "i": 0,
                "op": "sql",
                "ok": True,
                "res": {
                    "n": 1,
                    "cols": ["country", "year"],
                    "rows": [{"country": "France", "year": 2020}],
                },
            }
        ],
    }


class _FakeChatClient:
    def __init__(self, output: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(output))


class _FakeChatCompletions:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(kwargs)
        return _FakeChatResponse(self.output)


class _FakeChatResponse:
    id = "edge-review"

    def __init__(self, output: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {
            "id": self.id,
            "mode": mode,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


if __name__ == "__main__":
    unittest.main()
