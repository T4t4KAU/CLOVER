from __future__ import annotations

import unittest

from benchmarks.mmqa.eval import _score_mmqa_record, _update_record_from_result_metadata
from benchmarks.mmqa.metrics import flatten_mmqa_answer, score_mmqa_answer


class MMQAMetricsTest(unittest.TestCase):
    def test_semicolon_separates_values_that_contain_commas(self) -> None:
        text = "Schmidt, Kertzmann and Lubowitz; Schmitt-Lang"

        self.assertEqual(
            flatten_mmqa_answer(text),
            ["Schmidt, Kertzmann and Lubowitz", "Schmitt-Lang"],
        )
        score = score_mmqa_answer(
            expected=text,
            actual=["Schmitt-Lang", "Schmidt, Kertzmann and Lubowitz"],
        )
        self.assertTrue(score.correct)

    def test_last_first_name_matches_first_last_prediction(self) -> None:
        score = score_mmqa_answer(
            expected="Shivers, Olin",
            actual=["Olin Shivers"],
        )

        self.assertTrue(score.correct)

    def test_eval_scoring_uses_converted_answer_as_fallback(self) -> None:
        record = {
            "expected_raw": "Treasury, 115897",
            "expected_answer": ["Treasury", "115897"],
        }
        score = _score_mmqa_record(record, ["Treasury", "115897"])

        self.assertTrue(score.correct)

    def test_mmqa_metadata_keeps_raw_answer_for_scoring(self) -> None:
        record = {
            "expected_raw": "Schmidt, Kertzmann and Lubowitz; Schmitt-Lang",
        }
        _update_record_from_result_metadata(
            record,
            {
                "expected_answer": [
                    "Schmidt",
                    "Kertzmann and Lubowitz; Schmitt-Lang",
                ],
                "expected_raw": "Schmidt, Kertzmann and Lubowitz; Schmitt-Lang",
            },
        )

        self.assertEqual(
            record["expected_raw"],
            "Schmidt, Kertzmann and Lubowitz; Schmitt-Lang",
        )
        self.assertEqual(
            record["expected_answer"],
            ["Schmidt", "Kertzmann and Lubowitz; Schmitt-Lang"],
        )


if __name__ == "__main__":
    unittest.main()
