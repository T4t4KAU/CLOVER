from __future__ import annotations

import unittest

from benchmarks.tablebench.metrics import score_tablebench_answer


class TableBenchMetricsTest(unittest.TestCase):
    def test_numeric_em_accepts_gold_decimal_precision_rounding(self) -> None:
        first_five = score_tablebench_answer(
            expected=15.61,
            actual=15.614,
            qtype="NumericalReasoning",
            qsubtype="ArithmeticCalculation",
        )
        tumbling = score_tablebench_answer(
            expected=52.08,
            actual=52.083333333333336,
            qtype="NumericalReasoning",
            qsubtype="Aggregation",
        )

        self.assertTrue(first_five.correct)
        self.assertTrue(tumbling.correct)

    def test_numeric_em_keeps_integer_answers_strict(self) -> None:
        score = score_tablebench_answer(
            expected=29,
            actual=29.4,
            qtype="NumericalReasoning",
            qsubtype="Time-basedCalculation",
        )

        self.assertFalse(score.correct)

    def test_numeric_em_accepts_percent_equivalence(self) -> None:
        score = score_tablebench_answer(
            expected="1.9%",
            actual="0.019",
            qtype="NumericalReasoning",
            qsubtype="Aggregation",
        )

        self.assertTrue(score.correct)

    def test_boolean_em_accepts_common_synonyms(self) -> None:
        yes_score = score_tablebench_answer(
            expected="yes",
            actual="true",
            qtype="FactChecking",
            qsubtype="Single-hop FactChecking",
        )
        no_score = score_tablebench_answer(
            expected="refutes",
            actual="No",
            qtype="FactChecking",
            qsubtype="Single-hop FactChecking",
        )

        self.assertTrue(yes_score.correct)
        self.assertTrue(no_score.correct)

    def test_list_em_can_ignore_order_for_non_ranking_cases(self) -> None:
        score = score_tablebench_answer(
            expected=["red", "blue"],
            actual="blue, red",
            qtype="NumericalReasoning",
            qsubtype="MatchBased",
        )

        self.assertTrue(score.correct)

    def test_list_em_preserves_order_for_ranking_cases(self) -> None:
        score = score_tablebench_answer(
            expected=["red", "blue"],
            actual="blue, red",
            qtype="NumericalReasoning",
            qsubtype="Ranking",
        )

        self.assertFalse(score.correct)


if __name__ == "__main__":
    unittest.main()
