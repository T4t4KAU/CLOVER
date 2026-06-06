from __future__ import annotations

import unittest

from clover.runtime.final_answer import finalize_answer


class FinalAnswerTest(unittest.TestCase):
    def test_table_query_answer_is_passthrough(self) -> None:
        self.assertEqual(
            finalize_answer(
                task_type="table_reasoning.query",
                question="How many rows?",
                answer="5",
                explanation="There are 5 rows.",
            ),
            "5",
        )

    def test_document_numerical_answer_appends_existing_numeric_explanation(self) -> None:
        self.assertEqual(
            finalize_answer(
                task_type="document_reasoning",
                question="Which activity brought in the most cash flow?",
                answer="operating activities",
                explanation=(
                    "Operating activities generated $1,824 million, investing "
                    "activities used $962 million, and financing activities used "
                    "$1,806 million."
                ),
            ),
            (
                "operating activities. Operating activities generated $1,824 "
                "million, investing activities used $962 million, and financing "
                "activities used $1,806 million."
            ),
        )

    def test_document_numerical_answer_uses_evidence_when_explanation_has_no_number(
        self,
    ) -> None:
        self.assertEqual(
            finalize_answer(
                task_type="document_reasoning",
                question="Has debt increased between the two periods?",
                answer="No",
                explanation="The evidence shows a decrease.",
                observation={
                    "evidence_summary": (
                        "Debt decreased by $229 million between FY2021 and FY2022."
                    )
                },
            ),
            "No. Debt decreased by $229 million between FY2021 and FY2022.",
        )


if __name__ == "__main__":
    unittest.main()
