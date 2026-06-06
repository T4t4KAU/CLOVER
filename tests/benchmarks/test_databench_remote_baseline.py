from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.databench.remote_baseline import (
    execute_generated_answer_function,
    generated_answer_function_source,
    render_remote_code_prompt,
)


class DatabenchRemoteBaselineTest(unittest.TestCase):
    def test_generated_expression_is_wrapped_as_answer_function(self) -> None:
        source = generated_answer_function_source('df["value"].sum()')

        self.assertIn("def answer(df):", source)
        self.assertIn('return df["value"].sum()', source)

    def test_generated_mixed_indentation_body_is_normalized(self) -> None:
        source = generated_answer_function_source(
            "max_index = df['value'].idxmax()\n"
            "    return df.loc[max_index, 'value']"
        )

        self.assertIn("def answer(df):", source)
        self.assertIn("max_index = df['value'].idxmax()", source)
        self.assertIn("return df.loc[max_index, 'value']", source)

    def test_generated_fenced_function_is_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n", encoding="utf-8")
            source = generated_answer_function_source(
                "```python\n"
                "def answer(df):\n"
                "    return int(df['value'].sum())\n"
                "```"
            )

            result = execute_generated_answer_function(
                source,
                table_path=table_path,
                timeout_seconds=5.0,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], 3)

    def test_prompt_includes_columns_dtypes_and_answer_type(self) -> None:
        prompt = render_remote_code_prompt(
            question="How many rows are present?",
            answer_type="number",
            table_profile={
                "shape": {"rows": 2, "columns": 1},
                "columns": ["value"],
                "dtypes": {"value": "int64"},
            },
        )

        self.assertIn("def answer(df) -> float:", prompt)
        self.assertIn('"value"', prompt)
        self.assertIn("'value': dtype('int64')", prompt)
        self.assertIn("df.columns =", prompt)


if __name__ == "__main__":
    unittest.main()
