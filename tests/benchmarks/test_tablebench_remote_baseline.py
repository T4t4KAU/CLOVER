from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.tablebench.remote_baseline import (
    parse_tablebench_prediction,
    render_tablebench_instruction_prompt,
    tablebench_table_from_csv,
)


class TableBenchRemoteBaselineTests(unittest.TestCase):
    def test_dp_prompt_and_parse(self) -> None:
        table = {"columns": ["year", "value"], "data": [[2020, 3], [2021, 4]]}
        prompt = render_tablebench_instruction_prompt(
            table=table,
            question="What is the total value?",
            instruction_type="DP",
        )

        self.assertIn("Final Answer: AnswerName1", prompt)
        self.assertIn("Question: What is the total value?", prompt)
        parsed = parse_tablebench_prediction("Final Answer: 7", instruction_type="DP")
        self.assertEqual(parsed["parsed_prediction"], "7")
        self.assertIsNone(parsed["ecr_1"])

    def test_tcot_prompt_and_parse_uses_last_final_answer(self) -> None:
        table = {"columns": ["year", "value"], "data": [[2020, 3], [2021, 4]]}
        prompt = render_tablebench_instruction_prompt(
            table=table,
            question="What is the total value?",
            instruction_type="TCoT",
        )

        self.assertIn("Think step by step", prompt)
        self.assertIn("Final Answer: AnswerName1", prompt)
        parsed = parse_tablebench_prediction(
            "The first row has 3.\n"
            "Final Answer: 3\n"
            "Then add the second row.\n"
            "Final Answer: 7",
            instruction_type="TCoT",
        )
        self.assertEqual(parsed["parsed_prediction"], "7")
        self.assertIsNone(parsed["ecr_1"])

    def test_pot_executes_code_against_table_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            table_path = root / "source.csv"
            table_path.write_text("year,value\n2020,3\n2021,4\n", encoding="utf-8")
            prediction = (
                "```python\n"
                "import pandas as pd\n"
                "df = pd.read_csv('table.csv')\n"
                "print('Final Answer: ' + str(df['value'].sum()))\n"
                "```"
            )

            parsed = parse_tablebench_prediction(
                prediction,
                instruction_type="PoT",
                table_path=table_path,
                work_dir=root / "case",
                execution_timeout_seconds=5,
            )

        self.assertEqual(parsed["parsed_prediction"], "7")
        self.assertTrue(parsed["ecr_1"])
        self.assertIsNone(parsed["error"])

    def test_csv_table_uses_json_like_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "table.csv"
            path.write_text("name,count,ratio\nalpha,12,1.5\nbeta,\"1,200\",x\n")
            table = tablebench_table_from_csv(path)

        self.assertEqual(table["columns"], ["name", "count", "ratio"])
        self.assertEqual(table["data"][0], ["alpha", 12, 1.5])
        self.assertEqual(table["data"][1], ["beta", "1,200", "x"])


if __name__ == "__main__":
    unittest.main()
