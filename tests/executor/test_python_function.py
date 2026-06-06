from __future__ import annotations

import unittest

from clover.executor.python_function import (
    PythonFunctionParseError,
    PythonFunctionTask,
    parse_python_function_action,
    validate_python_function,
)


class PythonFunctionActionTest(unittest.TestCase):
    def test_parses_short_solve_json(self) -> None:
        action = parse_python_function_action(
            '{"s":"def solve(df):\\n    return df"}'
        )

        self.assertEqual(action["action"], "solve")
        self.assertIn("def solve", action["code"])

    def test_validates_only_expected_solve_function(self) -> None:
        task = PythonFunctionTask(
            name="solve",
            args=("df",),
            prompt_code="",
            contract={"kind": "dataframe"},
        )

        validate_python_function("def solve(df):\n    return df", task)

        with self.assertRaises(PythonFunctionParseError):
            validate_python_function(
                "x = 1\n"
                "def solve(df):\n"
                "    return df",
                task,
            )

        with self.assertRaises(PythonFunctionParseError):
            validate_python_function("def solve(table):\n    return table", task)


if __name__ == "__main__":
    unittest.main()
