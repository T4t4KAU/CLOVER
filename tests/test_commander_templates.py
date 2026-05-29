from __future__ import annotations

import unittest

from clover.commander import (
    available_task_types,
    initial_task_template_paths,
    render_followup_task_prompt,
    render_initial_task_prompt,
    render_task_prompt,
    template_paths_for_task_type,
)


class CommanderTemplateTest(unittest.TestCase):
    def test_initial_task_prompt_prepends_root_to_task_route(self) -> None:
        self.assertEqual(
            initial_task_template_paths("table_reasoning"),
            (
                "common/root.md",
                "table_reasoning/v1/system.md",
                "table_reasoning/v1/sql_constraints.md",
                "table_reasoning/v1/sql_generation.md",
            ),
        )
        self.assertEqual(
            template_paths_for_task_type("table_reasoning"),
            (
                "table_reasoning/v1/system.md",
                "table_reasoning/v1/sql_constraints.md",
                "table_reasoning/v1/sql_generation.md",
            ),
        )
        self.assertEqual(
            initial_task_template_paths("table_reasoning_v2"),
            (
                "common/root.md",
                "table_reasoning/v2/system.md",
                "table_reasoning/v2/sql_constraints.md",
                "table_reasoning/v2/sql_generation.md",
            ),
        )

    def test_renders_initial_table_reasoning_prompt_with_root_once(self) -> None:
        prompt = render_initial_task_prompt(
            {
                "task_type": "table_reasoning",
                "question": "How many passengers survived?",
                "sources": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "format": "csv",
                        "schema": {
                            "format": "csv",
                            "shape": {"rows": 3, "columns": 2},
                            "columns": [
                                {"name": "Survived", "type": "string"},
                                {"name": "Age", "type": "string"},
                            ],
                        },
                    }
                ],
                "answer": {"name": "answer", "type": "number"},
            }
        )

        self.assertIn("Commander does not generate the Logic DAG", prompt)
        self.assertIn("Commander returns SQL only", prompt)
        self.assertIn("translate the user question into a SQL query", prompt)
        self.assertIn("SQL generation constraints", prompt)
        self.assertIn('"question": "How many passengers survived?"', prompt)
        self.assertIn('"id": "table_1"', prompt)
        self.assertIn("SQL:", prompt)
        self.assertNotIn("{{TASK_DSL}}", prompt)
        self.assertNotIn("expected_answer", prompt)
        self.assertNotIn("/home/", prompt)

    def test_remote_prompt_strips_label_like_fields_even_if_supplied(self) -> None:
        prompt = render_initial_task_prompt(
            {
                "task_type": "table_reasoning",
                "question": "How many passengers survived?",
                "expected_answer": "SECRET_EXPECTED_ANSWER",
                "metadata": {
                    "case": {
                        "answer": "SECRET_EXPECTED_ANSWER",
                    }
                },
                "sources": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "format": "csv",
                        "file": "/home/user/private/table.csv",
                        "schema": {"columns": ["Survived"]},
                    }
                ],
                "answer": {"name": "answer", "type": "number"},
            }
        )

        self.assertIn('"answer": {', prompt)
        self.assertIn('"type": "number"', prompt)
        self.assertNotIn("expected_answer", prompt)
        self.assertNotIn("SECRET_EXPECTED_ANSWER", prompt)
        self.assertNotIn("/home/user/private/table.csv", prompt)

    def test_renders_followup_task_prompt_without_root(self) -> None:
        prompt = render_task_prompt(
            {
                "task_type": "table_reasoning",
                "question": "How many passengers survived?",
                "sources": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "format": "csv",
                        "schema": {
                            "format": "csv",
                            "shape": {"rows": 3, "columns": 2},
                            "columns": [
                                {"name": "Survived", "type": "string"},
                                {"name": "Age", "type": "string"},
                            ],
                        },
                    }
                ],
                "answer": {"name": "answer", "type": "number"},
            }
        )

        self.assertIn("translate the user question into a SQL query", prompt)
        self.assertNotIn("Commander makes the next-step decision", prompt)

    def test_renders_table_reasoning_v2_sql_list_prompt(self) -> None:
        prompt = render_initial_task_prompt(
            {
                "task_type": "table_reasoning_v2",
                "questions": [
                    "Is the person with the highest net worth self-made?",
                    "What is the country of the person with the highest net worth?",
                ],
                "sources": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "format": "csv",
                        "file": "/home/user/private/table.csv",
                        "schema": {
                            "columns": ["selfMade", "country", "finalWorth"],
                        },
                    }
                ],
                "answers": [
                    {"name": "answer_1", "type": "boolean"},
                    {"name": "answer_2", "type": "string"},
                ],
            }
        )

        self.assertIn("translate multiple user questions into SQL queries", prompt)
        self.assertIn("Return exactly one JSON array of SQL strings", prompt)
        self.assertIn("The SQL string at index i must answer `questions[i]`", prompt)
        self.assertIn('"questions": [', prompt)
        self.assertIn('"answers": [', prompt)
        self.assertIn('"answer_1"', prompt)
        self.assertIn('"answer_2"', prompt)
        self.assertIn("SQL JSON array:", prompt)
        self.assertNotIn("/home/user/private/table.csv", prompt)

    def test_renders_table_reasoning_v2_followup_without_schema(self) -> None:
        prompt = render_followup_task_prompt(
            {
                "task_type": "table_reasoning_v2",
                "questions": ["What is the country of the richest person?"],
                "answers": [{"name": "answer_37", "type": "string"}],
                "sources": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "format": "csv",
                        "schema": {"columns": ["country", "finalWorth"]},
                    }
                ],
            }
        )

        self.assertIn("Continue the existing table reasoning session", prompt)
        self.assertIn("schema is not repeated", prompt)
        self.assertIn('"answer_37"', prompt)
        self.assertNotIn('"columns"', prompt)
        self.assertIn("Do not rename, renumber, or invent answer names", prompt)

    def test_rejects_unsupported_task_type(self) -> None:
        self.assertEqual(
            available_task_types(),
            ("table_reasoning",),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported Commander task_type"):
            render_task_prompt({"task_type": "document_reasoning_v1"})


if __name__ == "__main__":
    unittest.main()
