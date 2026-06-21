from __future__ import annotations

import json
import unittest

from clover.optimizer import SqlParseError
from clover.optimizer.table_reasoning.sql_list_parser import parse_sql_list_response


class SqlListParserTest(unittest.TestCase):
    def test_reuses_single_sql_for_duplicate_batch_questions_with_retargeted_aliases(
        self,
    ) -> None:
        remote_dsl = {
            "task_type": "table_reasoning.query",
            "questions": [
                "List Washington players.",
                "Which Washington players are listed?",
            ],
            "sources": [
                {
                    "id": "players",
                    "type": "table",
                    "format": "csv",
                    "schema": {"columns": ["first_name", "last_name", "team"]},
                }
            ],
            "answers": [
                {"name": "answer_1", "type": "list[string]"},
                {"name": "answer_2", "type": "list[string]"},
            ],
        }
        response = json.dumps(
            [
                {
                    "sql": (
                        'SELECT "first_name" AS "answer_1" '
                        'FROM "players" WHERE "team" = \'Washington Nationals\''
                    )
                }
            ]
        )

        parsed = parse_sql_list_response(response, remote_dsl)

        self.assertEqual(len(parsed.sqls), 2)
        self.assertIn('AS "answer_1"', parsed.sqls[0])
        self.assertIn('AS "answer_2"', parsed.sqls[1])

    def test_retargets_non_answer_select_alias(self) -> None:
        remote_dsl = {
            "task_type": "table_reasoning.query",
            "questions": ["List authors."],
            "sources": [
                {
                    "id": "authors",
                    "type": "table",
                    "format": "csv",
                    "schema": {"columns": ["name"]},
                }
            ],
            "answers": [{"name": "answer_7", "type": "list[string]"}],
        }
        response = json.dumps(
            [{"sql": 'SELECT "name" AS "author_name" FROM "authors"'}]
        )

        parsed = parse_sql_list_response(response, remote_dsl)

        self.assertIn('AS "answer_7"', parsed.sqls[0])

    def test_rejects_duplicate_answer_aliases(self) -> None:
        remote_dsl = {
            "task_type": "table_reasoning.query",
            "questions": ["Who is the author?"],
            "sources": [
                {
                    "id": "authors",
                    "type": "table",
                    "format": "csv",
                    "schema": {"columns": ["fname", "lname"]},
                }
            ],
            "answers": [{"name": "answer_1", "type": "string"}],
        }
        response = json.dumps(
            [
                {
                    "sql": (
                        'SELECT "fname" AS "answer_1", '
                        '"lname" AS "answer_1" FROM "authors"'
                    )
                }
            ]
        )

        with self.assertRaisesRegex(SqlParseError, "aliases multiple SELECT items"):
            parse_sql_list_response(response, remote_dsl)


if __name__ == "__main__":
    unittest.main()
