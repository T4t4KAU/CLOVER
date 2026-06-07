from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.resource.dsl_builder import (
    BUILD_TABLE_DSL_TOOL_NAME,
    BuildTableDSLTool,
    build_table_task_dsl_with_builder_agent,
    parse_table_dsl_builder_output,
    table_profile_for_dsl_builder,
)


class TableDslBuilderTest(unittest.TestCase):
    def test_parses_fenced_json_object(self) -> None:
        parsed = parse_table_dsl_builder_output(
            '```json\n{"answer_type":"boolean","intent":["fact_check"]}\n```'
        )

        self.assertEqual(parsed["answer_type"], "boolean")
        self.assertEqual(parsed["intent"], ["fact_check"])

    def test_build_table_dsl_tool_builds_entity_lookup_dsl_without_slm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "Rank,Nation,Gold,Silver,Total\n"
                "1,Aland,18,30,57\n"
                "4,Borduria,4,3,13\n",
                encoding="utf-8",
            )

            result = BuildTableDSLTool().run(
                question="Which nation has a total of 13 medals?",
                table_path=table_path,
                source_file="table.csv",
            )

        self.assertEqual(result.builder_mode, "build_table_dsl_tool")
        self.assertEqual(result.task_dsl["answer"]["type"], "string")
        self.assertNotIn("hints", result.task_dsl)
        self.assertEqual(
            result.diagnostics["hints"]["columns"][:2],
            ["Nation", "Total"],
        )
        self.assertIn("lookup", result.diagnostics["hints"]["intent"])
        self.assertIn("filter", result.diagnostics["hints"]["intent"])
        self.assertEqual(result.response_payload, {})

    def test_build_table_dsl_tool_uses_target_column_kind_for_which_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "season,wins\n"
                "1999,8\n"
                "2003,13\n",
                encoding="utf-8",
            )

            result = BuildTableDSLTool().run(
                question="In which season did the driver win 13 races?",
                table_path=table_path,
            )

        self.assertEqual(result.task_dsl["answer"]["type"], "number")
        self.assertEqual(result.diagnostics["target_column"], "season")
        self.assertNotIn("hints", result.task_dsl)
        self.assertIn("season", result.diagnostics["hints"]["columns"])

    def test_build_table_dsl_tool_call_shape_is_minimal(self) -> None:
        call = BuildTableDSLTool().build_call(
            question="Which nation has a total of 13 medals?",
            source_id=0,
            source_file="table.csv",
        )

        self.assertEqual(call["tool"], BUILD_TABLE_DSL_TOOL_NAME)
        self.assertEqual(
            set(call["arguments"]),
            {"question", "source_id", "source_file"},
        )

    def test_builder_agent_selects_tool_without_table_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "secret_column,total\n"
                "Aland,57\n"
                "Borduria,13\n",
                encoding="utf-8",
            )

            with patch(
                "clover.resource.dsl_builder._generate_builder_slm_text",
                return_value=SimpleNamespace(
                    text=json.dumps(
                        {
                            "tool": "build_table_dsl",
                            "arguments": {"source_id": 0},
                        }
                    ),
                    response_payload={"usage": {"prompt_tokens": 12, "completion_tokens": 5}},
                ),
            ):
                result = build_table_task_dsl_with_builder_agent(
                    question="Which nation has a total of 13 medals?",
                    table_path=table_path,
                    source_file="table.csv",
                    slm_config={"api_type": "chat_completions", "model": "fake"},
                )

        self.assertEqual(result.builder_mode, "builder_agent")
        self.assertEqual(result.tool_call["tool"], BUILD_TABLE_DSL_TOOL_NAME)
        self.assertEqual(result.task_dsl["answer"]["type"], "string")
        self.assertNotIn("hints", result.task_dsl)
        self.assertNotIn("secret_column", result.prompt)
        self.assertEqual(result.response_payload["usage"]["prompt_tokens"], 12)

    def test_table_profile_truncates_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "a,b,c\n"
                "1,hello,extra\n"
                "2,world,extra\n",
                encoding="utf-8",
            )

            profile = table_profile_for_dsl_builder(
                table_path,
                max_preview_rows=1,
                max_columns=2,
            )

        self.assertEqual(profile["shape"], {"rows": 2, "columns": 3})
        self.assertEqual(profile["shown_columns"], ["a", "b"])
        self.assertEqual(profile["omitted_columns"], 1)
        self.assertEqual(len(profile["preview_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
