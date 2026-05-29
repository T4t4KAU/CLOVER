from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clover.planner import parse_remote_sql_to_logic_dag
from clover.tools import PandasExecutionError, execute_table_reasoning_plan


class TableReasoningPandasBackendTest(unittest.TestCase):
    def test_executes_filter_count_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")

            outputs = execute_table_reasoning_plan(
                {
                    "task_type": "table_reasoning_v1",
                    "resources": [_resource(table_path)],
                    "nodes": [
                        _scan_node("N0", "T0"),
                        {
                            "id": "N1",
                            "op": "Filter",
                            "dependency": ["T0"],
                            "input": [],
                            "params": {
                                "predicate": {
                                    "type": "binary_op",
                                    "op": ">",
                                    "left": {"type": "column", "name": "value"},
                                    "right": {"type": "literal", "value": 1},
                                }
                            },
                            "output": "T1",
                        },
                        {
                            "id": "N2",
                            "op": "Aggregate",
                            "dependency": ["T1"],
                            "input": [],
                            "params": {
                                "aggregations": [
                                    {
                                        "function": "COUNT",
                                        "argument": {"type": "wildcard"},
                                        "distinct": False,
                                        "alias": "answer",
                                    }
                                ],
                                "grouped": False,
                            },
                            "output": "T2",
                        },
                        _format_node("N3", "T2", "number"),
                    ],
                    "edges": [],
                }
            )

        self.assertEqual(outputs["answer"], 2)

    def test_executes_group_sort_limit_project_plan_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "category,value\n"
                "fruit,1\n"
                "tool,2\n"
                "fruit,3\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("category", ["category", "value"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT "category" AS answer FROM "table_1" '
                'GROUP BY "category" ORDER BY COUNT(*) DESC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], "fruit")

    def test_formats_last_column_when_grouped_aggregate_has_no_answer_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "player,season,points\n"
                "a,1,10\n"
                "a,1,20\n"
                "b,1,5\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("list[number]", ["player", "season", "points"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT SUM("points") AS total_points FROM "table_1" '
                'GROUP BY "player", "season" ORDER BY total_points DESC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], [30])

    def test_matches_databench_typed_column_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "Nationality<gx:category>,Overall<gx:number>\n"
                "France,90\n"
                "Spain,88\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("category", ["Nationality", "Overall"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT "Nationality" FROM "table_1" ORDER BY "Overall" DESC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], "France")

    def test_empty_grouped_aggregate_keeps_output_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("team,score\nred,1\nblue,2\n", encoding="utf-8")
            remote_dsl = _remote_dsl("category", ["team", "score"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT "team" FROM "table_1" WHERE "score" > 10 '
                'GROUP BY "team" ORDER BY COUNT(*) ASC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertIsNone(outputs["answer"])

    def test_executes_stddev_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            remote_dsl = _remote_dsl("number", ["value"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT STDDEV("value") AS answer FROM "table_1";',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], 1)

    def test_executes_percentile_cont_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n100\n", encoding="utf-8")
            remote_dsl = _remote_dsl("number", ["value"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT PERCENTILE_CONT(0.5) WITHIN GROUP '
                '(ORDER BY "value") AS answer FROM "table_1";',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], 2)

    def test_executes_ordered_limited_array_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "country,count\n"
                "US,3\n"
                "CN,2\n"
                "IN,1\n"
                "FR,1\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("list[category]", ["country", "count"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT ARRAY_AGG("country" ORDER BY "count" DESC LIMIT 3) AS answer '
                'FROM "table_1";',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], ["US", "CN", "IN"])

    def test_executes_split_cardinality_expression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "name,tags\n"
                "short,a\n"
                'long,"a,b,c"\n',
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("category", ["name", "tags"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT "name" FROM "table_1" '
                'ORDER BY ARRAY_LENGTH(STRING_TO_ARRAY("tags", \',\'), 1) DESC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], "long")

    def test_compares_numeric_strings_with_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("price\n$1,200\n$50\nmissing\n", encoding="utf-8")
            remote_dsl = _remote_dsl("number", ["price"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT COUNT(*) AS answer FROM "table_1" WHERE "price" > 100;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], 1)

    def test_executes_scalar_ref_plan_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "finalWorth,selfMade\n"
                "10,False\n"
                "20,True\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("boolean", ["finalWorth", "selfMade"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT "selfMade" FROM "table_1" '
                'WHERE "finalWorth" = (SELECT MAX("finalWorth") FROM "table_1");',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertIs(outputs["answer"], True)

    def test_binary_equality_against_single_row_set_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "author_id,lang,favorites\n"
                "1,en,10\n"
                "1,es,20\n"
                "2,en,1\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("boolean", ["author_id", "lang", "favorites"])
            logic_dag = parse_remote_sql_to_logic_dag(
                """
                SELECT EXISTS (
                    SELECT 1 FROM "table_1"
                    WHERE "author_id" = (
                        SELECT "author_id" FROM "table_1"
                        GROUP BY "author_id"
                        ORDER BY SUM("favorites") DESC
                        LIMIT 1
                    )
                    AND "lang" = 'es'
                ) AS answer;
                """,
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertIs(outputs["answer"], True)

    def test_executes_replace_function_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text('text\n"a,b,c"\n', encoding="utf-8")
            remote_dsl = _remote_dsl("string", ["text"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT REPLACE("text", \',\', \'\') AS answer FROM "table_1";',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], "abc")

    def test_rejects_unsatisfied_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(PandasExecutionError, "unsatisfied dependencies"):
                execute_table_reasoning_plan(
                    {
                        "task_type": "table_reasoning_v1",
                        "resources": [_resource(table_path)],
                        "nodes": [
                            {
                                "id": "N1",
                                "op": "Filter",
                                "dependency": ["T0"],
                                "input": [],
                                "params": {
                                    "predicate": {
                                        "type": "binary_op",
                                        "op": ">",
                                        "left": {"type": "column", "name": "value"},
                                        "right": {"type": "literal", "value": 0},
                                    }
                                },
                                "output": "T1",
                            }
                        ],
                        "edges": [],
                    }
                )

    def test_executes_recursive_repeat_union_plan_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "Unique ID,Parent,Tier 3\n"
                "200,150,\n"
                "300,200,leaf\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("boolean", ["Unique ID", "Parent", "Tier 3"])
            logic_dag = parse_remote_sql_to_logic_dag(
                """
                WITH RECURSIVE descendants AS (
                    SELECT "Unique ID", "Tier 3"
                    FROM "table_1"
                    WHERE "Parent" = 150
                    UNION ALL
                    SELECT t."Unique ID", t."Tier 3"
                    FROM "table_1" t
                    JOIN descendants d ON t."Parent" = d."Unique ID"
                )
                SELECT EXISTS(
                    SELECT 1 FROM descendants WHERE "Tier 3" IS NOT NULL
                ) AS answer;
                """,
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
                external_params={"max_iterations": 10},
            )

        self.assertIs(outputs["answer"], True)

    def test_join_coerces_numeric_string_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left.csv"
            right_path = Path(tmpdir) / "right.csv"
            left_path.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
            right_path.write_text("id,label\n1.0,x\n3.0,y\n", encoding="utf-8")
            plan = {
                "task_type": "table_reasoning_v1",
                "resources": [
                    {**_resource(left_path), "id": "table_1"},
                    {**_resource(right_path), "id": "table_2"},
                ],
                "nodes": [
                    _scan_node("N0", "T0"),
                    {
                        "id": "N1",
                        "op": "Join",
                        "dependency": ["T0"],
                        "input": [],
                        "params": {
                            "joins": [
                                {
                                    "kind": "JOIN",
                                    "source": "table_2",
                                    "on": {
                                        "type": "binary_op",
                                        "op": "=",
                                        "left": {"type": "column", "name": "id"},
                                        "right": {"type": "column", "name": "id"},
                                    },
                                }
                            ]
                        },
                        "output": "T1",
                    },
                    {
                        "id": "N2",
                        "op": "Project",
                        "dependency": ["T1"],
                        "input": [],
                        "params": {"expressions": [{"expr": {"type": "column", "name": "label"}}]},
                        "output": "T2",
                    },
                    _format_node("N3", "T2", "list[category]"),
                ],
                "edges": [],
            }

            outputs = execute_table_reasoning_plan(plan)

        self.assertEqual(outputs["answer"], ["x"])


def _remote_dsl(answer_type: str, columns: list[str]) -> dict:
    return {
        "task_type": "table_reasoning_v1",
        "question": "",
        "sources": [{"id": "table_1", "type": "table", "format": "csv", "schema": {"columns": columns}}],
        "answer": {"name": "answer", "type": answer_type},
    }


def _resource(table_path: Path) -> dict:
    return {
        "id": "table_1",
        "type": "table",
        "path": str(table_path),
        "format": "csv",
        "schema": {},
    }


def _scan_node(node_id: str, output: str) -> dict:
    return {
        "id": node_id,
        "op": "Scan",
        "dependency": [],
        "input": ["table_1"],
        "params": {"source": "table_1"},
        "output": output,
    }


def _format_node(node_id: str, dependency: str, answer_type: str) -> dict:
    return {
        "id": node_id,
        "op": "FormatAnswer",
        "dependency": [dependency],
        "input": [],
        "params": {"answer": {"name": "answer", "type": answer_type}},
        "output": "answer",
    }


if __name__ == "__main__":
    unittest.main()
