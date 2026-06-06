from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from clover.optimizer import parse_remote_sql_to_logic_dag
from clover.tools import PandasExecutionError, execute_table_reasoning_plan
from clover.tools.table_reasoning.pandas_backend import _read_resource_frame


class TableReasoningPandasBackendTest(unittest.TestCase):
    def test_table_cache_returns_shared_fast_path_frame_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n", encoding="utf-8")
            resource = _resource(table_path)
            table_cache: dict[str, pd.DataFrame] = {}

            first = _read_resource_frame(resource, table_cache)
            second = _read_resource_frame(resource, table_cache)

        self.assertIs(first, second)

    def test_table_cache_uses_one_read_for_parallel_cache_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n", encoding="utf-8")
            resource = _resource(table_path)
            table_cache: dict[str, pd.DataFrame] = {}
            read_count = 0
            read_count_lock = threading.Lock()

            def fake_read_csv(path: Path, *, low_memory: bool) -> pd.DataFrame:
                del path, low_memory
                nonlocal read_count
                with read_count_lock:
                    read_count += 1
                time.sleep(0.05)
                return pd.DataFrame({"value": [1, 2]})

            with patch(
                "clover.tools.table_reasoning.pandas_backend.pd.read_csv",
                side_effect=fake_read_csv,
            ):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    frames = list(
                        executor.map(
                            lambda _: _read_resource_frame(resource, table_cache),
                            range(4),
                        )
                    )

        self.assertEqual(read_count, 1)
        self.assertTrue(all(frame is frames[0] for frame in frames))

    def test_executes_filter_count_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")

            outputs = execute_table_reasoning_plan(
                {
                    "task_type": "table_reasoning.query",
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

    def test_executes_regression_forecast_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "year,price\n"
                "2000,388.88\n"
                "2001,388.88\n"
                "2002,388.88\n"
                "2003,398.88\n"
                "2004,398.88\n"
                "2005,398.88\n"
                "2006,448.88\n"
                "2007,498.95\n"
                "2008,508.95\n"
                "2009,638.88\n"
                "2010,555.55\n"
                "2011,638.88\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("number", ["year", "price"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT REGR_INTERCEPT("price", "year") '
                '+ REGR_SLOPE("price", "year") * 2012 AS answer '
                'FROM "table_1";',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertAlmostEqual(outputs["answer"], 627.9457575757551)

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

    def test_ignores_tablebench_alignment_total_rows_for_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "rank,country,growth\n"
                "1,egypt,2.29\n"
                "2,oman,8.8\n"
                "align = left|total,370989000,8763000\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("string", ["rank", "country", "growth"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT "country" FROM "table_1" ORDER BY "growth" DESC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], "oman")

    def test_aggregates_numeric_strings_with_spaced_signs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("profit\n- 16\n0.3\n0.3\n0.6\n- 1.4\n", encoding="utf-8")
            remote_dsl = _remote_dsl("number", ["profit"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT AVG("profit") AS answer FROM "table_1";',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertAlmostEqual(outputs["answer"], -3.24)

    def test_filter_literal_binding_requires_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "city,height,floors\n"
                "winnipeg,44,11\n"
                "winnipeg,50,13\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("number", ["city", "height", "floors"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT AVG("height") AS answer FROM "table_1" '
                'WHERE "city" = \'Winnipeg\' AND "floors" > 10;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertIsNone(outputs["answer"])

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
                        "task_type": "table_reasoning.query",
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
                "task_type": "table_reasoning.query",
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

    def test_executes_non_equality_self_join_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "location,enrollment\n"
                "nashville,100\n"
                "jackson,119\n"
                "knoxville,160\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("number", ["location", "enrollment"])
            logic_dag = parse_remote_sql_to_logic_dag(
                'SELECT ABS(t1."enrollment" - t2."enrollment") AS answer '
                'FROM "table_1" t1 '
                'JOIN "table_1" t2 ON t1."location" < t2."location" '
                'ORDER BY answer ASC LIMIT 1;',
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], 19)

    def test_executes_lead_window_function_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "SR No.,Date Built\n"
                "1,2020-01\n"
                "2,2020-07\n"
                "4,2021-01\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("number", ["SR No.", "Date Built"])
            logic_dag = parse_remote_sql_to_logic_dag(
                """
                WITH cte AS (
                    SELECT "SR No.",
                           CASE
                             WHEN LEAD("SR No.") OVER (ORDER BY "SR No.") = "SR No." + 1
                             THEN 1 ELSE 0
                           END AS consecutive
                    FROM "table_1"
                )
                SELECT SUM(consecutive) AS answer FROM cte;
                """,
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], 1)

    def test_executes_age_month_extract_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "Date Built\n"
                "2020-01-15\n"
                "2020-07-20\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("number", ["Date Built"])
            logic_dag = parse_remote_sql_to_logic_dag(
                """
                SELECT EXTRACT(YEAR FROM AGE(MAX("Date Built"), MIN("Date Built"))) * 12
                     + EXTRACT(MONTH FROM AGE(MAX("Date Built"), MIN("Date Built"))) AS answer
                FROM "table_1";
                """,
                remote_dsl,
            )

            outputs = execute_table_reasoning_plan(
                logic_dag,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(outputs["answer"], 6)

    def test_executes_rank_window_functions_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "team,score\n"
                "a,10\n"
                "b,20\n"
                "c,20\n"
                "d,5\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("number", ["team", "score"])
            rank_plan = parse_remote_sql_to_logic_dag(
                'SELECT RANK() OVER (ORDER BY "score" DESC) AS answer '
                'FROM "table_1" ORDER BY "team" ASC LIMIT 1;',
                remote_dsl,
            )
            dense_rank_plan = parse_remote_sql_to_logic_dag(
                'SELECT DENSE_RANK() OVER (ORDER BY "score" DESC) AS answer '
                'FROM "table_1" ORDER BY "team" ASC LIMIT 1;',
                remote_dsl,
            )

            rank_outputs = execute_table_reasoning_plan(
                rank_plan,
                resources={"table_1": _resource(table_path)},
            )
            dense_rank_outputs = execute_table_reasoning_plan(
                dense_rank_plan,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(rank_outputs["answer"], 3)
        self.assertEqual(dense_rank_outputs["answer"], 2)

    def test_executes_value_window_functions_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "team,score\n"
                "a,10\n"
                "b,20\n"
                "c,5\n",
                encoding="utf-8",
            )
            remote_dsl = _remote_dsl("string", ["team", "score"])
            first_plan = parse_remote_sql_to_logic_dag(
                'SELECT FIRST_VALUE("team") OVER (ORDER BY "score" DESC) AS answer '
                'FROM "table_1" LIMIT 1;',
                remote_dsl,
            )
            last_plan = parse_remote_sql_to_logic_dag(
                'SELECT LAST_VALUE("team") OVER (ORDER BY "score" DESC) AS answer '
                'FROM "table_1" LIMIT 1;',
                remote_dsl,
            )
            nth_plan = parse_remote_sql_to_logic_dag(
                'SELECT NTH_VALUE("team", 2) OVER (ORDER BY "score" DESC) AS answer '
                'FROM "table_1" LIMIT 1;',
                remote_dsl,
            )

            first_outputs = execute_table_reasoning_plan(
                first_plan,
                resources={"table_1": _resource(table_path)},
            )
            last_outputs = execute_table_reasoning_plan(
                last_plan,
                resources={"table_1": _resource(table_path)},
            )
            nth_outputs = execute_table_reasoning_plan(
                nth_plan,
                resources={"table_1": _resource(table_path)},
            )

        self.assertEqual(first_outputs["answer"], "b")
        self.assertEqual(last_outputs["answer"], "c")
        self.assertEqual(nth_outputs["answer"], "a")

    def test_executes_ntile_and_date_part_from_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            score_path = Path(tmpdir) / "scores.csv"
            score_path.write_text(
                "team,score\n"
                "a,40\n"
                "b,30\n"
                "c,20\n"
                "d,10\n",
                encoding="utf-8",
            )
            date_path = Path(tmpdir) / "dates.csv"
            date_path.write_text("date\n2020-05-17\n", encoding="utf-8")
            ntile_dsl = _remote_dsl("number", ["team", "score"])
            date_dsl = _remote_dsl("number", ["date"])
            ntile_plan = parse_remote_sql_to_logic_dag(
                'SELECT NTILE(2) OVER (ORDER BY "score" DESC) AS answer '
                'FROM "table_1" ORDER BY "team" DESC LIMIT 1;',
                ntile_dsl,
            )
            date_part_plan = parse_remote_sql_to_logic_dag(
                'SELECT DATE_PART(\'year\', "date") AS answer FROM "table_1" LIMIT 1;',
                date_dsl,
            )

            ntile_outputs = execute_table_reasoning_plan(
                ntile_plan,
                resources={"table_1": _resource(score_path)},
            )
            date_part_outputs = execute_table_reasoning_plan(
                date_part_plan,
                resources={"table_1": _resource(date_path)},
            )

        self.assertEqual(ntile_outputs["answer"], 2)
        self.assertEqual(date_part_outputs["answer"], 2020)


def _remote_dsl(answer_type: str, columns: list[str]) -> dict:
    return {
        "task_type": "table_reasoning.query",
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
