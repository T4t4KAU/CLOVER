from __future__ import annotations

import unittest

from clover.optimizer import (
    ALLOWED_OPS,
    SqlParseError,
    extract_sql_statement,
    parse_remote_sql_list_to_logic_dag,
    parse_remote_sql_to_logic_dag as _parse_remote_sql_to_logic_dag,
    parse_sql_list_response,
    parse_sql_response,
)


REMOTE_DSL = {
    "task_type": "table_reasoning.query",
    "question": "Is the person with the highest net worth self-made?",
    "sources": [
        {
            "id": "table_1",
            "type": "table",
            "format": "csv",
            "schema": {
                "format": "csv",
                "shape": {"rows": 3, "columns": 2},
                "columns": ["finalWorth", "selfMade"],
            },
        }
    ],
    "answer": {"name": "answer", "type": "boolean"},
}

BATCH_REMOTE_DSL = {
    "task_type": "table_reasoning.query",
    "questions": [
        "Is the person with the highest net worth self-made?",
        "What is the country of the person with the highest net worth?",
    ],
    "sources": [
        {
            "id": "table_1",
            "type": "table",
            "format": "csv",
            "schema": {
                "format": "csv",
                "shape": {"rows": 3, "columns": 3},
                "columns": ["finalWorth", "selfMade", "country"],
            },
        }
    ],
    "answers": [
        {"name": "answer_1", "type": "boolean"},
        {"name": "answer_2", "type": "string"},
    ],
}


def parse_remote_sql_to_logic_dag(
    remote_response: str,
    remote_dsl: dict,
) -> dict:
    """Return the single query_plan fragment for SQL lowering assertions."""

    logic_dag = _parse_remote_sql_to_logic_dag(remote_response, remote_dsl)
    query_plans = logic_dag.get("query_plans")
    if not isinstance(query_plans, list) or len(query_plans) != 1:
        raise AssertionError("expected one query_plan")
    return query_plans[0]


class SqlParserTest(unittest.TestCase):
    def test_allowed_ops_are_fixed(self) -> None:
        self.assertEqual(
            ALLOWED_OPS,
            {
                "Scan",
                "Filter",
                "Project",
                "Derive",
                "Aggregate",
                "Group",
                "Sort",
                "Limit",
                "Distinct",
                "Join",
                "SetOp",
                "RepeatUnion",
                "FormatAnswer",
            },
        )

    def test_extracts_plain_sql(self) -> None:
        sql = extract_sql_statement(
            'SELECT "selfMade" FROM "table_1" ORDER BY "finalWorth" DESC LIMIT 1;'
        )

        self.assertEqual(
            sql,
            'SELECT "selfMade" FROM "table_1" ORDER BY "finalWorth" DESC LIMIT 1',
        )

    def test_extracts_sql_from_short_json_plan(self) -> None:
        sql = extract_sql_statement(
            '{"sql":"SELECT COUNT(*) AS \\"answer\\" FROM \\"table_1\\";"}'
        )

        self.assertEqual(sql, 'SELECT COUNT(*) AS "answer" FROM "table_1"')

    def test_lowers_negative_literals_and_string_concatenation(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT SUBSTR("finalWorth", -4) || \'-\' AS answer '
            'FROM "table_1" WHERE "finalWorth" > -3 LIMIT 1;',
            REMOTE_DSL,
        )

        serialized = str(logic_dag)
        self.assertNotIn("sql_expr", serialized)
        self.assertIn("'function': 'CONCAT'", serialized)
        self.assertIn("'value': -4", serialized)
        self.assertIn("'value': -3", serialized)

    def test_rejects_final_protocol_marker(self) -> None:
        with self.assertRaises(SqlParseError):
            extract_sql_statement(
                '{"final":true,"sql":"SELECT COUNT(*) AS \\"answer\\" FROM \\"table_1\\";"}'
            )

    def test_extracts_fenced_sql(self) -> None:
        sql = extract_sql_statement(
            'Here is the query:\n```sql\nSELECT COUNT(*) AS answer FROM "table_1";\n```'
        )

        self.assertEqual(sql, 'SELECT COUNT(*) AS answer FROM "table_1"')

    def test_parses_sql_array_response(self) -> None:
        parsed = parse_sql_list_response(
            """
            [
              "SELECT \\"selfMade\\" AS \\"answer_1\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;",
              "SELECT \\"country\\" AS \\"answer_2\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;"
            ]
            """,
            BATCH_REMOTE_DSL,
        )

        self.assertEqual(len(parsed.sqls), 2)
        self.assertEqual(
            parsed.sqls[0],
            'SELECT "selfMade" AS "answer_1" FROM "table_1" '
            'ORDER BY "finalWorth" DESC LIMIT 1',
        )
        self.assertEqual(
            parsed.sqls[1],
            'SELECT "country" AS "answer_2" FROM "table_1" '
            'ORDER BY "finalWorth" DESC LIMIT 1',
        )

    def test_parses_short_plan_array_response(self) -> None:
        parsed = parse_sql_list_response(
            """
            [
              {"sql": "SELECT \\"selfMade\\" AS \\"answer_1\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;"},
              {"sql": "SELECT \\"country\\" AS \\"answer_2\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;"}
            ]
            """,
            BATCH_REMOTE_DSL,
        )

        self.assertEqual(len(parsed.sqls), 2)
        self.assertEqual(
            parsed.sqls[0],
            'SELECT "selfMade" AS "answer_1" FROM "table_1" '
            'ORDER BY "finalWorth" DESC LIMIT 1',
        )

    def test_rejects_final_protocol_marker_in_sql_array(self) -> None:
        with self.assertRaises(SqlParseError):
            parse_sql_list_response(
                """
                [
                  {"final": true, "sql": "SELECT \\"selfMade\\" AS \\"answer_1\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;"},
                  {"sql": "SELECT \\"country\\" AS \\"answer_2\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;"}
                ]
                """,
                BATCH_REMOTE_DSL,
            )

    def test_parses_sql_array_to_batch_logic_dag(self) -> None:
        logic_dag = parse_remote_sql_list_to_logic_dag(
            """
            ```json
            [
              "SELECT \\"selfMade\\" AS \\"answer_1\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;",
              "SELECT \\"country\\" AS \\"answer_2\\" FROM \\"table_1\\" ORDER BY \\"finalWorth\\" DESC LIMIT 1;"
            ]
            ```
            """,
            BATCH_REMOTE_DSL,
        )

        self.assertEqual(logic_dag["task_type"], "table_reasoning.query")
        self.assertEqual([item["id"] for item in logic_dag["query_plans"]], ["Q0", "Q1"])
        self.assertEqual(
            [item["answer"]["name"] for item in logic_dag["query_plans"]],
            ["answer_1", "answer_2"],
        )
        self.assertNotIn("logic_dag", logic_dag["query_plans"][0])
        self.assertNotIn("task_type", logic_dag["query_plans"][0])
        self.assertNotIn("question", logic_dag["query_plans"][0])
        self.assertNotIn("sql", logic_dag["query_plans"][0])
        self.assertEqual(
            logic_dag["query_plans"][0]["nodes"][-1]["params"]["answer"],
            {"name": "answer_1", "type": "boolean"},
        )
        self.assertEqual(
            logic_dag["query_plans"][1]["nodes"][-1]["params"]["answer"],
            {"name": "answer_2", "type": "string"},
        )

    def test_single_sql_uses_one_query_plan(self) -> None:
        logic_dag = _parse_remote_sql_to_logic_dag(
            'SELECT "selfMade" AS answer FROM "table_1" '
            'ORDER BY "finalWorth" DESC LIMIT 1;',
            REMOTE_DSL,
        )

        self.assertEqual(logic_dag["task_type"], "table_reasoning.query")
        self.assertEqual(len(logic_dag["query_plans"]), 1)
        self.assertEqual(logic_dag["query_plans"][0]["answer"]["name"], "answer")
        self.assertNotIn("nodes", logic_dag)

    def test_rejects_invalid_sql_array(self) -> None:
        with self.assertRaisesRegex(SqlParseError, "JSON array"):
            parse_sql_list_response(
                '{"sql": "SELECT 1"}',
                BATCH_REMOTE_DSL,
            )
        with self.assertRaisesRegex(SqlParseError, "length must equal"):
            parse_sql_list_response(
                '["SELECT \\"selfMade\\" AS \\"answer_1\\" FROM \\"table_1\\""]',
                BATCH_REMOTE_DSL,
            )
        with self.assertRaisesRegex(SqlParseError, 'alias its output as "answer_1"'):
            parse_sql_list_response(
                '["SELECT \\"selfMade\\" AS \\"wrong\\" FROM \\"table_1\\";", '
                '"SELECT \\"country\\" AS \\"answer_2\\" FROM \\"table_1\\";"]',
                BATCH_REMOTE_DSL,
            )

    def test_parses_to_logic_dag(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT "selfMade" FROM "table_1" ORDER BY "finalWorth" DESC LIMIT 1;',
            REMOTE_DSL,
        )

        self.assertEqual(
            set(logic_dag),
            {"id", "index", "answer", "nodes", "edges"},
        )
        self.assertEqual(logic_dag["answer"], {"name": "answer", "type": "boolean"})
        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Sort", "Limit", "Project", "FormatAnswer"],
        )
        self.assertEqual(logic_dag["nodes"][0]["id"], "N0")
        self.assertEqual(logic_dag["nodes"][0]["input"], ["table_1"])
        self.assertEqual(logic_dag["nodes"][0]["output"], "T0")
        self.assertEqual(logic_dag["nodes"][0]["params"]["source"], "table_1")
        self.assertEqual(logic_dag["nodes"][1]["dependency"], ["T0"])
        self.assertEqual(
            [node["input"] for node in logic_dag["nodes"][1:]],
            [[], [], [], []],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(
            logic_dag["nodes"][1]["params"]["keys"],
            [
                {
                    "expr": {"type": "column", "name": "finalWorth"},
                    "direction": "DESC",
                    "nulls": "LAST",
                }
            ],
        )
        self.assertEqual(logic_dag["nodes"][2]["params"]["count"], 1)
        self.assertEqual(
            logic_dag["nodes"][3]["params"]["expressions"],
            [{"expr": {"type": "column", "name": "selfMade"}}],
        )
        self.assertEqual(logic_dag["nodes"][-1]["output"], "answer")
        self.assertEqual(logic_dag["nodes"][-1]["params"]["answer"]["type"], "boolean")
        self.assertTrue(all(node["op"] in ALLOWED_OPS for node in logic_dag["nodes"]))
        self.assertEqual(
            logic_dag["edges"][0],
            {"from": "N0", "to": "N1"},
        )

    def test_parses_filter_aggregate_and_distinct_ops(self) -> None:
        aggregate_dag = parse_remote_sql_to_logic_dag(
            'SELECT COUNT(DISTINCT "country") AS answer '
            'FROM "table_1" WHERE "gender" = \'F\';',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in aggregate_dag["nodes"]],
            ["Scan", "Filter", "Distinct", "Aggregate", "Project", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(aggregate_dag)
        self.assertEqual(
            aggregate_dag["nodes"][1]["params"]["predicate"],
            {
                "type": "binary_op",
                "op": "=",
                "left": {"type": "column", "name": "gender"},
                "right": {"type": "literal", "value": "F", "value_type": "string"},
            },
        )
        self.assertEqual(
            aggregate_dag["nodes"][2]["params"]["on"],
            [{"type": "column", "name": "country"}],
        )
        self.assertEqual(
            aggregate_dag["nodes"][3]["params"]["aggregations"],
            [
                {
                    "function": "COUNT",
                    "argument": {"type": "column", "name": "country"},
                    "distinct": False,
                    "alias": "answer",
                }
            ],
        )

        distinct_dag = parse_remote_sql_to_logic_dag(
            'SELECT DISTINCT "country" FROM "table_1";',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in distinct_dag["nodes"]],
            ["Scan", "Distinct", "Project", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(distinct_dag)

    def test_parses_boolean_is_predicates(self) -> None:
        is_true_dag = parse_remote_sql_to_logic_dag(
            'SELECT COUNT(*) AS answer FROM "table_1" WHERE "selfMade" IS TRUE;',
            REMOTE_DSL,
        )

        self.assert_no_sql_expr(is_true_dag)
        self.assertEqual(
            is_true_dag["nodes"][1]["params"]["predicate"],
            {
                "type": "binary_op",
                "op": "=",
                "left": {"type": "column", "name": "selfMade"},
                "right": {"type": "literal", "value": True, "value_type": "boolean"},
            },
        )

        is_not_true_dag = parse_remote_sql_to_logic_dag(
            'SELECT NOT EXISTS (SELECT 1 FROM "table_1" '
            'WHERE "selfMade" IS NOT TRUE) AS answer;',
            REMOTE_DSL,
        )

        self.assert_no_sql_expr(is_not_true_dag)
        filter_node = next(
            node for node in is_not_true_dag["nodes"] if node["op"] == "Filter"
        )
        self.assertEqual(
            filter_node["params"]["predicate"],
            {
                "type": "binary_op",
                "op": "!=",
                "left": {"type": "column", "name": "selfMade"},
                "right": {"type": "literal", "value": True, "value_type": "boolean"},
            },
        )

    def test_parses_not_like_as_negated_like(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT "country" AS answer FROM "table_1" '
            'WHERE "country" NOT LIKE \'United%\';',
            REMOTE_DSL,
        )

        filter_node = next(node for node in logic_dag["nodes"] if node["op"] == "Filter")
        self.assertEqual(
            filter_node["params"]["predicate"],
            {
                "type": "not",
                "expr": {
                    "type": "like",
                    "case_sensitive": True,
                    "expr": {"type": "column", "name": "country"},
                    "pattern": {"type": "literal", "value": "United%", "value_type": "string"},
                },
            },
        )

    def test_preserves_binary_regression_aggregate_arguments(self) -> None:
        remote_dsl = {
            **REMOTE_DSL,
            "sources": [
                {
                    **REMOTE_DSL["sources"][0],
                    "schema": {
                        "format": "csv",
                        "columns": ["issue price", "year"],
                    },
                }
            ],
            "answer": {"name": "answer", "type": "number"},
        }
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT REGR_INTERCEPT("issue price", "year") '
            '+ REGR_SLOPE("issue price", "year") * 2012 AS "answer" '
            'FROM "table_1";',
            remote_dsl,
        )

        aggregate_node = next(
            node for node in logic_dag["nodes"] if node["op"] == "Aggregate"
        )
        aggregations = aggregate_node["params"]["aggregations"]
        self.assertEqual(
            aggregations[0],
            {
                "function": "REGR_INTERCEPT",
                "argument": {"type": "column", "name": "issue price"},
                "parameters": [{"type": "column", "name": "year"}],
                "distinct": False,
                "alias": "_agg_0",
            },
        )
        self.assertEqual(
            aggregations[1],
            {
                "function": "REGR_SLOPE",
                "argument": {"type": "column", "name": "issue price"},
                "parameters": [{"type": "column", "name": "year"}],
                "distinct": False,
                "alias": "_agg_1",
            },
        )

    def test_rejects_multiple_statements(self) -> None:
        with self.assertRaisesRegex(SqlParseError, "Expected one SQL statement"):
            parse_sql_response('SELECT * FROM "table_1"; SELECT 1;', REMOTE_DSL)

    def test_rejects_write_statement(self) -> None:
        with self.assertRaisesRegex(SqlParseError, "Only SELECT statements"):
            parse_sql_response('DELETE FROM "table_1";', REMOTE_DSL)

    def test_rejects_unknown_source(self) -> None:
        with self.assertRaisesRegex(SqlParseError, "unknown table sources"):
            parse_sql_response('SELECT * FROM "private_table";', REMOTE_DSL)

    def test_normalizes_backtick_quoted_identifiers(self) -> None:
        remote_dsl = {
            **REMOTE_DSL,
            "sources": [
                {
                    **REMOTE_DSL["sources"][0],
                    "schema": {
                        **REMOTE_DSL["sources"][0]["schema"],
                        "columns": ["season", "director (s)"],
                    },
                }
            ],
        }
        parsed = parse_sql_response(
            "SELECT COUNT(*) FROM table_1 "
            "WHERE season = 13 AND `director (s)` = 'william malone'",
            remote_dsl,
        )

        self.assertIn('"director (s)"', parsed.sql)
        logic_dag = parse_remote_sql_to_logic_dag(parsed.sql, remote_dsl)
        self.assert_logic_dag_io_contract(logic_dag)

    def test_parses_exists_subquery(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT EXISTS(SELECT 1 FROM "table_1" WHERE "selfMade" = \'True\') '
            'AS "answer";',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Filter", "Aggregate", "Derive", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(
            logic_dag["nodes"][1]["params"]["predicate"],
            {
                "type": "binary_op",
                "op": "=",
                "left": {"type": "column", "name": "selfMade"},
                "right": {"type": "literal", "value": "True", "value_type": "string"},
            },
        )
        self.assertEqual(
            logic_dag["nodes"][3]["params"]["expressions"][0]["expr"]["op"],
            ">",
        )

    def test_compiles_not_exists_to_basic_ops(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT NOT EXISTS (SELECT 1 FROM "table_1" '
            'WHERE "reviews_per_month" <= 5 OR "reviews_per_month" IS NULL) '
            'AS "answer";',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Filter", "Aggregate", "Derive", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(
            logic_dag["nodes"][1]["params"]["predicate"],
            {
                "type": "logical_op",
                "op": "OR",
                "operands": [
                    {
                        "type": "binary_op",
                        "op": "<=",
                        "left": {"type": "column", "name": "reviews_per_month"},
                        "right": {"type": "literal", "value": 5, "value_type": "number"},
                    },
                    {
                        "type": "is_null",
                        "expr": {"type": "column", "name": "reviews_per_month"},
                    },
                ],
            },
        )
        self.assertEqual(
            logic_dag["nodes"][3]["params"]["expressions"][0],
            {
                "alias": "answer",
                "expr": {
                    "type": "binary_op",
                    "op": "=",
                    "left": {"type": "column", "name": "_exists_count"},
                    "right": {"type": "literal", "value": 0, "value_type": "number"},
                },
            },
        )

    def test_ignores_supported_table_function_as_source(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT "verification_method" FROM "table_1" '
            'CROSS JOIN UNNEST("host_verifications") AS "verification_method" '
            'GROUP BY "verification_method" ORDER BY COUNT(*) DESC LIMIT 2;',
            REMOTE_DSL,
        )

        self.assertEqual(logic_dag["nodes"][0]["params"]["source"], "table_1")
        self.assertIn("Join", [node["op"] for node in logic_dag["nodes"]])
        self.assert_logic_dag_io_contract(logic_dag)

    def test_compiles_derived_table_subquery_with_group(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            """
            SELECT age_range AS answer
            FROM (
                SELECT
                    CASE
                        WHEN "Age" <= 18 THEN '0-18'
                        WHEN "Age" <= 30 THEN '18-30'
                        WHEN "Age" <= 50 THEN '30-50'
                        ELSE '50+'
                    END AS age_range,
                    COUNT(*) AS cnt
                FROM "table_1"
                WHERE "Age" IS NOT NULL
                GROUP BY age_range
            ) AS age_counts
            ORDER BY cnt DESC
            LIMIT 1
            """,
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            [
                "Scan",
                "Filter",
                "Derive",
                "Group",
                "Aggregate",
                "Project",
                "Sort",
                "Limit",
                "Project",
                "FormatAnswer",
            ],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(logic_dag["nodes"][2]["params"]["expressions"][0]["alias"], "age_range")
        self.assertEqual(
            logic_dag["nodes"][4]["params"]["aggregations"][0]["alias"],
            "cnt",
        )
        self.assertEqual(
            logic_dag["nodes"][6]["params"]["keys"][0]["expr"],
            {"type": "column", "name": "cnt"},
        )

    def test_compiles_scalar_aggregate_subquery_in_filter(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT "selfMade" FROM "table_1" '
            'WHERE "finalWorth" = (SELECT MAX("finalWorth") FROM "table_1");',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Aggregate", "Project", "Filter", "Project", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(logic_dag["nodes"][1]["dependency"], ["T0"])
        self.assertEqual(logic_dag["nodes"][3]["dependency"], ["T0", "T2"])
        self.assertEqual(
            logic_dag["nodes"][3]["params"]["predicate"],
            {
                "type": "binary_op",
                "op": "=",
                "left": {"type": "column", "name": "finalWorth"},
                "right": {
                    "type": "scalar_ref",
                    "source": "T2",
                    "name": "_agg_0",
                },
            },
        )
        self.assertEqual(
            logic_dag["nodes"][1]["params"]["aggregations"],
            [
                {
                    "function": "MAX",
                    "argument": {"type": "column", "name": "finalWorth"},
                    "distinct": False,
                    "alias": "_agg_0",
                }
            ],
        )
        self.assertIn({"from": "N0", "to": "N1"}, logic_dag["edges"])
        self.assertIn({"from": "N0", "to": "N3"}, logic_dag["edges"])
        self.assertIn({"from": "N2", "to": "N3"}, logic_dag["edges"])
        self.assert_no_sql_expr(logic_dag)

    def test_compiles_ordered_array_aggregate_from_derived_table(self) -> None:
        remote_dsl = {
            **REMOTE_DSL,
            "answer": {"name": "answer", "type": "list[category]"},
            "sources": [
                {
                    **REMOTE_DSL["sources"][0],
                    "schema": {"columns": ["country"]},
                }
            ],
        }
        logic_dag = parse_remote_sql_to_logic_dag(
            """
            SELECT ARRAY_AGG(area ORDER BY cnt DESC) AS answer
            FROM (
                SELECT "country" AS area, COUNT(*) AS cnt
                FROM "table_1"
                GROUP BY area
                ORDER BY cnt DESC
                LIMIT 3
            ) t;
            """,
            remote_dsl,
        )

        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)
        aggregation = logic_dag["nodes"][-3]["params"]["aggregations"][0]
        self.assertEqual(aggregation["function"], "ARRAY_AGG")
        self.assertEqual(aggregation["argument"], {"type": "column", "name": "area"})
        self.assertEqual(
            aggregation["order"],
            [
                {
                    "expr": {"type": "column", "name": "cnt"},
                    "direction": "DESC",
                    "nulls": "LAST",
                }
            ],
        )

    def test_compiles_ordered_limited_array_aggregate(self) -> None:
        remote_dsl = {
            **REMOTE_DSL,
            "answer": {"name": "answer", "type": "list[boolean]"},
        }
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT ARRAY_AGG("selfMade" ORDER BY "finalWorth" DESC LIMIT 2) AS answer '
            'FROM "table_1";',
            remote_dsl,
        )

        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)
        aggregation = logic_dag["nodes"][1]["params"]["aggregations"][0]
        self.assertEqual(aggregation["function"], "ARRAY_AGG")
        self.assertEqual(aggregation["argument"], {"type": "column", "name": "selfMade"})
        self.assertEqual(aggregation["limit"], 2)
        self.assertEqual(
            aggregation["order"],
            [
                {
                    "expr": {"type": "column", "name": "finalWorth"},
                    "direction": "DESC",
                    "nulls": "LAST",
                }
            ],
        )

    def test_lowers_common_sql_functions_and_between(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT LOWER(TRIM("country")) AS answer FROM "table_1" '
            'WHERE LENGTH("personName") BETWEEN 3 AND 30 '
            'ORDER BY EXTRACT(YEAR FROM "birthDate") DESC LIMIT 1;',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Filter", "Derive", "Sort", "Limit", "Project", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)
        self.assertEqual(
            logic_dag["nodes"][1]["params"]["predicate"]["type"],
            "logical_op",
        )
        self.assertEqual(
            logic_dag["nodes"][2]["params"]["expressions"][0]["expr"]["function"],
            "LOWER",
        )

    def test_compiles_scalar_subqueries_in_projection(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT ('
            '  (SELECT AVG("finalWorth") FROM "table_1" WHERE "gender" = \'M\') '
            '  > '
            '  (SELECT AVG("finalWorth") FROM "table_1" WHERE "gender" = \'F\')'
            ') AS answer;',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            [
                "Scan",
                "Filter",
                "Aggregate",
                "Project",
                "Filter",
                "Aggregate",
                "Project",
                "Derive",
                "Project",
                "FormatAnswer",
            ],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)
        self.assertEqual(logic_dag["nodes"][7]["dependency"], ["T3", "T6"])

    def test_compiles_scalar_subquery_arithmetic_without_alias(self) -> None:
        remote_dsl = {
            **REMOTE_DSL,
            "answer": {"name": "answer", "type": "number"},
            "sources": [
                {
                    **REMOTE_DSL["sources"][0],
                    "schema": {
                        "columns": [
                            "Composition",
                            "Expression",
                            "Drawing",
                            "Color",
                        ],
                    },
                }
            ],
        }
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT AVG("Composition") - ('
            '  SELECT AVG("Drawing") FROM ('
            '    SELECT "Drawing" FROM "table_1" ORDER BY "Color" ASC LIMIT 5'
            "  )"
            ') FROM ('
            '  SELECT "Composition" FROM "table_1" ORDER BY "Expression" DESC LIMIT 3'
            ")",
            remote_dsl,
        )

        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)
        derive_node = next(
            node
            for node in logic_dag["nodes"]
            if node["op"] == "Derive"
            and len(node["dependency"]) == 2
        )
        expr = derive_node["params"]["expressions"][0]["expr"]
        self.assertEqual(expr["type"], "binary_op")
        self.assertEqual(expr["op"], "-")
        self.assertEqual(expr["right"]["type"], "scalar_ref")
        self.assertEqual(logic_dag["nodes"][-1]["op"], "FormatAnswer")

    def test_compiles_cte_and_join_subquery(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            """
            SELECT EXISTS (
                SELECT 1
                FROM "table_1" t1
                INNER JOIN (
                    SELECT "country", MAX("finalWorth") AS max_worth
                    FROM "table_1"
                    GROUP BY "country"
                ) stats
                    ON t1."country" = stats."country"
                WHERE t1."finalWorth" = stats.max_worth
            ) AS answer;
            """,
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Group", "Aggregate", "Project", "Join", "Filter", "Aggregate", "Derive", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)

    def test_preserves_derived_subquery_columns_used_by_outer_filter(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT DISTINCT "month" FROM ('
            '  SELECT "month", PERCENT_RANK() OVER (ORDER BY "RH" ASC) AS rh_percentile '
            '  FROM "table_1"'
            ') sub WHERE rh_percentile <= 0.04;',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Derive", "Project", "Filter", "Distinct", "Project", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(
            logic_dag["nodes"][2]["params"]["expressions"],
            [
                {"expr": {"type": "column", "name": "month"}},
                {
                    "expr": {"type": "column", "name": "rh_percentile"},
                    "alias": "rh_percentile",
                },
            ],
        )
        self.assertEqual(
            logic_dag["nodes"][3]["params"]["predicate"]["left"],
            {"type": "column", "name": "rh_percentile"},
        )

    def test_materializes_group_by_expression_before_aggregate(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            'SELECT EXTRACT(YEAR FROM "Date Hired") AS hire_year '
            'FROM "table_1" GROUP BY EXTRACT(YEAR FROM "Date Hired") '
            'ORDER BY COUNT(*) DESC LIMIT 1;',
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["Scan", "Derive", "Group", "Aggregate", "Sort", "Limit", "Project", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assertEqual(logic_dag["nodes"][1]["params"]["expressions"][0]["alias"], "hire_year")
        self.assertEqual(
            logic_dag["nodes"][2]["params"]["keys"],
            [{"type": "column", "name": "hire_year"}],
        )
        self.assertEqual(
            logic_dag["nodes"][6]["params"]["expressions"],
            [{"expr": {"type": "column", "name": "hire_year"}, "alias": "hire_year"}],
        )

    def test_compiles_union_branches_inside_derived_table(self) -> None:
        logic_dag = parse_remote_sql_to_logic_dag(
            """
            SELECT service
            FROM (
              SELECT 'PhoneService' AS service, COUNT(*) AS count
              FROM "table_1" WHERE "PhoneService" = 'Yes'
              UNION ALL
              SELECT 'OnlineBackup' AS service, COUNT(*) AS count
              FROM "table_1" WHERE "OnlineBackup" = 'Yes'
            ) services
            ORDER BY count DESC
            LIMIT 1;
            """,
            REMOTE_DSL,
        )

        ops = [node["op"] for node in logic_dag["nodes"]]
        self.assertIn("SetOp", ops)
        self.assertEqual(ops.count("Aggregate"), 2)
        self.assertEqual(ops.count("Project"), 3)
        self.assert_logic_dag_io_contract(logic_dag)
        set_node = next(node for node in logic_dag["nodes"] if node["op"] == "SetOp")
        self.assertEqual(set_node["params"]["operator"], "UNION ALL")
        self.assertEqual(len(set_node["dependency"]), 2)

    def test_compiles_recursive_cte_to_repeat_union(self) -> None:
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
            REMOTE_DSL,
        )

        self.assertEqual(
            [node["op"] for node in logic_dag["nodes"]],
            ["RepeatUnion", "Filter", "Aggregate", "Derive", "FormatAnswer"],
        )
        self.assert_logic_dag_io_contract(logic_dag)
        self.assert_no_sql_expr(logic_dag)
        repeat_union = logic_dag["nodes"][0]
        self.assertEqual(repeat_union["input"], ["table_1"])
        self.assertEqual(repeat_union["params"]["name"], "descendants")
        self.assertTrue(repeat_union["params"]["all"])
        self.assertEqual(
            repeat_union["params"]["recursive_plan"]["nodes"][0]["params"],
            {
                "source": "descendants",
                "source_type": "transient",
                "read": "delta",
            },
        )
        self.assertEqual(logic_dag["nodes"][1]["dependency"], ["T0"])

    def assert_logic_dag_io_contract(self, logic_dag: dict) -> None:
        external_sources = {source["id"] for source in REMOTE_DSL["sources"]}
        produced_outputs = set()
        for node in logic_dag["nodes"]:
            self.assertTrue(set(node["input"]).issubset(external_sources))
            self.assertTrue(set(node["dependency"]).issubset(produced_outputs))
            produced_outputs.add(node["output"])

    def assert_no_sql_expr(self, payload: object) -> None:
        if isinstance(payload, dict):
            self.assertNotEqual(payload.get("type"), "sql_expr")
            for value in payload.values():
                self.assert_no_sql_expr(value)
        elif isinstance(payload, list):
            for value in payload:
                self.assert_no_sql_expr(value)


if __name__ == "__main__":
    unittest.main()
