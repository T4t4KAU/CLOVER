"""Parse remote table SQL responses into atomic Logic DAG nodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from clover.optimizer.errors import OptimizerParseError
from clover.task_types import is_table_task_type


ALLOWED_OPS = frozenset(
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
    }
)

COMPARISON_OPS = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}

ARITHMETIC_OPS = {
    exp.Add: "+",
    exp.Sub: "-",
    exp.Mul: "*",
    exp.Div: "/",
    exp.Mod: "%",
}


class SqlParseError(OptimizerParseError):
    """Raised when a Remote LLM SQL response cannot be safely parsed."""


@dataclass(frozen=True)
class ParsedSql:
    sql: str
    source_ids: tuple[str, ...]


def parse_sql_response(remote_response: str, remote_dsl: dict[str, Any]) -> ParsedSql:
    """Extract and validate one read-only SQL statement from Remote LLM text."""

    sql = extract_sql_statement(remote_response)
    expression = _parse_sql_ast(sql)
    _validate_read_only_select(sql, expression)

    allowed_source_ids = _source_ids(remote_dsl)
    source_ids = _extract_table_references(expression)
    if not source_ids:
        raise SqlParseError("SQL statement does not reference any table source")

    unknown_sources = sorted(set(source_ids) - set(allowed_source_ids))
    if unknown_sources:
        raise SqlParseError(f"SQL references unknown table sources: {unknown_sources}")

    return ParsedSql(sql=sql, source_ids=tuple(source_ids))


def parse_predicate_fragment(sql_fragment: str) -> dict[str, Any]:
    """Parse a SQL predicate fragment (e.g. ``WHERE "col" = 'val'``) into AST.

    Used by the Edge Agent ``rewrite_predicate`` action to convert an SLM's SQL
    predicate rewrite into the predicate AST understood by the pandas backend.
    """
    text = sql_fragment.strip()
    # Strip a leading WHERE keyword if present.
    if text.upper().startswith("WHERE "):
        text = text[6:].strip()
    # Wrap in a SELECT so sqlglot parses it as a WHERE clause.
    wrapped = f"SELECT * FROM __t__ WHERE {text}"
    expression = _parse_sql_ast(wrapped)
    select = expression
    if not isinstance(select, exp.Select):
        raise SqlParseError("Predicate fragment did not parse to a SELECT")
    where = select.args.get("where")
    if where is None:
        raise SqlParseError("Predicate fragment has no WHERE clause")
    return _expr_ast(where.this)


def parse_remote_sql_to_logic_dag(
    remote_response: str,
    remote_dsl: dict[str, Any],
) -> dict[str, Any]:
    """Parse Remote LLM SQL output into the table query-plan Logic DAG shape."""

    _validate_table_reasoning_dsl(remote_dsl)
    parsed_sql = parse_sql_response(remote_response, remote_dsl)
    return _parsed_sql_to_logic_dag(parsed_sql, remote_dsl)


def _parsed_sql_to_logic_dag(
    parsed_sql: ParsedSql,
    remote_dsl: dict[str, Any],
) -> dict[str, Any]:
    """Build a one-query batch DAG used by table query reasoning."""

    fragment = _parsed_sql_to_query_fragment(parsed_sql, remote_dsl)
    answer = remote_dsl.get("answer", {})
    answer_name = answer.get("name", "answer") if isinstance(answer, dict) else "answer"
    return {
        "task_type": remote_dsl["task_type"],
        "resource_processing": [],
        "source_sql": parsed_sql.sql,
        "query_plans": [
            {
                "id": str(answer_name),
                "index": 0,
                "answer": answer,
                **fragment,
            }
        ],
    }


def _parsed_sql_to_query_fragment(
    parsed_sql: ParsedSql,
    remote_dsl: dict[str, Any],
) -> dict[str, Any]:
    """Build the reusable query-plan fragment without task metadata."""

    expression = _parse_sql_ast(parsed_sql.sql)
    # Lower SQL through sqlglot AST nodes into the fixed atomic op set. The
    # validator below rejects any fallback raw SQL fragments that escaped lowering.
    nodes = _compile_query(expression, parsed_sql.source_ids, remote_dsl)
    _validate_ops(nodes)
    _validate_node_io(nodes, _source_ids(remote_dsl))
    _validate_no_unexpanded_sql(nodes)

    return {"nodes": nodes, "edges": _dependency_edges(nodes)}


def _validate_table_reasoning_dsl(remote_dsl: dict[str, Any]) -> None:
    if not is_table_task_type(remote_dsl.get("task_type")):
        raise SqlParseError("SQL parser requires a table_reasoning task_type")


def extract_sql_statement(remote_response: str) -> str:
    """Return one SQL statement from plain or fenced Remote LLM output."""

    if not isinstance(remote_response, str) or not remote_response.strip():
        raise SqlParseError("Remote response is empty")

    plan_sql = _extract_json_plan_sql(remote_response)
    if plan_sql is not None:
        return _clean_single_statement(plan_sql)

    candidates = _extract_fenced_blocks(remote_response)
    if not candidates:
        candidates = [remote_response]

    errors = []
    for candidate in candidates:
        try:
            return _clean_single_statement(candidate)
        except SqlParseError as exc:
            errors.append(str(exc))

    detail = "; ".join(errors) if errors else "no SQL candidate found"
    raise SqlParseError(f"Unable to extract one SQL statement: {detail}")


def _extract_json_plan_sql(remote_response: str) -> str | None:
    text = remote_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "final" in payload:
        raise SqlParseError("Remote SQL JSON must not include final")
    sql = payload.get("sql")
    if isinstance(sql, str) and sql.strip():
        return sql
    return None


def _compile_query(
    expression: exp.Expression,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(expression, exp.Select):
        _compile_select_into(
            nodes=nodes,
            select=expression,
            source_ids=source_ids,
            remote_dsl=remote_dsl,
        )
        return nodes
    if isinstance(expression, (exp.Union, exp.Intersect, exp.Except)):
        return _compile_set_operation(expression, source_ids, remote_dsl)
    raise SqlParseError(f"Unsupported SQL expression type: {type(expression).__name__}")


def _compile_select_into(
    nodes: list[dict[str, Any]],
    select: exp.Select,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    format_answer: bool = True,
    relation_outputs: dict[str, str] | None = None,
    reuse_existing_scan: bool = False,
) -> str:
    relation_outputs = dict(relation_outputs or {})
    # SELECT lowering follows relational execution order: CTE/source/join,
    # row filters, grouping/aggregation, post-aggregate filters, ordering,
    # limit, answer projection, and answer formatting.
    _append_cte_nodes(
        nodes=nodes,
        select=select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
    )

    exists_projection = _exists_projection(select)
    if exists_projection is not None:
        if not format_answer:
            raise SqlParseError("EXISTS projections in derived tables are not supported")
        exists_select, negated, alias = exists_projection
        return _append_exists_select_nodes(
            nodes=nodes,
            exists_select=exists_select,
            negated=negated,
            alias=alias or remote_dsl["answer"].get("name", "answer"),
            source_ids=source_ids,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )

    previous_output = _append_source_nodes(
        nodes=nodes,
        select=select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
        reuse_existing_scan=reuse_existing_scan,
    )

    previous_output = _append_join_nodes(
        nodes=nodes,
        select=select,
        input_ref=previous_output,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
    )

    where = select.args.get("where")
    if where is not None:
        if previous_output is None:
            raise SqlParseError("WHERE requires a table input")
        previous_output = _append_filter_node(
            nodes,
            input_ref=previous_output,
            predicate=where.this,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )

    select_items = [_select_item(item) for item in select.expressions]
    order = select.args.get("order")
    having = select.args.get("having")
    aggregate_calls = _collect_aggregate_calls(
        [item["expr"] for item in select_items],
        _order_expressions(order),
        [having.this] if having is not None else [],
    )
    aggregate_aliases = _aggregate_aliases(aggregate_calls, select_items)
    select_output_aliases = [
        _select_output_alias(item, index, aggregate_aliases)
        for index, item in enumerate(select_items)
    ]

    pre_group_derive_items = _pre_group_derive_items(
        group=select.args.get("group"),
        select_items=select_items,
        select_output_aliases=select_output_aliases,
        aggregate_aliases=aggregate_aliases,
    )
    materialized_select_aliases = {
        id(item["expr"]): alias for item, alias in pre_group_derive_items
    }
    materialized_group_aliases = {
        _expr_signature(item["expr"]): alias for item, alias in pre_group_derive_items
    }
    if pre_group_derive_items:
        previous_output, scalar_refs, scalar_dependencies = _append_expression_refs(
            nodes=nodes,
            input_ref=previous_output,
            expressions=[item["expr"] for item, _ in pre_group_derive_items],
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )
        previous_output = _append_node(
            nodes,
            op="Derive",
            input_ref=None,
            dependency_refs=_dependencies_for_node(previous_output, scalar_dependencies),
            params={
                "expressions": [
                    _output_expr_ast(
                        item,
                        aggregate_aliases,
                        scalar_subquery_refs=scalar_refs,
                        alias_override=alias,
                    )
                    for item, alias in pre_group_derive_items
                ]
            },
        )

    distinct_targets = _distinct_targets(select, aggregate_calls)
    if distinct_targets:
        if previous_output is None:
            raise SqlParseError("DISTINCT requires a table input")
        previous_output = _append_node(
            nodes,
            op="Distinct",
            input_ref=previous_output,
            params={"on": [_expr_ast(target) for target in distinct_targets]},
        )

    group = select.args.get("group")
    if group is not None and group.expressions:
        if previous_output is None:
            raise SqlParseError("GROUP BY requires a table input")
        previous_output = _append_node(
            nodes,
            op="Group",
            input_ref=previous_output,
            params={
                "keys": [
                    _group_key_ast(item, materialized_group_aliases)
                    for item in group.expressions
                ]
            },
        )

    if aggregate_calls:
        if previous_output is None:
            raise SqlParseError("Aggregate requires a table input")
        previous_output = _append_node(
            nodes,
            op="Aggregate",
            input_ref=previous_output,
            params={
                "aggregations": [
                    _aggregation_ast(call, _aggregate_alias(call, aggregate_aliases))
                    for call in aggregate_calls
                ],
                "grouped": bool(group and group.expressions),
            },
        )

    if having is not None:
        if previous_output is None:
            raise SqlParseError("HAVING requires a table input")
        previous_output = _append_filter_node(
            nodes,
            input_ref=previous_output,
            predicate=having.this,
            remote_dsl=remote_dsl,
            aggregate_aliases=aggregate_aliases,
            relation_outputs=relation_outputs,
        )

    derive_items = [
        (item, alias)
        for item, alias in zip(select_items, select_output_aliases)
        if _is_derived_select_item(item["expr"], aggregate_aliases)
        and id(item["expr"]) not in materialized_select_aliases
    ]
    if derive_items:
        previous_output, scalar_refs, scalar_dependencies = _append_expression_refs(
            nodes=nodes,
            input_ref=previous_output,
            expressions=[item["expr"] for item, _ in derive_items],
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )
        previous_output = _append_node(
            nodes,
            op="Derive",
            input_ref=None,
            dependency_refs=_dependencies_for_node(previous_output, scalar_dependencies),
            params={
                "expressions": [
                    _output_expr_ast(
                        item,
                        aggregate_aliases,
                        scalar_subquery_refs=scalar_refs,
                        alias_override=alias,
                    )
                    for item, alias in derive_items
                ]
            },
        )
        materialized_select_aliases.update(
            {id(item["expr"]): alias for item, alias in derive_items}
        )

    if order is not None and order.expressions:
        if previous_output is None:
            raise SqlParseError("ORDER BY requires a table input")
        previous_output = _append_node(
            nodes,
            op="Sort",
            input_ref=previous_output,
            params={
                "keys": [_sort_key_ast(item, aggregate_aliases) for item in order.expressions]
            },
        )

    limit = select.args.get("limit")
    if limit is not None:
        if previous_output is None:
            raise SqlParseError("LIMIT requires a table input")
        previous_output = _append_node(
            nodes,
            op="Limit",
            input_ref=previous_output,
            params=_limit_params(select),
        )

    if select_items:
        previous_output, scalar_refs, scalar_dependencies = _append_expression_refs(
            nodes=nodes,
            input_ref=previous_output,
            expressions=[
                item["expr"]
                for item in select_items
                if not _select_item_is_materialized(item, aggregate_aliases, materialized_select_aliases)
            ],
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )
        previous_output = _append_node(
            nodes,
            op="Project",
            input_ref=None,
            dependency_refs=_dependencies_for_node(previous_output, scalar_dependencies),
            params={
                "expressions": [
                    _final_project_expr_ast(
                        item=item,
                        alias=alias,
                        aggregate_aliases=aggregate_aliases,
                        materialized_select_aliases=materialized_select_aliases,
                        scalar_subquery_refs=scalar_refs,
                    )
                    for item, alias in zip(select_items, select_output_aliases)
                ]
            },
        )

    if format_answer:
        if previous_output is None:
            raise SqlParseError("Cannot format an empty SELECT plan")
        _append_node(
            nodes,
            op="FormatAnswer",
            input_ref=previous_output,
            output="answer",
            params={"answer": remote_dsl["answer"]},
        )
        return "answer"
    if previous_output is None:
        raise SqlParseError("SELECT did not produce an output")
    return previous_output


def _append_source_nodes(
    nodes: list[dict[str, Any]],
    select: exp.Select,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
    reuse_existing_scan: bool = False,
) -> str | None:
    from_clause = select.args.get("from_")
    source = from_clause.this if from_clause is not None else None
    if source is None:
        return None
    if isinstance(source, exp.Table):
        source_name = _table_name(source)
        if source_name in relation_outputs:
            return relation_outputs[source_name]
        source_ids = (source_name,)
    if isinstance(source, exp.Subquery) and isinstance(
        source.this, (exp.Select, exp.Union, exp.Intersect, exp.Except)
    ):
        output_ref = _append_query_expression_nodes(
            nodes=nodes,
            expression=source.this,
            source_ids=_extract_table_references(source.this),
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
            reuse_existing_scan=True,
        )
        alias = _table_alias(source)
        if alias:
            relation_outputs[alias] = output_ref
        return output_ref

    if reuse_existing_scan:
        existing_output = _find_equivalent_scan_output(nodes, source_ids)
        if existing_output is not None:
            return existing_output

    return _append_node(
        nodes,
        op="Scan",
        input_ref=None,
        input_values=list(source_ids),
        params=_scan_params(source_ids),
    )


def _append_cte_nodes(
    nodes: list[dict[str, Any]],
    select: exp.Select,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
) -> None:
    with_expression = select.args.get("with_")
    if with_expression is None:
        return

    # CTE aliases are treated as named relation outputs. Non-recursive CTEs are
    # inlined as nested plans; recursive CTEs become one RepeatUnion node.
    for cte in with_expression.expressions:
        alias = cte.alias
        if not alias:
            raise SqlParseError("CTE without an alias is not supported")
        if _is_recursive_cte(cte):
            output_ref = _append_repeat_union_node(
                nodes=nodes,
                cte=cte,
                remote_dsl=remote_dsl,
            )
        elif isinstance(cte.this, exp.Select):
            cte_source_ids = _extract_table_references(cte.this)
            output_ref = _compile_select_into(
                nodes=nodes,
                select=cte.this,
                source_ids=cte_source_ids or source_ids,
                remote_dsl=remote_dsl,
                format_answer=False,
                relation_outputs=relation_outputs,
                reuse_existing_scan=True,
            )
        elif isinstance(cte.this, (exp.Union, exp.Intersect, exp.Except)):
            cte_source_ids = _extract_table_references(cte.this)
            output_ref = _append_set_operation_nodes(
                nodes=nodes,
                expression=cte.this,
                source_ids=cte_source_ids or source_ids,
                remote_dsl=remote_dsl,
                relation_outputs=relation_outputs,
                reuse_existing_scan=True,
            )
        else:
            raise SqlParseError(
                f"Unsupported CTE expression type: {type(cte.this).__name__}"
            )
        relation_outputs[alias] = output_ref


def _is_recursive_cte(cte: exp.CTE) -> bool:
    alias = cte.alias
    if not alias or not isinstance(cte.this, exp.Union):
        return False
    return _contains_table_reference(cte.this.expression, alias)


def _append_repeat_union_node(
    nodes: list[dict[str, Any]],
    cte: exp.CTE,
    remote_dsl: dict[str, Any],
) -> str:
    alias = cte.alias
    union = cte.this
    if not alias or not isinstance(union, exp.Union):
        raise SqlParseError("Recursive CTE must be a UNION expression")
    if not isinstance(union.this, exp.Select) or not isinstance(union.expression, exp.Select):
        raise SqlParseError("Recursive CTE seed and step must be SELECT queries")

    seed_select = union.this
    recursive_select = union.expression
    source_ids = _external_source_ids(
        [seed_select, recursive_select],
        allowed_source_ids=_source_ids(remote_dsl),
    )
    if not source_ids:
        raise SqlParseError("Recursive CTE does not reference any table source")

    # Recursive CTE support stays inside one atomic RepeatUnion node so the
    # outer Logic DAG remains acyclic and downstream static tools can execute
    # the loop with explicit seed/recursive nested plans.
    seed_plan = _compile_repeat_union_branch(
        select=seed_select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        transient_table=None,
    )
    recursive_plan = _compile_repeat_union_branch(
        select=recursive_select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        transient_table=alias,
    )

    return _append_node(
        nodes,
        op="RepeatUnion",
        input_ref=None,
        input_values=list(source_ids),
        params={
            "name": alias,
            "transient_table": alias,
            "all": union.args.get("distinct") is False,
            "iteration_limit": -1,
            "termination": "until_empty_delta",
            "seed_plan": seed_plan,
            "recursive_plan": recursive_plan,
        },
    )


def _compile_repeat_union_branch(
    select: exp.Select,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    transient_table: str | None,
) -> dict[str, Any]:
    branch_nodes: list[dict[str, Any]] = []
    relation_outputs: dict[str, str] = {}
    if transient_table is not None:
        relation_outputs[transient_table] = _append_node(
            branch_nodes,
            op="Scan",
            input_ref=None,
            input_values=[],
            params={
                "source": transient_table,
                "source_type": "transient",
                "read": "delta",
            },
        )

    output_ref = _compile_select_into(
        nodes=branch_nodes,
        select=select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        format_answer=False,
        relation_outputs=relation_outputs,
        reuse_existing_scan=True,
    )
    return {
        "nodes": branch_nodes,
        "edges": _dependency_edges(branch_nodes),
        "output": output_ref,
    }


def _append_join_nodes(
    nodes: list[dict[str, Any]],
    select: exp.Select,
    input_ref: str | None,
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
) -> str | None:
    previous_output = input_ref
    for join in select.args.get("joins") or []:
        if previous_output is None:
            raise SqlParseError("JOIN requires a left table input")
        join_payload, extra_dependency, resource_inputs = _join_param(
            nodes=nodes,
            join=join,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )
        dependencies = [previous_output]
        if extra_dependency and extra_dependency not in dependencies:
            dependencies.append(extra_dependency)
        previous_output = _append_node(
            nodes,
            op="Join",
            input_ref=None,
            input_values=resource_inputs,
            dependency_refs=dependencies,
            params={"joins": [join_payload]},
        )
    return previous_output


def _append_filter_node(
    nodes: list[dict[str, Any]],
    input_ref: str,
    predicate: exp.Expression,
    remote_dsl: dict[str, Any],
    aggregate_aliases: dict[int, str] | None = None,
    relation_outputs: dict[str, str] | None = None,
) -> str:
    scalar_refs, scalar_dependencies = _append_scalar_subquery_nodes(
        nodes=nodes,
        expression=predicate,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs or {},
    )
    return _append_node(
        nodes,
        op="Filter",
        input_ref=None,
        dependency_refs=_dependencies_for_node(input_ref, scalar_dependencies),
        params={
            "predicate": _expr_ast(
                predicate,
                aggregate_aliases=aggregate_aliases,
                scalar_subquery_refs=scalar_refs,
            )
        },
    )


def _append_scalar_subquery_nodes(
    nodes: list[dict[str, Any]],
    expression: exp.Expression,
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    scalar_refs: dict[int, dict[str, Any]] = {}
    dependencies: list[str] = []
    seen_sql: dict[str, dict[str, Any]] = {}
    relation_outputs = relation_outputs or {}

    # Scalar and EXISTS subqueries are expanded into upstream nested plans, then
    # referenced by structured scalar_ref/set_ref expressions in the parent node.
    for exists in _outer_exists_expressions(expression):
        signature = exists.sql(dialect="duckdb")
        if signature in seen_sql:
            scalar_refs[id(exists)] = seen_sql[signature]
            if seen_sql[signature]["source"] not in dependencies:
                dependencies.append(seen_sql[signature]["source"])
            continue

        alias = f"_exists_{len(scalar_refs)}"
        output_ref, value_alias = _append_exists_value_subquery(
            nodes=nodes,
            exists=exists,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
            fallback_alias=alias,
        )
        ref = {"type": "scalar_ref", "source": output_ref, "name": value_alias}
        scalar_refs[id(exists)] = ref
        seen_sql[signature] = ref
        dependencies.append(output_ref)

    for subquery in _outer_subquery_expressions(expression):
        signature = subquery.sql(dialect="duckdb")
        if signature in seen_sql:
            scalar_refs[id(subquery)] = seen_sql[signature]
            if seen_sql[signature]["source"] not in dependencies:
                dependencies.append(seen_sql[signature]["source"])
            continue

        alias = f"_subquery_{len(scalar_refs)}"
        output_ref, value_alias, ref_type = _append_value_subquery(
            nodes=nodes,
            subquery=subquery,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
            fallback_alias=alias,
        )
        ref = {
            "type": ref_type,
            "source": output_ref,
            "name": value_alias,
        }
        scalar_refs[id(subquery)] = ref
        seen_sql[signature] = ref
        dependencies.append(output_ref)

    return scalar_refs, dependencies


def _append_expression_refs(
    nodes: list[dict[str, Any]],
    input_ref: str | None,
    expressions: list[exp.Expression],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
) -> tuple[str | None, dict[int, dict[str, Any]], list[str]]:
    scalar_refs: dict[int, dict[str, Any]] = {}
    scalar_dependencies: list[str] = []
    for expression in expressions:
        refs, dependencies = _append_scalar_subquery_nodes(
            nodes=nodes,
            expression=expression,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )
        scalar_refs.update(refs)
        for dependency in dependencies:
            if dependency not in scalar_dependencies:
                scalar_dependencies.append(dependency)
    return input_ref, scalar_refs, scalar_dependencies


def _dependencies_for_node(
    input_ref: str | None,
    dependencies: list[str],
) -> list[str]:
    refs = []
    if input_ref is not None:
        refs.append(input_ref)
    for dependency in dependencies:
        if dependency not in refs:
            refs.append(dependency)
    return refs


def _outer_exists_expressions(expression: exp.Expression) -> list[exp.Exists]:
    return [
        item
        for item in expression.find_all(exp.Exists)
        if not _has_expression_ancestor(item, expression, (exp.Subquery, exp.Exists))
    ]


def _outer_subquery_expressions(expression: exp.Expression) -> list[exp.Subquery]:
    return [
        item
        for item in expression.find_all(exp.Subquery)
        if not _has_expression_ancestor(item, expression, (exp.Subquery, exp.Exists))
    ]


def _contains_outer_subquery_expression(expression: exp.Expression) -> bool:
    return bool(
        _outer_subquery_expressions(expression)
        or _outer_exists_expressions(expression)
    )


def _has_expression_ancestor(
    expression: exp.Expression,
    root: exp.Expression,
    ancestor_types: tuple[type[exp.Expression], ...],
) -> bool:
    parent = expression.parent
    while parent is not None:
        if parent is root:
            return isinstance(parent, ancestor_types)
        if isinstance(parent, ancestor_types):
            return True
        parent = parent.parent
    return False


def _append_value_subquery(
    nodes: list[dict[str, Any]],
    subquery: exp.Subquery,
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
    fallback_alias: str,
) -> tuple[str, str, str]:
    if not isinstance(subquery.this, exp.Select):
        raise SqlParseError("Only SELECT scalar subqueries are supported in predicates")

    select = subquery.this
    if len(select.expressions) != 1:
        raise SqlParseError("Subqueries must project exactly one expression")
    source_ids = _extract_table_references(select)
    output_ref = _compile_select_into(
        nodes=nodes,
        select=select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        format_answer=False,
        relation_outputs=relation_outputs,
        reuse_existing_scan=True,
    )
    return (
        output_ref,
        _select_output_name(select, fallback_alias),
        "scalar_ref" if _is_scalar_value_subquery(select) else "set_ref",
    )


def _select_output_name(select: exp.Select, fallback_alias: str) -> str:
    item = _select_item(select.expressions[0])
    if item.get("alias"):
        return item["alias"]
    expression = item["expr"]
    if isinstance(expression, exp.Column):
        return expression.name
    aggregate_calls = _collect_aggregate_calls([expression])
    if _is_aggregate_only_select_item(expression) and aggregate_calls:
        aggregate_aliases = _aggregate_aliases(aggregate_calls, [item])
        return _aggregate_alias(aggregate_calls[0], aggregate_aliases)
    return fallback_alias


def _is_scalar_value_subquery(select: exp.Select) -> bool:
    group = select.args.get("group")
    if _collect_aggregate_calls([_select_item(select.expressions[0])["expr"]]) and not (
        group and group.expressions
    ):
        return True
    limit = select.args.get("limit")
    if limit is not None and _literal_value(limit.expression) == 1:
        return True
    return False


def _append_exists_value_subquery(
    nodes: list[dict[str, Any]],
    exists: exp.Exists,
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
    fallback_alias: str,
) -> tuple[str, str]:
    if not isinstance(exists.this, exp.Select):
        raise SqlParseError("Only SELECT EXISTS subqueries are supported")

    return _append_exists_core(
        nodes=nodes,
        exists_select=exists.this,
        negated=False,
        alias=fallback_alias,
        source_ids=_extract_table_references(exists.this),
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
    )


def _append_exists_select_nodes(
    nodes: list[dict[str, Any]],
    exists_select: exp.Select,
    negated: bool,
    alias: str,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
) -> str:
    output_ref, _ = _append_exists_core(
        nodes=nodes,
        exists_select=exists_select,
        negated=negated,
        alias=alias,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
    )
    _append_node(
        nodes,
        op="FormatAnswer",
        input_ref=output_ref,
        output="answer",
        params={"answer": remote_dsl["answer"]},
    )
    return "answer"


def _append_exists_core(
    nodes: list[dict[str, Any]],
    exists_select: exp.Select,
    negated: bool,
    alias: str,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
) -> tuple[str, str]:
    previous_output = _append_source_nodes(
        nodes=nodes,
        select=exists_select,
        source_ids=source_ids,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
        reuse_existing_scan=True,
    )
    previous_output = _append_join_nodes(
        nodes=nodes,
        select=exists_select,
        input_ref=previous_output,
        remote_dsl=remote_dsl,
        relation_outputs=relation_outputs,
    )
    if previous_output is None:
        raise SqlParseError("EXISTS requires a table input")
    where = exists_select.args.get("where")
    if where is not None:
        previous_output = _append_filter_node(
            nodes,
            input_ref=previous_output,
            predicate=where.this,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
        )

    count_alias = "_exists_count"
    previous_output = _append_node(
        nodes,
        op="Aggregate",
        input_ref=previous_output,
        params={
            "aggregations": [
                {
                    "function": "COUNT",
                    "argument": {"type": "wildcard"},
                    "distinct": False,
                    "alias": count_alias,
                }
            ],
            "grouped": False,
        },
    )
    previous_output = _append_node(
        nodes,
        op="Derive",
        input_ref=previous_output,
        params={
            "expressions": [
                {
                    "alias": alias,
                    "expr": {
                        "type": "binary_op",
                        "op": "=" if negated else ">",
                        "left": {"type": "column", "name": count_alias},
                        "right": {"type": "literal", "value": 0, "value_type": "number"},
                    },
                }
            ]
        },
    )
    return previous_output, alias


def _compile_set_operation(
    expression: exp.Expression,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    previous_output = _append_set_operation_nodes(
        nodes,
        expression,
        source_ids,
        remote_dsl=remote_dsl,
        relation_outputs={},
        reuse_existing_scan=False,
    )
    _append_node(
        nodes,
        op="FormatAnswer",
        input_ref=previous_output,
        output="answer",
        params={"answer": remote_dsl["answer"]},
    )
    return nodes


def _append_set_operation_nodes(
    nodes: list[dict[str, Any]],
    expression: exp.Expression,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
    reuse_existing_scan: bool,
) -> str:
    if not isinstance(expression, (exp.Union, exp.Intersect, exp.Except)):
        raise SqlParseError(f"Unsupported set operation: {type(expression).__name__}")

    dependencies = [
        _append_query_expression_nodes(
            nodes=nodes,
            expression=branch,
            source_ids=_extract_table_references(branch) or source_ids,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
            reuse_existing_scan=reuse_existing_scan,
        )
        for branch in (expression.this, expression.expression)
    ]
    return _append_node(
        nodes,
        op="SetOp",
        input_ref=None,
        dependency_refs=dependencies,
        params={
            "operator": _set_operator(expression),
            "branches": [
                {"side": "left", "query_type": type(expression.this).__name__},
                {"side": "right", "query_type": type(expression.expression).__name__},
            ],
        },
    )


def _set_operator(expression: exp.Expression) -> str:
    operator = expression.key.upper()
    if isinstance(expression, exp.Union) and expression.args.get("distinct") is False:
        return "UNION ALL"
    return operator


def _append_query_expression_nodes(
    nodes: list[dict[str, Any]],
    expression: exp.Expression,
    source_ids: tuple[str, ...],
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
    reuse_existing_scan: bool,
) -> str:
    if isinstance(expression, exp.Select):
        return _compile_select_into(
            nodes=nodes,
            select=expression,
            source_ids=source_ids,
            remote_dsl=remote_dsl,
            format_answer=False,
            relation_outputs=relation_outputs,
            reuse_existing_scan=reuse_existing_scan,
        )
    if isinstance(expression, (exp.Union, exp.Intersect, exp.Except)):
        return _append_set_operation_nodes(
            nodes=nodes,
            expression=expression,
            source_ids=source_ids,
            remote_dsl=remote_dsl,
            relation_outputs=relation_outputs,
            reuse_existing_scan=reuse_existing_scan,
        )
    raise SqlParseError(f"Unsupported query expression: {type(expression).__name__}")


def _exists_projection(select: exp.Select) -> tuple[exp.Select, bool, str | None] | None:
    if select.args.get("from_") is not None or len(select.expressions) != 1:
        return None

    item = select.expressions[0]
    alias = _alias_name(item)
    expr = _unwrap_parens(item.this if isinstance(item, exp.Alias) else item)
    negated = False
    if isinstance(expr, exp.Not):
        negated = True
        expr = _unwrap_parens(expr.this)
    if isinstance(expr, exp.Exists) and isinstance(expr.this, exp.Select):
        return expr.this, negated, alias
    return None


def _select_item(item: exp.Expression) -> dict[str, Any]:
    return {
        "expr": item.this if isinstance(item, exp.Alias) else item,
        "alias": _alias_name(item),
    }


def _select_output_alias(
    item: dict[str, Any],
    index: int,
    aggregate_aliases: dict[Any, str],
) -> str:
    if item.get("alias"):
        return item["alias"]
    expression = item["expr"]
    if _is_aggregate_only_select_item(expression):
        return _aggregate_alias(expression, aggregate_aliases)
    if isinstance(expression, exp.Column):
        return expression.name
    if _contains_outer_subquery_expression(expression):
        return f"_expr_{index}"
    return _expr_label(_expr_ast(expression, aggregate_aliases), f"_expr_{index}")


def _select_item_is_materialized(
    item: dict[str, Any],
    aggregate_aliases: dict[Any, str],
    materialized_select_aliases: dict[int, str],
) -> bool:
    expression = item["expr"]
    return (
        _is_aggregate_only_select_item(expression)
        or id(expression) in materialized_select_aliases
        or isinstance(expression, (exp.Column, exp.Star, exp.Literal))
    )


def _group_key_ast(
    expression: exp.Expression,
    materialized_group_aliases: dict[str, str],
) -> dict[str, Any]:
    alias = materialized_group_aliases.get(_expr_signature(expression))
    if alias:
        return {"type": "column", "name": alias}
    return _expr_ast(expression)


def _expr_label(expr: dict[str, Any], fallback: str) -> str:
    if expr.get("type") == "column":
        return expr.get("name") or fallback
    if expr.get("type") == "function_call":
        return str(expr.get("function", fallback)).lower()
    if expr.get("type") == "aggregate_call":
        return str(expr.get("function", fallback)).lower()
    return fallback


def _output_expr_ast(
    item: dict[str, Any],
    aggregate_aliases: dict[Any, str] | None = None,
    scalar_subquery_refs: dict[int, dict[str, Any]] | None = None,
    alias_override: str | None = None,
) -> dict[str, Any]:
    payload = {
        "expr": _expr_ast(
            item["expr"],
            aggregate_aliases,
            scalar_subquery_refs,
        )
    }
    alias = alias_override or item.get("alias")
    if alias:
        payload["alias"] = alias
    return payload


def _final_project_expr_ast(
    item: dict[str, Any],
    alias: str,
    aggregate_aliases: dict[Any, str],
    materialized_select_aliases: dict[int, str],
    scalar_subquery_refs: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expression = item["expr"]
    if _is_aggregate_only_select_item(expression):
        payload = {"expr": {"type": "column", "name": _aggregate_alias(expression, aggregate_aliases)}}
    elif id(expression) in materialized_select_aliases:
        payload = {"expr": {"type": "column", "name": materialized_select_aliases[id(expression)]}}
    else:
        payload = {
            "expr": _expr_ast(
                expression,
                aggregate_aliases,
                scalar_subquery_refs,
            )
        }

    item_alias = item.get("alias")
    if item_alias and _expr_label(payload["expr"], item_alias) != item_alias:
        payload["alias"] = item_alias
    elif item_alias:
        payload["alias"] = item_alias
    elif alias and payload["expr"].get("type") not in {"column", "wildcard"}:
        payload["alias"] = alias
    return payload


def _unwrap_parens(expression: exp.Expression) -> exp.Expression:
    while isinstance(expression, exp.Paren):
        expression = expression.this
    return expression


def _expr_ast(
    expression: exp.Expression,
    aggregate_aliases: dict[Any, str] | None = None,
    scalar_subquery_refs: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Expression lowering is intentionally closed over a small JSON-like AST
    # understood by the pandas executor. Unsupported SQL falls through to
    # sql_expr, which plan validation rejects before execution.
    if scalar_subquery_refs and id(expression) in scalar_subquery_refs:
        return scalar_subquery_refs[id(expression)]
    if aggregate_aliases:
        alias = _aggregate_alias_or_none(expression, aggregate_aliases)
        if alias is not None:
            return {"type": "column", "name": alias}
    if isinstance(expression, exp.Alias):
        return _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs)
    if isinstance(expression, exp.Paren):
        return _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs)
    if isinstance(expression, exp.Column):
        payload = {"type": "column", "name": expression.name}
        if expression.table:
            payload["table"] = expression.table
        return payload
    if isinstance(expression, exp.Star):
        return {"type": "wildcard"}
    if isinstance(expression, exp.Literal):
        return _literal_ast(expression)
    if isinstance(expression, exp.Neg):
        operand = _expr_ast(
            expression.this,
            aggregate_aliases,
            scalar_subquery_refs,
        )
        if (
            operand.get("type") == "literal"
            and operand.get("value_type") == "number"
            and isinstance(operand.get("value"), (int, float))
        ):
            return {
                "type": "literal",
                "value": -operand["value"],
                "value_type": "number",
            }
        return {
            "type": "binary_op",
            "op": "-",
            "left": {"type": "literal", "value": 0, "value_type": "number"},
            "right": operand,
        }
    if isinstance(expression, exp.Var):
        return {"type": "identifier", "name": str(expression.this)}
    if isinstance(expression, exp.Placeholder):
        return {"type": "placeholder", "name": str(expression.this)}
    if isinstance(expression, exp.Null):
        return {"type": "null"}
    if isinstance(expression, exp.Boolean):
        return {"type": "literal", "value": bool(expression.this), "value_type": "boolean"}
    if isinstance(expression, exp.And):
        return _logical_ast("AND", expression, aggregate_aliases, scalar_subquery_refs)
    if isinstance(expression, exp.Or):
        return _logical_ast("OR", expression, aggregate_aliases, scalar_subquery_refs)
    if isinstance(expression, exp.Not):
        if isinstance(expression.this, exp.Is) and isinstance(expression.this.expression, exp.Null):
            return {
                "type": "is_not_null",
                "expr": _expr_ast(
                    expression.this.this,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
            }
        if isinstance(expression.this, exp.Is) and isinstance(
            expression.this.expression,
            exp.Boolean,
        ):
            return {
                "type": "binary_op",
                "op": "!=",
                "left": _expr_ast(
                    expression.this.this,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
                "right": _expr_ast(
                    expression.this.expression,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
            }
        return {
            "type": "not",
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
        }
    if isinstance(expression, exp.Is) and isinstance(expression.expression, exp.Null):
        return {
            "type": "is_null",
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
        }
    if isinstance(expression, exp.Is) and isinstance(expression.expression, exp.Boolean):
        return {
            "type": "binary_op",
            "op": "=",
            "left": _expr_ast(
                expression.this,
                aggregate_aliases,
                scalar_subquery_refs,
            ),
            "right": _expr_ast(
                expression.expression,
                aggregate_aliases,
                scalar_subquery_refs,
            ),
        }
    if isinstance(expression, exp.Between):
        target = _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs)
        low = _expr_ast(expression.args["low"], aggregate_aliases, scalar_subquery_refs)
        high = _expr_ast(expression.args["high"], aggregate_aliases, scalar_subquery_refs)
        return {
            "type": "logical_op",
            "op": "AND",
            "operands": [
                {"type": "binary_op", "op": ">=", "left": target, "right": low},
                {"type": "binary_op", "op": "<=", "left": target, "right": high},
            ],
        }
    for klass, operator in COMPARISON_OPS.items():
        if isinstance(expression, klass):
            return {
                "type": "binary_op",
                "op": operator,
                "left": _expr_ast(
                    expression.this,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
                "right": _expr_ast(
                    expression.expression,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
            }
    for klass, operator in ARITHMETIC_OPS.items():
        if isinstance(expression, klass):
            return {
                "type": "binary_op",
                "op": operator,
                "left": _expr_ast(
                    expression.this,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
                "right": _expr_ast(
                    expression.expression,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
            }
    if isinstance(expression, exp.DPipe):
        return {
            "type": "function_call",
            "function": "CONCAT",
            "args": [
                _expr_ast(
                    expression.this,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
                _expr_ast(
                    expression.expression,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
            ],
        }
    if isinstance(expression, exp.In):
        query = expression.args.get("query")
        if query is not None:
            return {
                "type": "in_subquery",
                "expr": _expr_ast(
                    expression.this,
                    aggregate_aliases,
                    scalar_subquery_refs,
                ),
                "query": _expr_ast(query, aggregate_aliases, scalar_subquery_refs),
            }
        return {
            "type": "in",
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
            "values": [
                _expr_ast(item, aggregate_aliases, scalar_subquery_refs)
                for item in expression.expressions
            ],
        }
    if isinstance(expression, (exp.Like, exp.ILike)):
        payload = {
            "type": "like",
            "case_sensitive": isinstance(expression, exp.Like),
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
            "pattern": _expr_ast(
                expression.expression,
                aggregate_aliases,
                scalar_subquery_refs,
            ),
        }
        if expression.args.get("negate"):
            return {"type": "not", "expr": payload}
        return payload
    if isinstance(expression, exp.Cast):
        return {
            "type": "cast",
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
            "to": expression.args["to"].sql(dialect="duckdb"),
        }
    if isinstance(expression, exp.Tuple):
        return {
            "type": "tuple",
            "items": [
                _expr_ast(item, aggregate_aliases, scalar_subquery_refs)
                for item in expression.expressions
            ],
        }
    if isinstance(expression, exp.Case):
        return {
            "type": "case",
            "ifs": [
                {
                    "condition": _expr_ast(
                        item.this,
                        aggregate_aliases,
                        scalar_subquery_refs,
                    ),
                    "value": _expr_ast(
                        item.args["true"],
                        aggregate_aliases,
                        scalar_subquery_refs,
                    ),
                }
                for item in expression.args.get("ifs", [])
            ],
            "default": _expr_ast(
                expression.args["default"],
                aggregate_aliases,
                scalar_subquery_refs,
            )
            if expression.args.get("default") is not None
            else None,
        }
    if _is_aggregate_expression(expression):
        return _aggregate_call_ast(expression, distinct_override=None)
    if isinstance(expression, exp.Window):
        return _window_ast(expression, aggregate_aliases, scalar_subquery_refs)
    if isinstance(expression, exp.Func):
        return _function_call_ast(expression, aggregate_aliases, scalar_subquery_refs)
    if isinstance(expression, exp.Filter):
        return {
            "type": "filtered_expr",
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
            "filter": _expr_ast(
                expression.expression.this,
                aggregate_aliases,
                scalar_subquery_refs,
            )
            if isinstance(expression.expression, exp.Where)
            else _expr_ast(
                expression.expression,
                aggregate_aliases,
                scalar_subquery_refs,
            ),
        }
    if isinstance(expression, exp.WithinGroup):
        return {
            "type": "within_group",
            "expr": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
            "order": [
                _sort_key_ast(item, aggregate_aliases or {})
                for item in expression.expression.expressions
            ]
            if isinstance(expression.expression, exp.Order)
            else [],
        }
    if isinstance(expression, exp.Subquery):
        raise SqlParseError(
            "Unhandled subquery expression reached expression lowering"
        )
    if isinstance(expression, exp.Exists):
        raise SqlParseError("Unhandled EXISTS expression reached expression lowering")
    return {"type": "sql_expr", "sql": expression.sql(dialect="duckdb")}


def _logical_ast(
    operator: str,
    expression: exp.Expression,
    aggregate_aliases: dict[int, str] | None,
    scalar_subquery_refs: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    operands = []
    stack = [expression]
    klass = exp.And if operator == "AND" else exp.Or
    while stack:
        item = stack.pop(0)
        if isinstance(item, klass):
            stack.insert(0, item.expression)
            stack.insert(0, item.this)
        else:
            operands.append(_expr_ast(item, aggregate_aliases, scalar_subquery_refs))
    return {"type": "logical_op", "op": operator, "operands": operands}


def _literal_ast(literal: exp.Literal) -> dict[str, Any]:
    if literal.is_string:
        return {"type": "literal", "value": literal.this, "value_type": "string"}
    text = str(literal.this)
    try:
        value: int | float = int(text)
        value_type = "number"
    except ValueError:
        try:
            value = float(text)
            value_type = "number"
        except ValueError:
            value = text
            value_type = "string"
    return {"type": "literal", "value": value, "value_type": value_type}


def _literal_value(expression: exp.Expression | None) -> Any:
    if expression is None:
        return None
    if isinstance(expression, exp.Literal):
        return _literal_ast(expression)["value"]
    return expression.sql(dialect="duckdb")


def _limit_params(select: exp.Select) -> dict[str, Any]:
    limit = select.args.get("limit")
    params = {"count": _literal_value(limit.expression if limit is not None else None)}
    offset = select.args.get("offset")
    if offset is not None:
        params["offset"] = _literal_value(offset.expression)
    return params


def _is_aggregate_expression(expression: exp.Expression) -> bool:
    if isinstance(expression, exp.AggFunc):
        return True
    if isinstance(expression, exp.WithinGroup):
        return _is_aggregate_expression(expression.this)
    if isinstance(expression, exp.Filter):
        return _is_aggregate_expression(expression.this)
    return False


def _has_aggregate_ancestor(expression: exp.Expression) -> bool:
    parent = expression.parent
    while parent is not None:
        if _is_aggregate_expression(parent):
            return True
        parent = parent.parent
    return False


def _has_window_ancestor(expression: exp.Expression) -> bool:
    parent = expression.parent
    while parent is not None:
        if isinstance(parent, exp.Window):
            return True
        parent = parent.parent
    return False


def _has_subquery_ancestor_between(
    expression: exp.Expression,
    root: exp.Expression,
) -> bool:
    if expression is root:
        return False
    parent = expression.parent
    while parent is not None and parent is not root:
        if isinstance(parent, (exp.Subquery, exp.Exists)):
            return True
        parent = parent.parent
    return False


def _aggregate_call_ast(
    call: exp.Expression,
    distinct_override: bool | None,
) -> dict[str, Any]:
    distinct, argument = _aggregate_argument(call)
    parameters = _aggregate_parameters(call)
    payload = {
        "type": "aggregate_call",
        "function": _aggregate_name(call),
        "argument": _expr_ast(argument),
        "distinct": distinct if distinct_override is None else distinct_override,
    }
    if parameters:
        payload["parameters"] = [_expr_ast(parameter) for parameter in parameters]
    _add_ordered_aggregate_options(payload, call)
    return payload


def _aggregation_ast(call: exp.Expression, alias: str) -> dict[str, Any]:
    if isinstance(call, exp.Filter):
        payload = _aggregation_ast(call.this, alias)
        payload["filter"] = _expr_ast(
            call.expression.this
            if isinstance(call.expression, exp.Where)
            else call.expression
        )
        return payload
    if isinstance(call, exp.WithinGroup):
        aggregate = call.this
        payload = {
            "function": _aggregate_name(call),
            "argument": _within_group_argument(call),
            "distinct": False,
            "alias": alias,
        }
        if isinstance(aggregate, exp.AggFunc) and aggregate.this is not None:
            payload["parameters"] = [_expr_ast(aggregate.this)]
        if isinstance(call.expression, exp.Order):
            payload["order"] = [
                _sort_key_ast(item, {}) for item in call.expression.expressions
            ]
        return payload

    distinct, argument = _aggregate_argument(call)
    parameters = _aggregate_parameters(call)
    payload = {
        "function": _aggregate_name(call),
        "argument": _expr_ast(argument),
        "distinct": distinct,
        "alias": alias,
    }
    if parameters:
        payload["parameters"] = [_expr_ast(parameter) for parameter in parameters]
    _add_ordered_aggregate_options(payload, call)
    return payload


def _aggregate_argument(call: exp.Expression) -> tuple[bool, exp.Expression]:
    if isinstance(call, exp.Filter):
        return _aggregate_argument(call.this)
    if isinstance(call, exp.WithinGroup):
        return False, _within_group_argument_expression(call)
    if not isinstance(call, exp.AggFunc):
        raise SqlParseError(f"Unsupported aggregate expression: {type(call).__name__}")
    argument = call.this
    if isinstance(argument, exp.Distinct):
        expressions = list(argument.expressions)
        return True, expressions[0] if expressions else exp.Star()
    return False, _ordered_aggregate_argument(argument)


def _aggregate_parameters(call: exp.Expression) -> list[exp.Expression]:
    if isinstance(call, exp.Filter):
        return _aggregate_parameters(call.this)
    if isinstance(call, exp.WithinGroup):
        return []
    if not isinstance(call, exp.AggFunc):
        return []
    parameters: list[exp.Expression] = []
    for key in call.arg_types:
        if key == "this":
            continue
        value = call.args.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, exp.Expression):
            parameters.append(value)
            continue
        if isinstance(value, list):
            parameters.extend(item for item in value if isinstance(item, exp.Expression))
    return parameters


def _ordered_aggregate_argument(expression: exp.Expression | None) -> exp.Expression:
    if expression is None:
        return exp.Star()
    if isinstance(expression, exp.Limit):
        return _ordered_aggregate_argument(expression.this)
    if isinstance(expression, exp.Order):
        return expression.this if expression.this is not None else exp.Star()
    return expression


def _add_ordered_aggregate_options(
    payload: dict[str, Any],
    call: exp.Expression,
) -> None:
    order = _ordered_aggregate_order(call)
    if order:
        payload["order"] = [_sort_key_ast(item, {}) for item in order.expressions]
    limit = _ordered_aggregate_limit(call)
    if limit is not None:
        payload["limit"] = _literal_value(limit.expression)


def _ordered_aggregate_order(call: exp.Expression) -> exp.Order | None:
    if isinstance(call, exp.Filter):
        return _ordered_aggregate_order(call.this)
    if isinstance(call, exp.WithinGroup):
        return call.expression if isinstance(call.expression, exp.Order) else None
    if not isinstance(call, exp.AggFunc):
        return None
    expression = call.this
    if isinstance(expression, exp.Limit):
        expression = expression.this
    return expression if isinstance(expression, exp.Order) else None


def _ordered_aggregate_limit(call: exp.Expression) -> exp.Limit | None:
    if isinstance(call, exp.Filter):
        return _ordered_aggregate_limit(call.this)
    if not isinstance(call, exp.AggFunc):
        return None
    expression = call.this
    return expression if isinstance(expression, exp.Limit) else None


def _aggregate_name(call: exp.Expression) -> str:
    if isinstance(call, exp.Filter):
        return _aggregate_name(call.this)
    if isinstance(call, exp.WithinGroup):
        return _aggregate_name(call.this)
    if isinstance(call, exp.Func):
        return call.sql_name().upper()
    return call.key.upper()


def _within_group_argument(call: exp.WithinGroup) -> dict[str, Any]:
    return _expr_ast(_within_group_argument_expression(call))


def _within_group_argument_expression(call: exp.WithinGroup) -> exp.Expression:
    if isinstance(call.expression, exp.Order) and call.expression.expressions:
        return call.expression.expressions[0].this
    aggregate = call.this
    if isinstance(aggregate, exp.AggFunc) and aggregate.this is not None:
        return aggregate.this
    return exp.Star()


def _function_call_ast(
    expression: exp.Func,
    aggregate_aliases: dict[int, str] | None,
    scalar_subquery_refs: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "function": _function_name(expression),
        "args": _function_args(expression, aggregate_aliases, scalar_subquery_refs),
    }


def _function_name(expression: exp.Func) -> str:
    if isinstance(expression, exp.Anonymous):
        return expression.name.upper()
    return expression.sql_name().upper()


def _function_args(
    expression: exp.Func,
    aggregate_aliases: dict[int, str] | None,
    scalar_subquery_refs: dict[int, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    args: list[dict[str, Any]] = []
    for key in expression.arg_types:
        value = expression.args.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, exp.Expression):
            args.append(_expr_ast(value, aggregate_aliases, scalar_subquery_refs))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, exp.Expression):
                    args.append(_expr_ast(item, aggregate_aliases, scalar_subquery_refs))
                else:
                    args.append(
                        {
                            "type": "literal",
                            "value": item,
                            "value_type": type(item).__name__,
                        }
                    )
        else:
            args.append(
                {
                    "type": "literal",
                    "value": value,
                    "value_type": type(value).__name__,
                }
            )
    return args


def _window_ast(
    expression: exp.Window,
    aggregate_aliases: dict[int, str] | None,
    scalar_subquery_refs: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "type": "window_function",
        "function": _expr_ast(expression.this, aggregate_aliases, scalar_subquery_refs),
        "partition_by": [
            _expr_ast(item, aggregate_aliases, scalar_subquery_refs)
            for item in expression.args.get("partition_by") or []
        ],
        "order": [
            _sort_key_ast(item, aggregate_aliases or {})
            for item in (expression.args.get("order").expressions if expression.args.get("order") else [])
        ],
    }


def _sort_key_ast(
    ordered: exp.Ordered,
    aggregate_aliases: dict[int, str],
) -> dict[str, Any]:
    return {
        "expr": _expr_ast(ordered.this, aggregate_aliases),
        "direction": "DESC" if ordered.args.get("desc") else "ASC",
        "nulls": "FIRST" if ordered.args.get("nulls_first") else "LAST",
    }


def _join_param(
    nodes: list[dict[str, Any]],
    join: exp.Join,
    remote_dsl: dict[str, Any],
    relation_outputs: dict[str, str],
) -> tuple[dict[str, Any], str | None, list[str]]:
    source = join.this
    join_payload: dict[str, Any] = {
        "kind": (join.args.get("kind") or "JOIN").upper(),
    }
    dependency_ref: str | None = None
    resource_inputs: list[str] = []
    if isinstance(source, exp.Table):
        source_name = _table_name(source)
        if source_name in relation_outputs:
            dependency_ref = relation_outputs[source_name]
            join_payload["source_ref"] = dependency_ref
            join_payload["alias"] = _table_alias(source) or source_name
        else:
            join_payload["source"] = source_name
            resource_inputs.append(source_name)
            alias = _table_alias(source)
            if alias:
                join_payload["alias"] = alias
    elif isinstance(source, exp.Unnest):
        join_payload["source"] = {
            "type": "table_function",
            "function": "UNNEST",
            "args": [_expr_ast(item) for item in source.expressions],
            "alias": _table_alias(source),
            "output_column": _table_function_output_column(source),
        }
    elif isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select):
        dependency_ref = _compile_select_into(
            nodes=nodes,
            select=source.this,
            source_ids=_extract_table_references(source.this),
            remote_dsl=remote_dsl,
            format_answer=False,
            relation_outputs=relation_outputs,
            reuse_existing_scan=True,
        )
        alias = _table_alias(source)
        if alias:
            relation_outputs[alias] = dependency_ref
        join_payload["source_ref"] = dependency_ref
        if alias:
            join_payload["alias"] = alias
    else:
        join_payload["source"] = _expr_ast(source)
    if join.args.get("on") is None and join_payload["kind"] == "JOIN":
        join_payload["kind"] = "CROSS"
    if join.args.get("on") is not None:
        join_payload["on"] = _expr_ast(join.args["on"])
    return join_payload, dependency_ref, resource_inputs


def _table_function_output_column(expression: exp.Expression) -> str | None:
    alias = expression.args.get("alias")
    if alias is None:
        return None
    columns = alias.args.get("columns") or []
    if not columns:
        return None
    return columns[0].name


def _scan_params(source_ids: tuple[str, ...]) -> dict[str, Any]:
    if len(source_ids) == 1:
        return {"source": source_ids[0]}
    return {"sources": list(source_ids)}


def _find_equivalent_scan_output(
    nodes: list[dict[str, Any]],
    source_ids: tuple[str, ...],
) -> str | None:
    expected_input = list(source_ids)
    expected_params = _scan_params(source_ids)
    for node in nodes:
        if (
            node.get("op") == "Scan"
            and node.get("dependency") == []
            and node.get("input") == expected_input
            and node.get("params") == expected_params
        ):
            return node["output"]
    return None


def _collect_aggregate_calls(
    *expression_groups: list[exp.Expression],
) -> list[exp.Expression]:
    calls: list[exp.Expression] = []
    seen = set()
    for expressions in expression_groups:
        for expression in expressions:
            for call in expression.walk():
                if not _is_aggregate_expression(call):
                    continue
                if _has_window_ancestor(call):
                    continue
                if _has_subquery_ancestor_between(call, expression):
                    continue
                if _has_aggregate_ancestor(call):
                    continue
                signature = _expr_signature(call)
                if signature not in seen:
                    seen.add(signature)
                    calls.append(call)
    return calls


def _aggregate_aliases(
    aggregate_calls: list[exp.Expression],
    select_items: list[dict[str, Any]],
) -> dict[Any, str]:
    aliases: dict[Any, str] = {}
    for index, call in enumerate(aggregate_calls):
        alias = None
        for item in select_items:
            if item.get("alias") and _expr_signature(item["expr"]) == _expr_signature(call):
                alias = item["alias"]
                break
        output_alias = alias or f"_agg_{index}"
        aliases[id(call)] = output_alias
        aliases[_expr_signature(call)] = output_alias
    return aliases


def _aggregate_alias(
    expression: exp.Expression,
    aggregate_aliases: dict[Any, str],
) -> str:
    alias = _aggregate_alias_or_none(expression, aggregate_aliases)
    if alias is None:
        raise SqlParseError(
            f"Aggregate expression was not registered: {expression.sql(dialect='duckdb')}"
        )
    return alias


def _aggregate_alias_or_none(
    expression: exp.Expression,
    aggregate_aliases: dict[Any, str],
) -> str | None:
    return aggregate_aliases.get(id(expression)) or aggregate_aliases.get(
        _expr_signature(expression)
    )


def _expr_signature(expression: exp.Expression) -> str:
    return expression.sql(dialect="duckdb", normalize=True)


def _distinct_targets(
    select: exp.Select,
    aggregate_calls: list[exp.Expression],
) -> list[exp.Expression]:
    targets: list[exp.Expression] = []
    if select.args.get("distinct") is not None:
        targets.extend(item["expr"] for item in [_select_item(expr) for expr in select.expressions])
    return targets


def _pre_group_derive_items(
    group: exp.Group | None,
    select_items: list[dict[str, Any]],
    select_output_aliases: list[str],
    aggregate_aliases: dict[Any, str],
) -> list[tuple[dict[str, Any], str]]:
    if group is None or not group.expressions:
        return []

    group_aliases = {
        item.name
        for item in group.expressions
        if isinstance(item, exp.Column) and not item.table
    }
    group_signatures = {_expr_signature(item) for item in group.expressions}

    return [
        (item, alias)
        for item, alias in zip(select_items, select_output_aliases)
        if (
            item.get("alias") in group_aliases
            or _expr_signature(item["expr"]) in group_signatures
        )
        and _is_derived_select_item(item["expr"], aggregate_aliases)
    ]


def _is_aggregate_only_select_item(expression: exp.Expression) -> bool:
    return _is_aggregate_expression(expression)


def _is_derived_select_item(
    expression: exp.Expression,
    aggregate_aliases: dict[int, str],
) -> bool:
    if _is_aggregate_expression(expression):
        return False
    if id(expression) in aggregate_aliases:
        return False
    if isinstance(expression, (exp.Column, exp.Star)):
        return False
    if isinstance(expression, exp.Literal):
        return False
    return True


def _order_expressions(order: exp.Order | None) -> list[exp.Expression]:
    if order is None:
        return []
    return [item.this for item in order.expressions]


def _alias_name(expression: exp.Expression) -> str | None:
    if isinstance(expression, exp.Alias):
        return expression.alias
    return None


def _table_alias(expression: exp.Expression) -> str | None:
    alias = expression.args.get("alias")
    if alias is None:
        return None
    return alias.name


def _table_name(table: exp.Table) -> str:
    return table.name


def _contains_table_reference(expression: exp.Expression, table_name: str) -> bool:
    return any(_table_name(table) == table_name for table in expression.find_all(exp.Table))


def _external_source_ids(
    expressions: list[exp.Expression],
    allowed_source_ids: tuple[str, ...],
) -> tuple[str, ...]:
    allowed = set(allowed_source_ids)
    source_ids: list[str] = []
    for expression in expressions:
        for table in expression.find_all(exp.Table):
            source_id = _table_name(table)
            if source_id in allowed and source_id not in source_ids:
                source_ids.append(source_id)
    return tuple(source_ids)


def _parse_sql_ast(sql: str) -> exp.Expression:
    try:
        return sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.SqlglotError as exc:
        raise SqlParseError(f"Unable to parse SQL: {exc}") from exc


def _validate_read_only_select(sql: str, expression: exp.Expression) -> None:
    sql_without_strings = _mask_quoted_content(sql).lower()
    first_keyword = _first_keyword(sql_without_strings)
    if first_keyword not in {"select", "with"}:
        raise SqlParseError(f"Only SELECT statements are supported, found: {first_keyword}")
    if not isinstance(expression, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SqlParseError(f"Only SELECT statements are supported, found: {expression.key}")


def _validate_ops(nodes: list[dict[str, Any]]) -> None:
    unknown = sorted({node["op"] for node in nodes} - ALLOWED_OPS)
    if unknown:
        raise SqlParseError(f"Logic DAG contains unsupported ops: {unknown}")


def _validate_node_io(
    nodes: list[dict[str, Any]],
    source_ids: tuple[str, ...],
) -> None:
    external_inputs = set(source_ids)
    produced_outputs = set()
    for node in nodes:
        node_id = node["id"]
        dependency = node.get("dependency", [])
        node_input = node.get("input", [])

        unknown_dependencies = [
            item for item in dependency if item not in produced_outputs
        ]
        if unknown_dependencies:
            raise SqlParseError(
                f"Logic DAG node {node_id} has invalid dependencies: "
                f"{unknown_dependencies}"
            )

        non_external_inputs = [
            item for item in node_input if item not in external_inputs
        ]
        if non_external_inputs:
            raise SqlParseError(
                f"Logic DAG node {node_id} has non-external inputs: "
                f"{non_external_inputs}"
            )

        produced_outputs.add(node["output"])


def _validate_no_unexpanded_sql(payload: Any) -> None:
    if isinstance(payload, dict):
        if payload.get("type") == "sql_expr":
            raise SqlParseError(
                f"Logic DAG contains unexpanded SQL expression: {payload.get('sql')}"
            )
        for value in payload.values():
            _validate_no_unexpanded_sql(value)
    elif isinstance(payload, list):
        for item in payload:
            _validate_no_unexpanded_sql(item)


def _append_node(
    nodes: list[dict[str, Any]],
    op: str,
    input_ref: str | None,
    params: dict[str, Any],
    output: str | None = None,
    input_values: list[str] | None = None,
    dependency_refs: list[str] | None = None,
) -> str:
    if op not in ALLOWED_OPS:
        raise SqlParseError(f"Unsupported Logic DAG op: {op}")
    node_id = f"N{len(nodes)}"
    output_name = output or f"T{len(nodes)}"
    dependency = dependency_refs if dependency_refs is not None else (
        [] if input_ref is None else [input_ref]
    )
    inputs = input_values if input_values is not None else []
    nodes.append(
        {
            "id": node_id,
            "op": op,
            "dependency": dependency,
            "input": inputs,
            "params": params,
            "output": output_name,
        }
    )
    return output_name


def _dependency_edges(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    output_to_node = {node["output"]: node["id"] for node in nodes}
    edges = []
    seen = set()
    for node in nodes:
        for dependency in node.get("dependency", []):
            if dependency not in output_to_node:
                continue
            edge = {"from": output_to_node[dependency], "to": node["id"]}
            signature = (edge["from"], edge["to"])
            if signature not in seen:
                seen.add(signature)
                edges.append(edge)
    return edges


def _extract_fenced_blocks(text: str) -> list[str]:
    blocks = []
    fence_pattern = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", flags=re.DOTALL)
    for match in fence_pattern.finditer(text):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks


def _clean_single_statement(text: str) -> str:
    stripped = _strip_response_prefix(text.strip()).strip()
    if not stripped:
        raise SqlParseError("SQL candidate is empty")

    statements = _split_sql_statements(stripped)
    if len(statements) != 1:
        raise SqlParseError(f"Expected one SQL statement, found {len(statements)}")

    sql = statements[0].strip()
    if not sql:
        raise SqlParseError("SQL statement is empty")
    return sql


def _strip_response_prefix(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower() in {"sql:", "query:", "answer:"}:
        return "\n".join(lines[1:])
    return text


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if quote is None and char == "-" and next_char == "-":
            index = _consume_line_comment(sql, index)
            continue
        if quote is None and char == "/" and next_char == "*":
            index = _consume_block_comment(sql, index)
            continue

        current.append(char)
        if quote is None and char in {"'", '"', "`"}:
            quote = char
        elif quote == char:
            if next_char == char:
                current.append(next_char)
                index += 1
            else:
                quote = None
        elif quote is None and char == ";":
            statement = "".join(current[:-1]).strip()
            if statement:
                statements.append(statement)
            current = []
        index += 1

    if quote is not None:
        raise SqlParseError("SQL statement has an unterminated quoted identifier/string")

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _consume_line_comment(sql: str, start: int) -> int:
    newline_index = sql.find("\n", start)
    return len(sql) if newline_index == -1 else newline_index


def _consume_block_comment(sql: str, start: int) -> int:
    end = sql.find("*/", start + 2)
    if end == -1:
        raise SqlParseError("SQL statement has an unterminated block comment")
    return end + 2


def _first_keyword(sql: str) -> str:
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", sql)
    if not match:
        raise SqlParseError("SQL statement does not contain a keyword")
    return match.group(0).lower()


def _mask_quoted_content(sql: str) -> str:
    chars = list(sql)
    quote: str | None = None
    index = 0
    while index < len(chars):
        char = chars[index]
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        if quote is None and char in {"'", '"', "`"}:
            quote = char
        elif quote == char:
            if next_char == char:
                chars[index + 1] = " "
                index += 1
            else:
                quote = None
        elif quote is not None:
            chars[index] = " "
        index += 1
    return "".join(chars)


def _source_ids(remote_dsl: dict[str, Any]) -> tuple[str, ...]:
    source_ids = []
    for source in remote_dsl.get("sources", []):
        source_id = source.get("id")
        if isinstance(source_id, str):
            source_ids.append(source_id)
    if not source_ids:
        raise SqlParseError("Remote DSL does not define any source ids")
    return tuple(source_ids)


def _extract_table_references(expression: exp.Expression) -> tuple[str, ...]:
    local_aliases = _local_relation_aliases(expression)
    references: list[str] = []
    for table in expression.find_all(exp.Table):
        source_id = _table_name(table)
        if source_id in local_aliases:
            continue
        if source_id not in references:
            references.append(source_id)
    return tuple(references)


def _local_relation_aliases(expression: exp.Expression) -> set[str]:
    aliases: set[str] = set()
    for with_expression in expression.find_all(exp.With):
        for cte in with_expression.expressions:
            if cte.alias:
                aliases.add(cte.alias)
    for subquery in expression.find_all(exp.Subquery):
        alias = _table_alias(subquery)
        if alias:
            aliases.add(alias)
    return aliases
