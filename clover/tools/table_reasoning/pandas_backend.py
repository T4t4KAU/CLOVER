"""Pandas execution backend for table reasoning static tool calls."""

from __future__ import annotations

import ast
import copy
import itertools
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from clover.task_types import TABLE_REASONING_QUERY_TASK_TYPE, is_table_task_type

from .static_tools import TABLE_REASONING_STATIC_TOOLS, StaticToolError

_TABLE_CACHE_LOCKS_GUARD = threading.Lock()
_TABLE_CACHE_LOCKS: dict[tuple[int, str], threading.Lock] = {}


class PandasExecutionError(ValueError):
    """Raised when a table reasoning plan cannot be executed with pandas."""


@dataclass
class PandasTable:
    """A pandas table plus lightweight execution metadata."""

    frame: pd.DataFrame
    group_keys: list[dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "PandasTable":
        return PandasTable(
            frame=self.frame.copy(),
            group_keys=copy.deepcopy(self.group_keys),
        )


@dataclass(frozen=True)
class _DateInterval:
    years: int
    months: int
    days: int


def execute_table_reasoning_plan(
    plan: dict[str, Any],
    *,
    resources: dict[str, Any] | list[dict[str, Any]] | None = None,
    external_params: dict[str, Any] | None = None,
    table_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Execute a table reasoning Logic DAG or physical plan with pandas."""

    executor = PandasTableReasoningExecutor(
        resources=resources if resources is not None else plan.get("resources", []),
        external_params=external_params,
        table_cache=table_cache,
    )
    return executor.execute_plan(plan)


def execute_table_reasoning_call(call: dict[str, Any]) -> Any:
    """Execute one normalized table reasoning static tool call with pandas."""

    executor = PandasTableReasoningExecutor(
        resources=call.get("resources", {}),
        external_params=call.get("external_params", {}),
    )
    return executor.execute_call(call)


class PandasTableReasoningExecutor:
    """Concrete pandas runner for table reasoning physical plans."""

    def __init__(
        self,
        *,
        resources: dict[str, Any] | list[dict[str, Any]] | None = None,
        external_params: dict[str, Any] | None = None,
        transient_tables: dict[str, PandasTable] | None = None,
        table_cache: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.resources = _normalize_resources(resources)
        self.external_params = copy.deepcopy(external_params or {})
        self.transient_tables = transient_tables or {}
        self.table_cache = table_cache
        self.outputs: dict[str, Any] = {}

    def execute_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        task_type = plan.get("task_type")
        if not is_table_task_type(task_type):
            raise PandasExecutionError(
                f"Unsupported pandas execution task_type: {task_type!r}"
            )
        if "query_plans" in plan and not plan.get("nodes"):
            return self._execute_query_plan_batch(plan)

        # Physical plans are already topologically ordered by the optimizer parser; each
        # node consumes named dependency outputs produced earlier in this loop.
        for node in plan.get("nodes", []):
            call = self._build_call(node)
            call["task_type"] = task_type
            result = self.execute_call(call)
            output_name = call.get("output")
            if not output_name:
                raise PandasExecutionError(f"Node {node.get('id')} missing output")
            self.outputs[output_name] = result
        return dict(self.outputs)

    def _execute_query_plan_batch(self, plan: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for index, query_plan in enumerate(plan.get("query_plans", [])):
            if not isinstance(query_plan, dict):
                raise PandasExecutionError(f"query_plan {index} must be an object")
            answer = query_plan.get("answer", {})
            answer_name = answer.get("name") if isinstance(answer, dict) else None
            if not isinstance(answer_name, str) or not answer_name:
                raise PandasExecutionError(f"query_plan {index} missing answer.name")
            child = PandasTableReasoningExecutor(
                resources=self.resources,
                external_params=self.external_params,
                table_cache=self.table_cache,
            )
            query_outputs = child.execute_plan(
                {
                    "task_type": TABLE_REASONING_QUERY_TASK_TYPE,
                    "nodes": copy.deepcopy(query_plan.get("nodes", [])),
                    "edges": copy.deepcopy(query_plan.get("edges", [])),
                }
            )
            output_name = (
                "answer"
                if "answer" in query_outputs
                else query_plan.get("output") or answer_name
            )
            if output_name not in query_outputs:
                nodes = query_plan.get("nodes", [])
                if nodes and isinstance(nodes[-1], dict):
                    output_name = nodes[-1].get("output")
            outputs[answer_name] = query_outputs.get(output_name)
        self.outputs.update(outputs)
        return dict(self.outputs)

    def execute_call(self, call: dict[str, Any]) -> Any:
        op = call.get("op")
        _validate_call_dependencies(call)
        handlers = {
            "Scan": self._scan,
            "Filter": self._filter,
            "Project": self._project,
            "Derive": self._derive,
            "Aggregate": self._aggregate,
            "Group": self._group,
            "Sort": self._sort,
            "Limit": self._limit,
            "Distinct": self._distinct,
            "Join": self._join,
            "SetOp": self._set_op,
            "RepeatUnion": self._repeat_union,
            "FormatAnswer": self._format_answer,
            "AnalyzeEvidence": self._analyze_evidence,
        }
        try:
            handler = handlers[op]
        except KeyError as exc:
            raise PandasExecutionError(f"Unsupported pandas op: {op!r}") from exc
        return handler(call)

    def _build_call(self, node: dict[str, Any]) -> dict[str, Any]:
        try:
            tool = TABLE_REASONING_STATIC_TOOLS[node.get("op")]
        except KeyError as exc:
            raise PandasExecutionError(
                f"Unsupported pandas op: {node.get('op')!r}"
            ) from exc
        try:
            call = tool.build_call(
                node=node,
                resources=self.resources,
                upstream_outputs={},
                external_params=self.external_params,
            )
        except StaticToolError as exc:
            raise PandasExecutionError(str(exc)) from exc
        call["upstream_outputs"] = {
            output_name: self.outputs[output_name]
            for output_name in node.get("dependency", [])
            if output_name in self.outputs
        }
        return call

    def _scan(self, call: dict[str, Any]) -> PandasTable:
        params = call.get("params", {})
        source_id = params.get("source")
        if params.get("source_type") == "transient":
            if source_id not in self.transient_tables:
                raise PandasExecutionError(f"Unknown transient table: {source_id}")
            return self.transient_tables[source_id].copy()

        if not source_id:
            sources = params.get("sources") or list(call.get("resources", {}))
            if len(sources) != 1:
                raise PandasExecutionError(
                    f"Scan expects one source for pandas execution, got {sources}"
                )
            source_id = sources[0]
        resource = call.get("resources", {}).get(source_id) or self.resources.get(source_id)
        if resource is None:
            raise PandasExecutionError(f"Scan missing resource: {source_id}")
        return PandasTable(_read_resource_frame(resource, self.table_cache))

    def _filter(self, call: dict[str, Any]) -> PandasTable:
        table = _primary_table(call)
        predicate = call["params"]["predicate"]
        evaluator = _ExpressionEvaluator(table.frame, call.get("upstream_outputs", {}))
        mask = evaluator.eval(predicate)
        mask = _series_for_frame(mask, table.frame).fillna(False).astype(bool)
        if not table.frame.empty and not bool(mask.any()):
            repaired_mask = _empty_filter_literal_repair_mask(
                table.frame,
                predicate,
                evaluator,
            )
            if repaired_mask is not None and bool(repaired_mask.any()):
                return PandasTable(table.frame.loc[repaired_mask].copy())
        return PandasTable(table.frame.loc[mask].copy())

    def _project(self, call: dict[str, Any]) -> PandasTable:
        table = _optional_primary_table(call)
        frame = table.frame if table is not None else pd.DataFrame(index=range(1))
        evaluator = _ExpressionEvaluator(frame, call.get("upstream_outputs", {}))
        columns: dict[str, Any] = {}
        for index, item in enumerate(call["params"].get("expressions", [])):
            expr = item.get("expr", item)
            if expr.get("type") == "wildcard":
                for column in frame.columns:
                    columns[column] = frame[column].reset_index(drop=True)
                continue
            alias = item.get("alias") or _expr_label(expr, f"_expr_{index}")
            value = evaluator.eval(expr)
            columns[alias] = _series_for_frame(value, frame).reset_index(drop=True)
        return PandasTable(pd.DataFrame(columns))

    def _derive(self, call: dict[str, Any]) -> PandasTable:
        table = _optional_primary_table(call)
        frame = table.frame.copy() if table is not None else pd.DataFrame(index=range(1))
        evaluator = _ExpressionEvaluator(frame, call.get("upstream_outputs", {}))
        for index, item in enumerate(call["params"].get("expressions", [])):
            expr = item.get("expr", item)
            alias = item.get("alias") or _expr_label(expr, f"_expr_{index}")
            if _is_explode_expr(expr):
                values = _series_for_frame(evaluator.eval(expr["args"][0]), frame).apply(_parse_list_value)
                frame[alias] = values
                frame = frame.explode(alias).reset_index(drop=True)
                evaluator = _ExpressionEvaluator(frame, call.get("upstream_outputs", {}))
                continue
            frame[alias] = _series_for_frame(evaluator.eval(expr), frame).to_numpy()
        return PandasTable(frame)

    def _aggregate(self, call: dict[str, Any]) -> PandasTable:
        table = _primary_table(call)
        aggregations = call["params"].get("aggregations", [])
        grouped = bool(call["params"].get("grouped"))
        if grouped:
            return PandasTable(
                _aggregate_grouped_frame(
                    table.frame,
                    table.group_keys,
                    aggregations,
                    call.get("upstream_outputs", {}),
                )
            )
        row = {
            aggregation["alias"]: _aggregate_frame(
                table.frame,
                aggregation,
                call.get("upstream_outputs", {}),
            )
            for aggregation in aggregations
        }
        return PandasTable(pd.DataFrame([row]))

    def _group(self, call: dict[str, Any]) -> PandasTable:
        table = _primary_table(call)
        return PandasTable(
            table.frame.copy(),
            group_keys=copy.deepcopy(call["params"].get("keys", [])),
        )

    def _sort(self, call: dict[str, Any]) -> PandasTable:
        table = _primary_table(call)
        frame = table.frame.copy()
        evaluator = _ExpressionEvaluator(frame, call.get("upstream_outputs", {}))
        sort_columns = []
        ascending = []
        null_positions = []
        for index, key in enumerate(call["params"].get("keys", [])):
            expr = key["expr"]
            if expr.get("type") == "column" and expr.get("name") in frame.columns:
                column = expr["name"]
            else:
                column = f"__sort_key_{index}"
                frame[column] = _series_for_frame(evaluator.eval(expr), frame).to_numpy()
            sort_columns.append(column)
            ascending.append(key.get("direction", "ASC").upper() != "DESC")
            null_positions.append(key.get("nulls", "LAST").lower())

        if not sort_columns:
            return table.copy()

        # pandas sort_values only accepts a single na_position value. When
        # multiple sort keys specify conflicting null positions, partition the
        # sort into sequential single-key passes so each key respects its own
        # null ordering.  Stable sort (mergesort) preserves earlier key order.
        unique_null_positions = set(null_positions)
        if len(unique_null_positions) <= 1:
            sorted_frame = frame.sort_values(
                by=sort_columns,
                ascending=ascending,
                na_position=null_positions[0] if null_positions else "last",
                kind="mergesort",
            )
        else:
            # Sort from the least significant key to the most significant key
            # using stable sort so that earlier passes are preserved.
            sorted_frame = frame
            for i in reversed(range(len(sort_columns))):
                sorted_frame = sorted_frame.sort_values(
                    by=sort_columns[i],
                    ascending=ascending[i],
                    na_position=null_positions[i],
                    kind="mergesort",
                )
        return PandasTable(
            sorted_frame.drop(
                columns=[column for column in sort_columns if column.startswith("__sort_key_")],
                errors="ignore",
            ).reset_index(drop=True)
        )

    def _limit(self, call: dict[str, Any]) -> PandasTable:
        table = _primary_table(call)
        count = int(call["params"].get("count") or 0)
        offset = int(call["params"].get("offset") or 0)
        return PandasTable(table.frame.iloc[offset : offset + count].copy())

    def _distinct(self, call: dict[str, Any]) -> PandasTable:
        table = _primary_table(call)
        frame = table.frame.copy()
        evaluator = _ExpressionEvaluator(frame, call.get("upstream_outputs", {}))
        subset = []
        temp_columns = []
        for index, expr in enumerate(call["params"].get("on", [])):
            if expr.get("type") == "column" and expr.get("name") in frame.columns:
                subset.append(expr["name"])
                continue
            temp_column = f"__distinct_key_{index}"
            frame[temp_column] = _series_for_frame(evaluator.eval(expr), frame).to_numpy()
            subset.append(temp_column)
            temp_columns.append(temp_column)
        distinct = frame.drop_duplicates(subset=subset or None).drop(
            columns=temp_columns,
            errors="ignore",
        )
        return PandasTable(distinct.reset_index(drop=True))

    def _join(self, call: dict[str, Any]) -> PandasTable:
        left = _primary_table(call).frame.copy()
        upstream = call.get("upstream_outputs", {})
        for join in call["params"].get("joins", []):
            source = join.get("source")
            if isinstance(source, dict) and source.get("type") == "table_function":
                left = _join_table_function(left, source)
                continue

            if join.get("source_ref"):
                right = _as_table(upstream[join["source_ref"]]).frame.copy()
            elif isinstance(source, str):
                resource = call.get("resources", {}).get(source) or self.resources.get(source)
                if resource is None:
                    raise PandasExecutionError(f"Join missing resource: {source}")
                right = _read_resource_frame(resource, self.table_cache)
            else:
                raise PandasExecutionError(f"Unsupported join source: {source!r}")

            how = _join_how(join.get("kind"))
            right_alias = join.get("alias")
            if how == "cross" and join.get("on") is None:
                left = left.merge(
                    right,
                    how="cross",
                    suffixes=("", f"__{right_alias}" if right_alias else "__right"),
                )
                continue

            suffixes = ("", f"__{right_alias}" if right_alias else "__right")
            left_keys, right_keys, residual_predicate = _join_keys(join.get("on"), right_alias)
            if left_keys:
                left, right = _coerce_join_key_dtypes(left, right, left_keys, right_keys)
                # Pandas merge handles equality keys. Any remaining non-equality
                # conjunct is evaluated as a residual filter after the merge.
                left = left.merge(
                    right,
                    left_on=left_keys,
                    right_on=right_keys,
                    how=how,
                    suffixes=suffixes,
                )
            else:
                if how not in {"inner", "cross"}:
                    raise PandasExecutionError(
                        "Non-equality joins without equality keys only support inner joins"
                    )
                left = left.merge(
                    right,
                    how="cross",
                    suffixes=suffixes,
                )
            if residual_predicate is not None:
                evaluator = _ExpressionEvaluator(left, call.get("upstream_outputs", {}))
                mask = _series_for_frame(evaluator.eval(residual_predicate), left).fillna(False).astype(bool)
                left = left.loc[mask].copy()
        return PandasTable(left.reset_index(drop=True))

    def _set_op(self, call: dict[str, Any]) -> PandasTable:
        tables = [_as_table(value).frame for value in call.get("upstream_outputs", {}).values()]
        if not tables:
            raise PandasExecutionError("SetOp requires at least one table dependency")
        operator = call["params"].get("operator", "UNION").upper()
        if len(tables) == 1:
            return PandasTable(tables[0].copy())
        left, right = tables[0], tables[1]
        if operator == "UNION":
            result = pd.concat([left, right], ignore_index=True).drop_duplicates()
        elif operator == "UNION ALL":
            result = pd.concat([left, right], ignore_index=True)
        elif operator == "INTERSECT":
            result = left.merge(right.drop_duplicates())
        elif operator == "EXCEPT":
            merged = left.merge(right.drop_duplicates(), how="left", indicator=True)
            result = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        else:
            raise PandasExecutionError(f"Unsupported SetOp operator: {operator}")
        return PandasTable(result.reset_index(drop=True))

    def _repeat_union(self, call: dict[str, Any]) -> PandasTable:
        params = call["params"]
        transient_name = params["transient_table"]
        seed_table = self._execute_subplan(params["seed_plan"])
        accumulated = seed_table.frame.reset_index(drop=True)
        delta = accumulated.copy()

        iteration_limit = int(params.get("iteration_limit", -1))
        max_iterations = int(self.external_params.get("max_iterations", 100))
        if iteration_limit >= 0:
            max_iterations = min(max_iterations, iteration_limit)

        iteration = 0
        # RepeatUnion executes recursive CTE semantics with semi-naive deltas:
        # run the step plan on the latest delta, keep only new rows, and stop at
        # a fixpoint or configured safety limit.
        while not delta.empty and iteration < max_iterations:
            iteration += 1
            step_table = self._execute_subplan(
                params["recursive_plan"],
                transient_tables={transient_name: PandasTable(delta)},
            )
            step = step_table.frame.reset_index(drop=True)
            if step.empty:
                break
            new_delta = _anti_join_all_columns(step, accumulated)
            if new_delta.empty:
                break
            accumulated = pd.concat([accumulated, new_delta], ignore_index=True)
            delta = new_delta

        return PandasTable(accumulated.reset_index(drop=True))

    def _format_answer(self, call: dict[str, Any]) -> Any:
        table = _primary_table(call)
        frame = table.frame
        answer = call["params"].get("answer", {})
        answer_name = answer.get("name")
        answer_type = str(answer.get("type", "json")).lower()

        if answer_type == "table":
            return {
                "columns": [str(column) for column in frame.columns],
                "rows": [
                    {
                        str(column): _to_python_scalar(value)
                        for column, value in row.items()
                    }
                    for row in frame.to_dict(orient="records")
                ],
            }
        if frame.empty:
            return [] if answer_type.startswith("list") else None
        if answer_name and answer_name in frame.columns:
            series = frame[answer_name]
        else:
            series = frame.iloc[:, -1]

        if answer_type.startswith("list"):
            values = [_to_python_scalar(value) for value in series.tolist()]
            if len(values) == 1 and isinstance(values[0], list):
                return values[0]
            return values
        value = _to_python_scalar(series.iloc[0])
        if answer_type == "boolean":
            return _to_bool(value)
        if answer_type in {"number", "integer", "int", "float"}:
            return _to_number(value)
        if answer_type in {"string", "category"}:
            return None if value is None else str(value)
        return value

    def _analyze_evidence(self, call: dict[str, Any]) -> dict[str, Any]:
        table = _primary_table(call)
        frame = table.frame.copy()
        kind = str(call["params"].get("kind") or "").strip().lower()
        if kind in {"stat", "stats", "statistics"}:
            kind = "statistical"
        if kind not in {"statistical", "correlation"}:
            raise PandasExecutionError(f"Unsupported AnalyzeEvidence kind: {kind!r}")
        if kind == "statistical":
            return _statistical_evidence(frame)
        return _correlation_evidence(frame)

    def _execute_subplan(
        self,
        subplan: dict[str, Any],
        *,
        transient_tables: dict[str, PandasTable] | None = None,
    ) -> PandasTable:
        executor = PandasTableReasoningExecutor(
            resources=self.resources,
            external_params=self.external_params,
            transient_tables=transient_tables or self.transient_tables,
            table_cache=self.table_cache,
        )
        outputs = executor.execute_plan(
            {
                "task_type": TABLE_REASONING_QUERY_TASK_TYPE,
                "nodes": subplan.get("nodes", []),
                "edges": subplan.get("edges", []),
            }
        )
        output_name = subplan.get("output")
        if output_name not in outputs:
            raise PandasExecutionError(f"RepeatUnion subplan missing output {output_name}")
        return _as_table(outputs[output_name])


class _ExpressionEvaluator:
    """Evaluate the small structured expression AST emitted by the SQL parser."""

    def __init__(self, frame: pd.DataFrame, upstream_outputs: dict[str, Any]) -> None:
        self.frame = frame
        self.upstream_outputs = upstream_outputs

    def eval(self, expr: dict[str, Any] | None) -> Any:
        if expr is None:
            return None
        expr_type = expr.get("type")
        if expr_type == "column":
            return _column_series(self.frame, expr)
        if expr_type == "wildcard":
            return self.frame
        if expr_type == "literal":
            return expr.get("value")
        if expr_type == "null":
            return None
        if expr_type == "identifier":
            name = expr.get("name")
            if name in self.frame.columns:
                return self.frame[name]
            return name
        if expr_type == "placeholder":
            return None
        if expr_type == "scalar_ref":
            return _scalar_ref_value(self.upstream_outputs, expr)
        if expr_type == "set_ref":
            return _set_ref_values(self.upstream_outputs, expr)
        if expr_type == "binary_op":
            return self._binary_op(expr)
        if expr_type == "logical_op":
            return self._logical_op(expr)
        if expr_type == "not":
            return ~_series_for_frame(self.eval(expr["expr"]), self.frame).fillna(False).astype(bool)
        if expr_type == "is_null":
            return _series_for_frame(self.eval(expr["expr"]), self.frame).isna()
        if expr_type == "is_not_null":
            return _series_for_frame(self.eval(expr["expr"]), self.frame).notna()
        if expr_type == "in":
            values = [_to_python_scalar(self.eval(value)) for value in expr.get("values", [])]
            return _series_for_frame(self.eval(expr["expr"]), self.frame).isin(values)
        if expr_type == "in_subquery":
            values = self.eval(expr["query"])
            return _series_for_frame(self.eval(expr["expr"]), self.frame).isin(_as_list(values))
        if expr_type == "like":
            return self._like(expr)
        if expr_type == "cast":
            return _cast_value(self.eval(expr["expr"]), expr.get("to", ""))
        if expr_type == "tuple":
            return tuple(self.eval(item) for item in expr.get("items", []))
        if expr_type == "case":
            return self._case(expr)
        if expr_type == "function_call":
            return self._function_call(expr)
        if expr_type == "aggregate_call":
            aggregation = {
                "function": expr["function"],
                "argument": expr["argument"],
                "distinct": expr.get("distinct", False),
                "alias": "_value",
            }
            for key in ("filter", "order", "limit", "parameters"):
                if key in expr:
                    aggregation[key] = expr[key]
            return _aggregate_frame(
                self.frame,
                aggregation,
                self.upstream_outputs,
            )
        if expr_type == "filtered_expr":
            mask = _series_for_frame(self.eval(expr["filter"]), self.frame).fillna(False).astype(bool)
            value = self.eval(expr["expr"])
            return _series_for_frame(value, self.frame).where(mask)
        if expr_type == "window_function":
            return self._window_function(expr)
        raise PandasExecutionError(f"Unsupported expression type: {expr_type!r}")

    def _binary_op(self, expr: dict[str, Any]) -> Any:
        left = self.eval(expr["left"])
        right = self.eval(expr["right"])
        op = expr["op"]

        if op in {"=", "!=", ">", ">=", "<", "<="}:
            if op in {"=", "!="}:
                set_result = _compare_against_collection(left, right, op, self.frame)
                if set_result is not None:
                    return set_result
            left, right = _coerce_comparison_values(left, right)
            if op == "=":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            return left <= right

        if op == "+":
            left, right = _coerce_arithmetic_values(left, right)
            return left + right
        if op == "-":
            left, right = _coerce_arithmetic_values(left, right)
            return left - right
        if op == "*":
            left, right = _coerce_arithmetic_values(left, right)
            return left * right
        if op == "/":
            left, right = _coerce_arithmetic_values(left, right)
            return left / right
        if op == "%":
            left, right = _coerce_arithmetic_values(left, right)
            return left % right
        raise PandasExecutionError(f"Unsupported binary operator: {op}")

    def _logical_op(self, expr: dict[str, Any]) -> pd.Series:
        operands = [
            _series_for_frame(self.eval(item), self.frame).fillna(False).astype(bool)
            for item in expr.get("operands", [])
        ]
        if not operands:
            return pd.Series([True] * len(self.frame), index=self.frame.index)
        result = operands[0]
        if expr.get("op") == "AND":
            for operand in operands[1:]:
                result = result & operand
            return result
        if expr.get("op") == "OR":
            for operand in operands[1:]:
                result = result | operand
            return result
        raise PandasExecutionError(f"Unsupported logical operator: {expr.get('op')}")

    def _like(self, expr: dict[str, Any]) -> pd.Series:
        values = _series_for_frame(self.eval(expr["expr"]), self.frame).astype("string")
        pattern = _to_python_scalar(self.eval(expr["pattern"]))
        regex = _sql_like_to_regex(str(pattern))
        flags = 0 if expr.get("case_sensitive", True) else re.IGNORECASE
        return values.str.contains(regex, regex=True, flags=flags, na=False)

    def _case(self, expr: dict[str, Any]) -> pd.Series:
        default = self.eval(expr.get("default")) if expr.get("default") is not None else None
        result = _series_for_frame(default, self.frame)
        for branch in reversed(expr.get("ifs", [])):
            mask = _series_for_frame(self.eval(branch["condition"]), self.frame).fillna(False).astype(bool)
            value = _series_for_frame(self.eval(branch["value"]), self.frame)
            result = result.where(~mask, value)
        return result

    def _function_call(self, expr: dict[str, Any]) -> Any:
        function = expr.get("function", "").upper()
        args = [self.eval(arg) for arg in expr.get("args", [])]

        if function in {"LOWER", "UPPER", "TRIM", "LTRIM", "RTRIM"}:
            series = _series_for_frame(args[0], self.frame).astype("string")
            if function == "LOWER":
                return series.str.lower()
            if function == "UPPER":
                return series.str.upper()
            if function == "LTRIM":
                return series.str.lstrip()
            if function == "RTRIM":
                return series.str.rstrip()
            return series.str.strip()
        if function in {"LENGTH", "LEN"}:
            return _series_for_frame(args[0], self.frame).astype("string").str.len()
        if function in {"STR_POSITION", "STRPOS", "INSTR", "LOCATE", "POSITION"}:
            # SQL STR_POSITION(needle, haystack) returns 1-based index of the first
            # occurrence of needle in haystack, or 0 if not found. Either argument
            # may be a column (Series) or a literal; broadcast as needed.
            needle = args[0]
            haystack = args[1] if len(args) > 1 else args[0]
            haystack_series = _series_for_frame(haystack, self.frame).astype("string")
            needle_scalar = _to_python_scalar(needle)
            found = haystack_series.str.find(str(needle_scalar))
            return found + 1
        if function in {"ABS", "ROUND", "CEIL", "CEILING", "FLOOR"}:
            value = args[0]
            if function == "ABS":
                return value.abs() if isinstance(value, pd.Series) else abs(value)
            if function == "ROUND":
                digits = int(_to_python_scalar(args[1])) if len(args) > 1 else 0
                return value.round(digits) if isinstance(value, pd.Series) else round(value, digits)
            if function in {"CEIL", "CEILING"}:
                return value.apply(math.ceil) if isinstance(value, pd.Series) else math.ceil(value)
            return value.apply(math.floor) if isinstance(value, pd.Series) else math.floor(value)
        if function in {"COALESCE", "IFNULL"}:
            return _coalesce(args, self.frame)
        if function == "NULLIF":
            if len(args) < 2:
                raise PandasExecutionError("NULLIF requires two arguments")
            left = _series_for_frame(args[0], self.frame)
            right = _series_for_frame(args[1], self.frame)
            left, right = _coerce_comparison_values(left, right)
            return left.mask(left == right)
        if function == "CONCAT":
            result = _series_for_frame("", self.frame).astype("string")
            for arg in args:
                result = result + _series_for_frame(arg, self.frame).fillna("").astype("string")
            return result
        if function in {"SUBSTRING", "SUBSTR"}:
            series = _series_for_frame(args[0], self.frame).astype("string")
            start = int(_to_python_scalar(args[1])) - 1 if len(args) > 1 else 0
            if len(args) > 2 and args[2] is not None:
                length = int(_to_python_scalar(args[2]))
                return series.str.slice(start, start + length)
            return series.str.slice(start)
        if function == "LEFT":
            series = _series_for_frame(args[0], self.frame).astype("string")
            return series.str.slice(0, int(_to_python_scalar(args[1])))
        if function == "RIGHT":
            series = _series_for_frame(args[0], self.frame).astype("string")
            return series.str.slice(-int(_to_python_scalar(args[1])))
        if function in {"REPLACE", "REGEXP_REPLACE"}:
            series = _series_for_frame(args[0], self.frame).astype("string")
            pattern = str(_to_python_scalar(args[1])) if len(args) > 1 else ""
            replacement = str(_to_python_scalar(args[2])) if len(args) > 2 else ""
            return series.str.replace(
                pattern,
                replacement,
                regex=function == "REGEXP_REPLACE",
            )
        if function == "SPLIT_PART":
            values = _series_for_frame(args[0], self.frame)
            separator = str(_to_python_scalar(args[1])) if len(args) > 1 else ","
            part_index = int(_to_python_scalar(args[2])) if len(args) > 2 else 1
            return values.astype("string").str.split(separator).str[part_index - 1]
        if function in {"SPLIT", "STRING_TO_ARRAY", "REGEXP_SPLIT_TO_ARRAY"}:
            values = _series_for_frame(args[0], self.frame)
            separator = str(_to_python_scalar(args[1])) if len(args) > 1 else ","
            if function == "REGEXP_SPLIT_TO_ARRAY":
                return values.astype("string").str.split(separator, regex=True)
            return values.apply(lambda value: _split_value(value, separator))
        if function in {"ARRAY_SIZE", "ARRAY_LENGTH", "CARDINALITY"}:
            values = _series_for_frame(args[0], self.frame)
            return values.apply(_list_length_value)
        if function == "EXPLODE":
            values = _series_for_frame(args[0], self.frame)
            return values.apply(_parse_list_value)
        if function in {"DATE_PART", "DATEPART"}:
            date_part_args = list(args)
            if date_part_args and str(_first_scalar(date_part_args[0])).upper() in {"DATE_PART", "DATEPART"}:
                date_part_args = date_part_args[1:]
            if len(date_part_args) < 2:
                raise PandasExecutionError(f"{function} requires part and value")
            part = str(_to_python_scalar(date_part_args[0])).lower()
            source = date_part_args[1]
            if _contains_date_interval(source):
                return _extract_interval_part(source, part, self.frame)
            values = pd.to_datetime(_series_for_frame(source, self.frame), errors="coerce")
            return _extract_datetime_part(values, part)
        if function in {"YEAR", "MONTH", "DAY", "QUARTER"}:
            values = pd.to_datetime(_series_for_frame(args[0], self.frame), errors="coerce")
            return _extract_datetime_part(values, function.lower())
        if function == "EXTRACT":
            part = str(_to_python_scalar(args[0])).lower()
            if len(args) > 1 and _contains_date_interval(args[1]):
                return _extract_interval_part(args[1], part, self.frame)
            values = pd.to_datetime(_series_for_frame(args[1], self.frame), errors="coerce")
            return _extract_datetime_part(values, part)
        if function == "AGE":
            age_args = list(args)
            if age_args and str(_first_scalar(age_args[0])).upper() == "AGE":
                age_args = age_args[1:]
            if not age_args:
                raise PandasExecutionError("AGE requires at least one timestamp")
            end = age_args[0]
            start = age_args[1] if len(age_args) > 1 else pd.Timestamp.now().normalize()
            return _age_interval(end, start, self.frame)
        if function in {"STR_TO_TIME", "STRPTIME"}:
            values = _series_for_frame(args[0], self.frame)
            fmt = _to_python_scalar(args[1]) if len(args) > 1 else None
            return pd.to_datetime(values, format=fmt, errors="coerce")
        if function == "DATE":
            value = args[0] if args else None
            if isinstance(value, pd.Series):
                return pd.to_datetime(value, errors="coerce").dt.date
            parsed = pd.to_datetime(value, errors="coerce")
            return None if pd.isna(parsed) else parsed.date()
        if function == "CURRENT_DATE":
            return pd.Timestamp.now().normalize().date()
        if function in {"BOOL_OR", "LOGICAL_OR", "LOGICALOR"}:
            values = _series_for_frame(args[0], self.frame).dropna()
            if values.empty:
                return False
            return bool(values.map(_to_bool).fillna(False).any())
        if function in {"EVERY", "BOOL_AND", "LOGICAL_AND", "LOGICALAND"}:
            values = _series_for_frame(args[0], self.frame).dropna()
            if values.empty:
                return True
            return bool(values.map(_to_bool).fillna(False).all())
        raise PandasExecutionError(f"Unsupported function: {function}")

    def _window_function(self, expr: dict[str, Any]) -> pd.Series:
        function = expr.get("function", {})
        function_name = str(function.get("function", "")).upper()
        if function_name in {"ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE"}:
            return _evaluate_rank_window(self.frame, self.upstream_outputs, expr, function_name)
        if function_name in {"LEAD", "LAG"}:
            return _evaluate_offset_window(self.frame, self.upstream_outputs, expr, function_name)
        if function_name in {"FIRST_VALUE", "LAST_VALUE", "NTH_VALUE"}:
            return _evaluate_value_window(self.frame, self.upstream_outputs, expr, function_name)
        raise PandasExecutionError("Unsupported window function")


def _normalize_resources(
    resources: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if resources is None:
        return {}
    if isinstance(resources, dict):
        return dict(resources)
    return {resource["id"]: resource for resource in resources}


def _read_resource_frame(
    resource: dict[str, Any],
    table_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    path = Path(resource["path"])
    cache_key = str(path.resolve())
    if table_cache is None:
        return _read_frame_from_path(path, resource)

    lock = _table_cache_lock(table_cache, cache_key)
    with lock:
        cached = table_cache.get(cache_key)
        if cached is not None:
            return cached
        frame = _read_frame_from_path(path, resource)
        table_cache[cache_key] = frame
        return frame


def _read_frame_from_path(path: Path, resource: dict[str, Any]) -> pd.DataFrame:
    fmt = (resource.get("format") or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        return _clean_resource_frame(pd.read_csv(path, low_memory=False))
    elif fmt in {"parquet", "pq"}:
        return _clean_resource_frame(pd.read_parquet(path))
    elif fmt in {"json", "jsonl"}:
        return _clean_resource_frame(pd.read_json(path, lines=fmt == "jsonl"))
    raise PandasExecutionError(f"Unsupported table resource format: {fmt}")


def _clean_resource_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    text = frame.astype("string")
    marker_mask = text.apply(
        lambda column: column.str.contains(
            r"^\s*align\s*=\s*[^|]*\|",
            case=False,
            na=False,
            regex=True,
        )
    ).any(axis=1)
    if marker_mask.any():
        frame = frame.loc[~marker_mask].reset_index(drop=True)
    return frame


def _table_cache_lock(
    table_cache: dict[str, pd.DataFrame],
    cache_key: str,
) -> threading.Lock:
    lock_key = (id(table_cache), cache_key)
    with _TABLE_CACHE_LOCKS_GUARD:
        lock = _TABLE_CACHE_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _TABLE_CACHE_LOCKS[lock_key] = lock
        return lock


def _primary_table(call: dict[str, Any]) -> PandasTable:
    table = _optional_primary_table(call)
    if table is None:
        raise PandasExecutionError(f"{call.get('op')} requires a table dependency")
    return table


def _optional_primary_table(call: dict[str, Any]) -> PandasTable | None:
    upstream = call.get("upstream_outputs", {})
    for dependency in call.get("dependency", []):
        if dependency in upstream:
            return _as_table(upstream[dependency])
    return None


def _as_table(value: Any) -> PandasTable:
    if isinstance(value, PandasTable):
        return value
    if isinstance(value, pd.DataFrame):
        return PandasTable(value)
    raise PandasExecutionError(f"Expected pandas table output, got {type(value).__name__}")


def _statistical_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = _analysis_numeric_columns(frame)
    metrics = []
    for column, numeric in numeric_columns:
        values = numeric.dropna()
        if values.empty:
            continue
        item: dict[str, Any] = {
            "col": str(column),
            "n": int(len(values)),
            "mean": _analysis_scalar(values.mean()),
            "median": _analysis_scalar(values.median()),
            "min": _analysis_scalar(values.min()),
            "max": _analysis_scalar(values.max()),
        }
        if len(values) > 1:
            item["std"] = _analysis_scalar(values.std())
        metrics.append(item)
    evidence: dict[str, Any] = {
        "kind": "statistical",
        "rows": int(len(frame)),
        "cols": [str(column) for column in frame.columns[:40]],
        "metrics": metrics[:40],
    }
    extrema = _statistical_extrema(frame, numeric_columns)
    if extrema:
        evidence["extrema"] = extrema
    if not metrics:
        evidence["notes"] = ["no numeric columns"]
    return evidence


def _correlation_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = _analysis_numeric_columns(frame)
    metrics = []
    for left_index, (left_column, left) in enumerate(numeric_columns):
        for right_column, right in numeric_columns[left_index + 1 :]:
            pair = pd.DataFrame({"x": left, "y": right}).dropna()
            if len(pair) < 2:
                continue
            pearson = pair["x"].corr(pair["y"], method="pearson")
            spearman = pair["x"].rank().corr(pair["y"].rank(), method="pearson")
            if pd.isna(pearson) and pd.isna(spearman):
                continue
            metrics.append(
                {
                    "x": str(left_column),
                    "y": str(right_column),
                    "n": int(len(pair)),
                    "pearson": _analysis_scalar(pearson),
                    "spearman": _analysis_scalar(spearman),
                }
            )
    metrics.sort(
        key=lambda item: abs(float(item.get("pearson") or item.get("spearman") or 0.0)),
        reverse=True,
    )
    evidence: dict[str, Any] = {
        "kind": "correlation",
        "rows": int(len(frame)),
        "cols": [str(column) for column in frame.columns[:40]],
        "metrics": metrics[:20],
    }
    if len(numeric_columns) < 2:
        evidence["notes"] = ["fewer than two numeric columns"]
    elif not metrics:
        evidence["notes"] = ["no valid numeric pairs"]
    return evidence


def _analysis_numeric_columns(frame: pd.DataFrame) -> list[tuple[Any, pd.Series]]:
    columns: list[tuple[Any, pd.Series]] = []
    for column in frame.columns:
        numeric = _coerce_numeric_series(frame[column])
        if numeric.notna().any():
            columns.append((column, numeric))
    return columns


def _statistical_extrema(
    frame: pd.DataFrame,
    numeric_columns: list[tuple[Any, pd.Series]],
) -> dict[str, Any]:
    extrema: dict[str, Any] = {}
    column_std = _std_extrema(
        [
            {
                "col": str(column),
                "n": int(values.dropna().shape[0]),
                "std": _analysis_scalar(values.dropna().std()),
            }
            for column, values in numeric_columns
            if values.dropna().shape[0] > 1
        ]
    )
    if column_std:
        extrema["col_std"] = column_std

    row_std = _row_std_extrema(frame, numeric_columns)
    if row_std:
        extrema["row_std"] = row_std
    return extrema


def _row_std_extrema(
    frame: pd.DataFrame,
    numeric_columns: list[tuple[Any, pd.Series]],
) -> dict[str, list[dict[str, Any]]] | None:
    if len(numeric_columns) < 2 or frame.empty:
        return None
    numeric_frame = pd.DataFrame(
        {str(column): values for column, values in numeric_columns},
        index=frame.index,
    )
    counts = numeric_frame.notna().sum(axis=1)
    std_values = numeric_frame.std(axis=1, skipna=True)
    numeric_names = {column for column, _ in numeric_columns}
    label_col = next((column for column in frame.columns if column not in numeric_names), None)
    items = []
    for position, (row_index, std_value) in enumerate(std_values.items()):
        if counts.loc[row_index] <= 1:
            continue
        std = _analysis_scalar(std_value)
        if std is None:
            continue
        item: dict[str, Any] = {
            "row": int(position),
            "n": int(counts.loc[row_index]),
            "std": std,
        }
        if label_col is not None:
            item["label_col"] = str(label_col)
            item["label"] = _analysis_scalar(frame.iloc[position][label_col])
        items.append(item)
    return _std_extrema(items)


def _std_extrema(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]] | None:
    clean = [
        item
        for item in items
        if isinstance(item.get("std"), (int, float))
        and math.isfinite(float(item["std"]))
    ]
    if not clean:
        return None
    ordered = sorted(clean, key=lambda item: float(item["std"]))
    return {
        "min": ordered[:3],
        "max": list(reversed(ordered[-3:])),
    }


def _analysis_scalar(value: Any) -> Any:
    scalar = _to_python_scalar(value)
    if isinstance(scalar, float):
        if not math.isfinite(scalar):
            return None
        rounded = round(scalar, 12)
        return int(rounded) if rounded.is_integer() else rounded
    return scalar


def _empty_filter_literal_repair_mask(
    frame: pd.DataFrame,
    predicate: dict[str, Any],
    evaluator: _ExpressionEvaluator,
) -> pd.Series | None:
    options = _filter_mask_options(frame, predicate, evaluator)
    for mask, repairs in options:
        if repairs and bool(mask.any()):
            return mask
    return None


def _filter_mask_options(
    frame: pd.DataFrame,
    expr: dict[str, Any],
    evaluator: _ExpressionEvaluator,
) -> list[tuple[pd.Series, list[dict[str, Any]]]]:
    original = _bool_mask_for_filter(evaluator.eval(expr), frame)
    options: list[tuple[pd.Series, list[dict[str, Any]]]] = [(original, [])]
    expr_type = expr.get("type")
    if expr_type == "logical_op":
        operands = [
            item
            for item in expr.get("operands", [])
            if isinstance(item, dict)
        ]
        if not operands:
            return options
        operand_options = [
            _filter_mask_options(frame, operand, evaluator)
            for operand in operands
        ]
        for combo in itertools.islice(
            itertools.product(*operand_options),
            32,
        ):
            repairs = [
                repair
                for _, item_repairs in combo
                for repair in item_repairs
            ]
            if not repairs:
                continue
            masks = [mask for mask, _ in combo]
            combined = masks[0].copy()
            if expr.get("op") == "AND":
                for mask in masks[1:]:
                    combined = combined & mask
            elif expr.get("op") == "OR":
                for mask in masks[1:]:
                    combined = combined | mask
            else:
                continue
            options.append((combined, repairs))
        return options[:32]

    repaired = _literal_filter_leaf_repair_mask(frame, expr)
    if repaired is not None:
        mask, repair = repaired
        if not mask.equals(original):
            options.append((mask, [repair]))
    return options


def _bool_mask_for_filter(value: Any, frame: pd.DataFrame) -> pd.Series:
    return _series_for_frame(value, frame).fillna(False).astype(bool)


def _literal_filter_leaf_repair_mask(
    frame: pd.DataFrame,
    expr: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]] | None:
    expr_type = expr.get("type")
    if expr_type == "binary_op" and expr.get("op") == "=":
        pair = _column_literal_pair(expr.get("left"), expr.get("right"))
        if pair is None:
            return None
        column, literal = pair
        mask = _normalized_literal_equality_mask(frame, column, [literal])
        if mask is None:
            return None
        return mask, {"op": "=", "column": column, "value": literal}
    if expr_type == "in":
        target = expr.get("expr")
        if not isinstance(target, dict) or target.get("type") != "column":
            return None
        column = target.get("name")
        if column not in frame.columns:
            return None
        literals = []
        for item in expr.get("values", []):
            if not isinstance(item, dict) or item.get("type") != "literal":
                return None
            value = item.get("value")
            if not isinstance(value, str):
                return None
            literals.append(value)
        if not literals:
            return None
        mask = _normalized_literal_equality_mask(frame, str(column), literals)
        if mask is None:
            return None
        return mask, {"op": "IN", "column": str(column), "values": literals}
    if expr_type == "like":
        target = expr.get("expr")
        pattern = expr.get("pattern")
        if not isinstance(target, dict) or target.get("type") != "column":
            return None
        if not isinstance(pattern, dict) or pattern.get("type") != "literal":
            return None
        column = target.get("name")
        literal = pattern.get("value")
        if column not in frame.columns or not isinstance(literal, str):
            return None
        regex = _sql_like_to_regex(literal)
        values = frame[str(column)].astype("string")
        mask = values.str.contains(regex, regex=True, flags=re.IGNORECASE, na=False)
        if not bool(mask.any()):
            return None
        return mask, {"op": "LIKE", "column": str(column), "value": literal}
    return None


def _column_literal_pair(left: Any, right: Any) -> tuple[str, str] | None:
    if (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("type") == "column"
        and right.get("type") == "literal"
        and isinstance(right.get("value"), str)
        and isinstance(left.get("name"), str)
    ):
        return left["name"], right["value"]
    if (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("type") == "literal"
        and right.get("type") == "column"
        and isinstance(left.get("value"), str)
        and isinstance(right.get("name"), str)
    ):
        return right["name"], left["value"]
    return None


def _normalized_literal_equality_mask(
    frame: pd.DataFrame,
    column: str,
    literals: list[str],
) -> pd.Series | None:
    if column not in frame.columns:
        return None
    normalized_targets = [_normalize_filter_text(value) for value in literals]
    if any(not value for value in normalized_targets):
        return None
    series = frame[column]
    normalized_cells = series.map(_normalize_filter_text)
    for target in normalized_targets:
        matched_originals = {
            str(original)
            for original, normalized in zip(series.tolist(), normalized_cells.tolist(), strict=False)
            if normalized == target
        }
        if len(matched_originals) != 1:
            return None
    mask = normalized_cells.isin(normalized_targets)
    return mask.fillna(False).astype(bool)


def _normalize_filter_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    return re.sub(r"\s+", " ", text)


def _validate_call_dependencies(call: dict[str, Any]) -> None:
    upstream = call.get("upstream_outputs", {})
    missing = [
        dependency
        for dependency in call.get("dependency", [])
        if dependency not in upstream
    ]
    if missing:
        raise PandasExecutionError(
            f"Node {call.get('node_id')} has unsatisfied dependencies: {missing}"
        )


def _column_series(frame: pd.DataFrame, expr: dict[str, Any]) -> pd.Series:
    name = expr.get("name")
    table = expr.get("table")
    if table and f"{name}__{table}" in frame.columns:
        return frame[f"{name}__{table}"]
    if name in frame.columns:
        return frame[name]

    lower_matches = [
        column for column in frame.columns if str(column).lower() == str(name).lower()
    ]
    if len(lower_matches) == 1:
        return frame[lower_matches[0]]

    normalized_name = _normalize_column_name(name)
    normalized_matches = [
        column
        for column in frame.columns
        if _normalize_column_name(column) == normalized_name
    ]
    if len(normalized_matches) == 1:
        return frame[normalized_matches[0]]
    raise PandasExecutionError(f"Unknown column: {name}")


def _normalize_column_name(name: Any) -> str:
    base = _column_name_base(name)
    return re.sub(r"[^a-z0-9]+", "", base.lower())


def _column_name_base(name: Any) -> str:
    base = str(name).split("<gx:", 1)[0]
    return (
        base
        .replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\\t", " ")
    )


def _series_for_frame(value: Any, frame: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(frame.index)
    if isinstance(value, pd.DataFrame):
        if len(value.columns) != 1:
            raise PandasExecutionError("Cannot coerce multi-column frame to Series")
        return value.iloc[:, 0].reindex(frame.index)
    return pd.Series([value] * len(frame), index=frame.index)


def _is_explode_expr(expr: dict[str, Any]) -> bool:
    return (
        expr.get("type") == "function_call"
        and str(expr.get("function", "")).upper() == "EXPLODE"
        and bool(expr.get("args"))
    )


def _expr_label(expr: dict[str, Any], fallback: str) -> str:
    if expr.get("type") == "column":
        return expr.get("name") or fallback
    if expr.get("type") == "function_call":
        return expr.get("function", fallback).lower()
    return fallback


def _scalar_ref_value(upstream_outputs: dict[str, Any], expr: dict[str, Any]) -> Any:
    table = _as_table(upstream_outputs[expr["source"]]).frame
    column = expr.get("name")
    if column not in table.columns:
        column = table.columns[0]
    if table.empty:
        return None
    return _to_python_scalar(table[column].iloc[0])


def _set_ref_values(upstream_outputs: dict[str, Any], expr: dict[str, Any]) -> list[Any]:
    table = _as_table(upstream_outputs[expr["source"]]).frame
    column = expr.get("name")
    if column not in table.columns:
        column = table.columns[0]
    return [_to_python_scalar(value) for value in table[column].tolist()]


def _compare_against_collection(
    left: Any,
    right: Any,
    op: str,
    frame: pd.DataFrame,
) -> pd.Series | None:
    if isinstance(left, pd.Series) and _is_collection_for_membership(right):
        result = left.isin(_as_list(right))
        return result if op == "=" else ~result
    if isinstance(right, pd.Series) and _is_collection_for_membership(left):
        result = right.isin(_as_list(left))
        return result if op == "=" else ~result
    return None


def _is_collection_for_membership(value: Any) -> bool:
    if isinstance(value, (pd.Series, pd.DataFrame, str, bytes, dict)):
        return False
    return isinstance(value, (list, tuple, set))


def _coerce_comparison_values(left: Any, right: Any) -> tuple[Any, Any]:
    left = _unwrap_singleton_collection(left)
    right = _unwrap_singleton_collection(right)
    if isinstance(left, pd.Series) and _is_number_like(right):
        numeric_left = _coerce_numeric_series(left)
        if numeric_left.notna().any():
            left = numeric_left
    if isinstance(right, pd.Series) and _is_number_like(left):
        numeric_right = _coerce_numeric_series(right)
        if numeric_right.notna().any():
            right = numeric_right
    if isinstance(left, pd.Series):
        right = _coerce_scalar_for_series(left, right)
    if isinstance(right, pd.Series):
        left = _coerce_scalar_for_series(right, left)
    return left, right


def _coerce_arithmetic_values(left: Any, right: Any) -> tuple[Any, Any]:
    left = _unwrap_singleton_collection(left)
    right = _unwrap_singleton_collection(right)
    if isinstance(left, pd.Series):
        left = _coerce_numeric_series(left)
    if isinstance(right, pd.Series):
        right = _coerce_numeric_series(right)
    if isinstance(left, str):
        left = _to_number(left)
    if isinstance(right, str):
        right = _to_number(right)
    return left, right


def _unwrap_singleton_collection(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)) and len(value) == 1:
        return next(iter(value))
    return value


def _is_number_like(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_scalar_for_series(series: pd.Series, value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value
    if pd.api.types.is_bool_dtype(series.dtype) and isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    if pd.api.types.is_numeric_dtype(series.dtype) and isinstance(value, str):
        numeric = _coerce_numeric_series(pd.Series([value])).iloc[0]
        if not pd.isna(numeric):
            return numeric
    return value


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    # DataBench CSV values often carry formatting such as commas, percents, or
    # currency prefixes; strip those before numeric comparison/aggregation.
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"([+-])\s+(?=\d|\.)", r"\1", regex=True)
        .str.replace(r"^[^\d+\-.]+", "", regex=True)
    )
    extracted = cleaned.str.extract(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce")
    if numeric.notna().any():
        return numeric
    bool_numeric = _coerce_bool_numeric_series(series)
    if bool_numeric.notna().any():
        return bool_numeric
    return numeric


def _numeric_series_if_convertible(series: pd.Series) -> pd.Series | None:
    numeric = _coerce_numeric_series(series)
    non_null = series.dropna()
    if non_null.empty:
        return numeric
    non_empty = non_null.astype("string").str.strip() != ""
    required = int(non_empty.sum())
    if required == 0:
        return numeric
    if int(numeric.reindex(non_null.index).notna().sum()) == required:
        return numeric
    return None


def _datetime_series_if_convertible(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return pd.to_datetime(series, errors="coerce")

    non_null = series.dropna()
    if non_null.empty:
        return None
    text = non_null.astype("string").str.strip()
    non_empty = text != ""
    required = int(non_empty.sum())
    if required == 0:
        return None

    candidate_text = text[non_empty]
    date_like = candidate_text.str.contains(
        r"(?:^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?$)|(?:^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$)",
        regex=True,
    )
    if not bool(date_like.all()):
        return None

    parsed = pd.to_datetime(series, errors="coerce")
    if int(parsed.reindex(non_null.index).notna().sum()) == required:
        return parsed
    return None


def _coerce_bool_numeric_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": 1,
        "t": 1,
        "yes": 1,
        "y": 1,
        "false": 0,
        "f": 0,
        "no": 0,
        "n": 0,
    }
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.map(mapping).astype("float64")


def _cast_value(value: Any, target_type: str) -> Any:
    target = target_type.lower()
    if "int" in target:
        if isinstance(value, pd.Series):
            numeric = _coerce_numeric_series(value)
            truncated = numeric.map(
                lambda item: math.trunc(item) if not pd.isna(item) else pd.NA
            )
            return truncated.astype("Int64")
        number = _to_number(value)
        return None if number is None else int(number)
    if any(name in target for name in ("double", "float", "real", "decimal", "numeric")):
        if isinstance(value, pd.Series):
            return _coerce_numeric_series(value)
        number = _to_number(value)
        return None if number is None else float(number)
    if any(name in target for name in ("varchar", "text", "string")):
        return value.astype("string") if isinstance(value, pd.Series) else str(value)
    if "bool" in target:
        return value.astype(bool) if isinstance(value, pd.Series) else bool(value)
    if "date" in target or "time" in target:
        return pd.to_datetime(value, errors="coerce")
    return value


def _coalesce(values: list[Any], frame: pd.DataFrame) -> pd.Series:
    if not values:
        return _series_for_frame(None, frame)
    result = _series_for_frame(values[0], frame)
    for value in values[1:]:
        result = result.combine_first(_series_for_frame(value, frame))
    return result


def _age_interval(end: Any, start: Any, frame: pd.DataFrame) -> Any:
    if isinstance(end, pd.Series) or isinstance(start, pd.Series):
        end_series = _series_for_frame(end, frame)
        start_series = _series_for_frame(start, frame)
        return pd.Series(
            [
                _date_interval_between(end_value, start_value)
                for end_value, start_value in zip(end_series.tolist(), start_series.tolist())
            ],
            index=frame.index,
            dtype="object",
        )
    return _date_interval_between(end, start)


def _date_interval_between(end: Any, start: Any) -> _DateInterval | None:
    end_ts = pd.to_datetime(_first_scalar(end), errors="coerce")
    start_ts = pd.to_datetime(_first_scalar(start), errors="coerce")
    if pd.isna(end_ts) or pd.isna(start_ts):
        return None

    sign = 1
    if end_ts < start_ts:
        end_ts, start_ts = start_ts, end_ts
        sign = -1

    total_months = (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month)
    if end_ts.day < start_ts.day:
        total_months -= 1
    years = total_months // 12
    months = total_months % 12

    anchor = start_ts + pd.DateOffset(months=total_months)
    days = int((end_ts.normalize() - anchor.normalize()).days)
    return _DateInterval(sign * years, sign * months, sign * days)


def _contains_date_interval(value: Any) -> bool:
    if isinstance(value, _DateInterval):
        return True
    if isinstance(value, pd.Series):
        return any(isinstance(item, _DateInterval) for item in value.dropna().tolist())
    return False


def _extract_interval_part(value: Any, part: str, frame: pd.DataFrame) -> Any:
    def extract_one(item: Any) -> int | None:
        if item is None:
            return None
        if not isinstance(item, _DateInterval):
            raise PandasExecutionError("EXTRACT interval source must be AGE interval")
        if part == "year":
            return item.years
        if part == "month":
            return item.months
        if part == "day":
            return item.days
        raise PandasExecutionError(f"Unsupported interval extract part: {part}")

    if isinstance(value, pd.Series):
        return pd.Series([extract_one(item) for item in value.tolist()], index=frame.index)
    return extract_one(value)


def _extract_datetime_part(values: pd.Series, part: str) -> pd.Series:
    normalized = part.lower()
    if normalized == "year":
        return values.dt.year
    if normalized == "month":
        return values.dt.month
    if normalized == "day":
        return values.dt.day
    if normalized == "quarter":
        return values.dt.quarter
    if normalized in {"dow", "dayofweek"}:
        return values.dt.dayofweek
    if normalized in {"doy", "dayofyear"}:
        return values.dt.dayofyear
    raise PandasExecutionError(f"Unsupported date extract part: {part}")


def _first_scalar(value: Any) -> Any:
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return _to_python_scalar(value.iloc[0])
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        return _to_python_scalar(value.iloc[0, 0])
    return _to_python_scalar(value)


def _split_value(value: Any, separator: str) -> list[Any]:
    parsed = _parse_list_value(value)
    if len(parsed) != 1 or not isinstance(parsed[0], str):
        return parsed
    text = parsed[0]
    if not separator:
        return list(text)
    return [part.strip() for part in text.split(separator) if part.strip()]


def _list_length_value(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return len(_parse_list_value(value))


def _evaluate_rank_window(
    frame: pd.DataFrame,
    upstream_outputs: dict[str, Any],
    expr: dict[str, Any],
    function_name: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64", index=frame.index)

    working = pd.DataFrame(index=frame.index)
    evaluator = _ExpressionEvaluator(frame, upstream_outputs)
    partition_columns = []
    for index, partition_expr in enumerate(expr.get("partition_by", [])):
        column = f"__partition_{index}"
        working[column] = _series_for_frame(evaluator.eval(partition_expr), frame)
        partition_columns.append(column)

    sort_columns = []
    ascending = []
    null_position = "last"
    for index, order_item in enumerate(expr.get("order", [])):
        column = f"__order_{index}"
        working[column] = _series_for_frame(evaluator.eval(order_item["expr"]), frame)
        sort_columns.append(column)
        ascending.append(order_item.get("direction", "ASC").upper() != "DESC")
        null_position = order_item.get("nulls", "LAST").lower()

    result = pd.Series(index=frame.index, dtype="float64")
    if partition_columns:
        partitions = working.groupby(partition_columns, dropna=False, sort=False).indices.values()
    else:
        partitions = [frame.index]

    for partition_index in partitions:
        partition_working = working.loc[partition_index].copy()
        if sort_columns:
            partition_working = partition_working.sort_values(
                by=sort_columns,
                ascending=ascending,
                na_position=null_position,
                kind="mergesort",
            )
        row_numbers = list(range(1, len(partition_working) + 1))
        if function_name == "ROW_NUMBER":
            result.loc[partition_working.index] = row_numbers
            continue

        if function_name == "NTILE":
            function = expr.get("function", {})
            buckets = _to_number(
                _ExpressionEvaluator(frame, upstream_outputs).eval(function.get("argument"))
            )
            buckets = max(1, int(buckets or 1))
            size = len(partition_working)
            result.loc[partition_working.index] = [
                min(buckets, ((position - 1) * buckets) // size + 1)
                for position in row_numbers
            ]
            continue

        if function_name == "CUME_DIST":
            if not sort_columns:
                result.loc[partition_working.index] = 1.0
                continue
            counts: dict[tuple[Any, ...], int] = {}
            for _, row in partition_working.iterrows():
                key = tuple(_to_python_scalar(row[column]) for column in sort_columns)
                counts[key] = counts.get(key, 0) + 1
            cumulative = 0
            key_to_cume: dict[tuple[Any, ...], float] = {}
            for _, row in partition_working.iterrows():
                key = tuple(_to_python_scalar(row[column]) for column in sort_columns)
                if key in key_to_cume:
                    continue
                cumulative += counts[key]
                key_to_cume[key] = cumulative / len(partition_working)
            result.loc[partition_working.index] = [
                key_to_cume[tuple(_to_python_scalar(row[column]) for column in sort_columns)]
                for _, row in partition_working.iterrows()
            ]
            continue

        if function_name == "DENSE_RANK":
            dense_ranks = []
            previous_key: tuple[Any, ...] | None = None
            current_rank = 0
            for _, row in partition_working.iterrows():
                key = tuple(_to_python_scalar(row[column]) for column in sort_columns)
                if previous_key is None or key != previous_key:
                    current_rank += 1
                    previous_key = key
                dense_ranks.append(current_rank)
            result.loc[partition_working.index] = dense_ranks
            continue

        ranks = _peer_min_ranks(partition_working, sort_columns)
        if function_name == "RANK":
            result.loc[partition_working.index] = ranks
            continue

        if function_name == "PERCENT_RANK":
            if len(partition_working) <= 1:
                result.loc[partition_working.index] = 0.0
                continue
            result.loc[partition_working.index] = range(1, len(partition_working) + 1)
            result.loc[partition_working.index] = [
                (rank - 1) / (len(partition_working) - 1) for rank in ranks
            ]
    return result


def _evaluate_offset_window(
    frame: pd.DataFrame,
    upstream_outputs: dict[str, Any],
    expr: dict[str, Any],
    function_name: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object", index=frame.index)

    function = expr.get("function", {})
    evaluator = _ExpressionEvaluator(frame, upstream_outputs)
    values = _series_for_frame(evaluator.eval(function.get("argument")), frame)
    parameters = function.get("parameters") or []
    offset = 1
    default: Any = None
    if parameters:
        parsed_offset = _to_number(evaluator.eval(parameters[0]))
        offset = 1 if parsed_offset is None else int(parsed_offset)
    if len(parameters) > 1:
        default = _to_python_scalar(evaluator.eval(parameters[1]))

    working = pd.DataFrame({"__value": values}, index=frame.index)
    partition_columns = []
    for index, partition_expr in enumerate(expr.get("partition_by", [])):
        column = f"__partition_{index}"
        working[column] = _series_for_frame(evaluator.eval(partition_expr), frame)
        partition_columns.append(column)

    sort_columns = []
    ascending = []
    null_position = "last"
    for index, order_item in enumerate(expr.get("order", [])):
        column = f"__order_{index}"
        working[column] = _series_for_frame(evaluator.eval(order_item["expr"]), frame)
        sort_columns.append(column)
        ascending.append(order_item.get("direction", "ASC").upper() != "DESC")
        null_position = order_item.get("nulls", "LAST").lower()

    result = pd.Series(index=frame.index, dtype="object")
    if partition_columns:
        partitions = working.groupby(partition_columns, dropna=False, sort=False).indices.values()
    else:
        partitions = [frame.index]

    shift = -offset if function_name == "LEAD" else offset
    for partition_index in partitions:
        partition_working = working.loc[partition_index].copy()
        if sort_columns:
            partition_working = partition_working.sort_values(
                by=sort_columns,
                ascending=ascending,
                na_position=null_position,
                kind="mergesort",
            )
        shifted = partition_working["__value"].shift(shift)
        if default is not None:
            shifted = shifted.fillna(default)
        result.loc[partition_working.index] = shifted
    return result


def _evaluate_value_window(
    frame: pd.DataFrame,
    upstream_outputs: dict[str, Any],
    expr: dict[str, Any],
    function_name: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object", index=frame.index)

    function = expr.get("function", {})
    evaluator = _ExpressionEvaluator(frame, upstream_outputs)
    values = _series_for_frame(evaluator.eval(function.get("argument")), frame)
    parameters = function.get("parameters") or []
    nth = 1
    if function_name == "NTH_VALUE":
        nth_value = _to_number(evaluator.eval(parameters[0])) if parameters else 1
        nth = max(1, int(nth_value or 1))

    working = pd.DataFrame({"__value": values}, index=frame.index)
    partition_columns = []
    for index, partition_expr in enumerate(expr.get("partition_by", [])):
        column = f"__partition_{index}"
        working[column] = _series_for_frame(evaluator.eval(partition_expr), frame)
        partition_columns.append(column)

    sort_columns = []
    ascending = []
    null_position = "last"
    for index, order_item in enumerate(expr.get("order", [])):
        column = f"__order_{index}"
        working[column] = _series_for_frame(evaluator.eval(order_item["expr"]), frame)
        sort_columns.append(column)
        ascending.append(order_item.get("direction", "ASC").upper() != "DESC")
        null_position = order_item.get("nulls", "LAST").lower()

    result = pd.Series(index=frame.index, dtype="object")
    if partition_columns:
        partitions = working.groupby(partition_columns, dropna=False, sort=False).indices.values()
    else:
        partitions = [frame.index]

    for partition_index in partitions:
        partition_working = working.loc[partition_index].copy()
        if sort_columns:
            partition_working = partition_working.sort_values(
                by=sort_columns,
                ascending=ascending,
                na_position=null_position,
                kind="mergesort",
            )
        if function_name == "FIRST_VALUE":
            value = partition_working["__value"].iloc[0] if not partition_working.empty else None
        elif function_name == "LAST_VALUE":
            value = partition_working["__value"].iloc[-1] if not partition_working.empty else None
        else:
            value = (
                partition_working["__value"].iloc[nth - 1]
                if len(partition_working) >= nth
                else None
            )
        result.loc[partition_working.index] = value
    return result


def _peer_min_ranks(frame: pd.DataFrame, sort_columns: list[str]) -> list[int]:
    if not sort_columns:
        return list(range(1, len(frame) + 1))
    ranks: list[int] = []
    previous_key: tuple[Any, ...] | None = None
    current_rank = 1
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        key = tuple(_to_python_scalar(row[column]) for column in sort_columns)
        if previous_key is None or key != previous_key:
            current_rank = position
            previous_key = key
        ranks.append(current_rank)
    return ranks


def _sql_like_to_regex(pattern: str) -> str:
    regex = ""
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "%":
            regex += ".*"
        elif char == "_":
            regex += "."
        else:
            regex += re.escape(char)
        index += 1
    return f"^{regex}$"


def _aggregate_grouped_frame(
    frame: pd.DataFrame,
    group_keys: list[dict[str, Any]],
    aggregations: list[dict[str, Any]],
    upstream_outputs: dict[str, Any],
) -> pd.DataFrame:
    if not group_keys:
        raise PandasExecutionError("Grouped Aggregate missing Group keys")

    working = frame.copy()
    evaluator = _ExpressionEvaluator(working, upstream_outputs)
    key_columns = []
    output_key_columns = []
    for index, key in enumerate(group_keys):
        if key.get("type") == "column" and key.get("name") in working.columns:
            key_column = key["name"]
            output_column = key["name"]
        else:
            key_column = f"__group_key_{index}"
            output_column = _expr_label(key, key_column)
            working[key_column] = _series_for_frame(evaluator.eval(key), working).to_numpy()
        key_columns.append(key_column)
        output_key_columns.append(output_column)

    result_columns = output_key_columns + [
        aggregation["alias"] for aggregation in aggregations
    ]
    if working.empty:
        return pd.DataFrame(columns=result_columns)

    rows = []
    grouped = working.groupby(key_columns, dropna=False, sort=False)
    for key_values, group_frame in grouped:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {
            output_column: _to_python_scalar(value)
            for output_column, value in zip(output_key_columns, key_values)
        }
        for aggregation in aggregations:
            row[aggregation["alias"]] = _aggregate_frame(
                group_frame,
                aggregation,
                upstream_outputs,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_frame(
    frame: pd.DataFrame,
    aggregation: dict[str, Any],
    upstream_outputs: dict[str, Any],
) -> Any:
    evaluator = _ExpressionEvaluator(frame, upstream_outputs)
    filtered_frame = frame
    if aggregation.get("filter") is not None:
        mask = _series_for_frame(evaluator.eval(aggregation["filter"]), frame).fillna(False).astype(bool)
        filtered_frame = frame.loc[mask]
        evaluator = _ExpressionEvaluator(filtered_frame, upstream_outputs)

    if aggregation.get("order"):
        filtered_frame = _sort_frame_for_aggregation(
            filtered_frame,
            aggregation["order"],
            upstream_outputs,
        )
        evaluator = _ExpressionEvaluator(filtered_frame, upstream_outputs)

    if aggregation.get("limit") is not None:
        filtered_frame = filtered_frame.iloc[: int(aggregation["limit"])].copy()
        evaluator = _ExpressionEvaluator(filtered_frame, upstream_outputs)

    function = aggregation.get("function", "").upper()
    argument = aggregation.get("argument", {"type": "wildcard"})
    if argument.get("type") == "wildcard":
        values = pd.Series([1] * len(filtered_frame), index=filtered_frame.index)
        wildcard = True
    else:
        values = _series_for_frame(evaluator.eval(argument), filtered_frame)
        wildcard = False

    if function in {"REGR_SLOPE", "REGR_INTERCEPT"}:
        return _regression_aggregate(
            function,
            y_values=values,
            aggregation=aggregation,
            frame=filtered_frame,
            upstream_outputs=upstream_outputs,
        )

    if aggregation.get("distinct"):
        values = values.drop_duplicates()

    if function == "COUNT":
        return int(len(filtered_frame) if wildcard else values.count())
    if function in {"SUM", "TOTAL"}:
        return _to_python_scalar(_coerce_numeric_series(values).sum())
    if function in {"AVG", "MEAN"}:
        return _to_python_scalar(_coerce_numeric_series(values).mean())
    if function == "MIN":
        datetime_values = _datetime_series_if_convertible(values)
        if datetime_values is not None:
            return _to_python_scalar(datetime_values.min())
        numeric_values = _numeric_series_if_convertible(values)
        if numeric_values is not None:
            return _to_python_scalar(numeric_values.min())
        try:
            return _to_python_scalar(values.min())
        except TypeError:
            numeric_values = _coerce_numeric_series(values)
            if numeric_values.notna().any():
                return _to_python_scalar(numeric_values.min())
            raise
    if function == "MAX":
        datetime_values = _datetime_series_if_convertible(values)
        if datetime_values is not None:
            return _to_python_scalar(datetime_values.max())
        numeric_values = _numeric_series_if_convertible(values)
        if numeric_values is not None:
            return _to_python_scalar(numeric_values.max())
        try:
            return _to_python_scalar(values.max())
        except TypeError:
            numeric_values = _coerce_numeric_series(values)
            if numeric_values.notna().any():
                return _to_python_scalar(numeric_values.max())
            raise
    if function == "MEDIAN":
        return _to_python_scalar(_coerce_numeric_series(values).median())
    if function in {"PERCENTILE_CONT", "PERCENTILE_DISC"}:
        numeric_values = _coerce_numeric_series(values).dropna()
        if numeric_values.empty:
            return None
        percentile = _aggregation_percentile(aggregation, frame, upstream_outputs)
        if function == "PERCENTILE_DISC":
            sorted_values = numeric_values.sort_values(kind="mergesort").reset_index(drop=True)
            position = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(sorted_values)) - 1))
            return _to_python_scalar(sorted_values.iloc[position])
        return _to_python_scalar(numeric_values.quantile(percentile, interpolation="linear"))
    if function in {"STDDEV", "STDDEV_SAMP", "STDEV"}:
        return _to_python_scalar(_coerce_numeric_series(values).std(ddof=1))
    if function in {"STDDEV_POP", "VAR_POP"}:
        numeric_values = _coerce_numeric_series(values)
        if function == "STDDEV_POP":
            return _to_python_scalar(numeric_values.std(ddof=0))
        return _to_python_scalar(numeric_values.var(ddof=0))
    if function in {"VARIANCE", "VAR_SAMP"}:
        return _to_python_scalar(_coerce_numeric_series(values).var(ddof=1))
    if function in {"ARRAY_AGG", "LIST"}:
        return [_to_python_scalar(value) for value in values.tolist()]
    if function in {"STRING_AGG", "GROUP_CONCAT"}:
        return ",".join(str(value) for value in values.dropna().tolist())
    if function in {"BOOL_OR", "LOGICAL_OR", "LOGICALOR"}:
        clean_values = values.dropna()
        return False if clean_values.empty else bool(clean_values.map(_to_bool).fillna(False).any())
    if function in {"EVERY", "BOOL_AND", "LOGICAL_AND", "LOGICALAND"}:
        clean_values = values.dropna()
        return True if clean_values.empty else bool(clean_values.map(_to_bool).fillna(False).all())
    raise PandasExecutionError(f"Unsupported aggregation function: {function}")


def _regression_aggregate(
    function: str,
    *,
    y_values: pd.Series,
    aggregation: dict[str, Any],
    frame: pd.DataFrame,
    upstream_outputs: dict[str, Any],
) -> Any:
    parameters = aggregation.get("parameters") or []
    if not parameters:
        raise PandasExecutionError(f"{function} requires x and y arguments")
    if aggregation.get("distinct"):
        raise PandasExecutionError(f"{function} does not support DISTINCT")

    evaluator = _ExpressionEvaluator(frame, upstream_outputs)
    x_values = _series_for_frame(evaluator.eval(parameters[0]), frame)
    xy = pd.DataFrame(
        {
            "x": _coerce_numeric_series(x_values),
            "y": _coerce_numeric_series(y_values),
        }
    ).dropna()
    if len(xy) < 2:
        return None

    x = xy["x"]
    y = xy["y"]
    x_delta = x - x.mean()
    denominator = (x_delta * x_delta).sum()
    if denominator == 0 or pd.isna(denominator):
        return None

    slope = ((x_delta) * (y - y.mean())).sum() / denominator
    if function == "REGR_SLOPE":
        return _to_python_scalar(slope)
    return _to_python_scalar(y.mean() - slope * x.mean())


def _sort_frame_for_aggregation(
    frame: pd.DataFrame,
    order: list[dict[str, Any]],
    upstream_outputs: dict[str, Any],
) -> pd.DataFrame:
    if not order or frame.empty:
        return frame

    working = frame.copy()
    evaluator = _ExpressionEvaluator(working, upstream_outputs)
    sort_columns = []
    ascending = []
    null_positions = []
    for index, key in enumerate(order):
        expr = key["expr"]
        if expr.get("type") == "column" and expr.get("name") in working.columns:
            column = expr["name"]
        else:
            column = f"__agg_sort_key_{index}"
            working[column] = _series_for_frame(evaluator.eval(expr), working).to_numpy()
        sort_columns.append(column)
        ascending.append(key.get("direction", "ASC").upper() != "DESC")
        null_positions.append(key.get("nulls", "LAST").lower())

    sorted_frame = working.sort_values(
        by=sort_columns,
        ascending=ascending,
        na_position=null_positions[0] if null_positions else "last",
        kind="mergesort",
    )
    return sorted_frame.drop(
        columns=[column for column in sort_columns if column.startswith("__agg_sort_key_")],
        errors="ignore",
    )


def _aggregation_percentile(
    aggregation: dict[str, Any],
    frame: pd.DataFrame,
    upstream_outputs: dict[str, Any],
) -> float:
    parameters = aggregation.get("parameters") or []
    percentile = 0.5
    if parameters:
        percentile = _to_number(_ExpressionEvaluator(frame, upstream_outputs).eval(parameters[0]))
        if percentile is None:
            percentile = 0.5
    return max(0.0, min(1.0, float(percentile)))


def _join_table_function(left: pd.DataFrame, source: dict[str, Any]) -> pd.DataFrame:
    function = source.get("function")
    if function != "UNNEST":
        raise PandasExecutionError(f"Unsupported table function: {function}")
    alias = source.get("output_column") or source.get("alias") or "unnest"
    args = source.get("args", [])
    if not args:
        raise PandasExecutionError("UNNEST requires one argument")
    evaluator = _ExpressionEvaluator(left, {})
    values = _series_for_frame(evaluator.eval(args[0]), left).apply(_parse_list_value)
    frame = left.copy()
    frame[alias] = values
    return frame.explode(alias).reset_index(drop=True)


def _parse_list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    try:
        if bool(is_missing):
            return []
    except (TypeError, ValueError):
        # pd.isna can return an array-like value for nested list inputs.
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
        if stripped.startswith("{") and stripped.endswith("}"):
            stripped = stripped[1:-1]
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [value]


def _join_keys(
    on_expr: dict[str, Any] | None,
    right_alias: str | None,
) -> tuple[list[str], list[str], dict[str, Any] | None]:
    if not on_expr:
        return [], [], None
    predicates = (
        on_expr.get("operands", [])
        if on_expr.get("type") == "logical_op" and on_expr.get("op") == "AND"
        else [on_expr]
    )
    left_keys = []
    right_keys = []
    residual_predicates = []
    for predicate in predicates:
        if predicate.get("type") == "binary_op" and predicate.get("op") == "=":
            left_key, right_key = _single_join_key(predicate, right_alias)
            left_keys.append(left_key)
            right_keys.append(right_key)
        else:
            residual_predicates.append(predicate)
    residual = None
    if len(residual_predicates) == 1:
        residual = residual_predicates[0]
    elif residual_predicates:
        residual = {"type": "logical_op", "op": "AND", "operands": residual_predicates}
    return left_keys, right_keys, residual


def _single_join_key(on_expr: dict[str, Any], right_alias: str | None) -> tuple[str, str]:
    left_expr = on_expr["left"]
    right_expr = on_expr["right"]
    if right_alias and left_expr.get("table") == right_alias:
        left_expr, right_expr = right_expr, left_expr
    return left_expr["name"], right_expr["name"]


def _coerce_join_key_dtypes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for left_key, right_key in zip(left_keys, right_keys):
        left_values = _coerce_numeric_series(left[left_key])
        right_values = _coerce_numeric_series(right[right_key])
        if left_values.notna().any() and right_values.notna().any():
            left = left.copy()
            right = right.copy()
            left[left_key] = left_values
            right[right_key] = right_values
            continue

        if left[left_key].dtype != right[right_key].dtype:
            left = left.copy()
            right = right.copy()
            left[left_key] = left[left_key].astype("string")
            right[right_key] = right[right_key].astype("string")
    return left, right


def _join_how(kind: str | None) -> str:
    normalized = (kind or "JOIN").upper()
    if normalized in {"JOIN", "INNER"}:
        return "inner"
    if normalized == "LEFT":
        return "left"
    if normalized == "RIGHT":
        return "right"
    if normalized in {"FULL", "FULL OUTER"}:
        return "outer"
    if normalized == "CROSS":
        return "cross"
    raise PandasExecutionError(f"Unsupported join kind: {kind}")


def _anti_join_all_columns(candidates: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return candidates.drop_duplicates().reset_index(drop=True)
    common_columns = [column for column in candidates.columns if column in existing.columns]
    if not common_columns:
        return candidates.drop_duplicates().reset_index(drop=True)
    merged = candidates.merge(
        existing[common_columns].drop_duplicates(),
        on=common_columns,
        how="left",
        indicator=True,
    )
    return (
        merged[merged["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, pd.Series):
        return [_to_python_scalar(item) for item in value.tolist()]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    try:
        if bool(is_missing):
            return None
    except (TypeError, ValueError):
        # Array-like missingness checks are ambiguous; keep the original value.
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _to_number(value: Any) -> int | float | None:
    if value is None:
        return None
    numeric = _coerce_numeric_series(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return None
    python_value = _to_python_scalar(numeric)
    if isinstance(python_value, float) and python_value.is_integer():
        return int(python_value)
    return python_value
