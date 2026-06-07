"""Static tool declarations for table reasoning nodes.

The tools in this module intentionally build normalized call payloads instead of
executing data operations. A later executor can map each call payload to a local
library, process boundary, or dedicated static tool implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


class StaticToolError(ValueError):
    """Raised when a Logic DAG node cannot be mapped to a static tool."""


@dataclass(frozen=True)
class TableReasoningStaticTool:
    """Base static tool declaration for a table reasoning Logic DAG op."""

    op: str
    tool_name: str
    required_params: tuple[str, ...] = ()

    def build_call(
        self,
        *,
        node: dict[str, Any],
        resources: dict[str, Any],
        upstream_outputs: dict[str, Any],
        external_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one normalized static tool call."""

        self._validate_node(node)
        params = copy.deepcopy(node.get("params", {}))
        # Tool calls are pure payloads: execution backends decide how to run
        # them, while this layer only normalizes IO and validates op shape.
        return {
            "task_type": "table_reasoning.query",
            "tool": self.tool_name,
            "op": self.op,
            "node_id": node.get("id"),
            "input": list(node.get("input", [])),
            "dependency": list(node.get("dependency", [])),
            "resources": self._resource_payload(node, resources),
            "upstream_outputs": self._upstream_payload(node, upstream_outputs),
            "params": params,
            "external_params": copy.deepcopy(external_params),
            "output": node.get("output"),
        }

    def _validate_node(self, node: dict[str, Any]) -> None:
        node_op = node.get("op")
        if node_op != self.op:
            raise StaticToolError(
                f"{self.tool_name} cannot handle op {node_op!r}; expected {self.op!r}"
            )
        params = node.get("params", {})
        missing = [name for name in self.required_params if name not in params]
        if missing:
            raise StaticToolError(
                f"Node {node.get('id')} missing params for {self.op}: {missing}"
            )

    def _resource_payload(
        self,
        node: dict[str, Any],
        resources: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            source_id: copy.deepcopy(resources[source_id])
            for source_id in node.get("input", [])
            if source_id in resources
        }

    def _upstream_payload(
        self,
        node: dict[str, Any],
        upstream_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            output_name: copy.deepcopy(upstream_outputs[output_name])
            for output_name in node.get("dependency", [])
            if output_name in upstream_outputs
        }


class ScanTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Scan", "table_reasoning.scan")


class FilterTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Filter", "table_reasoning.filter", ("predicate",))


class ProjectTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Project", "table_reasoning.project", ("expressions",))


class DeriveTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Derive", "table_reasoning.derive", ("expressions",))


class AggregateTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Aggregate", "table_reasoning.aggregate", ("aggregations",))


class GroupTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Group", "table_reasoning.group", ("keys",))


class SortTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Sort", "table_reasoning.sort", ("keys",))


class LimitTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Limit", "table_reasoning.limit", ("count",))


class DistinctTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Distinct", "table_reasoning.distinct", ("on",))


class JoinTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("Join", "table_reasoning.join", ("joins",))


class SetOpTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("SetOp", "table_reasoning.set_op", ("operator",))


class RepeatUnionTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__(
            "RepeatUnion",
            "table_reasoning.repeat_union",
            (
                "transient_table",
                "termination",
                "seed_plan",
                "recursive_plan",
            ),
        )


class FormatAnswerTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("FormatAnswer", "table_reasoning.format_answer", ("answer",))


class AnalyzeEvidenceTool(TableReasoningStaticTool):
    def __init__(self) -> None:
        super().__init__("AnalyzeEvidence", "table_reasoning.analyze_evidence", ("kind",))


TABLE_REASONING_STATIC_TOOLS = {
    tool.op: tool
    for tool in (
        ScanTool(),
        FilterTool(),
        ProjectTool(),
        DeriveTool(),
        AggregateTool(),
        GroupTool(),
        SortTool(),
        LimitTool(),
        DistinctTool(),
        JoinTool(),
        SetOpTool(),
        RepeatUnionTool(),
        FormatAnswerTool(),
        AnalyzeEvidenceTool(),
    )
}
