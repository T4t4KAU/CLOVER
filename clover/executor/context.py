"""Execution contexts built on executor resource views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clover.executor.resources import (
    NodeResourceView,
    ResourceStore,
    normalize_resource_specs,
)


@dataclass
class ExecutionContext:
    """Shared context owned by the DAG executor."""

    task_type: str
    resource_store: ResourceStore
    external_params: dict[str, Any]
    table_cache: dict[str, Any] | None = None
    slm_config: dict[str, Any] | None = None
    slm_client: Any | None = None
    agent_loop_max_iterations: int = 3

    def node_context(self, node: dict[str, Any]) -> "NodeExecutionContext":
        # The view pins the exact dependency/source resources this node can see.
        # The existing pandas Fast Path receives plain source specs and
        # materialized dependency values derived from the resource view.
        resource_view = self.resource_store.node_view(
            node_id=str(node.get("id")),
            dependencies=list(node.get("dependency", [])),
            sources=list(node.get("input", [])),
        )
        resource_view.pin()
        return NodeExecutionContext(
            task_type=self.task_type,
            node=node,
            resource_view=resource_view,
            resources=resource_view.materialize_sources(target="resource_spec"),
            upstream_outputs=resource_view.materialize_dependencies(target="pandas"),
            external_params=self.external_params,
            table_cache=self.table_cache,
            slm_config=self.slm_config,
            slm_client=self.slm_client,
            agent_loop_max_iterations=self.agent_loop_max_iterations,
        )


@dataclass(frozen=True)
class NodeExecutionContext:
    """Read-only context passed into one NodeAgent invocation."""

    task_type: str
    node: dict[str, Any]
    resource_view: NodeResourceView
    resources: dict[str, Any]
    upstream_outputs: dict[str, Any]
    external_params: dict[str, Any]
    table_cache: dict[str, Any] | None = None
    slm_config: dict[str, Any] | None = None
    slm_client: Any | None = None
    agent_loop_max_iterations: int = 3


def normalize_resources(
    resources: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize physical-plan resources into an id-keyed mapping."""

    return normalize_resource_specs(resources)
