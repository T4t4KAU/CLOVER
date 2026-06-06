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
    slm_dispatcher: Any | None = None
    slm_node_job_slots: Any | None = None
    agent_loop_max_iterations: int = 3
    node_timeout_seconds: float | None = None

    def node_context(self, unit: Any) -> "NodeExecutionContext":
        # The view pins the exact dependency/source resources this node can see.
        # Materialization is intentionally lazy; task-specific NodeAgents choose
        # their required target formats.
        node = unit.node
        resource_view = self.resource_store.node_view(
            node_id=str(node.get("id")),
            dependencies=list(unit.dependencies),
            sources=list(unit.resources),
        )
        resource_view.pin()
        return NodeExecutionContext(
            task_type=unit.task_type,
            node=node,
            resource_view=resource_view,
            external_params=self.external_params,
            table_cache=self.table_cache,
            slm_config=self.slm_config,
            slm_client=self.slm_client,
            slm_dispatcher=self.slm_dispatcher,
            slm_node_job_slots=self.slm_node_job_slots,
            agent_loop_max_iterations=self.agent_loop_max_iterations,
            node_timeout_seconds=self.node_timeout_seconds,
        )


@dataclass
class NodeExecutionContext:
    """Read-only context passed into one NodeAgent invocation."""

    task_type: str
    node: dict[str, Any]
    resource_view: NodeResourceView
    external_params: dict[str, Any]
    table_cache: dict[str, Any] | None = None
    slm_config: dict[str, Any] | None = None
    slm_client: Any | None = None
    slm_dispatcher: Any | None = None
    slm_node_job_slots: Any | None = None
    agent_loop_max_iterations: int = 3
    node_timeout_seconds: float | None = None
    _source_cache: dict[str, dict[str, Any]] | None = None
    _dependency_cache: dict[str, dict[str, Any]] | None = None

    @property
    def resources(self) -> dict[str, Any]:
        """Default source view used by table Fast Path agents."""

        return self.materialize_sources(target="resource_spec")

    @property
    def upstream_outputs(self) -> dict[str, Any]:
        """Default dependency view used by table Fast Path agents."""

        return self.materialize_dependencies(target="pandas")

    def source_specs(self) -> dict[str, Any]:
        """Return optimizer-visible source specifications."""

        return self.materialize_sources(target="resource_spec")

    def materialize_sources(self, *, target: str) -> dict[str, Any]:
        """Materialize source resources in a task-selected target format."""

        if self._source_cache is None:
            self._source_cache = {}
        if target not in self._source_cache:
            self._source_cache[target] = self.resource_view.materialize_sources(
                target=target
            )
        return dict(self._source_cache[target])

    def materialize_dependencies(self, *, target: str) -> dict[str, Any]:
        """Materialize dependency outputs in a task-selected target format."""

        if self._dependency_cache is None:
            self._dependency_cache = {}
        if target not in self._dependency_cache:
            self._dependency_cache[target] = self.resource_view.materialize_dependencies(
                target=target
            )
        return dict(self._dependency_cache[target])

    def project_sources(self, projector: Any) -> dict[str, Any]:
        """Project source resources into a task-specific sandbox view."""

        return self.resource_view.project_sources(projector)

    def project_dependencies(self, projector: Any) -> dict[str, Any]:
        """Project dependency resources into a task-specific sandbox view."""

        return self.resource_view.project_dependencies(projector)

    def acquire_slm_node_job(self) -> None:
        """Enter the Local SLM Node Job limiter."""

        if self.slm_node_job_slots is not None:
            self.slm_node_job_slots.acquire()

    def release_slm_node_job(self) -> None:
        """Leave the Local SLM Node Job limiter."""

        if self.slm_node_job_slots is not None:
            self.slm_node_job_slots.release()


def normalize_resources(
    resources: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize physical-plan resources into an id-keyed mapping."""

    return normalize_resource_specs(resources)
