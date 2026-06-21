"""Runtime scheduling for physical-plan NodeAgent execution."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from clover.executor.errors import PlanValidationError
from clover.executor.prompt_prefix import document_worker_prefix_metadata


class ResourceState(Protocol):
    """Resource availability view required by the scheduler."""

    def has_artifact(self, name: str) -> bool:
        """Return whether an intermediate artifact is available."""

    def has_source(self, name: str) -> bool:
        """Return whether an external source is available."""


@dataclass(frozen=True)
class ExecutionUnit:
    """One ready-checkable unit passed to a task-specific NodeAgent."""

    id: str
    task_type: str
    op: str
    node: dict[str, Any]
    dependencies: tuple[str, ...]
    resources: tuple[str, ...]
    output: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectorSpec:
    """Static post-execution collector over node outputs."""

    id: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    """Executor-ready units plus static collection metadata."""

    units: tuple[ExecutionUnit, ...]
    resources: tuple[dict[str, Any], ...] = ()
    collectors: tuple[CollectorSpec, ...] = ()
    retained_artifacts: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


class PlanNodeBuilder(Protocol):
    """Build executable node units from one physical-plan representation."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        start_index: int,
    ) -> "PlanNodeBuildResult":
        """Return execution units produced from this plan representation."""


class PlanResourceBuilder(Protocol):
    """Normalize executor-visible resources before node/collector construction."""

    def build(self, physical_plan: dict[str, Any]) -> dict[str, Any]:
        """Return a plan whose resources are ready for execution."""


class PlanCollectorBuilder(Protocol):
    """Build static collectors over execution unit outputs."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        units: tuple[ExecutionUnit, ...],
    ) -> "PlanCollectorBuildResult":
        """Return collector specs produced from this plan representation."""


@dataclass(frozen=True)
class PlanNodeBuildResult:
    """Units created by one PlanNodeBuilder."""

    units: tuple[ExecutionUnit, ...] = ()
    retained_artifacts: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PlanCollectorBuildResult:
    """Collectors and retained node outputs produced by one collector builder."""

    collectors: tuple[CollectorSpec, ...] = ()
    retained_artifacts: frozenset[str] = frozenset()


class PreparedResourceBuilder:
    """Executor-side no-op resource builder.

    Resource-heavy work is done by clover.resource before Executor submission.
    This builder keeps the PlanBuilder shape explicit for all task types.
    """

    def build(self, physical_plan: dict[str, Any]) -> dict[str, Any]:
        return physical_plan


@dataclass
class ExecutionPlanBuilder:
    """Create a task-neutral execution plan from a physical plan."""

    resource_builders: list[PlanResourceBuilder] = field(default_factory=list)
    node_builders: list[PlanNodeBuilder] = field(default_factory=list)
    collector_builders: list[PlanCollectorBuilder] = field(default_factory=list)

    @classmethod
    def default(cls) -> "ExecutionPlanBuilder":
        return cls(
            resource_builders=[PreparedResourceBuilder()],
            node_builders=[
                PhysicalNodeBuilder(),
                MapGroupNodeBuilder(),
            ],
            collector_builders=[
                StaticCollectorBuilder(),
                MapGroupCollectorBuilder(),
                QueryOutputsCollectorBuilder(),
                TableAnswerCollectorBuilder(),
                FinalAnswerCollectorBuilder(),
            ],
        )

    def build(self, physical_plan: dict[str, Any]) -> ExecutionPlan:
        plan = physical_plan
        for builder in self.resource_builders:
            plan = builder.build(plan)

        # Build in layers: normalize resources, expose scheduler-visible units,
        # then attach collectors that define compact observations/final answers.
        units: list[ExecutionUnit] = []
        retained_artifacts: set[str] = set()
        for builder in self.node_builders:
            result = builder.build(plan, start_index=len(units))
            units.extend(result.units)
            retained_artifacts.update(result.retained_artifacts)

        _validate_execution_units(units)
        retained_artifacts.update(_retained_output_names(plan, units))

        collectors: list[CollectorSpec] = []
        for builder in self.collector_builders:
            result = builder.build(plan, units=tuple(units))
            collectors.extend(result.collectors)
            retained_artifacts.update(result.retained_artifacts)
        for collector in collectors:
            retained_artifacts.update(collector.inputs)
        return ExecutionPlan(
            units=tuple(units),
            resources=_resource_specs(plan),
            collectors=tuple(collectors),
            retained_artifacts=frozenset(retained_artifacts),
            metadata={
                "task_type": plan.get("task_type"),
                "source": "physical_plan",
            },
        )

    def build_many(
        self,
        physical_plans: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        namespaces: list[str] | tuple[str, ...] | None = None,
    ) -> ExecutionPlan:
        """Build one executor-ready plan from multiple single-task physical plans."""

        if not physical_plans:
            return ExecutionPlan(units=(), metadata={"source": "physical_plan_list"})
        if namespaces is not None and len(namespaces) != len(physical_plans):
            raise PlanValidationError("namespaces length must match physical_plans")
        execution_plans = []
        for index, physical_plan in enumerate(physical_plans):
            namespace = (
                namespaces[index]
                if namespaces is not None
                else f"plan_{index}"
            )
            execution_plans.append(
                _namespace_execution_plan(
                    self.build(physical_plan),
                    namespace=namespace,
                )
            )
        return merge_execution_plans(
            execution_plans,
            metadata={"source": "physical_plan_list"},
        )


def merge_execution_plans(
    execution_plans: list[ExecutionPlan] | tuple[ExecutionPlan, ...],
    *,
    metadata: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """Merge already executor-ready plans into one scheduler-visible graph."""

    resources: list[dict[str, Any]] = []
    units: list[ExecutionUnit] = []
    collectors: list[CollectorSpec] = []
    retained_artifacts: set[str] = set()
    for execution_plan in execution_plans:
        resources.extend(copy.deepcopy(resource) for resource in execution_plan.resources)
        index_offset = len(units)
        for unit in execution_plan.units:
            units.append(_reindex_unit(unit, index=unit.index + index_offset))
        collectors.extend(copy.deepcopy(collector) for collector in execution_plan.collectors)
        retained_artifacts.update(execution_plan.retained_artifacts)
    _validate_resource_specs(resources)
    _validate_execution_units(units)
    return ExecutionPlan(
        units=tuple(units),
        resources=tuple(resources),
        collectors=tuple(collectors),
        retained_artifacts=frozenset(retained_artifacts),
        metadata=copy.deepcopy(metadata or {"source": "execution_plan_merge"}),
    )


def _resource_specs(physical_plan: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    resources = physical_plan.get("resources", [])
    if not isinstance(resources, list):
        raise PlanValidationError("Physical plan resources must be a list")
    return tuple(copy.deepcopy(resource) for resource in resources)


def _namespace_execution_plan(
    execution_plan: ExecutionPlan,
    *,
    namespace: str,
) -> ExecutionPlan:
    prefix = _safe_id(namespace)
    if not prefix:
        return execution_plan

    resource_map = {
        str(resource.get("id")): f"{prefix}__{resource.get('id')}"
        for resource in execution_plan.resources
        if isinstance(resource.get("id"), str) and resource.get("id")
    }
    artifact_map = {
        unit.output: f"{prefix}__{unit.output}"
        for unit in execution_plan.units
        if unit.output
    }
    unit_id_map = {
        unit.id: f"{prefix}__{unit.id}"
        for unit in execution_plan.units
    }

    resources = tuple(
        _namespace_resource_spec(resource, resource_map=resource_map)
        for resource in execution_plan.resources
    )
    units = tuple(
        _namespace_unit(
            unit,
            resource_map=resource_map,
            artifact_map=artifact_map,
            unit_id_map=unit_id_map,
        )
        for unit in execution_plan.units
    )
    collectors = tuple(
        _namespace_collector(
            collector,
            resource_map=resource_map,
            artifact_map=artifact_map,
            namespace=prefix,
        )
        for collector in execution_plan.collectors
    )
    return ExecutionPlan(
        units=units,
        resources=resources,
        collectors=collectors,
        retained_artifacts=frozenset(
            _mapped_artifact(name, artifact_map=artifact_map)
            for name in execution_plan.retained_artifacts
        ),
        metadata={
            **copy.deepcopy(execution_plan.metadata),
            "namespace": prefix,
        },
    )


def _namespace_resource_spec(
    resource: dict[str, Any],
    *,
    resource_map: dict[str, str],
) -> dict[str, Any]:
    updated = copy.deepcopy(resource)
    resource_id = updated.get("id")
    if isinstance(resource_id, str):
        updated["id"] = resource_map.get(resource_id, resource_id)
    return updated


def _namespace_unit(
    unit: ExecutionUnit,
    *,
    resource_map: dict[str, str],
    artifact_map: dict[str, str],
    unit_id_map: dict[str, str],
) -> ExecutionUnit:
    node = copy.deepcopy(unit.node)
    original_id = unit.id
    node["id"] = unit_id_map.get(unit.id, unit.id)
    node["dependency"] = [
        _mapped_artifact(name, artifact_map=artifact_map)
        for name in unit.dependencies
    ]
    node["input"] = [
        _mapped_resource(name, resource_map=resource_map)
        for name in unit.resources
    ]
    node["output"] = _mapped_artifact(unit.output, artifact_map=artifact_map)
    params = node.get("params")
    if isinstance(params, dict):
        node["params"] = _namespace_node_params(
            params,
            resource_map=resource_map,
            artifact_map=artifact_map,
        )
    metadata = _namespace_metadata(
        unit.metadata,
        resource_map=resource_map,
        artifact_map=artifact_map,
    )
    node_metadata = node.get("metadata")
    if isinstance(node_metadata, dict):
        node["metadata"] = _namespace_metadata(
            node_metadata,
            resource_map=resource_map,
            artifact_map=artifact_map,
        )
    metadata.setdefault("original_unit_id", original_id)
    return ExecutionUnit(
        id=node["id"],
        task_type=unit.task_type,
        op=unit.op,
        node=node,
        dependencies=tuple(node["dependency"]),
        resources=tuple(node["input"]),
        output=node["output"],
        index=unit.index,
        metadata=metadata,
    )


def _namespace_collector(
    collector: CollectorSpec,
    *,
    resource_map: dict[str, str],
    artifact_map: dict[str, str],
    namespace: str,
) -> CollectorSpec:
    return CollectorSpec(
        id=f"{namespace}__{collector.id}",
        kind=collector.kind,
        inputs=tuple(
            _mapped_artifact(name, artifact_map=artifact_map)
            for name in collector.inputs
        ),
        output=_mapped_collector_output(
            collector.output,
            artifact_map=artifact_map,
            namespace=namespace,
        ),
        params=_namespace_collector_params(
            collector.params,
            resource_map=resource_map,
            artifact_map=artifact_map,
        ),
        metadata=_namespace_metadata(
            collector.metadata,
            resource_map=resource_map,
            artifact_map=artifact_map,
        ),
    )


def _namespace_node_params(
    params: dict[str, Any],
    *,
    resource_map: dict[str, str],
    artifact_map: dict[str, str],
) -> dict[str, Any]:
    updated = copy.deepcopy(params)
    source = updated.get("source")
    if isinstance(source, str):
        updated["source"] = _mapped_resource(source, resource_map=resource_map)
    source_ref = updated.get("source_ref")
    if isinstance(source_ref, str):
        updated["source_ref"] = _mapped_artifact(
            source_ref,
            artifact_map=artifact_map,
        )
    joins = updated.get("joins")
    if isinstance(joins, list):
        for join in joins:
            if not isinstance(join, dict):
                continue
            join_source = join.get("source")
            if isinstance(join_source, str):
                join["source"] = _mapped_resource(
                    join_source,
                    resource_map=resource_map,
                )
            join_source_ref = join.get("source_ref")
            if isinstance(join_source_ref, str):
                join["source_ref"] = _mapped_artifact(
                    join_source_ref,
                    artifact_map=artifact_map,
                )
    return updated


def _namespace_collector_params(
    params: dict[str, Any],
    *,
    resource_map: dict[str, str],
    artifact_map: dict[str, str],
) -> dict[str, Any]:
    updated = copy.deepcopy(params)
    query_outputs = updated.get("query_outputs")
    if isinstance(query_outputs, list):
        for item in query_outputs:
            if isinstance(item, dict) and isinstance(item.get("output"), str):
                item["output"] = _mapped_artifact(
                    item["output"],
                    artifact_map=artifact_map,
                )
    items = updated.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("output"), str):
                item["output"] = _mapped_artifact(
                    item["output"],
                    artifact_map=artifact_map,
                )
            if isinstance(item.get("chunk_resource_id"), str):
                item["chunk_resource_id"] = _mapped_resource(
                    item["chunk_resource_id"],
                    resource_map=resource_map,
                )
    return updated


def _namespace_metadata(
    metadata: dict[str, Any],
    *,
    resource_map: dict[str, str],
    artifact_map: dict[str, str],
) -> dict[str, Any]:
    updated = copy.deepcopy(metadata)
    for key in ("chunk_resource_id",):
        if isinstance(updated.get(key), str):
            updated[key] = _mapped_resource(updated[key], resource_map=resource_map)
    for key in ("group_output",):
        if isinstance(updated.get(key), str):
            updated[key] = _mapped_artifact(updated[key], artifact_map=artifact_map)
    return updated


def _reindex_unit(unit: ExecutionUnit, *, index: int) -> ExecutionUnit:
    return ExecutionUnit(
        id=unit.id,
        task_type=unit.task_type,
        op=unit.op,
        node=copy.deepcopy(unit.node),
        dependencies=unit.dependencies,
        resources=unit.resources,
        output=unit.output,
        index=index,
        metadata=copy.deepcopy(unit.metadata),
    )


def _mapped_resource(name: str, *, resource_map: dict[str, str]) -> str:
    return resource_map.get(name, name)


def _mapped_artifact(name: str, *, artifact_map: dict[str, str]) -> str:
    return artifact_map.get(name, name)


def _mapped_collector_output(
    name: str,
    *,
    artifact_map: dict[str, str],
    namespace: str,
) -> str:
    return artifact_map.get(name, f"{namespace}__{name}")


def _validate_resource_specs(resources: list[dict[str, Any]]) -> None:
    duplicate_ids = _duplicates(
        [
            resource.get("id")
            for resource in resources
            if isinstance(resource.get("id"), str) and resource.get("id")
        ]
    )
    if duplicate_ids:
        raise PlanValidationError(f"Duplicate execution resource id: {duplicate_ids}")


class PhysicalNodeBuilder:
    """Build units from the canonical physical-plan ``nodes`` field."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        start_index: int,
    ) -> PlanNodeBuildResult:
        task_type = str(physical_plan["task_type"])
        source_sql = physical_plan.get("source_sql")
        units: list[ExecutionUnit] = []
        for offset, node in enumerate(physical_plan.get("nodes", [])):
            if not isinstance(node, dict):
                raise PlanValidationError(f"Physical node must be an object: {node!r}")
            node_id = _required_string(node, "id", label="physical node")
            output = _required_string(node, "output", label=f"physical node {node_id}")
            # Propagate plan-level source_sql into each node so the Edge Agent
            # sandbox can inject the original cloud SQL as a hint.
            if source_sql and "source_sql" not in node:
                node = {**node, "source_sql": source_sql}
            unit_metadata = {
                "source_kind": "node",
                "output_type": node.get("output_type"),
            }
            unit_metadata.update(copy.deepcopy(node.get("metadata", {})))
            units.append(
                ExecutionUnit(
                    id=node_id,
                    task_type=task_type,
                    op=str(node["op"]),
                    node=copy.deepcopy(node),
                    dependencies=tuple(
                        str(name) for name in node.get("dependency", [])
                    ),
                    resources=tuple(str(name) for name in node.get("input", [])),
                    output=output,
                    index=start_index + offset,
                    metadata=unit_metadata,
                )
            )
        return PlanNodeBuildResult(units=tuple(units))


class MapGroupNodeBuilder:
    """Build chunk-local map units from compact ``map_groups``."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        start_index: int,
    ) -> PlanNodeBuildResult:
        task_type = str(physical_plan["task_type"])
        units: list[ExecutionUnit] = []
        retained_artifacts: set[str] = set()
        for group_index, group in enumerate(physical_plan.get("map_groups", [])):
            if not isinstance(group, dict):
                raise PlanValidationError(f"map_group must be an object: {group!r}")
            group_id = _required_string(group, "id", label="map_group")
            op = str(group.get("op") or "map")
            if op != "map":
                raise PlanValidationError(f"map_group {group_id} op must be map")
            group_output = str(group.get("output") or group_id)
            chunk_resources = _map_group_chunk_resources(group)
            dependencies = tuple(str(name) for name in group.get("dependency", []))
            replicas = _positive_int(group.get("replicas", 1), label=f"map_group {group_id} replicas")
            for chunk_index, resource_id in enumerate(chunk_resources):
                for replica_index in range(replicas):
                    unit_id = _map_unit_id(
                        group_id,
                        resource_id,
                        chunk_index,
                        replica_index=replica_index,
                        replicas=replicas,
                    )
                    output = unit_id
                    retained_artifacts.add(output)
                    node = _map_unit_node(
                        group=group,
                        unit_id=unit_id,
                        resource_id=resource_id,
                        dependencies=dependencies,
                        output=output,
                        chunk_index=chunk_index,
                        replica_index=replica_index,
                    )
                    unit_metadata = copy.deepcopy(node.get("metadata", {}))
                    unit_metadata["map_group_index"] = group_index
                    units.append(
                        ExecutionUnit(
                            id=unit_id,
                            task_type=task_type,
                            op=op,
                            node=node,
                            dependencies=dependencies,
                            resources=(resource_id,),
                            output=output,
                            index=start_index + len(units),
                            metadata=unit_metadata,
                        )
                    )
            retained_artifacts.add(group_output)
        return PlanNodeBuildResult(
            units=tuple(units),
            retained_artifacts=frozenset(retained_artifacts),
        )


class StaticCollectorBuilder:
    """Build collectors explicitly carried by the physical plan."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        units: tuple[ExecutionUnit, ...],
    ) -> PlanCollectorBuildResult:
        collectors: list[CollectorSpec] = []
        for index, payload in enumerate(physical_plan.get("static_collectors", [])):
            if not isinstance(payload, dict):
                raise PlanValidationError(
                    f"static_collector must be an object: {payload!r}"
                )
            collector_id = _required_string(
                payload,
                "id",
                label=f"static_collector {index}",
            )
            kind = _required_string(
                payload,
                "kind",
                label=f"static_collector {collector_id}",
            )
            output = str(payload.get("output") or collector_id)
            if kind == "minions_transform_outputs":
                collectors.append(
                    _minions_transform_collector_spec(
                        payload,
                        physical_plan=physical_plan,
                        units=units,
                        collector_id=collector_id,
                        output=output,
                    )
                )
                continue
            inputs = payload.get("inputs", [])
            if not isinstance(inputs, list) or not all(
                isinstance(item, str) and item for item in inputs
            ):
                raise PlanValidationError(
                    f"static_collector {collector_id} inputs must be a string list"
                )
            collectors.append(
                CollectorSpec(
                    id=collector_id,
                    kind=kind,
                    inputs=tuple(inputs),
                    output=output,
                    params=copy.deepcopy(payload.get("params", {})),
                    metadata=copy.deepcopy(payload.get("metadata", {})),
                )
            )
        return _collector_result(collectors)


class MapGroupCollectorBuilder:
    """Build a collector for map-group worker outputs."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        units: tuple[ExecutionUnit, ...],
    ) -> PlanCollectorBuildResult:
        if _has_static_collectors(physical_plan) or not _has_map_groups(physical_plan):
            return PlanCollectorBuildResult()
        return _collector_result(_map_group_collector_specs(physical_plan, list(units)))


class QueryOutputsCollectorBuilder:
    """Build answer-map collectors for merged table query outputs."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        units: tuple[ExecutionUnit, ...],
    ) -> PlanCollectorBuildResult:
        del units
        if (
            _has_static_collectors(physical_plan)
            or _has_map_groups(physical_plan)
            or not _has_query_outputs(physical_plan)
        ):
            return PlanCollectorBuildResult()
        inputs: list[str] = []
        for item in physical_plan.get("query_outputs", []):
            if not isinstance(item, dict):
                continue
            output_name = item.get("output")
            answer = item.get("answer")
            answer_name = answer.get("name") if isinstance(answer, dict) else None
            if isinstance(output_name, str):
                inputs.append(output_name)
            elif isinstance(answer_name, str):
                inputs.append(answer_name)
        return _collector_result(
            [
                CollectorSpec(
                    id="answer",
                    kind="answer_map",
                    inputs=tuple(dict.fromkeys(inputs)),
                    output="answer",
                    params={
                        "query_outputs": copy.deepcopy(
                            physical_plan.get("query_outputs", [])
                        )
                    },
                    metadata={"source": "query_outputs"},
                )
            ]
        )


class TableAnswerCollectorBuilder:
    """Build table answer collectors from answer formatting node metadata."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        units: tuple[ExecutionUnit, ...],
    ) -> PlanCollectorBuildResult:
        if (
            _has_static_collectors(physical_plan)
            or _has_map_groups(physical_plan)
            or _has_query_outputs(physical_plan)
        ):
            return PlanCollectorBuildResult()
        answer = physical_plan.get("answer")
        final_unit = _default_final_unit(list(units))
        if not isinstance(answer, dict) and final_unit is not None:
            params = final_unit.node.get("params")
            answer = params.get("answer") if isinstance(params, dict) else None
        if not isinstance(answer, dict):
            return PlanCollectorBuildResult()
        final_output = _default_final_output(list(units))
        if final_output is None:
            return PlanCollectorBuildResult()
        return _collector_result(
            [
                CollectorSpec(
                    id="answer",
                    kind="table_answer",
                    inputs=(final_output,),
                    output="answer",
                    params={"answer": copy.deepcopy(answer)},
                    metadata={"source": "answer_metadata"},
                )
            ]
        )


class FinalAnswerCollectorBuilder:
    """Build the fallback collector for plans whose output is already an answer."""

    def build(
        self,
        physical_plan: dict[str, Any],
        *,
        units: tuple[ExecutionUnit, ...],
    ) -> PlanCollectorBuildResult:
        if (
            _has_static_collectors(physical_plan)
            or _has_map_groups(physical_plan)
            or _has_query_outputs(physical_plan)
            or isinstance(physical_plan.get("answer"), dict)
        ):
            return PlanCollectorBuildResult()
        final_output = _default_final_output(list(units))
        if final_output is None:
            return PlanCollectorBuildResult()
        return _collector_result(
            [
                CollectorSpec(
                    id="answer",
                    kind="final_answer",
                    inputs=(final_output,),
                    output="answer",
                    params={},
                    metadata={"source": "final_output"},
                )
            ]
        )


class Scheduler:
    """Schedule physical-plan nodes without executing them."""

    def __init__(
        self,
        *,
        units: list[ExecutionUnit],
        resource_state: ResourceState,
        consumers_by_artifact: dict[str, int],
        retained_artifacts: set[str],
        collectors: tuple[CollectorSpec, ...] = (),
    ) -> None:
        self._units = list(units)
        self._units_by_id = {unit.id: unit for unit in units}
        if len(self._units_by_id) != len(units):
            duplicates = _duplicates([unit.id for unit in units])
            raise PlanValidationError(f"Duplicate execution unit id: {duplicates}")
        self._resource_state = resource_state
        self._pending = {unit.id for unit in units}
        self._running: set[str] = set()
        self._succeeded: set[str] = set()
        self._failed: dict[str, Any] = {}
        self._consumers_by_artifact = dict(consumers_by_artifact)
        self._retained_artifacts = set(retained_artifacts)
        self._collectors = tuple(collectors)

    @classmethod
    def from_execution_plan(
        cls,
        execution_plan: ExecutionPlan,
        *,
        resource_state: ResourceState,
    ) -> "Scheduler":
        """Build a scheduler from a pre-built execution plan."""

        units = list(execution_plan.units)
        return cls(
            units=units,
            resource_state=resource_state,
            consumers_by_artifact=_dependency_consumer_counts(units),
            retained_artifacts=set(execution_plan.retained_artifacts),
            collectors=execution_plan.collectors,
        )

    @property
    def consumers_by_artifact(self) -> dict[str, int]:
        return dict(self._consumers_by_artifact)

    @property
    def retained_artifacts(self) -> set[str]:
        return set(self._retained_artifacts)

    @property
    def collectors(self) -> tuple[CollectorSpec, ...]:
        return self._collectors

    @property
    def done(self) -> bool:
        return not self._pending and not self._running

    def next_ready(self) -> ExecutionUnit | None:
        """Move the next ready unit into RUNNING state."""

        batch = self.next_ready_batch(max_units=1)
        return batch[0] if batch else None

    def next_ready_batch(self, *, max_units: int) -> list[ExecutionUnit]:
        """Move up to ``max_units`` ready units into RUNNING state."""

        if max_units <= 0:
            raise PlanValidationError("max_units must be positive")
        if self._running:
            return []
        # Readiness is deliberately checkable at the edge: every dependency
        # artifact and source resource must already be available locally.
        ready_units = [
            unit
            for unit in self._units
            if unit.id in self._pending and self._unit_ready(unit)
        ]
        if not ready_units:
            return []
        selected = _select_ready_units(ready_units, max_units=max_units)
        for unit in selected:
            self._pending.remove(unit.id)
            self._running.add(unit.id)
        return selected

    def mark_succeeded(self, unit_id: str) -> None:
        self._mark_finished(unit_id)
        self._succeeded.add(unit_id)

    def mark_failed(self, unit_id: str, error: Any | None = None) -> None:
        self._mark_finished(unit_id)
        self._failed[unit_id] = error

    def blocked_unit_ids(self) -> list[str]:
        """Return pending ids in deterministic plan order."""

        return [unit.id for unit in self._units if unit.id in self._pending]

    def _unit_ready(self, unit: ExecutionUnit) -> bool:
        return all(
            self._resource_state.has_artifact(dependency)
            for dependency in unit.dependencies
        ) and all(
            self._resource_state.has_source(resource_id)
            for resource_id in unit.resources
        )

    def _mark_finished(self, unit_id: str) -> None:
        if unit_id not in self._units_by_id:
            raise PlanValidationError(f"Unknown execution unit id: {unit_id}")
        if unit_id not in self._running:
            raise PlanValidationError(f"Execution unit is not running: {unit_id}")
        self._running.remove(unit_id)


def _select_ready_units(
    ready_units: list[ExecutionUnit],
    *,
    max_units: int,
) -> list[ExecutionUnit]:
    """Pick a deterministic workflow batch in plan order."""

    return ready_units[:max_units]


def _dependency_consumer_counts(units: list[ExecutionUnit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        for dependency in unit.dependencies:
            counts[dependency] = counts.get(dependency, 0) + 1
    return counts


def _retained_output_names(
    physical_plan: dict[str, Any],
    units: list[ExecutionUnit],
) -> set[str]:
    retained: set[str] = set()
    if units and units[-1].output:
        retained.add(units[-1].output)
    for unit in units:
        if unit.op == "FormatAnswer" and unit.output:
            retained.add(unit.output)
    for item in physical_plan.get("query_outputs", []):
        if not isinstance(item, dict):
            continue
        output_name = item.get("output")
        answer = item.get("answer")
        answer_name = answer.get("name") if isinstance(answer, dict) else None
        if isinstance(output_name, str):
            retained.add(output_name)
        if isinstance(answer_name, str):
            retained.add(answer_name)
    return retained


def _default_final_output(units: list[ExecutionUnit]) -> str | None:
    unit = _default_final_unit(units)
    return unit.output if unit is not None else None


def _default_final_unit(units: list[ExecutionUnit]) -> ExecutionUnit | None:
    for unit in units:
        if unit.output == "answer":
            return unit
    if units and units[-1].output:
        return units[-1]
    return None


def _map_group_collector_specs(
    physical_plan: dict[str, Any],
    units: list[ExecutionUnit],
) -> list[CollectorSpec]:
    groups_by_id = {
        str(group.get("id")): group
        for group in physical_plan.get("map_groups", [])
        if isinstance(group, dict) and group.get("id")
    }
    units_by_group: dict[str, list[ExecutionUnit]] = {}
    for unit in units:
        if unit.metadata.get("source_kind") != "map_group":
            continue
        group_id = unit.metadata.get("group_id")
        if isinstance(group_id, str):
            units_by_group.setdefault(group_id, []).append(unit)

    collectors: list[CollectorSpec] = []
    for group_index, group in enumerate(physical_plan.get("map_groups", [])):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id"))
        group_units = sorted(units_by_group.get(group_id, []), key=lambda unit: unit.index)
        if not group_units:
            continue
        group_output = str(group.get("output") or group_id)
        collectors.append(
            CollectorSpec(
                id=group_id,
                kind="map_group_evidence",
                inputs=tuple(unit.output for unit in group_units),
                output=group_output,
                params={
                    **copy.deepcopy(group.get("collector", {})),
                    "items": [
                        _collector_item_for_map_unit(
                            unit,
                            group=groups_by_id.get(group_id, group),
                            job_index=item_index,
                        )
                        for item_index, unit in enumerate(group_units)
                    ],
                },
                metadata={
                    "group_id": group_id,
                    "map_group_index": group_index,
                    "output_type": group.get("output_type"),
                },
            )
        )
    return collectors


def _minions_transform_collector_spec(
    payload: dict[str, Any],
    *,
    physical_plan: dict[str, Any],
    units: tuple[ExecutionUnit, ...],
    collector_id: str,
    output: str,
) -> CollectorSpec:
    map_units = [
        unit
        for unit in sorted(units, key=lambda item: item.index)
        if unit.metadata.get("source_kind") == "map_group"
    ]
    if not map_units:
        raise PlanValidationError(
            f"static_collector {collector_id} requires map_group units"
        )
    groups_by_id = {
        str(group.get("id")): group
        for group in physical_plan.get("map_groups", [])
        if isinstance(group, dict) and group.get("id")
    }
    return CollectorSpec(
        id=collector_id,
        kind="minions_transform_outputs",
        inputs=tuple(unit.output for unit in map_units),
        output=output,
        params={
            "function_name": payload.get("function_name", "transform_outputs"),
            "source": payload.get("source", ""),
            "items": [
                _collector_item_for_map_unit(
                    unit,
                    group=groups_by_id.get(str(unit.metadata.get("group_id")), {}),
                    job_index=index,
                )
                for index, unit in enumerate(map_units)
            ],
        },
        metadata={
            **copy.deepcopy(payload.get("metadata", {})),
            "source": "static_collectors",
        },
    )


def _collector_item_for_map_unit(
    unit: ExecutionUnit,
    *,
    group: dict[str, Any],
    job_index: int,
) -> dict[str, Any]:
    params = group.get("params", {}) if isinstance(group, dict) else {}
    if not isinstance(params, dict):
        params = {}
    group_id = str(unit.metadata.get("group_id") or group.get("id") or "")
    group_index = _int_or_zero(unit.metadata.get("map_group_index"))
    chunk_index = _int_or_zero(unit.metadata.get("chunk_index"))
    replica_index = _int_or_zero(unit.metadata.get("replica_index"))
    return {
        "output": unit.output,
        "chunk_resource_id": unit.metadata.get("chunk_resource_id"),
        "chunk_index": chunk_index,
        "replica_index": replica_index,
        "task_id": group_index,
        "job_id": job_index,
        "group_id": group_id,
        "task": params.get("local_instruction") or params.get("task") or "",
        "advice": params.get("local_guidance") or params.get("advice") or "",
    }


def _collector_result(collectors: list[CollectorSpec]) -> PlanCollectorBuildResult:
    retained_artifacts = {
        output_name for collector in collectors for output_name in collector.inputs
    }
    return PlanCollectorBuildResult(
        collectors=tuple(collectors),
        retained_artifacts=frozenset(retained_artifacts),
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _has_map_groups(physical_plan: dict[str, Any]) -> bool:
    return bool(physical_plan.get("map_groups"))


def _has_query_outputs(physical_plan: dict[str, Any]) -> bool:
    return bool(physical_plan.get("query_outputs"))


def _has_static_collectors(physical_plan: dict[str, Any]) -> bool:
    return bool(physical_plan.get("static_collectors"))


def _validate_execution_units(units: list[ExecutionUnit]) -> None:
    duplicate_ids = _duplicates([unit.id for unit in units])
    if duplicate_ids:
        raise PlanValidationError(f"Duplicate execution unit id: {duplicate_ids}")
    duplicate_artifacts = _duplicates([unit.output for unit in units if unit.output])
    if duplicate_artifacts:
        raise PlanValidationError(f"Duplicate execution unit output: {duplicate_artifacts}")


def _map_group_chunk_resources(group: dict[str, Any]) -> list[str]:
    group_id = group.get("id")
    inputs = group.get("input")
    if not isinstance(inputs, dict):
        raise PlanValidationError(f"map_group {group_id} missing input")
    chunks = inputs.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise PlanValidationError(
            f"map_group {group_id} must reference prepared chunk resources"
        )
    resource_ids = []
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk:
            raise PlanValidationError(
                f"map_group {group_id} has invalid chunk resource: {chunk!r}"
            )
        resource_ids.append(chunk)
    return resource_ids


def _map_unit_node(
    *,
    group: dict[str, Any],
    unit_id: str,
    resource_id: str,
    dependencies: tuple[str, ...],
    output: str,
    chunk_index: int,
    replica_index: int,
) -> dict[str, Any]:
    group_id = str(group["id"])
    params = copy.deepcopy(group.get("params", {}))
    metadata = copy.deepcopy(group.get("metadata", {}))
    metadata.update(document_worker_prefix_metadata(params))
    metadata.update(
        {
            "source_kind": "map_group",
            "group_id": group_id,
            "group_output": str(group.get("output") or group_id),
            "chunk_resource_id": resource_id,
            "chunk_index": chunk_index,
            "replica_index": replica_index,
            "output_type": group.get("output_type"),
        }
    )
    return {
        "id": unit_id,
        "op": "map",
        "dependency": list(dependencies),
        "input": [resource_id],
        "params": params,
        "output": output,
        "output_type": group.get("output_type"),
        "metadata": metadata,
    }


def _map_unit_id(
    group_id: str,
    resource_id: str,
    chunk_index: int,
    *,
    replica_index: int = 0,
    replicas: int = 1,
) -> str:
    base = f"{_safe_id(group_id)}__{chunk_index}__{_safe_id(resource_id)}"
    if replicas <= 1:
        return base
    return f"{base}__sample_{replica_index}"


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "unit"


def _required_string(payload: dict[str, Any], field_name: str, *, label: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise PlanValidationError(f"{label} missing {field_name}")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise PlanValidationError(f"{label} must be a positive integer")
    return value


def _duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates
