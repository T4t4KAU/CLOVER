"""Static result collectors that run after NodeAgent execution."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol

from clover.executor.errors import CollectorExecutionError
from clover.executor.resources import ResourceStore
from clover.executor.result import json_ready
from clover.executor.scheduler import CollectorSpec


class ResultCollector(Protocol):
    """Collect node outputs into an executor-level result artifact."""

    def collect(
        self,
        spec: CollectorSpec,
        *,
        resource_store: ResourceStore,
        physical_plan: dict[str, Any],
    ) -> Any:
        """Return the collected artifact."""


class FinalAnswerCollector:
    """Return the single answer output."""

    def collect(
        self,
        spec: CollectorSpec,
        *,
        resource_store: ResourceStore,
        physical_plan: dict[str, Any],
    ) -> Any:
        for output_name in spec.inputs:
            if resource_store.has_artifact(output_name):
                return resource_store.get_artifact(output_name).materialize(target="python")
        raise CollectorExecutionError(
            f"final_answer collector {spec.id} has no available inputs",
            collector=_collector_payload(spec),
        )


class AnswerMapCollector:
    """Build a multi-answer map from table reasoning query_plan outputs."""

    def collect(
        self,
        spec: CollectorSpec,
        *,
        resource_store: ResourceStore,
        physical_plan: dict[str, Any],
    ) -> dict[str, Any]:
        answer_map: dict[str, Any] = {}
        for item in spec.params.get("query_outputs", []):
            if not isinstance(item, dict):
                continue
            answer = item.get("answer", {})
            answer_name = answer.get("name") if isinstance(answer, dict) else None
            output_name = item.get("output") or answer_name
            if isinstance(answer_name, str) and isinstance(output_name, str):
                if resource_store.has_artifact(output_name):
                    value = resource_store.get_artifact(output_name).materialize(
                        target="python"
                    )
                    answer_map[answer_name] = _table_answer_value(value, answer)
        return answer_map


class TableAnswerCollector:
    """Extract a typed answer value from a table or scalar node output."""

    def collect(
        self,
        spec: CollectorSpec,
        *,
        resource_store: ResourceStore,
        physical_plan: dict[str, Any],
    ) -> Any:
        for output_name in spec.inputs:
            if not resource_store.has_artifact(output_name):
                continue
            value = resource_store.get_artifact(output_name).materialize(target="python")
            return _table_answer_value(value, spec.params.get("answer", {}))
        raise CollectorExecutionError(
            f"table_answer collector {spec.id} has no available inputs",
            collector=_collector_payload(spec),
        )


class MapGroupEvidenceCollector:
    """Collect chunk-local worker results into a MinionS-style evidence view."""

    def collect(
        self,
        spec: CollectorSpec,
        *,
        resource_store: ResourceStore,
        physical_plan: dict[str, Any],
    ) -> dict[str, Any]:
        units_by_output = {
            str(item.get("output")): item
            for item in spec.params.get("items", [])
            if isinstance(item, dict) and item.get("output")
        }
        items: list[dict[str, Any]] = []
        for output_name in spec.inputs:
            if not resource_store.has_artifact(output_name):
                raise CollectorExecutionError(
                    f"map_group collector {spec.id} missing node output {output_name}",
                    collector=_collector_payload(spec),
                )
            resource = resource_store.get_artifact(output_name)
            result = resource.materialize(target="python")
            unit = units_by_output.get(output_name, {})
            chunk_resource_id = unit.get("chunk_resource_id")
            items.append(
                {
                    "output": output_name,
                    "chunk_resource_id": chunk_resource_id,
                    "chunk_index": unit.get("chunk_index"),
                    "result": json_ready(result),
                    "include": _include_worker_result(result),
                }
            )
        included_items = [item for item in items if item["include"]]
        return {
            "kind": "map_group_evidence",
            "group_id": spec.metadata.get("group_id") or spec.id,
            "items": items,
            "worker_count": len(items),
            "included_count": len(included_items),
            "evidence_summary": _format_evidence_summary(
                spec=spec,
                items=included_items,
                physical_plan=physical_plan,
            ),
        }


@dataclass
class JobManifest:
    """MinionS-compatible manifest passed to transform_outputs()."""

    chunk: str
    task: str
    advice: str
    chunk_id: Any = None
    task_id: Any = None
    job_id: Any = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass
class JobOutput:
    """MinionS-compatible worker output passed to transform_outputs()."""

    explanation: str
    citation: str | None
    answer: str | None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass
class Job:
    """MinionS-compatible transform input."""

    manifest: JobManifest
    output: JobOutput
    sample: str
    include: bool | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


class MinionsTransformOutputsCollector:
    """Run Remote-generated transform_outputs(jobs) over worker outputs."""

    def collect(
        self,
        spec: CollectorSpec,
        *,
        resource_store: ResourceStore,
        physical_plan: dict[str, Any],
    ) -> dict[str, Any]:
        jobs = _transform_jobs_from_outputs(
            spec,
            resource_store=resource_store,
        )
        transform_error: str | None = None
        fallback_used = False
        try:
            evidence_summary = _run_transform_outputs(
                source=str(spec.params.get("source") or ""),
                function_name=str(spec.params.get("function_name") or "transform_outputs"),
                jobs=jobs,
            )
        except Exception as exc:  # noqa: BLE001 - MinionS falls back on transform failure.
            transform_error = f"{type(exc).__name__}: {exc}"
            fallback_used = True
            evidence_summary = _fallback_minions_transform(jobs)

        included_count = sum(1 for job in jobs if job.include)
        return {
            "kind": "minions_transform_outputs",
            "collector_id": spec.id,
            "worker_count": len(jobs),
            "included_count": included_count,
            "evidence_summary": evidence_summary,
            "job_manifests": [job.manifest.model_dump() for job in jobs],
            "job_outputs": [job.output.model_dump() for job in jobs],
            "jobs": [job.model_dump() for job in jobs],
            "fallback_used": fallback_used,
            **({"transform_error": transform_error} if transform_error else {}),
        }


COLLECTOR_REGISTRY: dict[str, ResultCollector] = {
    "final_answer": FinalAnswerCollector(),
    "answer_map": AnswerMapCollector(),
    "table_answer": TableAnswerCollector(),
    "map_group_evidence": MapGroupEvidenceCollector(),
    "minions_transform_outputs": MinionsTransformOutputsCollector(),
}


def run_collectors(
    collectors: tuple[CollectorSpec, ...] | list[CollectorSpec],
    *,
    resource_store: ResourceStore,
    physical_plan: dict[str, Any],
) -> dict[str, Any]:
    """Run all static collectors and return artifacts keyed by collector output."""

    outputs: dict[str, Any] = {}
    for spec in collectors:
        collector = COLLECTOR_REGISTRY.get(spec.kind)
        if collector is None:
            raise CollectorExecutionError(
                f"Unsupported collector kind: {spec.kind}",
                collector=_collector_payload(spec),
            )
        try:
            outputs[spec.output] = collector.collect(
                spec,
                resource_store=resource_store,
                physical_plan=physical_plan,
            )
        except CollectorExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize collector failures.
            raise CollectorExecutionError(
                f"Collector {spec.id} failed: {exc}",
                collector=_collector_payload(spec),
                original=exc,
            ) from exc
    return outputs


def _table_answer_value(value: Any, answer: Any) -> Any:
    if not _looks_like_pandas_table(value):
        return _cast_answer_value(value, answer)
    frame = value.frame
    answer_payload = answer if isinstance(answer, dict) else {}
    answer_name = answer_payload.get("name")
    answer_type = str(answer_payload.get("type", "json")).lower()

    if answer_type == "table":
        return {
            "columns": [str(column) for column in frame.columns],
            "rows": json_ready(frame.to_dict(orient="records")),
        }
    if frame.empty:
        return [] if answer_type.startswith("list") else None
    if isinstance(answer_name, str) and answer_name in frame.columns:
        series = frame[answer_name]
    else:
        series = frame.iloc[:, -1]

    if answer_type.startswith("list"):
        values = [_to_python_scalar(item) for item in series.tolist()]
        if len(values) == 1 and isinstance(values[0], list):
            return values[0]
        return values
    return _cast_answer_value(_to_python_scalar(series.iloc[0]), answer_payload)


def _looks_like_pandas_table(value: Any) -> bool:
    frame = getattr(value, "frame", None)
    return hasattr(frame, "empty") and hasattr(frame, "columns") and hasattr(frame, "iloc")


def _cast_answer_value(value: Any, answer: Any) -> Any:
    answer_payload = answer if isinstance(answer, dict) else {}
    answer_type = str(answer_payload.get("type", "json")).lower()
    if answer_type == "boolean":
        return _to_bool(value)
    if answer_type in {"number", "integer", "int", "float"}:
        return _to_number(value)
    if answer_type in {"string", "category"}:
        return None if value is None else str(value)
    return value


def _to_python_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return bool(text)


def _to_number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def _transform_jobs_from_outputs(
    spec: CollectorSpec,
    *,
    resource_store: ResourceStore,
) -> list[Job]:
    items_by_output = {
        str(item.get("output")): item
        for item in spec.params.get("items", [])
        if isinstance(item, dict) and item.get("output")
    }
    jobs: list[Job] = []
    for output_name in spec.inputs:
        if not resource_store.has_artifact(output_name):
            raise CollectorExecutionError(
                f"transform_outputs collector {spec.id} missing node output {output_name}",
                collector=_collector_payload(spec),
            )
        result = resource_store.get_artifact(output_name).materialize(target="python")
        item = items_by_output.get(output_name, {})
        job = _job_from_worker_result(result, item=item, resource_store=resource_store)
        job.include = _include_job(job)
        jobs.append(job)
    return jobs


def _job_from_worker_result(
    result: Any,
    *,
    item: dict[str, Any],
    resource_store: ResourceStore,
) -> Job:
    chunk_record = _chunk_record_for_item(item, resource_store=resource_store)
    manifest = JobManifest(
        chunk=_chunk_label_from_result(result, item=item, chunk_record=chunk_record),
        task=_string_value_from_any(item.get("task")),
        advice=_string_value_from_any(item.get("advice")),
        chunk_id=item.get("chunk_index"),
        task_id=item.get("task_id"),
        job_id=item.get("job_id"),
    )
    output = JobOutput(
        explanation=_result_field(result, "explanation") or "",
        citation=_result_field(result, "citation"),
        answer=_result_field(result, "answer"),
    )
    return Job(
        manifest=manifest,
        output=output,
        sample=_worker_sample(result),
        include=None,
    )


def _chunk_record_for_item(
    item: dict[str, Any],
    *,
    resource_store: ResourceStore,
) -> dict[str, Any]:
    resource_id = item.get("chunk_resource_id")
    if not isinstance(resource_id, str) or not resource_store.has_source(resource_id):
        return {}
    try:
        record = resource_store.get_source(resource_id).materialize(target="chunk_record")
    except Exception:  # noqa: BLE001 - chunk metadata is best effort for collectors.
        return {}
    return record if isinstance(record, dict) else {}


def _chunk_label_from_result(
    result: Any,
    *,
    item: dict[str, Any],
    chunk_record: dict[str, Any],
) -> str:
    chunk = result.get("chunk") if isinstance(result, dict) else None
    if isinstance(chunk, dict) and chunk.get("chunk_id") is not None:
        return str(chunk["chunk_id"])
    if chunk_record.get("chunk_id") is not None:
        return str(chunk_record["chunk_id"])
    if item.get("chunk_resource_id") is not None:
        return str(item["chunk_resource_id"])
    return f"chunk_{item.get('chunk_index', 'unknown')}"


def _result_field(result: Any, field_name: str) -> str | None:
    if not isinstance(result, dict):
        return None
    return _optional_string_value(result.get(field_name))


def _worker_sample(result: Any) -> str:
    if isinstance(result, dict) and isinstance(result.get("sample"), str):
        return result["sample"].strip()
    if isinstance(result, dict):
        sample_payload = {
            key: result.get(key)
            for key in ("answer", "citation", "explanation")
            if result.get(key) is not None
        }
        return json.dumps(json_ready(sample_payload), ensure_ascii=False)
    return json.dumps(json_ready(result), ensure_ascii=False)


def _run_transform_outputs(
    *,
    source: str,
    function_name: str,
    jobs: list[Job],
) -> str:
    if not source.strip():
        raise CollectorExecutionError("transform_outputs source is empty")
    namespace: dict[str, Any] = _transform_namespace()
    exec(compile(source, "<transform_outputs>", "exec"), namespace)  # noqa: S102
    function = namespace.get(function_name)
    if not callable(function):
        raise CollectorExecutionError(f"transform function not found: {function_name}")
    value = function(jobs)
    if isinstance(value, str):
        return value
    return json.dumps(json_ready(value), ensure_ascii=False, indent=2)


def _transform_namespace() -> dict[str, Any]:
    return {
        "__builtins__": {
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        },
        "Any": Any,
        "Dict": Dict,
        "Job": Job,
        "JobManifest": JobManifest,
        "JobOutput": JobOutput,
        "List": List,
        "Optional": Optional,
        "defaultdict": defaultdict,
        "json": json,
        "math": math,
        "re": re,
        "statistics": statistics,
    }


def _fallback_minions_transform(jobs: list[Job]) -> str:
    tasks: dict[Any, dict[str, Any]] = {}
    for job in jobs:
        job.include = _include_job(job)
        task_id = job.manifest.task_id
        if task_id not in tasks:
            tasks[task_id] = {
                "task_id": task_id,
                "task": job.manifest.task,
                "chunks": {},
            }
        tasks[task_id]["chunks"].setdefault(job.manifest.chunk_id, []).append(job)

    chunks: list[str] = []
    for task_id, task_info in tasks.items():
        chunks.append(f"## Task (task_id=`{task_id}`): {task_info['task']}\n")
        for chunk_id, chunk_jobs in task_info["chunks"].items():
            chunks.append(f"### Chunk # {chunk_id}")
            filtered_jobs = [job for job in chunk_jobs if job.include]
            if not filtered_jobs:
                chunks.append("   No jobs returned successfully for this chunk.\n")
                continue
            for index, job in enumerate(filtered_jobs, start=1):
                chunks.append(f"   -- Job {index} (job_id=`{job.manifest.job_id}`):")
                chunks.append(f"   {job.sample}\n")
        chunks.append("-----------------------\n")
    return "\n".join(chunks).strip()


def _include_job(job: Job) -> bool:
    answer = job.output.answer
    if answer is None or str(answer).lower().strip() == "none":
        return False
    return True


def _optional_string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        return stripped
    if isinstance(value, list):
        parts = [_optional_string_value(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return str(value)


def _string_value_from_any(value: Any) -> str:
    normalized = _optional_string_value(value)
    return normalized or ""


def _include_worker_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return result is not None
    return _non_empty_value(result.get("answer")) or _non_empty_value(
        result.get("citation")
    )


def _format_evidence_summary(
    *,
    spec: CollectorSpec,
    items: list[dict[str, Any]],
    physical_plan: dict[str, Any],
) -> str:
    question = _plan_question(physical_plan)
    lines = [
        f"## Task (group_id=`{spec.metadata.get('group_id') or spec.id}`)",
    ]
    if question:
        lines.extend(["", f"Question: {question}"])
    if not items:
        lines.extend(["", "No worker outputs contained a non-empty answer."])
        return "\n".join(lines)

    for item in items:
        result = item["result"]
        lines.extend(
            [
                "",
                f"### {_chunk_heading(result, item)}",
                f"answer: {_compact_field(result, 'answer')}",
                f"citation: {_compact_field(result, 'citation')}",
                f"explanation: {_compact_field(result, 'explanation')}",
            ]
        )
    return "\n".join(lines)


def _non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null"}
    if isinstance(value, list):
        return any(_non_empty_value(item) for item in value)
    return True


def _chunk_heading(result: Any, item: dict[str, Any]) -> str:
    if isinstance(result, dict) and isinstance(result.get("chunk"), dict):
        chunk = result["chunk"]
        label = str(chunk.get("chunk_id") or "unknown chunk")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if page_start is not None and page_end is not None:
            return f"Chunk {label} (pages {page_start}-{page_end})"
        if page_start is not None:
            return f"Chunk {label} (page {page_start})"
        return f"Chunk {label}"
    if item.get("chunk_index") is not None:
        return f"Chunk {item['chunk_index']}"
    return "Chunk unknown"


def _compact_field(result: Any, field_name: str) -> str:
    if not isinstance(result, dict):
        return json.dumps(json_ready(result), ensure_ascii=False)
    value = result.get(field_name)
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(json_ready(value), ensure_ascii=False)


def _plan_question(physical_plan: dict[str, Any]) -> str | None:
    question = physical_plan.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    task = physical_plan.get("task")
    if isinstance(task, dict):
        question = task.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return None


def _collector_payload(spec: CollectorSpec) -> dict[str, Any]:
    return asdict(spec)
