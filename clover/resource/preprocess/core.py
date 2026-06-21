"""Generic DSL preprocessing."""

from __future__ import annotations

import copy
import csv
import itertools
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from clover.reasoning_profiles import (
    HINTS_KEY,
    PROFILE_KEY,
    table_reasoning_profile_from_dsl,
)
from clover.resource.preprocess.csv_schema import extract_csv_schema
from clover.resource.preprocess.pdf_schema import extract_pdf_schema
from clover.task_types import is_table_task_type, task_type_spec


SUPPORTED_SOURCE_TYPES = {"pdf", "table"}
SUPPORTED_ANSWER_TYPES = {
    "boolean",
    "category",
    "date",
    "list",
    "list[category]",
    "list[number]",
    "list[string]",
    "number",
    "string",
    "table",
}

MAX_JOIN_CANDIDATES = 12
MAX_JOIN_PATHS = 16
MAX_JOIN_SAMPLE_ROWS = 1000
MAX_JOIN_PROFILE_COLUMNS = 48
MIN_JOIN_VALUE_OVERLAP = 2
MIN_JOIN_CANDIDATE_SCORE = 0.50


def preprocess_task_dsl(
    task_dsl: dict[str, Any],
    base_dir: str | Path,
) -> dict[str, Any]:
    """Bind local resource details and produce local/remote DSL variants.

    This step is deterministic and local-only: it validates the task, resolves
    source files, extracts structural schemas, and does not call any model.
    """

    _validate_task_dsl(task_dsl)
    task_type = task_dsl["task_type"]

    base_path = Path(base_dir).expanduser().resolve()

    # The local DSL is for local execution and keeps absolute resource paths.
    local_dsl = {
        "task_type": task_type,
        "question": task_dsl["question"],
        "sources": [],
        "answer": copy.deepcopy(task_dsl["answer"]),
    }

    # The remote DSL is the payload intended for remote planning. It includes
    # structural metadata only, not local filesystem paths.
    remote_dsl = {
        "task_type": task_type,
        "question": task_dsl["question"],
        "sources": [],
        "answer": copy.deepcopy(task_dsl["answer"]),
    }
    _copy_optional_task_fields(task_dsl, local_dsl, remote_dsl)

    # Context is local runtime state used to map remote-visible source ids back
    # to concrete resources during later planning/execution stages.
    context: dict[str, Any] = {
        "task_type": task_type,
        "base_dir": str(base_path),
        "source_map": {},
    }

    for source_index, source in enumerate(task_dsl["sources"], start=1):
        local_source, remote_source, source_map, source_context = _preprocess_source(
            source=source,
            source_index=source_index,
            base_dir=base_path,
        )
        local_dsl["sources"].append(local_source)
        remote_dsl["sources"].append(remote_source)
        source_id = local_source["id"]
        context["source_map"][source_id] = source_map
        _merge_source_context(context, source_id, source_context)

    join_candidates = _build_join_candidates(
        local_dsl.get("sources", []),
        hints=task_dsl.get(HINTS_KEY),
    )
    if join_candidates:
        join_hints: dict[str, Any] = {"join_candidates": join_candidates}
        join_paths = _build_join_paths(join_candidates)
        if join_paths:
            join_hints["join_paths"] = join_paths
        _merge_hints(
            local_dsl,
            remote_dsl,
            join_hints,
        )

    return {
        "local_dsl": local_dsl,
        "remote_dsl": remote_dsl,
        "context": context,
    }


def _copy_optional_task_fields(
    task_dsl: dict[str, Any],
    local_dsl: dict[str, Any],
    remote_dsl: dict[str, Any],
) -> None:
    for key in (PROFILE_KEY, HINTS_KEY):
        if key in task_dsl:
            local_dsl[key] = copy.deepcopy(task_dsl[key])
            remote_dsl[key] = copy.deepcopy(task_dsl[key])
    if PROFILE_KEY not in task_dsl and is_table_task_type(task_dsl.get("task_type")):
        profile = table_reasoning_profile_from_dsl(task_dsl)
        local_dsl[PROFILE_KEY] = profile
        remote_dsl[PROFILE_KEY] = profile


def _merge_hints(
    local_dsl: dict[str, Any],
    remote_dsl: dict[str, Any],
    hints: dict[str, Any],
) -> None:
    for dsl in (local_dsl, remote_dsl):
        existing = dsl.get(HINTS_KEY)
        merged = copy.deepcopy(existing) if isinstance(existing, dict) else {}
        merged.update(copy.deepcopy(hints))
        dsl[HINTS_KEY] = merged


def _validate_task_dsl(task_dsl: dict[str, Any]) -> None:
    required_fields = ("task_type", "question", "sources", "answer")
    missing = [field for field in required_fields if field not in task_dsl]
    if missing:
        raise ValueError(f"Task DSL missing required fields: {missing}")

    if task_type_spec(task_dsl["task_type"]) is None:
        raise ValueError(f"Unsupported task_type: {task_dsl['task_type']}")

    if not isinstance(task_dsl["sources"], list) or not task_dsl["sources"]:
        raise ValueError("Task DSL requires at least one source")

    answer = task_dsl["answer"]
    if not isinstance(answer, dict) or "type" not in answer:
        raise ValueError("Task DSL answer must include a type")
    if answer["type"] not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"Unsupported answer type: {answer['type']}")

    for source in task_dsl["sources"]:
        source_type = source.get("type")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source type: {source_type}")
        if "file" not in source:
            raise ValueError("Source must include a file path")


def _preprocess_source(
    source: dict[str, Any],
    source_index: int,
    base_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_type = source["type"]
    resource_path = _resolve_source_path(source["file"], base_dir)
    source_id = _resolve_source_id(source, source_type, source_index)

    if source_type == "table":
        return _preprocess_table_source(
            source,
            source_id,
            resource_path,
        )
    if source_type == "pdf":
        return _preprocess_pdf_source(
            source=source,
            source_id=_resolve_source_id(source, "document", source_index),
            resource_path=resource_path,
        )

    raise ValueError(f"Unsupported source type: {source_type}")


def _resolve_source_id(
    source: dict[str, Any],
    source_type: str,
    source_index: int,
) -> str:
    """Use the source's explicit id when present, else fall back to ``<type>_<n>``."""

    raw_id = source.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    return _normal_source_id(source_type, source_index)


def _preprocess_table_source(
    source: dict[str, Any],
    source_id: str,
    resource_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if resource_path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV table sources are supported now: {resource_path}")

    # CSV is structured data, but its cells are text. The schema therefore
    # records shape and column names only; it does not infer logical types.
    schema = extract_csv_schema(resource_path)

    local_source = {
        "id": source_id,
        "original_id": source.get("id"),
        "type": "table",
        "file": source["file"],
        "path": str(resource_path),
        "format": "csv",
        "schema": schema,
    }

    remote_source = {
        "id": source_id,
        "type": "table",
        "format": "csv",
        "schema": copy.deepcopy(schema),
    }

    source_map = {
        "original_id": source.get("id"),
        "type": "table",
        "file": source["file"],
        "path": str(resource_path),
        "format": "csv",
    }
    return local_source, remote_source, source_map, {}


def _build_join_candidates(
    sources: list[dict[str, Any]],
    *,
    hints: Any,
) -> list[dict[str, Any]]:
    """Infer compact multi-table join candidates from local CSV evidence."""

    table_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("type") == "table"
        and isinstance(source.get("path"), str)
    ]
    if len(table_sources) < 2:
        return []

    hint_roles = _join_hint_roles(hints)
    profiles = [_join_table_profile(source) for source in table_sources]
    candidates: list[dict[str, Any]] = []
    for left, right in itertools.combinations(profiles, 2):
        candidates.extend(
            _join_candidates_for_pair(
                left,
                right,
                hint_roles=hint_roles,
            )
        )
    candidates.sort(
        key=lambda item: (
            float(item.get("score") or 0),
            int(item.get("overlap") or 0),
            float(item.get("left_coverage") or 0),
            float(item.get("right_coverage") or 0),
        ),
        reverse=True,
    )
    return candidates[:MAX_JOIN_CANDIDATES]


def _build_join_paths(join_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact table-to-table paths from locally supported join edges."""

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge_index, candidate in enumerate(join_candidates):
        left_table = candidate.get("left_table")
        right_table = candidate.get("right_table")
        if not isinstance(left_table, str) or not isinstance(right_table, str):
            continue
        if not left_table or not right_table or left_table == right_table:
            continue
        edge = {
            "edge_index": edge_index,
            "left_table": left_table,
            "left_column": candidate.get("left_column"),
            "right_table": right_table,
            "right_column": candidate.get("right_column"),
            "score": candidate.get("score"),
        }
        adjacency.setdefault(left_table, []).append(edge)
        adjacency.setdefault(right_table, []).append(_reverse_join_edge(edge))
    if len(adjacency) < 2:
        return []

    paths: list[dict[str, Any]] = []
    tables = sorted(adjacency)
    for start_index, start in enumerate(tables):
        for goal in tables[start_index + 1 :]:
            path = _shortest_join_path(start, goal, adjacency)
            if path:
                paths.append(path)
    paths.sort(
        key=lambda item: (
            int(item.get("length") or 0),
            float(item.get("score") or 0),
        )
    )
    return paths[:MAX_JOIN_PATHS]


def _reverse_join_edge(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_index": edge["edge_index"],
        "left_table": edge.get("right_table"),
        "left_column": edge.get("right_column"),
        "right_table": edge.get("left_table"),
        "right_column": edge.get("left_column"),
        "score": edge.get("score"),
    }


def _shortest_join_path(
    start: str,
    goal: str,
    adjacency: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    queue: list[tuple[str, list[dict[str, Any]], set[str], set[int]]] = [
        (start, [], {start}, set())
    ]
    best_edges: list[dict[str, Any]] | None = None
    best_score = -1.0
    while queue:
        table, edges, visited_tables, used_edge_indices = queue.pop(0)
        if best_edges is not None and len(edges) >= len(best_edges):
            continue
        for edge in adjacency.get(table, []):
            edge_index = edge.get("edge_index")
            next_table = edge.get("right_table")
            if not isinstance(edge_index, int) or not isinstance(next_table, str):
                continue
            if edge_index in used_edge_indices or next_table in visited_tables:
                continue
            next_edges = edges + [edge]
            if next_table == goal:
                score = _join_path_score(next_edges)
                if best_edges is None or len(next_edges) < len(best_edges) or score > best_score:
                    best_edges = next_edges
                    best_score = score
                continue
            queue.append(
                (
                    next_table,
                    next_edges,
                    visited_tables | {next_table},
                    used_edge_indices | {edge_index},
                )
            )
    if not best_edges:
        return {}
    tables = [start] + [
        str(edge.get("right_table"))
        for edge in best_edges
        if isinstance(edge.get("right_table"), str)
    ]
    joins = []
    for edge in best_edges:
        joins.append(
            {
                "left_table": edge.get("left_table"),
                "left_column": edge.get("left_column"),
                "right_table": edge.get("right_table"),
                "right_column": edge.get("right_column"),
            }
        )
    return {
        "tables": tables,
        "joins": joins,
        "length": len(joins),
        "score": round(best_score, 3),
    }


def _join_path_score(edges: list[dict[str, Any]]) -> float:
    scores = [
        float(edge.get("score") or 0.0)
        for edge in edges
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _join_table_profile(source: dict[str, Any]) -> dict[str, Any]:
    schema = source.get("schema")
    columns = schema.get("columns") if isinstance(schema, dict) else None
    if not isinstance(columns, list):
        columns = []
    selected_columns = [str(column) for column in columns[:MAX_JOIN_PROFILE_COLUMNS]]
    values: dict[str, set[str]] = {column: set() for column in selected_columns}
    path = Path(str(source["path"]))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if row_index >= MAX_JOIN_SAMPLE_ROWS:
                    break
                if not isinstance(row, dict):
                    continue
                for column in selected_columns:
                    normalized = _normalize_join_value(row.get(column))
                    if normalized is not None:
                        values[column].add(normalized)
    except OSError:
        values = {column: set() for column in selected_columns}
    return {
        "id": str(source.get("id") or ""),
        "columns": selected_columns,
        "values": values,
    }


def _join_candidates_for_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    hint_roles: dict[str, set[str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    left_values_by_column = left.get("values") if isinstance(left.get("values"), dict) else {}
    right_values_by_column = right.get("values") if isinstance(right.get("values"), dict) else {}
    for left_column in left.get("columns", []):
        left_values = left_values_by_column.get(left_column)
        if not isinstance(left_values, set) or not left_values:
            continue
        for right_column in right.get("columns", []):
            right_values = right_values_by_column.get(right_column)
            if not isinstance(right_values, set) or not right_values:
                continue
            overlap_values = left_values & right_values
            overlap = len(overlap_values)
            if overlap < MIN_JOIN_VALUE_OVERLAP:
                continue
            left_coverage = overlap / max(1, len(left_values))
            right_coverage = overlap / max(1, len(right_values))
            value_score = (left_coverage + right_coverage) / 2
            name_score = _column_name_similarity(left_column, right_column)
            hint_score = _join_hint_score(
                left_column,
                right_column,
                hint_roles=hint_roles,
            )
            cross_entity_id_overlap = _is_cross_entity_id_overlap(
                left_column,
                right_column,
            )
            score = 0.80 * value_score + 0.18 * name_score + 0.02 * hint_score
            if cross_entity_id_overlap:
                # Independent entity ids in normalized tables often share the
                # small integers 1, 2, 3, ... without representing a valid
                # relationship.  Keep exact/same-entity id matches such as
                # party.Party_ID -> party_host.Party_ID, but demote Party_ID
                # -> Host_ID style overlaps so they do not distract planning.
                score *= 0.35
            if score < MIN_JOIN_CANDIDATE_SCORE:
                continue
            reasons = ["value_overlap"]
            if name_score >= 0.7:
                reasons.append("column_name_similarity")
            if hint_score >= 0.7 and name_score >= 0.35 and not cross_entity_id_overlap:
                reasons.append("primary_foreign_key_hint")
            output.append(
                {
                    "left_table": left.get("id"),
                    "left_column": left_column,
                    "right_table": right.get("id"),
                    "right_column": right_column,
                    "score": round(score, 3),
                    "overlap": overlap,
                    "left_coverage": round(left_coverage, 3),
                    "right_coverage": round(right_coverage, 3),
                    "evidence": reasons,
                    "sample_matches": sorted(overlap_values)[:5],
                }
            )
    return output


def _normalize_join_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "nan", "na", "n/a"}:
        return None
    numeric = _canonical_number(text)
    if numeric is not None:
        return numeric
    return re.sub(r"\s+", " ", text.casefold())


def _canonical_number(text: str) -> str | None:
    compact = text.replace(",", "").strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
        return None
    try:
        number = float(compact)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def _column_name_similarity(left: str, right: str) -> float:
    left_norm = _normalize_join_name(left)
    right_norm = _normalize_join_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _join_hint_roles(hints: Any) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    if not isinstance(hints, dict):
        return roles
    for key, role in (("primary_keys", "primary"), ("foreign_keys", "foreign")):
        values = hints.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            normalized = _normalize_join_name(str(value))
            if normalized:
                roles.setdefault(normalized, set()).add(role)
    return roles


def _join_hint_score(
    left_column: str,
    right_column: str,
    *,
    hint_roles: dict[str, set[str]],
) -> float:
    left_roles = _roles_for_column(left_column, hint_roles)
    right_roles = _roles_for_column(right_column, hint_roles)
    left_primary_only = "primary" in left_roles and "foreign" not in left_roles
    right_primary_only = "primary" in right_roles and "foreign" not in right_roles
    left_foreign_only = "foreign" in left_roles and "primary" not in left_roles
    right_foreign_only = "foreign" in right_roles and "primary" not in right_roles
    if (
        (left_primary_only and right_foreign_only)
        or (left_foreign_only and right_primary_only)
    ):
        return 1.0
    if "primary" in left_roles and "primary" in right_roles:
        return 0.25
    if "foreign" in left_roles and "foreign" in right_roles:
        return 0.15
    return 0.0


def _roles_for_column(column: str, hint_roles: dict[str, set[str]]) -> set[str]:
    normalized = _normalize_join_name(column)
    output: set[str] = set()
    for hint, roles in hint_roles.items():
        if not hint:
            continue
        if normalized == hint or normalized.endswith(hint) or hint.endswith(normalized):
            output.update(roles)
    return output


def _normalize_join_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _is_cross_entity_id_overlap(left_column: str, right_column: str) -> bool:
    left_entity = _id_column_entity(left_column)
    right_entity = _id_column_entity(right_column)
    return bool(left_entity and right_entity and left_entity != right_entity)


def _id_column_entity(column: str) -> str:
    normalized = _normalize_join_name(column)
    if not normalized.endswith("id"):
        return ""
    entity = normalized[:-2]
    for suffix in ("identifier", "number", "num", "code", "key"):
        if entity.endswith(suffix):
            entity = entity[: -len(suffix)]
            break
    return entity


def _preprocess_pdf_source(
    *,
    source: dict[str, Any],
    source_id: str,
    resource_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if resource_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF document sources are supported now: {resource_path}")

    metadata = _source_metadata(source)
    extracted = extract_pdf_schema(resource_path)
    schema = extracted["schema"]

    local_source = {
        "id": source_id,
        "original_id": source.get("id"),
        "type": "document",
        "source_type": "pdf",
        "file": source["file"],
        "path": str(resource_path),
        "format": "pdf",
        "schema": schema,
        **metadata,
    }

    remote_source = {
        "id": source_id,
        "type": "document",
        "source_type": "pdf",
        "format": "pdf",
        "schema": copy.deepcopy(schema),
        **metadata,
    }

    source_map = {
        "original_id": source.get("id"),
        "type": "document",
        "source_type": "pdf",
        "file": source["file"],
        "path": str(resource_path),
        "format": "pdf",
        "resource_cache": copy.deepcopy(extracted["resource_cache"]),
        **metadata,
    }
    source_context = {
        "chunk_map": extracted["chunk_map"],
        "resource_cache": extracted["resource_cache"],
    }
    return local_source, remote_source, source_map, source_context


def _merge_source_context(
    context: dict[str, Any],
    source_id: str,
    source_context: dict[str, Any],
) -> None:
    if "chunk_map" in source_context:
        context.setdefault("chunk_map", {})[source_id] = source_context["chunk_map"]
    if "resource_cache" in source_context:
        context.setdefault("resource_cache", {})[source_id] = source_context[
            "resource_cache"
        ]


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    reserved_keys = {
        "file",
        "format",
        "id",
        "original_id",
        "path",
        "schema",
        "source_type",
        "type",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key not in reserved_keys and _is_prompt_safe_scalar(value)
    }


def _is_prompt_safe_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _normal_source_id(source_type: str, source_index: int) -> str:
    return f"{source_type}_{source_index}"


def _resolve_source_path(file_name: str, base_dir: Path) -> Path:
    # Task DSL paths are resolved relative to the dataset/case directory.
    path = Path(file_name).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source file not found: {resolved}")
    return resolved
