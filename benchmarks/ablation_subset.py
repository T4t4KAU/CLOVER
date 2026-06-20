"""Build fixed, mechanism-stratified subsets for CLOVER ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

WIKITQ_WEIGHTS = {
    "aggregation": 20,
    "comparison": 15,
    "direct_lookup": 15,
    "extrema": 15,
    "multi_answer": 15,
    "temporal_order": 15,
    "boolean": 5,
}

TABLEBENCH_WEIGHTS = {
    "FactChecking/MatchBased": 15,
    "FactChecking/Multi-hop FactChecking": 15,
    "NumericalReasoning/Aggregation": 9,
    "NumericalReasoning/ArithmeticCalculation": 9,
    "NumericalReasoning/Comparison": 9,
    "NumericalReasoning/Counting": 9,
    "NumericalReasoning/Multi-hop NumericalReasoing": 9,
    "NumericalReasoning/Ranking": 9,
    "NumericalReasoning/Domain-Specific": 8,
    "NumericalReasoning/Time-basedCalculation": 8,
}

TABLEBENCH_QTYPES = frozenset({"FactChecking", "NumericalReasoning"})

WIKITQ_EDGE_OPPORTUNITY_WEIGHTS = {
    "field_selection": 30,
    "value_normalization": 15,
    "list_assembly": 15,
    "candidate_selection": 25,
    "deterministic_control": 15,
}

TABLEBENCH_EDGE_OPPORTUNITY_WEIGHTS = {
    "field_selection": 30,
    "value_normalization": 10,
    "candidate_selection": 30,
    "deterministic_control": 30,
}

SELECTION_POLICIES = frozenset({"representative", "edge_opportunity"})

_DETERMINISTIC_OPERATION_RE = re.compile(
    r"\b(how many|number of|count|total|sum|average|mean|difference|"
    r"most|least|highest|lowest|maximum|minimum|top|bottom|rank|"
    r"more|less|greater|fewer|higher|lower|above|below|longer|shorter|"
    r"older|younger|before|after|earliest|latest|first|last|"
    r"sort|order|ratio|percent(?:age)?)\b",
    re.IGNORECASE,
)

_FORMAT_RISK_RE = re.compile(
    r"""[%$£€¥]|["'“”‘’].+["'“”‘’]|\([^)]*\)|"""
    r"\b\d+(?:\.\d+)?\s*(?:km|kg|lb|lbs|miles?|years?|points?|"
    r"minutes?|seconds?|cm|mm|m)\b|^[^:=]{1,40}\s*[:=]\s*\S",
    re.IGNORECASE,
)


def build_ablation_subset(
    *,
    dataset: str,
    dataset_root: Path,
    output_path: Path,
    size: int = 100,
    seed: int = 20260619,
    selection_policy: str = "representative",
) -> dict[str, Any]:
    """Write one deterministic JSONL manifest and its summary."""

    if size <= 0:
        raise ValueError("size must be positive")
    normalized = _normalize_dataset(dataset)
    policy = _normalize_selection_policy(selection_policy)
    cases = _load_cases(dataset_root)
    if normalized == "tablebench":
        cases = [
            case
            for case in cases
            if str(case.get("qtype") or "") in TABLEBENCH_QTYPES
        ]
        case_id_counts = Counter(str(case.get("case_id") or "") for case in cases)
        cases = [
            case
            for case in cases
            if case_id_counts[str(case.get("case_id") or "")] == 1
        ]
        if policy == "edge_opportunity":
            assign = lambda case: _edge_opportunity_stratum(  # noqa: E731
                case,
                dataset=normalized,
            )
            weights = TABLEBENCH_EDGE_OPPORTUNITY_WEIGHTS
        else:
            assign = _tablebench_stratum
            weights = TABLEBENCH_WEIGHTS
    else:
        if policy == "edge_opportunity":
            assign = lambda case: _edge_opportunity_stratum(  # noqa: E731
                case,
                dataset=normalized,
            )
            weights = WIKITQ_EDGE_OPPORTUNITY_WEIGHTS
        else:
            assign = _wikitq_stratum
            weights = WIKITQ_WEIGHTS

    selected, quotas = _stratified_select(
        cases=cases,
        assign=assign,
        weights=weights,
        size=size,
        seed=seed,
        dataset=normalized,
    )
    records = [
        _manifest_record(
            case,
            dataset=normalized,
            stratum=assign(case),
            selection_policy=policy,
        )
        for case in selected
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, records)
    summary = {
        "dataset": normalized,
        "selection_policy": f"{policy}_outcome_blind",
        "selection_basis": [
            "question_text",
            "answer_type",
            "dataset_metadata",
            "table_shape",
            "cell_representation",
        ],
        "uses_model_predictions": False,
        "uses_answer_correctness": False,
        "size": len(records),
        "requested_size": size,
        "seed": seed,
        "dataset_root": dataset_root.expanduser().as_posix(),
        "manifest": output_path.expanduser().as_posix(),
        "eligible_cases": len(cases),
        "quotas": quotas,
        "strata": dict(sorted(Counter(record["stratum"] for record in records).items())),
        "answer_types": dict(
            sorted(
                Counter(
                    str(record.get("answer_type") or "unknown")
                    for record in records
                ).items()
            )
        ),
        "mechanism_tags": dict(
            sorted(
                Counter(
                    tag
                    for record in records
                    for tag in record.get("mechanism_tags", [])
                ).items()
            )
        ),
        "unique_tables": len({record["dataset_id"] for record in records}),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def read_manifest_case_ids(path: Path) -> list[str]:
    """Read case ids from a generated manifest."""

    return [
        str(record["case_id"])
        for record in _read_jsonl(path)
    ]


def _stratified_select(
    *,
    cases: list[dict[str, Any]],
    assign: Callable[[dict[str, Any]], str],
    weights: dict[str, int],
    size: int,
    seed: int,
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        pools[assign(case)].append(case)
    quotas = _scaled_quotas(weights, size)
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()
    used_case_ids: set[str] = set()
    used_tables: set[str] = set()

    for stratum, quota in quotas.items():
        pool = sorted(
            pools.get(stratum, []),
            key=lambda case: _stable_rank(case, seed=seed, dataset=dataset),
        )
        picked = _pick_diverse(
            pool,
            quota=quota,
            used_keys=used_keys,
            used_case_ids=used_case_ids,
            used_tables=used_tables,
        )
        selected.extend(picked)

    if len(selected) < size:
        remaining = sorted(
            cases,
            key=lambda case: _stable_rank(case, seed=seed + 1, dataset=dataset),
        )
        selected.extend(
            _pick_diverse(
                remaining,
                quota=size - len(selected),
                used_keys=used_keys,
                used_case_ids=used_case_ids,
                used_tables=used_tables,
            )
        )
    if len(selected) != size:
        raise ValueError(
            f"Unable to select {size} unique {dataset} cases; selected {len(selected)}"
        )
    return selected, quotas


def _pick_diverse(
    pool: list[dict[str, Any]],
    *,
    quota: int,
    used_keys: set[tuple[str, str]],
    used_case_ids: set[str],
    used_tables: set[str],
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for prefer_new_table in (True, False):
        for case in pool:
            key = _case_key(case)
            case_id = key[1]
            table_id = key[0]
            if key in used_keys or case_id in used_case_ids:
                continue
            if prefer_new_table and table_id in used_tables:
                continue
            picked.append(case)
            used_keys.add(key)
            used_case_ids.add(case_id)
            used_tables.add(table_id)
            if len(picked) >= quota:
                return picked
    return picked


def _scaled_quotas(weights: dict[str, int], size: int) -> dict[str, int]:
    total = sum(weights.values())
    raw = {
        key: size * weight / total
        for key, weight in weights.items()
    }
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = size - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda key: (raw[key] - quotas[key], weights[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _wikitq_stratum(case: dict[str, Any]) -> str:
    question = str(case.get("question") or "").strip().lower()
    answer = case.get("answer")
    answer_type = str(case.get("type") or "")
    if (
        isinstance(answer, list)
        and len(answer) > 1
    ) or answer_type.startswith("list"):
        return "multi_answer"
    if re.search(
        r"\b(more|less|higher|lower|greater|fewer|difference|between|"
        r"compared|longer|shorter|older|younger|same|equal)\b",
        question,
    ):
        return "comparison"
    if re.search(
        r"\b(how many|number of|total|sum|average|mean|combined|"
        r"altogether|percentage|percent)\b",
        question,
    ):
        return "aggregation"
    if re.search(
        r"\b(most|least|highest|lowest|maximum|minimum|top|largest|"
        r"smallest|best|worst)\b",
        question,
    ):
        return "extrema"
    if re.search(
        r"\b(before|after|previous|next|first|last|earliest|latest|"
        r"prior|following|year|date|season|when)\b",
        question,
    ):
        return "temporal_order"
    if re.match(
        r"^(is|are|was|were|did|does|do|has|have|had|can|could|would|will)\b",
        question,
    ):
        return "boolean"
    return "direct_lookup"


def _tablebench_stratum(case: dict[str, Any]) -> str:
    return (
        f"{str(case.get('qtype') or 'unknown')}/"
        f"{str(case.get('qsubtype') or 'unknown')}"
    )


def _manifest_record(
    case: dict[str, Any],
    *,
    dataset: str,
    stratum: str,
    selection_policy: str,
) -> dict[str, Any]:
    features = case.get("_selection_features")
    public_features = {}
    if isinstance(features, dict):
        public_features = {
            key: features[key]
            for key in (
                "table_rows",
                "table_columns",
                "matched_body_cells",
                "matched_format_risk_cells",
                "requires_deterministic_operation",
            )
            if key in features
        }
    record = {
        "dataset": dataset,
        "dataset_id": str(case.get("dataset_id") or ""),
        "case_id": str(case.get("case_id") or case.get("id") or ""),
        "question": case.get("question"),
        "answer_type": case.get("type"),
        "stratum": stratum,
        "selection_policy": selection_policy,
        "mechanism_tags": _mechanism_tags(
            dataset,
            stratum,
            selection_policy=selection_policy,
        ),
        "selection_features": public_features,
    }
    for field in ("qtype", "qsubtype", "split"):
        if case.get(field) is not None:
            record[field] = case[field]
    return record


def _mechanism_tags(
    dataset: str,
    stratum: str,
    *,
    selection_policy: str,
) -> list[str]:
    if selection_policy == "edge_opportunity":
        mapping = {
            "field_selection": [
                "edge_local_review",
                "field_selection",
                "static_fallback",
            ],
            "value_normalization": [
                "edge_local_review",
                "value_normalization",
                "contract_gate",
            ],
            "list_assembly": [
                "edge_local_review",
                "list_assembly",
                "contract_gate",
            ],
            "candidate_selection": [
                "edge_local_review",
                "candidate_selection",
                "node_review",
            ],
            "deterministic_control": [
                "static_finalization",
                "cloud_recovery",
                "negative_control",
            ],
        }
        return mapping[stratum]
    if dataset == "tablebench":
        if stratum == "FactChecking/MatchBased":
            return ["static_finalization", "local_semantic_binding"]
        if stratum == "FactChecking/Multi-hop FactChecking":
            return ["edge_local_review", "cloud_recovery"]
        subtype = stratum.partition("/")[2]
        if subtype in {"Aggregation", "ArithmeticCalculation", "Counting"}:
            return ["static_finalization", "contract_gate"]
        if subtype in {"Comparison", "Ranking", "Time-basedCalculation"}:
            return ["node_review", "cloud_recovery"]
        return ["contract_gate", "cloud_recovery"]
    mapping = {
        "aggregation": ["static_finalization", "contract_gate"],
        "boolean": ["contract_gate", "edge_local_review"],
        "comparison": ["node_review", "cloud_recovery"],
        "direct_lookup": ["static_finalization", "local_semantic_binding"],
        "extrema": ["node_review", "cloud_recovery"],
        "multi_answer": ["contract_gate", "cloud_synthesis"],
        "temporal_order": ["node_review", "cloud_recovery"],
    }
    return mapping[stratum]


def _load_cases(root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(root.expanduser().resolve().glob("*/cases.jsonl")):
        table_profile = _load_table_profile(path.parent / "table.csv")
        for record in _read_jsonl(path):
            payload = dict(record)
            payload.setdefault("dataset_id", path.parent.name)
            payload["_selection_features"] = _selection_features(
                payload,
                table_profile=table_profile,
            )
            cases.append(payload)
    if not cases:
        raise FileNotFoundError(f"No converted cases found under: {root}")
    return cases


def _load_table_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "table_rows": 0,
            "table_columns": 0,
            "body_values": (),
        }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    columns = max((len(row) for row in rows), default=0)
    body_values = tuple(
        str(value).strip()
        for row in rows[1:]
        for value in row
        if str(value).strip()
    )
    return {
        "table_rows": max(0, len(rows) - 1),
        "table_columns": columns,
        "body_values": body_values,
    }


def _selection_features(
    case: dict[str, Any],
    *,
    table_profile: dict[str, Any],
) -> dict[str, Any]:
    question = str(case.get("question") or "")
    normalized_question = f" {_normalize_phrase(question)} "
    matched_values: list[str] = []
    format_risk_values: list[str] = []
    for value in table_profile.get("body_values", ()):
        normalized_value = _normalize_phrase(value)
        if not _eligible_cell_mention(normalized_value):
            continue
        if f" {normalized_value} " not in normalized_question:
            continue
        matched_values.append(value)
        if _FORMAT_RISK_RE.search(value):
            format_risk_values.append(value)
    return {
        "table_rows": int(table_profile.get("table_rows", 0) or 0),
        "table_columns": int(table_profile.get("table_columns", 0) or 0),
        "matched_body_cells": len(matched_values),
        "matched_format_risk_cells": len(format_risk_values),
        "requires_deterministic_operation": bool(
            _DETERMINISTIC_OPERATION_RE.search(question)
        ),
    }


def _edge_opportunity_stratum(case: dict[str, Any], *, dataset: str) -> str:
    features = case.get("_selection_features")
    if not isinstance(features, dict):
        features = {}
    deterministic = bool(features.get("requires_deterministic_operation"))
    answer_type = str(case.get("type") or "").strip().lower()
    if dataset == "tablebench":
        if (
            str(case.get("qtype") or "") == "FactChecking"
            and int(features.get("matched_format_risk_cells", 0) or 0) > 0
        ):
            return "value_normalization"
        subtype = str(case.get("qsubtype") or "")
        if subtype == "MatchBased":
            return "field_selection"
        if subtype == "Multi-hop FactChecking":
            return "candidate_selection"
        return "deterministic_control"

    if answer_type.startswith("list") and not deterministic:
        return "list_assembly"
    if (
        not deterministic
        and int(features.get("matched_format_risk_cells", 0) or 0) > 0
    ):
        return "value_normalization"
    if not deterministic and int(features.get("matched_body_cells", 0) or 0) > 0:
        return "field_selection"
    if (
        not deterministic
        and 0 < int(features.get("table_rows", 0) or 0) <= 15
        and 0 < int(features.get("table_columns", 0) or 0) <= 10
    ):
        return "candidate_selection"
    return "deterministic_control"


def _normalize_phrase(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _eligible_cell_mention(value: str) -> bool:
    if len(value) >= 4:
        return True
    return bool(re.fullmatch(r"\d{4}", value))


def _stable_rank(case: dict[str, Any], *, seed: int, dataset: str) -> str:
    key = "::".join((*_case_key(case), str(seed), dataset))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _case_key(case: dict[str, Any]) -> tuple[str, str]:
    return (
        str(case.get("dataset_id") or ""),
        str(case.get("case_id") or case.get("id") or ""),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_dataset(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"wikitq", "wikitablequestions"}:
        return "wikitq"
    if normalized == "tablebench":
        return normalized
    raise ValueError(f"Unsupported ablation dataset: {value!r}")


def _normalize_selection_policy(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "mechanism_stratified": "representative",
        "edge": "edge_opportunity",
        "opportunity": "edge_opportunity",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SELECTION_POLICIES:
        available = ", ".join(sorted(SELECTION_POLICIES))
        raise ValueError(
            f"Unsupported selection policy: {value!r}. Available: {available}"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fixed mechanism-stratified ablation subset."
    )
    parser.add_argument("--dataset", required=True, choices=("tablebench", "wikitq"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument(
        "--selection-policy",
        choices=tuple(sorted(SELECTION_POLICIES)),
        default="representative",
    )
    parser.add_argument("--print-case-ids", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_case_ids:
        for case_id in read_manifest_case_ids(args.output):
            print(case_id)
        return 0
    summary = build_ablation_subset(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        output_path=args.output,
        size=args.size,
        seed=args.seed,
        selection_policy=args.selection_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
