"""Build fixed, mechanism-stratified subsets for CLOVER ablations."""

from __future__ import annotations

import argparse
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


def build_ablation_subset(
    *,
    dataset: str,
    dataset_root: Path,
    output_path: Path,
    size: int = 100,
    seed: int = 20260619,
) -> dict[str, Any]:
    """Write one deterministic JSONL manifest and its summary."""

    if size <= 0:
        raise ValueError("size must be positive")
    normalized = _normalize_dataset(dataset)
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
        assign = _tablebench_stratum
        weights = TABLEBENCH_WEIGHTS
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
        )
        for case in selected
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, records)
    summary = {
        "dataset": normalized,
        "selection_policy": "mechanism_stratified_outcome_blind",
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
) -> dict[str, Any]:
    record = {
        "dataset": dataset,
        "dataset_id": str(case.get("dataset_id") or ""),
        "case_id": str(case.get("case_id") or case.get("id") or ""),
        "question": case.get("question"),
        "answer_type": case.get("type"),
        "stratum": stratum,
        "mechanism_tags": _mechanism_tags(dataset, stratum),
    }
    for field in ("qtype", "qsubtype", "split"):
        if case.get(field) is not None:
            record[field] = case[field]
    return record


def _mechanism_tags(dataset: str, stratum: str) -> list[str]:
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
        for record in _read_jsonl(path):
            payload = dict(record)
            payload.setdefault("dataset_id", path.parent.name)
            cases.append(payload)
    if not cases:
        raise FileNotFoundError(f"No converted cases found under: {root}")
    return cases


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fixed mechanism-stratified ablation subset."
    )
    parser.add_argument("--dataset", required=True, choices=("tablebench", "wikitq"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260619)
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
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
