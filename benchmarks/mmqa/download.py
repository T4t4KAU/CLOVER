"""Convert a local MMQA multi-table release into CLOVER's table layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.mmqa.metrics import flatten_mmqa_answer, parse_number


DEFAULT_SOURCE_ROOT = Path("datasets") / "mmqa_source"
DEFAULT_OUTPUT_ROOT = Path("datasets") / "mmqa"

# Map source file names to split labels. MMQA's multi-table subset ships as
# Synthesized_two_table.json / Synthesized_three_table.json.
SPLIT_FILE_MAP = {
    "two_table": "Synthesized_two_table.json",
    "three_table": "Synthesized_three_table.json",
}
DEFAULT_SPLITS = ("two_table", "three_table")


def download_and_convert_mmqa(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    splits: Sequence[str] = DEFAULT_SPLITS,
    case_ids: Sequence[int] | None = None,
    limit_cases: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert an existing MMQA multi-table release into CLOVER-ready folders.

    Each unique table-set becomes one dataset directory holding the shared
    ``table_1.csv``/``table_2.csv``/... files plus a ``cases.jsonl`` of every
    question that uses those tables. This mirrors WikiTQ's layout (one dir per
    table, many cases) and avoids duplicating CSVs for synthesized data where
    many cases share the same underlying tables.
    """

    source_path = _resolve_path(source_root)
    output_path = _resolve_path(output_root)
    if not source_path.is_dir():
        raise FileNotFoundError(f"MMQA source root not found: {source_path}")
    if limit_cases is not None and limit_cases <= 0:
        raise ValueError("limit_cases must be positive")

    normalized_splits = _normalize_splits(splits)
    selected_case_ids = set(case_ids) if case_ids is not None else None

    output_path.mkdir(parents=True, exist_ok=True)

    split_summaries: list[dict[str, Any]] = []
    total_cases = 0
    total_datasets = 0
    answer_type_counts: dict[str, int] = defaultdict(int)
    dict_answer_count = 0

    for split in normalized_splits:
        source_file = source_path / SPLIT_FILE_MAP[split]
        if not source_file.is_file():
            raise FileNotFoundError(f"MMQA split file not found: {source_file}")
        with source_file.open("r", encoding="utf-8") as handle:
            raw_cases = json.load(handle)
        if not isinstance(raw_cases, list):
            raise ValueError(f"MMQA split file is not a list: {source_file}")

        selected = _select_cases(
            raw_cases,
            case_ids=selected_case_ids,
            limit_cases=limit_cases,
        )

        split_dir = output_path / split
        if split_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    "Output split directory already exists: "
                    f"{_portable_path(split_dir)}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        grouped = _group_cases_by_tables(selected)
        dataset_summaries: list[dict[str, Any]] = []
        split_case_count = 0

        for table_hash, table_cases in sorted(grouped.items()):
            dataset_id = f"mmqa_{_split_short(split)}_{table_hash}"
            dataset_dir = split_dir / dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)

            first_case = table_cases[0]
            table_names = list(first_case.get("table_names") or [])
            tables = first_case.get("tables") or []
            source_files: list[str] = []
            for index, table in enumerate(tables, start=1):
                source_file_name = f"table_{index}.csv"
                source_files.append(source_file_name)
                _write_table_csv(table, dataset_dir / source_file_name)

            case_records: list[dict[str, Any]] = []
            for case_index, case in enumerate(table_cases):
                record = _case_record(
                    case=case,
                    dataset_id=dataset_id,
                    split=split,
                    table_names=table_names,
                    source_files=source_files,
                    case_index=case_index,
                )
                case_records.append(record)
                answer_type_counts[str(record["type"])] += 1
                if isinstance(case.get("answer"), dict):
                    dict_answer_count += 1

            _write_jsonl(dataset_dir / "cases.jsonl", case_records)
            split_case_count += len(case_records)
            dataset_summaries.append(
                {
                    "dataset_id": dataset_id,
                    "case_count": len(case_records),
                    "table_count": len(tables),
                    "table_names": table_names,
                    "source_files": source_files,
                }
            )

        total_cases += split_case_count
        total_datasets += len(dataset_summaries)
        split_summaries.append(
            {
                "split": split,
                "source_file": _portable_path(source_file),
                "source_case_count": len(raw_cases),
                "selected_case_count": len(selected),
                "dataset_count": len(dataset_summaries),
                "case_count": split_case_count,
                "datasets": dataset_summaries,
            }
        )

    summary = {
        "stage": "mmqa_local_conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": _portable_path(source_path),
        "output_root": _portable_path(output_path),
        "splits": normalized_splits,
        "dataset_count": total_datasets,
        "case_count": total_cases,
        "dict_answer_cases": dict_answer_count,
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "split_summaries": split_summaries,
    }
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local MMQA multi-table release for CLOVER eval."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split labels: two_table, three_table.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="MMQA case id (integer) to include. Repeat or comma-separate.",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = download_and_convert_mmqa(
        source_root=args.source_root,
        output_root=args.output_root,
        splits=_expand_csv_values([args.splits]),
        case_ids=_expand_int_values(args.case_id),
        limit_cases=args.limit_cases,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _normalize_splits(splits: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for split in splits:
        label = str(split or "").strip()
        if not label:
            continue
        if label not in SPLIT_FILE_MAP:
            raise ValueError(
                f"Unsupported MMQA split: {label!r}. "
                f"Expected one of {sorted(SPLIT_FILE_MAP)}"
            )
        normalized.append(label)
    if not normalized:
        raise ValueError("MMQA conversion requires at least one split")
    return normalized


def _split_short(split: str) -> str:
    return split.replace("_table", "")


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: set[int] | None,
    limit_cases: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for case in cases:
        case_id = _case_id_value(case)
        if case_id is not None:
            seen_ids.add(case_id)
        if case_ids is not None and (case_id is None or case_id not in case_ids):
            continue
        _required_field(case, "Question")
        _required_field(case, "tables")
        _required_field(case, "answer")
        selected.append(case)
        if limit_cases is not None and len(selected) >= limit_cases:
            break
    if case_ids is not None:
        missing = sorted(case_ids - seen_ids)
        if missing:
            raise ValueError(f"Requested MMQA case ids not found: {missing}")
    return selected


def _group_cases_by_tables(
    cases: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        table_hash = _table_set_hash(case.get("tables") or [])
        grouped[table_hash].append(case)
    return dict(grouped)


def _table_set_hash(tables: list[dict[str, Any]]) -> str:
    payload = json.dumps(tables, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _case_record(
    *,
    case: dict[str, Any],
    dataset_id: str,
    split: str,
    table_names: list[str],
    source_files: list[str],
    case_index: int,
) -> dict[str, Any]:
    raw_answer = case.get("answer")
    flat_answer = flatten_mmqa_answer(raw_answer)
    case_id_value = _case_id_value(case)
    case_id = (
        f"mmqa_{_split_short(split)}_{case_id_value:06d}"
        if case_id_value is not None
        else f"mmqa_{_split_short(split)}_{dataset_id}_{case_index:04d}"
    )
    return {
        "case_id": case_id,
        "original_id": case_id_value,
        "dataset_id": dataset_id,
        "question": str(case.get("Question") or "").strip(),
        "answer": flat_answer,
        "answer_raw": raw_answer,
        "type": _infer_answer_type(flat_answer),
        "table_names": list(table_names),
        "foreign_keys": list(case.get("foreign_keys") or []),
        "primary_keys": list(case.get("primary_keys") or []),
        "gold_sql": case.get("SQL"),
        "source_files": list(source_files),
        "table_count": len(source_files),
        "split": split,
    }


def _write_table_csv(table: dict[str, Any], target: Path) -> tuple[int, int]:
    columns = list(table.get("table_columns") or [])
    rows = list(table.get("table_content") or [])
    if not columns:
        raise ValueError(f"MMQA table has no columns: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell_to_str(cell) for cell in row])
    return len(rows), len(columns)


def _cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        # Drop trailing ".0" for integer-valued floats so "3.0" aligns with
        # the string "3" used in questions and gold answers.
        if value == int(value):
            return str(int(value))
        return repr(value)
    return str(value)


def _infer_answer_type(answer: list[str]) -> str:
    if len(answer) > 1:
        if answer and all(parse_number(value) is not None for value in answer):
            return "list[number]"
        return "list[string]"
    if answer and parse_number(answer[0]) is not None:
        return "number"
    return "string"


def _case_id_value(case: dict[str, Any]) -> int | None:
    value = case.get("id_")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _required_field(case: dict[str, Any], field: str) -> None:
    if field not in case or case[field] is None:
        raise ValueError(f"MMQA case missing required field: {field}")


def _expand_csv_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    return expanded


def _expand_int_values(values: Sequence[str] | None) -> list[int] | None:
    expanded = _expand_csv_values(values)
    if not expanded:
        return None
    result: list[int] = []
    for value in expanded:
        try:
            result.append(int(value))
        except ValueError as exc:
            raise ValueError(f"MMQA case id must be an integer: {value!r}") from exc
    return result


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _portable_path(path: Path, *, base: Path | None = None) -> str:
    resolved = path.expanduser().resolve()
    base_path = (base or Path.cwd()).expanduser().resolve()
    try:
        return resolved.relative_to(base_path).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
