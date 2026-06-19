"""Convert a local Table-Fact-Checking release into CLOVER's table layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path("datasets") / "tablefact_source"
DEFAULT_OUTPUT_ROOT = Path("datasets") / "tablefact"
DEFAULT_SPLITS = ("test",)
SUPPORTED_SPLITS = frozenset(
    {"train", "dev", "val", "test", "simple_test", "complex_test", "small_test"}
)
SPLIT_DATA_FILES = {
    "train": "train_examples.json",
    "dev": "val_examples.json",
    "test": "test_examples.json",
}
SUBSET_ID_FILES = {
    "simple_test": "simple_test_id.json",
    "complex_test": "complex_test_id.json",
    "small_test": "small_test_id.json",
}
CAPTION_USAGE = (
    "This is the Wikipedia page or table caption. Use it to resolve entities "
    "mentioned by the statement, but do not treat it as a table row."
)


def convert_tablefact_release(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    splits: Sequence[str] = DEFAULT_SPLITS,
    case_ids: Sequence[str] | None = None,
    limit_cases: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert official TableFact JSON splits and hash-delimited tables."""

    source_path = _resolve_path(source_root)
    output_path = _resolve_path(output_root)
    if not source_path.is_dir():
        raise FileNotFoundError(f"TableFact source root not found: {source_path}")
    if limit_cases is not None and limit_cases <= 0:
        raise ValueError("limit_cases must be positive")

    selected_splits = _normalize_splits(splits)
    requested_case_ids = _normalize_ids(case_ids)
    test_subsets = _load_test_subsets(source_path)
    rows = _load_selected_rows(
        source_path=source_path,
        splits=selected_splits,
        test_subsets=test_subsets,
    )
    selected_rows = _select_rows(
        rows,
        case_ids=requested_case_ids,
        limit_cases=limit_cases,
    )
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped_rows[row["table_file"]].append(row)

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_summaries = []
    split_counts: dict[str, int] = defaultdict(int)
    subset_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)

    for table_file, table_rows in sorted(grouped_rows.items()):
        dataset_id = _dataset_id(table_file)
        dataset_dir = output_path / dataset_id
        if dataset_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output dataset already exists: {dataset_dir}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        source_table = source_path / "data" / "all_csv" / table_file
        if not source_table.is_file():
            raise FileNotFoundError(f"TableFact table not found: {source_table}")
        row_count, column_count = _write_normalized_csv(
            source_table,
            dataset_dir / "table.csv",
        )

        case_records = []
        for case_index, row in enumerate(table_rows):
            record = _case_record(
                row=row,
                dataset_id=dataset_id,
                case_index=case_index,
            )
            case_records.append(record)
            split_counts[record["split"]] += 1
            subset_counts[record["qsubtype"]] += 1
            label_counts["entailed" if record["answer"] else "refuted"] += 1
        _write_jsonl(dataset_dir / "cases.jsonl", case_records)

        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "source_table": table_file,
                "case_count": len(case_records),
                "table_csv": f"{dataset_id}/table.csv",
                "rows": row_count,
                "columns": column_count,
            }
        )

    summary = {
        "stage": "tablefact_local_conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_official_name": "TabFact",
        "source_root": _portable_path(source_path),
        "output_root": _portable_path(output_path),
        "splits": list(selected_splits),
        "dataset_count": len(dataset_summaries),
        "case_count": len(selected_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "subset_counts": dict(sorted(subset_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "datasets": dataset_summaries,
    }
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local Table-Fact-Checking release for CLOVER eval."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=None,
        choices=tuple(sorted(SUPPORTED_SPLITS)),
        help="Split to convert. Repeat for multiple splits; default: test.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Converted case id to include. Repeat or comma-separate.",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = convert_tablefact_release(
        source_root=args.source_root,
        output_root=args.output_root,
        splits=tuple(args.splits or DEFAULT_SPLITS),
        case_ids=_expand_csv_values(args.case_id),
        limit_cases=args.limit_cases,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_selected_rows(
    *,
    source_path: Path,
    splits: Sequence[str],
    test_subsets: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows = []
    for requested_split in splits:
        canonical_split = "dev" if requested_split == "val" else requested_split
        base_split = (
            "test" if canonical_split in SUBSET_ID_FILES else canonical_split
        )
        data_file = source_path / "tokenized_data" / SPLIT_DATA_FILES[base_split]
        if not data_file.is_file():
            raise FileNotFoundError(f"TableFact split file not found: {data_file}")
        payload = json.loads(data_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"TableFact split must contain an object: {data_file}")

        allowed_tables = None
        if canonical_split in SUBSET_ID_FILES:
            allowed_tables = test_subsets[canonical_split]
        for table_file, value in payload.items():
            if allowed_tables is not None and table_file not in allowed_tables:
                continue
            statements, labels, caption = _table_examples(value, table_file=table_file)
            for statement_index, (statement, label) in enumerate(
                zip(statements, labels, strict=True)
            ):
                rows.append(
                    {
                        "table_file": table_file,
                        "statement_index": statement_index,
                        "statement": statement,
                        "label": label,
                        "caption": caption,
                        "split": base_split,
                        "requested_split": canonical_split,
                        "subset": _table_subset(
                            table_file,
                            base_split=base_split,
                            test_subsets=test_subsets,
                        ),
                        "is_small_test": table_file in test_subsets["small_test"],
                    }
                )
    return rows


def _load_test_subsets(source_path: Path) -> dict[str, set[str]]:
    subsets = {}
    for subset, filename in SUBSET_ID_FILES.items():
        path = source_path / "data" / filename
        if not path.is_file():
            raise FileNotFoundError(f"TableFact subset id file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"TableFact subset id file must contain a list: {path}")
        subsets[subset] = {str(value) for value in payload}
    return subsets


def _table_examples(value: Any, *, table_file: str) -> tuple[list[str], list[int], str]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"Invalid TableFact examples for table {table_file}")
    statements, labels, caption = value[:3]
    if not isinstance(statements, list) or not isinstance(labels, list):
        raise ValueError(f"Invalid TableFact statements or labels for {table_file}")
    if len(statements) != len(labels):
        raise ValueError(f"TableFact statement/label length mismatch for {table_file}")
    normalized_labels = []
    for label in labels:
        try:
            normalized = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid TableFact label {label!r} for {table_file}") from exc
        if normalized not in {0, 1}:
            raise ValueError(f"Invalid TableFact label {label!r} for {table_file}")
        normalized_labels.append(normalized)
    return [str(item) for item in statements], normalized_labels, str(caption or "")


def _table_subset(
    table_file: str,
    *,
    base_split: str,
    test_subsets: dict[str, set[str]],
) -> str:
    if base_split != "test":
        return base_split
    if table_file in test_subsets["simple_test"]:
        return "simple"
    if table_file in test_subsets["complex_test"]:
        return "complex"
    return "test"


def _case_record(
    *,
    row: dict[str, Any],
    dataset_id: str,
    case_index: int,
) -> dict[str, Any]:
    statement = str(row["statement"]).strip()
    case_id = _case_id(row["table_file"], int(row["statement_index"]))
    caption = str(row.get("caption") or "").strip()
    hints = {}
    if caption:
        hints = {
            "source_context": caption,
            "source_context_usage": CAPTION_USAGE,
        }
    return {
        "case_id": case_id,
        "original_id": case_id,
        "dataset_id": dataset_id,
        "question": (
            "Determine whether the following statement is entailed by the table. "
            "Answer true if it is entailed and false if it is refuted: "
            f"{statement}"
        ),
        "statement": statement,
        "answer": bool(row["label"]),
        "label": int(row["label"]),
        "label_text": "entailed" if row["label"] else "refuted",
        "type": "boolean",
        "qtype": "FactChecking",
        "qsubtype": row["subset"],
        "split": row["split"],
        "requested_split": row["requested_split"],
        "is_small_test": bool(row["is_small_test"]),
        "caption": caption,
        "source_table": row["table_file"],
        "case_index": case_index,
        "hints": hints,
    }


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    case_ids: set[str] | None,
    limit_cases: int | None,
) -> list[dict[str, Any]]:
    selected = []
    seen_case_ids = set()
    for row in rows:
        case_id = _case_id(row["table_file"], int(row["statement_index"]))
        if case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)
        if case_ids is not None and case_id not in case_ids:
            continue
        selected.append(row)
        if limit_cases is not None and len(selected) >= limit_cases:
            break
    if case_ids is not None:
        missing = sorted(case_ids - seen_case_ids)
        if missing:
            raise ValueError(f"Requested TableFact case ids not found: {missing}")
    return selected


def _write_normalized_csv(source: Path, target: Path) -> tuple[int, int]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="#"))
    if not rows:
        raise ValueError(f"TableFact table is empty: {source}")
    max_columns = max(len(row) for row in rows)
    header = list(rows[0]) + [
        f"column_{index + 1}" for index in range(len(rows[0]), max_columns)
    ]
    header = _unique_columns(header)
    data_rows = [row + [""] * (max_columns - len(row)) for row in rows[1:]]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data_rows)
    return len(data_rows), len(header)


def _unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for index, column in enumerate(columns):
        base = str(column or "").strip() or f"column_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for value in splits or DEFAULT_SPLITS:
        split = str(value).strip().lower()
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported TableFact split: {value!r}")
        normalized.append(split)
    return tuple(dict.fromkeys(normalized))


def _dataset_id(table_file: str) -> str:
    stem = re.sub(r"\.html\.csv$", "", table_file, flags=re.IGNORECASE)
    return f"tablefact_{_safe_id(stem)}"


def _case_id(table_file: str, statement_index: int) -> str:
    stem = re.sub(r"\.html\.csv$", "", table_file, flags=re.IGNORECASE)
    return f"tablefact_{_safe_id(stem)}_{statement_index:04d}"


def _safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return text[:120]


def _normalize_ids(values: Sequence[str] | None) -> set[str] | None:
    expanded = _expand_csv_values(values)
    return set(expanded) if expanded else None


def _expand_csv_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    expanded = []
    for value in values:
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    return expanded


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
