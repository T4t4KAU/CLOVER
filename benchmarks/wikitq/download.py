"""Convert a local WikiTableQuestions release into CLOVER's table layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.wikitq.metrics import parse_number, tsv_unescape_list


DEFAULT_SOURCE_ROOT = Path("datasets") / "wikitq_source"
DEFAULT_OUTPUT_ROOT = Path("datasets") / "wikitq"
DEFAULT_SPLIT = "pristine-unseen-tables"


def download_and_convert_wikitq(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    split: str = DEFAULT_SPLIT,
    case_ids: Sequence[str] | None = None,
    limit_cases: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert an existing WikiTQ release into CLOVER-ready folders."""

    source_path = _resolve_path(source_root)
    output_path = _resolve_path(output_root)
    if not source_path.is_dir():
        raise FileNotFoundError(f"WikiTQ source root not found: {source_path}")
    if limit_cases is not None and limit_cases <= 0:
        raise ValueError("limit_cases must be positive")

    split_label = _split_label(split)
    data_path = source_path / "data" / f"{split_label}.tsv"
    if not data_path.is_file():
        raise FileNotFoundError(f"WikiTQ split file not found: {data_path}")

    tagged_by_id = _load_tagged_rows(source_path)
    rows = list(_read_tsv_rows(data_path))
    selected_rows = _select_rows(
        rows,
        case_ids=_normalize_ids(case_ids),
        limit_cases=limit_cases,
    )
    grouped_rows = _group_rows_by_context(selected_rows)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    total_cases = 0
    missing_canon = 0
    multi_answer_count = 0
    answer_type_counts: dict[str, int] = defaultdict(int)

    for context, table_rows in sorted(grouped_rows.items()):
        table_id = _table_id_from_context(context)
        dataset_dir = output_path / table_id
        if dataset_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    "Output dataset already exists: "
                    f"{_portable_path(dataset_dir)}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        source_table = source_path / context
        if not source_table.is_file():
            raise FileNotFoundError(f"WikiTQ table not found: {source_table}")
        rows_written, columns_written = _write_normalized_csv(
            source_table,
            dataset_dir / "table.csv",
        )
        case_records = []
        for case_index, row in enumerate(table_rows):
            tagged_row = tagged_by_id.get(row["id"])
            if tagged_row is None:
                missing_canon += 1
            record = _case_record(
                row=row,
                tagged_row=tagged_row,
                table_id=table_id,
                split_label=split_label,
                case_index=case_index,
            )
            case_records.append(record)
            if len(record["answer"]) > 1:
                multi_answer_count += 1
            answer_type_counts[str(record["type"])] += 1

        _write_jsonl(dataset_dir / "cases.jsonl", case_records)
        total_cases += len(case_records)
        dataset_summaries.append(
            {
                "dataset_id": table_id,
                "context": context,
                "case_count": len(case_records),
                "table_csv": f"{table_id}/table.csv",
                "rows": rows_written,
                "columns": columns_written,
            }
        )

    summary = {
        "stage": "wikitq_local_conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": _portable_path(source_path),
        "output_root": _portable_path(output_path),
        "split": split_label,
        "source_case_count": len(rows),
        "dataset_count": len(dataset_summaries),
        "case_count": total_cases,
        "missing_canonical_answer_cases": missing_canon,
        "multi_answer_cases": multi_answer_count,
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "datasets": dataset_summaries,
    }
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local WikiTableQuestions release for CLOVER eval."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="WikiTQ case id to include. Repeat or comma-separate.",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = download_and_convert_wikitq(
        source_root=args.source_root,
        output_root=args.output_root,
        split=args.split,
        case_ids=_expand_csv_values(args.case_id),
        limit_cases=args.limit_cases,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_tsv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield {str(key): str(value or "") for key, value in row.items()}


def _load_tagged_rows(source_root: Path) -> dict[str, dict[str, str]]:
    tagged_root = source_root / "tagged" / "data"
    rows: dict[str, dict[str, str]] = {}
    if not tagged_root.is_dir():
        return rows
    for path in sorted(tagged_root.glob("*.tagged")):
        for row in _read_tsv_rows(path):
            rows[row["id"]] = row
    return rows


def _select_rows(
    rows: list[dict[str, str]],
    *,
    case_ids: set[str] | None,
    limit_cases: int | None,
) -> list[dict[str, str]]:
    selected = []
    seen_case_ids = set()
    for row in rows:
        case_id = _required_text(row, "id")
        seen_case_ids.add(case_id)
        if case_ids is not None and case_id not in case_ids:
            continue
        _required_text(row, "utterance")
        _required_text(row, "context")
        _required_text(row, "targetValue")
        selected.append(row)
        if limit_cases is not None and len(selected) >= limit_cases:
            break
    if case_ids is not None:
        missing = sorted(case_ids - seen_case_ids)
        if missing:
            raise ValueError(f"Requested WikiTQ case ids not found: {missing}")
    return selected


def _group_rows_by_context(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["context"]].append(row)
    return dict(grouped)


def _case_record(
    *,
    row: dict[str, str],
    tagged_row: dict[str, str] | None,
    table_id: str,
    split_label: str,
    case_index: int,
) -> dict[str, Any]:
    answer = tsv_unescape_list(row["targetValue"])
    canon = (
        tsv_unescape_list(tagged_row["targetCanon"])
        if tagged_row is not None and tagged_row.get("targetCanon")
        else None
    )
    canon_type = tagged_row.get("targetCanonType") if tagged_row is not None else None
    return {
        "case_id": _safe_id(row["id"]) or f"{table_id}_{case_index:04d}",
        "original_id": row["id"],
        "dataset_id": table_id,
        "question": tsv_unescape(row["utterance"]),
        "answer": answer,
        "answer_canon": canon,
        "answer_canon_type": canon_type,
        "type": _infer_answer_type(answer, canon, canon_type),
        "split": split_label,
        "context": row["context"],
    }


def _write_normalized_csv(source: Path, target: Path) -> tuple[int, int]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, escapechar="\\"))
    if not rows:
        raise ValueError(f"WikiTQ table CSV is empty: {source}")
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


def _infer_answer_type(
    answer: list[str],
    canon: list[str] | None,
    canon_type: str | None,
) -> str:
    values = canon if canon is not None and len(canon) == len(answer) else answer
    canon_type_text = str(canon_type or "").strip().lower()
    if len(answer) > 1:
        if values and all(parse_number(value) is not None for value in values):
            return "list[number]"
        return "list[string]"
    if canon_type_text == "number" and values and parse_number(values[0]) is not None:
        return "number"
    if values and parse_number(values[0]) is not None:
        return "number"
    return "string"


def _unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for index, column in enumerate(columns):
        base = str(column or "").strip() or f"column_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def tsv_unescape(text: str) -> str:
    return text.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


def _table_id_from_context(context: str) -> str:
    match = re.fullmatch(r"csv/(\d+)-csv/(\d+)\.csv", context)
    if match:
        return f"wikitq_{match.group(1)}_{match.group(2)}"
    return f"wikitq_{_safe_id(context)}"


def _split_label(split: str) -> str:
    text = str(split or DEFAULT_SPLIT).strip()
    if text.endswith(".tsv"):
        text = text[:-4]
    if not text:
        raise ValueError("WikiTQ split must not be empty")
    return text


def _required_text(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"WikiTQ row missing required field: {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"WikiTQ row has empty required field: {field}")
    return text


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text[:120]


def _normalize_ids(values: Sequence[str] | None) -> set[str] | None:
    expanded = _expand_csv_values(values)
    return set(expanded) if expanded else None


def _expand_csv_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    return expanded


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
