"""Download HuggingFace DataBench and convert it to CLOVER's local layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "cardiffnlp/databench"
DEFAULT_CONFIG_NAME = "qa"
DEFAULT_SPLITS = ("train",)
DEFAULT_TABLE_KIND = "all"


TableLoader = Callable[[str], Any]


def download_and_convert_databench(
    *,
    output_root: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
    table_kind: str = DEFAULT_TABLE_KIND,
    dataset_ids: Sequence[str] | None = None,
    limit_datasets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download DataBench from HuggingFace and write CLOVER dataset folders."""

    rows = load_huggingface_qa_rows(
        repo_id=repo_id,
        config_name=config_name,
        splits=splits,
    )
    table_loader = lambda dataset_id: load_huggingface_table(
        dataset_id=dataset_id,
        repo_id=repo_id,
        table_kind=table_kind,
    )
    return convert_databench_rows(
        rows=rows,
        output_root=output_root,
        table_loader=table_loader,
        repo_id=repo_id,
        config_name=config_name,
        splits=splits,
        table_kind=table_kind,
        dataset_ids=dataset_ids,
        limit_datasets=limit_datasets,
        overwrite=overwrite,
    )


def load_huggingface_qa_rows(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> list[dict[str, Any]]:
    """Load QA rows from the HuggingFace DataBench dataset."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by user envs.
        raise RuntimeError(
            "Missing optional dependency 'datasets'. Install requirements.txt first."
        ) from exc

    rows: list[dict[str, Any]] = []
    for split in splits:
        dataset = load_dataset(repo_id, name=config_name, split=split)
        for row in dataset:
            payload = dict(row)
            payload.setdefault("split", split)
            rows.append(payload)
    return rows


def load_huggingface_table(
    *,
    dataset_id: str,
    repo_id: str = DEFAULT_REPO_ID,
    table_kind: str = DEFAULT_TABLE_KIND,
) -> Any:
    """Load one DataBench table from HuggingFace as a pandas DataFrame."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised by user envs.
        raise RuntimeError(
            "Missing optional dependency 'pandas'. Install requirements.txt first."
        ) from exc

    parquet_path = f"hf://datasets/{repo_id}/data/{dataset_id}/{table_kind}.parquet"
    return pd.read_parquet(parquet_path)


def convert_databench_rows(
    *,
    rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    table_loader: TableLoader,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
    table_kind: str = DEFAULT_TABLE_KIND,
    dataset_ids: Sequence[str] | None = None,
    limit_datasets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert DataBench QA rows into CLOVER's table reasoning dataset layout."""

    output_path = Path(output_root).expanduser().resolve()
    selected_ids = _normalize_dataset_ids(dataset_ids)
    grouped_rows = _group_rows_by_dataset(rows, selected_ids)
    dataset_order = sorted(grouped_rows)
    if limit_datasets is not None:
        dataset_order = dataset_order[:limit_datasets]

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_summaries = []
    total_cases = 0
    answer_field = "sample_answer" if table_kind == "sample" else "answer"

    for dataset_id in dataset_order:
        dataset_dir = output_path / dataset_id
        if dataset_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output dataset already exists: {dataset_dir}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(dataset_dir)

        task_specs_dir = dataset_dir / "task_specs"
        task_specs_dir.mkdir(parents=True, exist_ok=True)

        table = table_loader(dataset_id)
        table_path = dataset_dir / "table.csv"
        table.to_csv(table_path, index=False)

        case_records = []
        for case_index, row in enumerate(grouped_rows[dataset_id]):
            case_id = _case_id(row, dataset_id, case_index)
            case_record = _case_record(
                row,
                dataset_id,
                case_id,
                answer_field=answer_field,
            )
            case_records.append(case_record)
            _write_json(
                task_specs_dir / f"{case_id}.json",
                _task_spec_from_case(case_record),
            )

        _write_jsonl(dataset_dir / "cases.jsonl", case_records)
        total_cases += len(case_records)
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "case_count": len(case_records),
                "table_csv": str(table_path),
                "rows": _table_shape_value(table, 0),
                "columns": _table_shape_value(table, 1),
            }
        )

    summary = {
        "stage": "databench_huggingface_download",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "config_name": config_name,
        "splits": list(splits),
        "table_kind": table_kind,
        "answer_field": answer_field,
        "output_root": str(output_path),
        "dataset_count": len(dataset_summaries),
        "case_count": total_cases,
        "datasets": dataset_summaries,
    }
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download HuggingFace DataBench and convert it for CLOVER eval."
    )
    parser.add_argument("--output-root", default="datasets/databench")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=None,
        help="HuggingFace split to load. Repeat for multiple splits.",
    )
    parser.add_argument(
        "--table-kind",
        choices=("all", "sample"),
        default=DEFAULT_TABLE_KIND,
        help="Use full DataBench tables or 20-row sample tables.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Dataset id to include, for example 001_Forbes. Repeat or comma-separate.",
    )
    parser.add_argument("--limit-datasets", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = download_and_convert_databench(
        output_root=args.output_root,
        repo_id=args.repo_id,
        config_name=args.config_name,
        splits=tuple(args.splits or DEFAULT_SPLITS),
        table_kind=args.table_kind,
        dataset_ids=_expand_csv_values(args.dataset_id),
        limit_datasets=args.limit_datasets,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _group_rows_by_dataset(
    rows: Iterable[dict[str, Any]],
    selected_ids: set[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_id = _required_text(row, "dataset")
        if selected_ids is not None and dataset_id not in selected_ids:
            continue
        grouped[dataset_id].append(dict(row))
    if selected_ids is not None:
        missing = sorted(selected_ids - set(grouped))
        if missing:
            raise ValueError(f"Requested DataBench dataset ids not found: {missing}")
    return dict(grouped)


def _case_record(
    row: dict[str, Any],
    dataset_id: str,
    case_id: str,
    *,
    answer_field: str,
) -> dict[str, Any]:
    record = {
        "case_id": case_id,
        "dataset_id": dataset_id,
        "question": _required_text(row, "question"),
        "answer": _jsonable(row.get(answer_field)),
        "type": _required_text(row, "type"),
    }
    if answer_field != "answer" and row.get("answer") is not None:
        record["full_answer"] = _jsonable(row.get("answer"))
    for field in ("columns_used", "column_types", "sample_answer", "split"):
        value = row.get(field)
        if value is not None:
            record[field] = _jsonable(value)
    return record


def _task_spec_from_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning",
        "question": case["question"],
        "sources": [
            {
                "id": 0,
                "type": "table",
                "file": "table.csv",
            }
        ],
        "answer": {
            "name": "answer",
            "type": case["type"],
        },
    }


def _case_id(row: dict[str, Any], dataset_id: str, case_index: int) -> str:
    value = row.get("case_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"{dataset_id}_{case_index:04d}"


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"DataBench row missing required field: {field}")
    text = str(value)
    if not text:
        raise ValueError(f"DataBench row has empty required field: {field}")
    return text


def _normalize_dataset_ids(values: Sequence[str] | None) -> set[str] | None:
    expanded = _expand_csv_values(values)
    return set(expanded) if expanded else None


def _expand_csv_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return expanded


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _table_shape_value(table: Any, index: int) -> int | None:
    shape = getattr(table, "shape", None)
    if not isinstance(shape, tuple) or len(shape) <= index:
        return None
    value = shape[index]
    return int(value) if value is not None else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
