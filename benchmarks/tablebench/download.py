"""Download TableBench and convert it to CLOVER's local table layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.modelscope_utils import (
    download_modelscope_dataset_snapshot,
    load_modelscope_rows as load_modelscope_dataset_rows,
)

DEFAULT_REPO_ID = "Multilingual-Multimodal-NLP/TableBench"
DEFAULT_CONFIG_NAME = "table_bench"
DEFAULT_SPLITS = ("TQA_test",)
DEFAULT_SOURCE_ROOT = Path("datasets") / "tablebench_source"
DEFAULT_OUTPUT_ROOT = Path("datasets") / "tablebench"
DEFAULT_RAW_FILENAMES = {
    "TQA_test": "TableBench.jsonl",
}
DATASET_SOURCES = ("huggingface", "modelscope")
DEFAULT_DATASET_SOURCE = "huggingface"
REPO_ROOT = Path(__file__).resolve().parents[2]


def download_and_convert_tablebench(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
    dataset_source: str = DEFAULT_DATASET_SOURCE,
    modelscope_cache_dir: str | Path | None = None,
    case_ids: Sequence[str] | None = None,
    qtypes: Sequence[str] | None = None,
    qsubtypes: Sequence[str] | None = None,
    include_visualization: bool = False,
    limit_cases: int | None = None,
    overwrite: bool = False,
    download_overwrite: bool = False,
) -> dict[str, Any]:
    """Download TableBench and write CLOVER dataset folders."""

    source = _normalize_dataset_source(dataset_source)
    source_path = _resolve_path(source_root)
    if source == "modelscope":
        rows = load_modelscope_rows(
            repo_id=repo_id,
            config_name=config_name,
            splits=splits,
            modelscope_cache_dir=modelscope_cache_dir,
        )
    else:
        rows = load_huggingface_rows(
            repo_id=repo_id,
            config_name=config_name,
            splits=splits,
        )
    source_summary = write_source_rows(
        rows=rows,
        source_root=source_path,
        repo_id=repo_id,
        config_name=config_name,
        splits=splits,
        dataset_source=source,
        overwrite=download_overwrite,
    )
    return convert_tablebench_rows(
        rows=rows,
        output_root=output_root,
        repo_id=repo_id,
        config_name=config_name,
        splits=splits,
        dataset_source=source,
        case_ids=case_ids,
        qtypes=qtypes,
        qsubtypes=qsubtypes,
        include_visualization=include_visualization,
        limit_cases=limit_cases,
        overwrite=overwrite,
        source_summary=source_summary,
    )


def load_huggingface_rows(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> list[dict[str, Any]]:
    """Load TableBench rows from HuggingFace.

    The official dataset exposes config ``table_bench`` with split
    ``TQA_test``. We first use ``datasets.load_dataset`` and fall back to the
    raw JSONL file for environments where the packaged dataset metadata is not
    available.
    """

    try:
        from datasets import load_dataset
    except ImportError:
        return _load_raw_huggingface_rows(repo_id=repo_id, splits=splits)

    rows: list[dict[str, Any]] = []
    try:
        for split in splits:
            dataset = load_dataset(repo_id, name=config_name, split=split)
            for row in dataset:
                payload = _json_ready(dict(row))
                payload.setdefault("split", split)
                rows.append(payload)
        return rows
    except Exception:
        return _load_raw_huggingface_rows(repo_id=repo_id, splits=splits)


def load_modelscope_rows(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
    modelscope_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load TableBench rows from ModelScope.

    MsDataset handles standard dataset repositories. The raw JSONL fallback
    covers mirrors that keep the Hugging Face-style file layout.
    """

    try:
        return load_modelscope_dataset_rows(
            repo_id=repo_id,
            subset_name=config_name,
            splits=splits,
        )
    except Exception:
        return _load_raw_modelscope_rows(
            repo_id=repo_id,
            splits=splits,
            modelscope_cache_dir=modelscope_cache_dir,
        )


def _load_raw_huggingface_rows(
    *,
    repo_id: str,
    splits: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        filename = DEFAULT_RAW_FILENAMES.get(split)
        if filename is None:
            raise ValueError(
                f"No raw TableBench filename is known for split {split!r}"
            )
        url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{filename}"
        with urllib.request.urlopen(url, timeout=120) as response:
            for line in response:
                stripped = line.decode("utf-8").strip()
                if stripped:
                    payload = json.loads(stripped)
                    payload.setdefault("split", split)
                    rows.append(payload)
    return rows


def _load_raw_modelscope_rows(
    *,
    repo_id: str,
    splits: Sequence[str],
    modelscope_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    snapshot_root = download_modelscope_dataset_snapshot(
        repo_id=repo_id,
        local_dir=modelscope_cache_dir,
    )
    rows: list[dict[str, Any]] = []
    for split in splits:
        filename = DEFAULT_RAW_FILENAMES.get(split)
        if filename is None:
            raise ValueError(
                f"No raw TableBench filename is known for split {split!r}"
            )
        path = _find_raw_modelscope_file(snapshot_root, filename)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    payload.setdefault("split", split)
                    rows.append(payload)
    return rows


def write_source_rows(
    *,
    rows: Iterable[dict[str, Any]],
    source_root: str | Path,
    repo_id: str,
    config_name: str,
    splits: Sequence[str],
    dataset_source: str = DEFAULT_DATASET_SOURCE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist downloaded rows locally for auditability and offline reuse."""

    source_path = _resolve_path(source_root)
    source_path.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_split[str(row.get("split") or "TQA_test")].append(dict(row))

    written_files = []
    skipped_files = []
    for split, split_rows in sorted(rows_by_split.items()):
        filename = DEFAULT_RAW_FILENAMES.get(split, f"{_safe_id(split)}.jsonl")
        path = source_path / filename
        if path.exists() and not overwrite:
            skipped_files.append(_portable_path(path))
            continue
        _write_jsonl(path, split_rows)
        written_files.append(_portable_path(path))

    summary = {
        "stage": f"tablebench_{dataset_source}_source_download",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": dataset_source,
        "repo_id": repo_id,
        "config_name": config_name,
        "splits": list(splits),
        "source_root": _portable_path(source_path),
        "written_files": written_files,
        "skipped_files": skipped_files,
        "row_count": sum(len(items) for items in rows_by_split.values()),
    }
    _write_json(source_path / "download_summary.json", summary)
    return summary


def convert_tablebench_rows(
    *,
    rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    splits: Sequence[str] = DEFAULT_SPLITS,
    dataset_source: str = DEFAULT_DATASET_SOURCE,
    case_ids: Sequence[str] | None = None,
    qtypes: Sequence[str] | None = None,
    qsubtypes: Sequence[str] | None = None,
    include_visualization: bool = False,
    limit_cases: int | None = None,
    overwrite: bool = False,
    source_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert TableBench rows into CLOVER's table reasoning dataset layout."""

    if limit_cases is not None and limit_cases <= 0:
        raise ValueError("limit_cases must be positive")

    output_path = _resolve_path(output_root)
    all_rows = [_json_ready(dict(row)) for row in rows]
    visualization_case_count = sum(
        1 for row in all_rows if _is_visualization_row(row)
    )
    selected_rows = _select_rows(
        all_rows,
        case_ids=_normalize_ids(case_ids),
        qtypes=_normalize_ids(qtypes),
        qsubtypes=_normalize_ids(qsubtypes),
        include_visualization=include_visualization,
        limit_cases=limit_cases,
    )
    grouped_rows = _group_rows_by_table(selected_rows)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    total_cases = 0
    qtype_counts: dict[str, int] = defaultdict(int)
    qsubtype_counts: dict[str, int] = defaultdict(int)

    for table_id, table_rows in sorted(grouped_rows.items()):
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

        frame = _table_frame(table_rows[0]["table"])
        table_path = dataset_dir / "table.csv"
        frame.to_csv(table_path, index=False)

        case_records = []
        for case_index, row in enumerate(table_rows):
            case_record = _case_record(row, table_id, case_index)
            case_records.append(case_record)
            if case_record.get("qtype"):
                qtype_counts[str(case_record["qtype"])] += 1
            if case_record.get("qsubtype"):
                qsubtype_counts[str(case_record["qsubtype"])] += 1

        _write_jsonl(dataset_dir / "cases.jsonl", case_records)
        total_cases += len(case_records)
        dataset_summaries.append(
            {
                "dataset_id": table_id,
                "case_count": len(case_records),
                "table_csv": f"{table_id}/table.csv",
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
            }
        )

    summary = {
        "stage": f"tablebench_{dataset_source}_download",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": dataset_source,
        "repo_id": repo_id,
        "config_name": config_name,
        "splits": list(splits),
        "output_root": _portable_path(output_path),
        "include_visualization": include_visualization,
        "source_case_count": len(all_rows),
        "visualization_case_count": visualization_case_count,
        "visualization_excluded": not include_visualization,
        "dataset_count": len(dataset_summaries),
        "case_count": total_cases,
        "qtype_counts": dict(sorted(qtype_counts.items())),
        "qsubtype_counts": dict(sorted(qsubtype_counts.items())),
        "datasets": dataset_summaries,
    }
    if source_summary is not None:
        summary["source_summary"] = source_summary
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TableBench and convert it for CLOVER eval."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Output dataset root. Relative paths are resolved from the "
            "repository root."
        ),
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Directory used to store downloaded raw TableBench JSONL files.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--dataset-source",
        choices=DATASET_SOURCES,
        default=DEFAULT_DATASET_SOURCE,
        help="Dataset hub to use for TableBench rows.",
    )
    parser.add_argument(
        "--modelscope-cache-dir",
        default=None,
        help="Optional local_dir passed to ModelScope dataset_snapshot_download.",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=None,
        help="Dataset split to load. Repeat for multiple splits.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="TableBench case id to include. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--qtype",
        action="append",
        default=None,
        help="Question type to include, for example NumericalReasoning.",
    )
    parser.add_argument(
        "--qsubtype",
        action="append",
        default=None,
        help="Question subtype to include, for example Aggregation.",
    )
    parser.add_argument(
        "--include-visualization",
        action="store_true",
        help=(
            "Include TableBench Visualization cases. By default conversion "
            "excludes chart-generation cases because CLOVER evaluates "
            "non-visual table reasoning here."
        ),
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = download_and_convert_tablebench(
        output_root=args.output_root,
        source_root=args.source_root,
        repo_id=args.repo_id,
        config_name=args.config_name,
        splits=tuple(args.splits or DEFAULT_SPLITS),
        dataset_source=args.dataset_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        case_ids=_expand_csv_values(args.case_id),
        qtypes=_expand_csv_values(args.qtype),
        qsubtypes=_expand_csv_values(args.qsubtype),
        include_visualization=args.include_visualization,
        limit_cases=args.limit_cases,
        overwrite=args.overwrite,
        download_overwrite=args.download_overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    case_ids: set[str] | None,
    qtypes: set[str] | None,
    qsubtypes: set[str] | None,
    include_visualization: bool,
    limit_cases: int | None,
) -> list[dict[str, Any]]:
    selected = []
    seen_case_ids = set()
    for row in rows:
        payload = _json_ready(dict(row))
        original_id = _required_text(payload, "id")
        seen_case_ids.add(original_id)
        if case_ids is not None and original_id not in case_ids:
            continue
        if not include_visualization and _is_visualization_row(payload):
            continue
        if qtypes is not None and str(payload.get("qtype")) not in qtypes:
            continue
        if qsubtypes is not None and str(payload.get("qsubtype")) not in qsubtypes:
            continue
        if "table" not in payload:
            raise ValueError(f"TableBench row missing table: {original_id}")
        selected.append(payload)
        if limit_cases is not None and len(selected) >= limit_cases:
            break
    if case_ids is not None:
        missing = sorted(case_ids - seen_case_ids)
        if missing:
            raise ValueError(f"Requested TableBench case ids not found: {missing}")
    return selected


def _normalize_dataset_source(value: str) -> str:
    source = str(value or DEFAULT_DATASET_SOURCE).strip().lower()
    if source not in DATASET_SOURCES:
        raise ValueError(f"Unsupported dataset source: {value!r}")
    return source


def _find_raw_modelscope_file(snapshot_root: Path, filename: str) -> Path:
    expected = snapshot_root / filename
    if expected.exists():
        return expected
    matches = sorted(snapshot_root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find TableBench raw JSONL in ModelScope snapshot: "
        f"filename={filename!r}, snapshot_root={snapshot_root}"
    )


def _is_visualization_row(row: dict[str, Any]) -> bool:
    return str(row.get("qtype") or "").strip().lower() == "visualization"


def _group_rows_by_table(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_table_id(row["table"])].append(row)
    return dict(grouped)


def _case_record(row: dict[str, Any], table_id: str, case_index: int) -> dict[str, Any]:
    original_id = _required_text(row, "id")
    record = {
        "case_id": _case_id(original_id, table_id, case_index),
        "original_id": original_id,
        "dataset_id": table_id,
        "question": _required_text(row, "question"),
        "answer": _json_ready(row.get("answer")),
        "type": _infer_answer_type(row),
        "qtype": row.get("qtype"),
        "qsubtype": row.get("qsubtype"),
    }
    for field in ("chart_type", "split"):
        value = row.get(field)
        if value is not None:
            record[field] = _json_ready(value)
    return record


def _table_frame(table: Any) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised by user envs.
        raise RuntimeError(
            "Missing optional dependency 'pandas'. Install requirements.txt first."
        ) from exc

    normalized = _normalize_table(table)
    columns = _unique_columns([str(column) for column in normalized["columns"]])
    data = normalized["data"]
    if all(isinstance(row, dict) for row in data):
        frame = pd.DataFrame(data)
        frame = frame.rename(columns={column: str(column) for column in frame.columns})
        return frame
    return pd.DataFrame(data, columns=columns)


def _normalize_table(table: Any) -> dict[str, Any]:
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except json.JSONDecodeError as exc:
            raise ValueError("TableBench table field is not valid JSON") from exc
    if not isinstance(table, dict):
        raise ValueError("TableBench table must be an object with columns and data")
    columns = table.get("columns")
    data = table.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        raise ValueError("TableBench table requires list fields: columns and data")
    return {"columns": columns, "data": data}


def _unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for index, column in enumerate(columns):
        base = column.strip() or f"column_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def _table_id(table: Any) -> str:
    normalized = _normalize_table(table)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"table_{digest}"


def _case_id(original_id: str, table_id: str, case_index: int) -> str:
    safe = _safe_id(original_id)
    if safe:
        return safe
    return f"{table_id}_{case_index:04d}"


def _infer_answer_type(row: dict[str, Any]) -> str:
    answer = _parse_jsonish(row.get("answer"))
    if isinstance(answer, bool):
        return "boolean"
    if isinstance(answer, list):
        if answer and all(_number_or_none(item) is not None for item in answer):
            return "list[number]"
        return "list[string]"
    if _number_or_none(answer) is not None:
        return "number"
    if str(row.get("qtype") or "").lower() == "factchecking":
        text = str(answer).strip().lower()
        if text in {"true", "false", "yes", "no", "y", "n", "1", "0"}:
            return "boolean"
    return "string"


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.endswith("%"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"TableBench row missing required field: {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"TableBench row has empty required field: {field}")
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
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return expanded


def _resolve_path(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = REPO_ROOT / selected
    return selected.resolve()


def _portable_path(path: Path, *, base: Path = REPO_ROOT) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return (Path("<external>") / resolved.name).as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
