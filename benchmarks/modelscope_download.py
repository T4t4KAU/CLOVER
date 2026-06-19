"""Convert ModelScope WikiTableQuestions and TableFact rows for CLOVER."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.modelscope_utils import load_modelscope_rows
from benchmarks.wikitq.metrics import parse_number


DEFAULT_WIKITQ_REPO = "stanfordnlp/wikitablequestions"
DEFAULT_WIKITQ_SUBSET = "random-split-1"
DEFAULT_TABLEFACT_REPO = "ibm-research/tab_fact"


def download_and_convert_modelscope_wikitq(
    *,
    output_root: str | Path,
    cache_dir: str | Path | None = None,
    repo_id: str = DEFAULT_WIKITQ_REPO,
    subset_name: str = DEFAULT_WIKITQ_SUBSET,
    source_split: str = "test",
    output_split: str = "pristine-unseen-tables",
    limit_cases: int | None = None,
    overwrite: bool = False,
    force_redownload: bool = False,
) -> dict[str, Any]:
    rows = load_modelscope_rows(
        repo_id=repo_id,
        subset_name=subset_name,
        splits=(source_split,),
        cache_dir=cache_dir,
        trust_remote_code=True,
        force_redownload=force_redownload,
    )
    if limit_cases is not None:
        rows = rows[:limit_cases]
    return convert_modelscope_wikitq_rows(
        rows=rows,
        output_root=output_root,
        repo_id=repo_id,
        subset_name=subset_name,
        source_split=source_split,
        output_split=output_split,
        overwrite=overwrite,
    )


def convert_modelscope_wikitq_rows(
    *,
    rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    repo_id: str,
    subset_name: str,
    source_split: str,
    output_split: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = _prepare_output_root(output_root, overwrite=overwrite)
    source_root = output_path / "_modelscope_source"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        table = _required_mapping(row, "table")
        context = _wikitq_context(table.get("name"))
        grouped[context].append(dict(row))

    datasets = []
    total_cases = 0
    for context, table_rows in sorted(grouped.items()):
        table = _required_mapping(table_rows[0], "table")
        header, data_rows = _normalized_table(table)
        dataset_id = _wikitq_dataset_id(context)
        dataset_dir = output_path / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(dataset_dir / "table.csv", header, data_rows)
        _write_csv(source_root / context, header, data_rows)

        cases = []
        for case_index, row in enumerate(table_rows):
            answers = [str(value) for value in row.get("answers") or []]
            original_id = str(row.get("id") or f"{dataset_id}_{case_index:04d}")
            cases.append(
                {
                    "case_id": _safe_id(original_id),
                    "original_id": original_id,
                    "dataset_id": dataset_id,
                    "question": str(row.get("question") or "").strip(),
                    "answer": answers,
                    "answer_canon": None,
                    "answer_canon_type": None,
                    "type": _answer_type(answers),
                    "split": output_split,
                    "context": context,
                    "modelscope_source_split": source_split,
                }
            )
        _write_jsonl(dataset_dir / "cases.jsonl", cases)
        total_cases += len(cases)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "context": context,
                "case_count": len(cases),
                "rows": len(data_rows),
                "columns": len(header),
            }
        )

    summary = {
        "stage": "wikitq_modelscope_conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "modelscope",
        "repo_id": repo_id,
        "subset_name": subset_name,
        "source_split": source_split,
        "split": output_split,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_path.resolve()),
        "dataset_count": len(datasets),
        "case_count": total_cases,
        "datasets": datasets,
    }
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def download_and_convert_modelscope_tablefact(
    *,
    output_root: str | Path,
    cache_dir: str | Path | None = None,
    repo_id: str = DEFAULT_TABLEFACT_REPO,
    splits: Sequence[str] = ("test",),
    limit_cases: int | None = None,
    overwrite: bool = False,
    force_redownload: bool = False,
) -> dict[str, Any]:
    rows = load_modelscope_rows(
        repo_id=repo_id,
        subset_name=None,
        splits=splits,
        cache_dir=cache_dir,
        trust_remote_code=True,
        force_redownload=force_redownload,
    )
    if limit_cases is not None:
        rows = rows[:limit_cases]
    return convert_modelscope_tablefact_rows(
        rows=rows,
        output_root=output_root,
        repo_id=repo_id,
        requested_splits=splits,
        overwrite=overwrite,
    )


def convert_modelscope_tablefact_rows(
    *,
    rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    repo_id: str,
    requested_splits: Sequence[str],
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = _prepare_output_root(output_root, overwrite=overwrite)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        table = _required_mapping(row, "table")
        table_id = str(table.get("id") or "").strip()
        if not table_id:
            raise ValueError("ModelScope TableFact row has no table.id")
        grouped[table_id].append(dict(row))

    datasets = []
    split_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    total_cases = 0
    for table_id, table_rows in sorted(grouped.items()):
        table = _required_mapping(table_rows[0], "table")
        header, data_rows = _normalized_table(table)
        dataset_id = f"tablefact_{_safe_id(Path(table_id).stem)}"
        dataset_dir = output_path / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(dataset_dir / "table.csv", header, data_rows)

        cases = []
        for statement_index, row in enumerate(table_rows):
            row_table = _required_mapping(row, "table")
            statement = str(row.get("statement") or "").strip()
            label = int(row.get("label") or 0)
            source_split = str(row.get("split") or "test")
            split = "dev" if source_split == "validation" else source_split
            caption = str(row_table.get("caption") or "").strip()
            case_id = f"{dataset_id}_{statement_index:04d}"
            hints = {}
            if caption:
                hints = {
                    "source_context": caption,
                    "source_context_usage": (
                        "This is the Wikipedia page or table caption. Use it to "
                        "resolve entities mentioned by the statement, but do not "
                        "treat it as a table row."
                    ),
                }
            cases.append(
                {
                    "case_id": case_id,
                    "original_id": str(row.get("id") or case_id),
                    "dataset_id": dataset_id,
                    "question": (
                        "Determine whether the following statement is entailed "
                        "by the table. Answer true if it is entailed and false "
                        f"if it is refuted: {statement}"
                    ),
                    "statement": statement,
                    "answer": bool(label),
                    "label": label,
                    "label_text": "entailed" if label else "refuted",
                    "type": "boolean",
                    "qtype": "FactChecking",
                    "qsubtype": split,
                    "split": split,
                    "requested_split": source_split,
                    "is_small_test": False,
                    "caption": caption,
                    "source_table": table_id,
                    "case_index": statement_index,
                    "hints": hints,
                }
            )
            split_counts[split] += 1
            label_counts["entailed" if label else "refuted"] += 1
        _write_jsonl(dataset_dir / "cases.jsonl", cases)
        total_cases += len(cases)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "source_table": table_id,
                "case_count": len(cases),
                "rows": len(data_rows),
                "columns": len(header),
            }
        )

    summary = {
        "stage": "tablefact_modelscope_conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "modelscope",
        "repo_id": repo_id,
        "requested_splits": list(requested_splits),
        "output_root": str(output_path.resolve()),
        "dataset_count": len(datasets),
        "case_count": total_cases,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "datasets": datasets,
    }
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ModelScope WikiTQ/TableFact and convert for CLOVER."
    )
    parser.add_argument("--dataset", choices=("wikitq", "tablefact"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--subset-name", default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--output-split", default=None)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "wikitq":
        source_splits = args.split or ["test"]
        if len(source_splits) != 1:
            raise SystemExit("WikiTQ conversion accepts exactly one --split.")
        summary = download_and_convert_modelscope_wikitq(
            output_root=args.output_root,
            cache_dir=args.cache_dir,
            repo_id=args.repo_id or DEFAULT_WIKITQ_REPO,
            subset_name=args.subset_name or DEFAULT_WIKITQ_SUBSET,
            source_split=source_splits[0],
            output_split=args.output_split or "pristine-unseen-tables",
            limit_cases=args.limit_cases,
            overwrite=args.overwrite,
            force_redownload=args.force_redownload,
        )
    else:
        summary = download_and_convert_modelscope_tablefact(
            output_root=args.output_root,
            cache_dir=args.cache_dir,
            repo_id=args.repo_id or DEFAULT_TABLEFACT_REPO,
            splits=tuple(args.split or ["test"]),
            limit_cases=args.limit_cases,
            overwrite=args.overwrite,
            force_redownload=args.force_redownload,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prepare_output_root(path: str | Path, *, overwrite: bool) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output}. Pass --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _normalized_table(table: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    header = [str(value or "").strip() for value in table.get("header") or []]
    rows = [
        [str(value or "").strip() for value in row]
        for row in table.get("rows") or []
    ]
    width = max([len(header), *(len(row) for row in rows)], default=0)
    if width == 0:
        raise ValueError("ModelScope table is empty")
    header.extend(f"column_{index + 1}" for index in range(len(header), width))
    header = _unique_columns(header)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return header, rows


def _unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for index, column in enumerate(columns):
        base = str(column or "").strip() or f"column_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def _wikitq_context(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if text.endswith(".tsv"):
        text = text[:-4] + ".csv"
    if not text:
        raise ValueError("ModelScope WikiTQ row has no table.name")
    return text


def _wikitq_dataset_id(context: str) -> str:
    match = re.fullmatch(r"csv/(\d+)-csv/(\d+)\.csv", context)
    if match:
        return f"wikitq_{match.group(1)}_{match.group(2)}"
    return f"wikitq_{_safe_id(context)}"


def _answer_type(answers: Sequence[str]) -> str:
    if len(answers) > 1:
        if answers and all(parse_number(value) is not None for value in answers):
            return "list[number]"
        return "list[string]"
    if answers and parse_number(answers[0]) is not None:
        return "number"
    return "string"


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"ModelScope row has invalid {key}: {type(value)!r}")
    return value


def _safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return text[:160] or "item"


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


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
