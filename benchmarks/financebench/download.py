"""Download FinanceBench sources and convert them into CLOVER's dataset layout."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.financebench.adapter import (
    DOCUMENT_INFO_PATH,
    PDFS_DIR,
    QUESTIONS_PATH,
    find_pdf_path,
    read_jsonl,
    task_dsl_from_row,
)

DEFAULT_SOURCE_ROOT = Path("datasets") / "financebench_source"
DEFAULT_OUTPUT_ROOT = Path("datasets") / "financebench"
DEFAULT_GITHUB_REPO = "patronus-ai/financebench"
DEFAULT_GITHUB_REF = "main"
DEFAULT_QUESTION_REASONING = "Numerical reasoning"
DEFAULT_PDF_MODE = "symlink"
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 120.0
PDF_MODES = ("symlink", "copy")

UrlDownloader = Callable[[str, Path, bool, float], bool]


def download_and_convert_financebench(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    download: bool = True,
    github_repo: str = DEFAULT_GITHUB_REPO,
    github_ref: str = DEFAULT_GITHUB_REF,
    download_overwrite: bool = False,
    download_timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    question_reasoning: str | None = DEFAULT_QUESTION_REASONING,
    case_ids: Sequence[str] | None = None,
    limit_cases: int | None = None,
    pdf_mode: str = DEFAULT_PDF_MODE,
    overwrite: bool = False,
    downloader: UrlDownloader | None = None,
) -> dict[str, Any]:
    """Download FinanceBench if needed and write CLOVER-ready dataset folders."""

    source_path = _resolve_path(source_root)
    selected_ids = _normalize_case_ids(case_ids)
    download_summary: dict[str, Any] | None = None

    if download:
        download_summary = download_financebench_sources(
            source_root=source_path,
            github_repo=github_repo,
            github_ref=github_ref,
            doc_names=None,
            overwrite=download_overwrite,
            timeout_seconds=download_timeout_seconds,
            downloader=downloader,
        )

    rows = read_jsonl(source_path / QUESTIONS_PATH)
    selected_rows = _select_rows(
        [dict(row) for row in rows],
        selected_ids=selected_ids,
        question_reasoning=question_reasoning,
        limit_cases=limit_cases,
    )

    if download:
        selected_doc_names = sorted({str(row["doc_name"]) for row in selected_rows})
        pdf_summary = download_financebench_sources(
            source_root=source_path,
            github_repo=github_repo,
            github_ref=github_ref,
            doc_names=selected_doc_names,
            include_metadata=False,
            overwrite=download_overwrite,
            timeout_seconds=download_timeout_seconds,
            downloader=downloader,
        )
        download_summary = _merge_download_summaries(download_summary, pdf_summary)

    document_infos = read_jsonl(source_path / DOCUMENT_INFO_PATH)
    return convert_financebench_rows(
        rows=rows,
        document_infos=document_infos,
        source_root=source_path,
        output_root=output_root,
        question_reasoning=question_reasoning,
        case_ids=case_ids,
        limit_cases=limit_cases,
        pdf_mode=pdf_mode,
        overwrite=overwrite,
        download_summary=download_summary,
    )


def download_financebench_sources(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    github_repo: str = DEFAULT_GITHUB_REPO,
    github_ref: str = DEFAULT_GITHUB_REF,
    doc_names: Sequence[str] | None = None,
    include_metadata: bool = True,
    overwrite: bool = False,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    downloader: UrlDownloader | None = None,
) -> dict[str, Any]:
    """Download FinanceBench JSONL metadata and selected PDFs from GitHub."""

    source_path = _resolve_path(source_root)
    fetch = downloader or _download_url_to_path
    files: list[tuple[Path, str]] = []
    if include_metadata:
        files.extend(
            [
                (QUESTIONS_PATH, "questions"),
                (DOCUMENT_INFO_PATH, "document_info"),
            ]
        )
    if doc_names is not None:
        files.extend(
            (Path(PDFS_DIR) / f"{doc_name}.pdf", "pdf")
            for doc_name in sorted({str(name) for name in doc_names})
        )

    downloaded: list[str] = []
    skipped: list[str] = []
    for relative_path, kind in files:
        destination = source_path / relative_path
        url = _github_raw_url(github_repo, github_ref, relative_path)
        did_download = fetch(url, destination, overwrite, timeout_seconds)
        target = relative_path.as_posix()
        if did_download:
            downloaded.append(target)
        else:
            skipped.append(target)

    return {
        "source": "github",
        "github_repo": github_repo,
        "github_ref": github_ref,
        "source_root": _source_path(source_path, source_path),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "pdf_count": sum(1 for path, kind in files if kind == "pdf"),
    }


def convert_financebench_rows(
    *,
    rows: Iterable[dict[str, Any]],
    document_infos: Iterable[dict[str, Any]],
    source_root: str | Path,
    output_root: str | Path,
    question_reasoning: str | None = DEFAULT_QUESTION_REASONING,
    case_ids: Sequence[str] | None = None,
    limit_cases: int | None = None,
    pdf_mode: str = DEFAULT_PDF_MODE,
    overwrite: bool = False,
    download_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert FinanceBench rows into CLOVER's document reasoning layout."""

    if pdf_mode not in PDF_MODES:
        raise ValueError(f"Unsupported pdf_mode: {pdf_mode}")
    if limit_cases is not None and limit_cases <= 0:
        raise ValueError("limit_cases must be positive")

    source_path = _resolve_path(source_root)
    output_path = _resolve_path(output_root)
    if source_path == output_path:
        raise ValueError("source_root and output_root must be different directories")

    selected_ids = _normalize_case_ids(case_ids)
    all_rows = [dict(row) for row in rows]
    info_by_doc = {
        str(info["doc_name"]): dict(info)
        for info in document_infos
        if info.get("doc_name") is not None
    }
    selected_rows = _select_rows(
        all_rows,
        selected_ids=selected_ids,
        question_reasoning=question_reasoning,
        limit_cases=limit_cases,
    )
    _prepare_output_dir(output_path, overwrite=overwrite)

    data_dir = output_path / "data"
    pdfs_dir = output_path / PDFS_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    case_records: list[dict[str, Any]] = []
    selected_doc_names = {str(row["doc_name"]) for row in selected_rows}
    selected_infos = [
        info_by_doc[doc_name]
        for doc_name in sorted(selected_doc_names)
        if doc_name in info_by_doc
    ]

    pdf_outputs: dict[str, Path] = {}
    for doc_name in sorted(selected_doc_names):
        pdf_source = find_pdf_path(source_path, doc_name)
        pdf_output = pdfs_dir / pdf_source.name
        _materialize_pdf(pdf_source, pdf_output, pdf_mode=pdf_mode)
        pdf_outputs[doc_name] = pdf_output

    for sample_index, row in enumerate(selected_rows):
        _required_text(row, "financebench_id")
        doc_name = _required_text(row, "doc_name")
        case_dir = output_path / f"case_{sample_index + 1}"
        case_dir.mkdir(parents=True, exist_ok=True)

        pdf_output = pdf_outputs[doc_name]
        _materialize_case_document(pdf_output, case_dir / "document.pdf")

        doc_info = info_by_doc.get(doc_name, {})
        task_dsl = task_dsl_from_row(row, pdf_path=Path("document.pdf"))
        metadata = _metadata_from_row(
            row,
            doc_info=doc_info,
            source_root=source_path,
            source_pdf=find_pdf_path(source_path, doc_name),
            output_pdf=pdf_output,
            case_dir=case_dir,
            sample_index=sample_index,
            selected_count=len(selected_rows),
            question_reasoning=question_reasoning,
        )
        _write_json(case_dir / "task.json", task_dsl)
        _write_json(case_dir / "metadata.json", metadata)
        case_records.append(
            _index_record(
                row,
                doc_info=doc_info,
                metadata=metadata,
                example_id=f"case_{sample_index + 1}",
                example_dir=case_dir.name,
            )
        )

    _write_jsonl(data_dir / QUESTIONS_PATH.name, selected_rows)
    _write_jsonl(data_dir / DOCUMENT_INFO_PATH.name, selected_infos)
    _write_jsonl(output_path / "index.jsonl", case_records)

    summary = {
        "stage": (
            "financebench_download_and_conversion"
            if download_summary is not None
            else "financebench_local_conversion"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": _source_path(source_path, source_path),
        "output_root": _output_path(output_path, output_path),
        "question_reasoning": question_reasoning,
        "pdf_mode": pdf_mode,
        "total_rows": len(all_rows),
        "case_count": len(selected_rows),
        "document_count": len(selected_doc_names),
        "cases_index": _output_path(output_path / "index.jsonl", output_path),
        "questions_file": _output_path(data_dir / QUESTIONS_PATH.name, output_path),
        "document_info_file": _output_path(
            data_dir / DOCUMENT_INFO_PATH.name,
            output_path,
        ),
        "pdfs_dir": _output_path(pdfs_dir, output_path),
        "cases": [
            {
                "case_id": record["case_id"],
                "example_id": record["example_id"],
                "doc_name": record.get("doc_name"),
                "expected_answer": record.get("expected_answer"),
            }
            for record in case_records
        ],
    }
    if download_summary is not None:
        summary["download"] = download_summary
    _write_json(output_path / "conversion_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download FinanceBench and convert it for CLOVER eval."
    )
    parser.add_argument(
        "--source-root",
        default=os.environ.get("FINANCEBENCH_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT)),
        help=(
            "Local cache/source root for raw FinanceBench files. Relative paths "
            "are resolved from the repository root. Can also be set with "
            "FINANCEBENCH_SOURCE_ROOT."
        ),
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
        "--question-reasoning",
        default=DEFAULT_QUESTION_REASONING,
        help=(
            "FinanceBench question_reasoning value to include. "
            "Use an empty string to include all reasoning types."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="FinanceBench case id to include. Repeat or comma-separate.",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--pdf-mode", choices=PDF_MODES, default=DEFAULT_PDF_MODE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-download",
        dest="download",
        action="store_false",
        help="Use an existing source root without downloading missing files.",
    )
    parser.set_defaults(download=True)
    parser.add_argument(
        "--github-repo",
        default=DEFAULT_GITHUB_REPO,
        help="GitHub repository that hosts the FinanceBench files.",
    )
    parser.add_argument(
        "--github-ref",
        default=DEFAULT_GITHUB_REF,
        help="Git ref, branch, or commit to download from.",
    )
    parser.add_argument(
        "--download-overwrite",
        action="store_true",
        help="Redownload raw source files even when cached files already exist.",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        help="Timeout for each GitHub raw file download.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = download_and_convert_financebench(
        output_root=args.output_root,
        source_root=args.source_root,
        download=args.download,
        github_repo=args.github_repo,
        github_ref=args.github_ref,
        download_overwrite=args.download_overwrite,
        download_timeout_seconds=args.download_timeout_seconds,
        question_reasoning=args.question_reasoning or None,
        case_ids=_expand_csv_values(args.case_id),
        limit_cases=args.limit_cases,
        pdf_mode=args.pdf_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _download_url_to_path(
    url: str,
    destination: Path,
    overwrite: bool,
    timeout_seconds: float,
) -> bool:
    if destination.exists() and not overwrite:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CLOVER-FinanceBench-Downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            "Failed to download FinanceBench source file "
            f"{_portable_path(destination)} from {url}"
        ) from exc
    temporary.replace(destination)
    return True


def _github_raw_url(repo: str, ref: str, relative_path: Path) -> str:
    quoted_path = "/".join(
        urllib.parse.quote(part)
        for part in relative_path.as_posix().split("/")
    )
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{quoted_path}"


def _merge_download_summaries(
    first: dict[str, Any] | None,
    second: dict[str, Any],
) -> dict[str, Any]:
    if first is None:
        return dict(second)
    merged = dict(first)
    merged["downloaded_files"] = [
        *first.get("downloaded_files", []),
        *second.get("downloaded_files", []),
    ]
    merged["skipped_files"] = [
        *first.get("skipped_files", []),
        *second.get("skipped_files", []),
    ]
    merged["pdf_count"] = int(first.get("pdf_count", 0)) + int(
        second.get("pdf_count", 0)
    )
    return merged


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    selected_ids: set[str] | None,
    question_reasoning: str | None,
    limit_cases: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        case_id = _required_text(row, "financebench_id")
        if selected_ids is not None and case_id not in selected_ids:
            continue
        if (
            question_reasoning is not None
            and row.get("question_reasoning") != question_reasoning
        ):
            continue
        _required_text(row, "doc_name")
        _required_text(row, "question")
        selected.append(dict(row))
        if limit_cases is not None and len(selected) >= limit_cases:
            break
    if selected_ids is not None:
        found = {str(row["financebench_id"]) for row in selected}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Requested FinanceBench case ids not found: {missing}")
    return selected


def _metadata_from_row(
    row: dict[str, Any],
    *,
    doc_info: dict[str, Any],
    source_root: Path,
    source_pdf: Path,
    output_pdf: Path,
    case_dir: Path,
    sample_index: int,
    selected_count: int,
    question_reasoning: str | None,
) -> dict[str, Any]:
    evidence = _evidence_records(row)
    relative_output_pdf = Path(os.path.relpath(output_pdf, start=case_dir))
    metadata = {
        "case_id": row.get("financebench_id"),
        "dataset": "financebench",
        "dataset_subset_label": row.get("dataset_subset_label"),
        "source_dataset_dir": _source_path(source_root, source_root),
        "source_questions_file": _source_path(source_root / QUESTIONS_PATH, source_root),
        "source_document_info_file": _source_path(
            source_root / DOCUMENT_INFO_PATH,
            source_root,
        ),
        "original_source_pdf": _source_path(source_pdf, source_root),
        "source_pdf": relative_output_pdf.as_posix(),
        "pdf_size_bytes": output_pdf.stat().st_size if output_pdf.exists() else None,
        "doc_name": row.get("doc_name"),
        "company": row.get("company") or doc_info.get("company"),
        "gics_sector": doc_info.get("gics_sector"),
        "doc_type": doc_info.get("doc_type"),
        "doc_period": doc_info.get("doc_period"),
        "doc_link": doc_info.get("doc_link"),
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "question_reasoning": row.get("question_reasoning"),
        "answer_type": "string",
        "expected_answer": row.get("answer"),
        "justification": row.get("justification"),
        "evidence_page_nums": _evidence_page_nums(row),
        "evidence": evidence,
        "sampling": {
            "pool_filter": (
                f"question_reasoning == {question_reasoning}"
                if question_reasoning is not None
                else "all"
            ),
            "pool_size": selected_count,
            "sample_size": selected_count,
            "sample_index": sample_index,
        },
        "example_id": f"case_{sample_index + 1}",
        "example_dir": f"case_{sample_index + 1}",
    }
    return {key: _jsonable(value) for key, value in metadata.items()}


def _index_record(
    row: dict[str, Any],
    *,
    doc_info: dict[str, Any],
    metadata: dict[str, Any],
    example_id: str,
    example_dir: str,
) -> dict[str, Any]:
    return {
        "case_id": row.get("financebench_id"),
        "dataset": "financebench",
        "doc_name": row.get("doc_name"),
        "company": row.get("company") or doc_info.get("company"),
        "doc_type": doc_info.get("doc_type"),
        "doc_period": doc_info.get("doc_period"),
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "question_reasoning": row.get("question_reasoning"),
        "answer_type": "string",
        "expected_answer": row.get("answer"),
        "evidence_page_nums": metadata.get("evidence_page_nums"),
        "source_pdf": metadata.get("source_pdf"),
        "pdf_size_bytes": metadata.get("pdf_size_bytes"),
        "example_id": example_id,
        "example_dir": example_dir,
    }


def _evidence_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in row.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "doc_name": item.get("doc_name") or row.get("doc_name"),
                "page_num": item.get("evidence_page_num"),
                "text": item.get("evidence_text") or item.get("evidence_text_full_page"),
            }
        )
    return records


def _evidence_page_nums(row: dict[str, Any]) -> list[int]:
    pages = []
    for item in row.get("evidence") or []:
        if isinstance(item, dict) and item.get("evidence_page_num") is not None:
            pages.append(int(item["evidence_page_num"]))
    return sorted(set(pages))


def _prepare_output_dir(output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "Output directory already exists: "
                f"{_portable_path(output_path)}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def _materialize_pdf(source: Path, destination: Path, *, pdf_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if pdf_mode == "copy":
        shutil.copy2(source, destination)
    else:
        _relative_symlink(source, destination)


def _materialize_case_document(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    _relative_symlink(source, destination)


def _relative_symlink(source: Path, destination: Path) -> None:
    relative_source = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_source)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _portable_path(path: Path, *, base: Path = REPO_ROOT) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return (Path("<external>") / resolved.name).as_posix()


def _source_path(path: Path, source_root: Path) -> str:
    resolved = path.expanduser().resolve()
    source_base = source_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(source_base)
    except ValueError:
        return _portable_path(resolved)
    if not relative.parts:
        return "<source>"
    return (Path("<source>") / relative).as_posix()


def _output_path(path: Path, output_root: Path) -> str:
    resolved = path.expanduser().resolve()
    output_base = output_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(output_base)
    except ValueError:
        return _portable_path(resolved)
    if not relative.parts:
        return "<output>"
    return (Path("<output>") / relative).as_posix()


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"FinanceBench row missing required field: {field}")
    text = str(value)
    if not text:
        raise ValueError(f"FinanceBench row has empty required field: {field}")
    return text


def _normalize_case_ids(values: Sequence[str] | None) -> set[str] | None:
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
