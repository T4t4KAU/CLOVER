"""Download/convert the MMQA multi-table release into CLOVER's table layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from benchmarks.mmqa.metrics import flatten_mmqa_answer, parse_number


DEFAULT_SOURCE_ROOT = Path("datasets") / "mmqa_source"
DEFAULT_OUTPUT_ROOT = Path("datasets") / "mmqa"
DEFAULT_GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1XQ9djKSK4yjxLWAmHMzsyPAKMqdCIpXo"
)

# Map source file names to split labels. MMQA's multi-table subset ships as
# Synthesized_two_table.json / Synthesized_three_table.json.
SPLIT_FILE_MAP = {
    "two_table": "Synthesized_two_table.json",
    "three_table": "Synthesized_three_table.json",
}
DEFAULT_SPLITS = ("two_table", "three_table")
DEFAULT_GOOGLE_DRIVE_FILE_IDS = {
    # Public file IDs observed in the official multi_table_data Drive folder.
    # The downloader still tries to rediscover IDs from the folder page first,
    # so this mapping is a stable fallback rather than a per-case shortcut.
    "two_table": "1PWq95gk_8Fs46XiSFl9d7JckWBDvpcEc",
    "three_table": "1MkArlHyNSZ5rnHBl0117T6BJ6tIl5C_d",
}
DriveDownloader = Callable[[str, Path], None]


def download_and_convert_mmqa(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    splits: Sequence[str] = DEFAULT_SPLITS,
    case_ids: Sequence[int] | None = None,
    limit_cases: int | None = None,
    overwrite: bool = False,
    download: bool = True,
    download_overwrite: bool = False,
    drive_folder_url: str = DEFAULT_GOOGLE_DRIVE_FOLDER_URL,
    drive_file_ids: dict[str, str] | None = None,
    downloader: DriveDownloader | None = None,
) -> dict[str, Any]:
    """Download and convert the MMQA multi-table release into CLOVER-ready folders.

    Each unique table-set becomes one dataset directory holding the shared
    ``table_1.csv``/``table_2.csv``/... files plus a ``cases.jsonl`` of every
    question that uses those tables. This mirrors WikiTQ's layout (one dir per
    table, many cases) and avoids duplicating CSVs for synthesized data where
    many cases share the same underlying tables.
    """

    source_path = _resolve_path(source_root)
    output_path = _resolve_path(output_root)
    if limit_cases is not None and limit_cases <= 0:
        raise ValueError("limit_cases must be positive")

    normalized_splits = _normalize_splits(splits)
    download_summary = _ensure_source_files(
        source_path=source_path,
        splits=normalized_splits,
        download=download,
        download_overwrite=download_overwrite,
        drive_folder_url=drive_folder_url,
        drive_file_ids=drive_file_ids,
        downloader=downloader,
    )
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
        "stage": "mmqa_google_drive_download_and_conversion"
        if download_summary.get("download_enabled")
        else "mmqa_local_conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": _portable_path(source_path),
        "output_root": _portable_path(output_path),
        "download": download_summary,
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
        description="Download/convert the MMQA multi-table release for CLOVER eval."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--drive-folder-url",
        default=DEFAULT_GOOGLE_DRIVE_FOLDER_URL,
        help="Public Google Drive folder URL for the MMQA multi_table_data release.",
    )
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
    parser.add_argument(
        "--download-overwrite",
        action="store_true",
        help="Redownload raw Google Drive files even when they already exist.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only convert existing files under --source-root.",
    )
    parser.add_argument(
        "--drive-file-id",
        action="append",
        default=None,
        metavar="SPLIT=FILE_ID",
        help=(
            "Override a Google Drive file id for a split. "
            "May be repeated, e.g. --drive-file-id two_table=..."
        ),
    )
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
        download=not args.no_download,
        download_overwrite=args.download_overwrite,
        drive_folder_url=args.drive_folder_url,
        drive_file_ids=_expand_drive_file_ids(args.drive_file_id),
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


def _ensure_source_files(
    *,
    source_path: Path,
    splits: Sequence[str],
    download: bool,
    download_overwrite: bool,
    drive_folder_url: str,
    drive_file_ids: dict[str, str] | None,
    downloader: DriveDownloader | None,
) -> dict[str, Any]:
    source_path.mkdir(parents=True, exist_ok=True)
    expected_files = {split: source_path / SPLIT_FILE_MAP[split] for split in splits}
    missing = [
        split
        for split, path in expected_files.items()
        if download_overwrite or not path.is_file()
    ]
    if not missing:
        summary = {
            "stage": "mmqa_source_files",
            "download_enabled": bool(download),
            "source": "existing",
            "source_root": _portable_path(source_path),
            "splits": list(splits),
            "downloaded_files": [],
            "skipped_files": [
                _portable_path(path)
                for path in expected_files.values()
                if path.is_file()
            ],
        }
        _write_json(source_path / "download_summary.json", summary)
        return summary
    if not download:
        missing_paths = ", ".join(_portable_path(expected_files[split]) for split in missing)
        raise FileNotFoundError(
            "MMQA source files are missing and --no-download was set: "
            f"{missing_paths}"
        )

    file_ids = dict(DEFAULT_GOOGLE_DRIVE_FILE_IDS)
    discovered_ids = _discover_google_drive_file_ids(
        drive_folder_url=drive_folder_url,
        filenames=[SPLIT_FILE_MAP[split] for split in missing],
    )
    for filename, file_id in discovered_ids.items():
        split = _split_for_filename(filename)
        if split is not None:
            file_ids[split] = file_id
    if drive_file_ids:
        file_ids.update(drive_file_ids)

    active_downloader = downloader or _download_google_drive_file
    downloaded_files: list[str] = []
    skipped_files: list[str] = []
    used_file_ids: dict[str, str] = {}
    for split in splits:
        target = expected_files[split]
        if split not in missing:
            skipped_files.append(_portable_path(target))
            continue
        file_id = file_ids.get(split)
        if not file_id:
            raise ValueError(
                f"No Google Drive file id is known for MMQA split {split!r}. "
                "Pass --drive-file-id SPLIT=FILE_ID."
            )
        active_downloader(file_id, target)
        _validate_downloaded_json(target)
        downloaded_files.append(_portable_path(target))
        used_file_ids[split] = file_id

    summary = {
        "stage": "mmqa_google_drive_source_download",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "download_enabled": True,
        "source": "google_drive",
        "drive_folder_url": drive_folder_url,
        "source_root": _portable_path(source_path),
        "splits": list(splits),
        "downloaded_files": downloaded_files,
        "skipped_files": skipped_files,
        "file_ids": used_file_ids,
    }
    _write_json(source_path / "download_summary.json", summary)
    return summary


def _discover_google_drive_file_ids(
    *,
    drive_folder_url: str,
    filenames: Sequence[str],
) -> dict[str, str]:
    """Best-effort discovery of file IDs from a public Drive folder page."""

    if not drive_folder_url:
        return {}
    url = _drive_folder_page_url(drive_folder_url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (CLOVER MMQA downloader)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError):
        return {}
    page = html.unescape(page)
    discovered: dict[str, str] = {}
    for filename in filenames:
        index = page.find(filename)
        if index < 0:
            continue
        window = page[max(0, index - 1600) : index + 1600]
        candidates = re.findall(r"\b[0-9A-Za-z_-]{20,}\b", window)
        for candidate in candidates:
            if candidate == _drive_folder_id(drive_folder_url):
                continue
            if candidate.startswith("1"):
                discovered[filename] = candidate
                break
    return discovered


def _drive_folder_page_url(value: str) -> str:
    folder_id = _drive_folder_id(value)
    if not folder_id:
        return value
    return f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"


def _drive_folder_id(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"/folders/([0-9A-Za-z_-]+)", text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{20,}", text):
        return text
    return ""


def _download_google_drive_file(file_id: str, destination: Path) -> None:
    """Download one Google Drive file, using gdown when available."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError:
        _download_google_drive_file_with_urllib(file_id, destination)
        return
    url = f"https://drive.google.com/uc?id={urllib.parse.quote(file_id)}"
    result = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
    if not result:
        raise RuntimeError(f"gdown failed to download Google Drive file: {file_id}")


def _download_google_drive_file_with_urllib(file_id: str, destination: Path) -> None:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    base_url = "https://drive.google.com/uc?export=download"
    first_url = f"{base_url}&id={urllib.parse.quote(file_id)}"
    with opener.open(_drive_request(first_url), timeout=120) as response:
        payload = response.read()
        confirm = _drive_confirm_token(response, payload)
        form_url = _drive_download_form_url(payload)
        if confirm is None and not _looks_like_drive_html(payload):
            destination.write_bytes(payload)
            return
    if form_url is not None:
        second_url = form_url
    elif confirm is not None:
        second_url = (
            f"{base_url}&confirm={urllib.parse.quote(confirm)}"
            f"&id={urllib.parse.quote(file_id)}"
        )
    else:
        second_url = None
    if second_url is None:
        raise RuntimeError(
            f"Google Drive did not return a downloadable JSON file for id {file_id}."
        )
    with opener.open(_drive_request(second_url), timeout=120) as response:
        destination.write_bytes(response.read())


def _drive_download_form_url(payload: bytes) -> str | None:
    text = payload[:300000].decode("utf-8", errors="ignore")
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    if not form_match:
        return None
    action = html.unescape(form_match.group(1))
    params: dict[str, str] = {}
    for input_match in re.finditer(r"<input\b[^>]*>", text, flags=re.IGNORECASE):
        tag = input_match.group(0)
        name_match = re.search(r'\bname="([^"]+)"', tag, flags=re.IGNORECASE)
        value_match = re.search(r'\bvalue="([^"]*)"', tag, flags=re.IGNORECASE)
        if name_match and value_match:
            params[html.unescape(name_match.group(1))] = html.unescape(
                value_match.group(1)
            )
    if not params:
        return None
    return f"{action}?{urllib.parse.urlencode(params)}"


def _drive_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (CLOVER MMQA downloader)"},
    )


def _drive_confirm_token(response: Any, payload: bytes) -> str | None:
    cookies = response.headers.get_all("Set-Cookie") or []
    for cookie in cookies:
        match = re.search(r"download_warning[^=]*=([^;]+)", cookie)
        if match:
            return urllib.parse.unquote(match.group(1))
    text = payload[:200000].decode("utf-8", errors="ignore")
    for pattern in (
        r"confirm=([0-9A-Za-z_-]+)",
        r'name="confirm"\s+value="([^"]+)"',
    ):
        match = re.search(pattern, text)
        if match:
            return html.unescape(match.group(1))
    return None


def _looks_like_drive_html(payload: bytes) -> bool:
    prefix = payload[:4096].lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _validate_downloaded_json(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        preview = path.read_text(encoding="utf-8", errors="ignore")[:300]
        raise ValueError(
            f"Downloaded MMQA file is not JSON: {_portable_path(path)}. "
            f"Preview: {preview!r}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"Downloaded MMQA file is not a JSON list: {_portable_path(path)}")


def _split_for_filename(filename: str) -> str | None:
    for split, expected in SPLIT_FILE_MAP.items():
        if filename == expected:
            return split
    return None


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


def _expand_drive_file_ids(values: Sequence[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    overrides: dict[str, str] = {}
    for value in values:
        split, sep, file_id = str(value).partition("=")
        if not sep:
            raise ValueError(f"--drive-file-id must use SPLIT=FILE_ID: {value!r}")
        split = split.strip()
        file_id = file_id.strip()
        _normalize_splits([split])
        if not file_id:
            raise ValueError(f"Google Drive file id is empty for split {split!r}")
        overrides[split] = file_id
    return overrides


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
