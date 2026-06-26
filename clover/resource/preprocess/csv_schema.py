"""Static CSV schema extraction."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any


COMMON_DELIMITERS = {",", "\t", ";", "|"}


_SAMPLE_MAX_ROWS = 5
_SAMPLE_TAIL_ROWS = 2
_SAMPLE_MAX_CHARS = 80
_ROW_SAMPLE_MAX_ROWS = 18
_ROW_SAMPLE_HEAD_ROWS = 10
_ROW_SAMPLE_TAIL_ROWS = 4
_ROW_SAMPLE_MAX_COLUMNS = 12
_ROW_SAMPLE_MAX_CHARS = 60


def extract_csv_schema(path: str | Path) -> dict[str, Any]:
    """Extract structural CSV schema.

    CSV files do not carry a strict column type system. The schema therefore
    records only the table shape and header names, plus a small sample of
    values per column so that the Supervisor can detect patterns (e.g. country
    codes hidden inside parentheses). The sample combines the first
    ``_SAMPLE_MAX_ROWS`` rows and the last ``_SAMPLE_TAIL_ROWS`` rows so that
    trailing summary rows (e.g. Total, Turnout) are visible to the Supervisor.
    """

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    csv.field_size_limit(sys.maxsize)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        dialect = _detect_dialect(sample)

        # Databench tables include long quoted text fields. Keep standard CSV
        # double-quote handling even when Sniffer guesses otherwise.
        reader = csv.DictReader(handle, dialect=dialect, doublequote=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        fieldnames = reader.fieldnames
        rows = list(reader)
        row_count = len(rows)

        column_samples: dict[str, list[str]] = {col: [] for col in fieldnames}

        # Indices of rows to sample: first _SAMPLE_MAX_ROWS + last _SAMPLE_TAIL_ROWS.
        head_end = min(_SAMPLE_MAX_ROWS, row_count)
        tail_start = max(head_end, row_count - _SAMPLE_TAIL_ROWS)
        sampled_indices = set(range(head_end)) | set(range(tail_start, row_count))

        for idx in sorted(sampled_indices):
            row = rows[idx]
            for col in fieldnames:
                val = row.get(col, "")
                if val is None:
                    continue
                truncated = str(val)[:_SAMPLE_MAX_CHARS]
                # Skip duplicates to keep the sample compact when head and tail
                # overlap or when the same value repeats.
                samples = column_samples[col]
                if truncated not in samples:
                    samples.append(truncated)

        row_samples = _compact_row_samples(rows, fieldnames)

    columns_detail = []
    for col in fieldnames:
        entry: dict[str, Any] = {"name": col}
        if column_samples.get(col):
            entry["sample"] = column_samples[col]
        columns_detail.append(entry)

    return {
        "format": "csv",
        "shape": {
            "rows": row_count,
            "columns": len(fieldnames),
        },
        "columns": list(fieldnames),
        "columns_detail": columns_detail,
        "row_samples": row_samples,
    }


def _compact_row_samples(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> list[dict[str, Any]]:
    """Return bounded row-level examples for preserving column relationships.

    Column-wise samples are compact, but they lose the row relationships needed
    by table fact-checking statements such as "street X has milepost Y".  This
    sample is intentionally small and character-limited so prompts gain those
    relationships without serialising full benchmark tables.
    """

    row_count = len(rows)
    if row_count == 0:
        return []
    if row_count <= _ROW_SAMPLE_MAX_ROWS:
        sampled_indices = list(range(row_count))
    else:
        head_end = min(_ROW_SAMPLE_HEAD_ROWS, row_count)
        tail_start = max(head_end, row_count - _ROW_SAMPLE_TAIL_ROWS)
        sampled_indices = list(range(head_end)) + list(range(tail_start, row_count))
    sampled_columns = list(fieldnames[:_ROW_SAMPLE_MAX_COLUMNS])
    samples: list[dict[str, Any]] = []
    for idx in sampled_indices:
        row = rows[idx]
        values = {
            col: _truncate_cell(row.get(col, ""))
            for col in sampled_columns
            if row.get(col, "") not in (None, "")
        }
        samples.append({"row": idx, "values": values})
    return samples


def _truncate_cell(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= _ROW_SAMPLE_MAX_CHARS:
        return text
    return text[: _ROW_SAMPLE_MAX_CHARS - 1].rstrip() + "…"


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.get_dialect("excel")

    # Sniffer can overfit text-heavy samples; reject unusual delimiters and
    # fall back to standard comma-separated CSV.
    if dialect.delimiter not in COMMON_DELIMITERS:
        return csv.get_dialect("excel")
    return dialect
