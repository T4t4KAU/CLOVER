"""Static CSV schema extraction."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any


COMMON_DELIMITERS = {",", "\t", ";", "|"}


def extract_csv_schema(path: str | Path) -> dict[str, Any]:
    """Extract structural CSV schema.

    CSV files do not carry a strict column type system. The schema therefore
    records only the table shape and header names.
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

        row_count = 0

        for _row in reader:
            row_count += 1

    return {
        "format": "csv",
        "shape": {
            "rows": row_count,
            "columns": len(reader.fieldnames),
        },
        "columns": list(reader.fieldnames),
    }


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
