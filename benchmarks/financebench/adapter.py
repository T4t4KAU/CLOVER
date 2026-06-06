"""FinanceBench dataset adapter."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUESTIONS_PATH = Path("data") / "financebench_open_source.jsonl"
DOCUMENT_INFO_PATH = Path("data") / "financebench_document_information.jsonl"
PDFS_DIR = "pdfs"


@dataclass(frozen=True)
class FinanceBenchTask:
    """One FinanceBench question normalized for benchmark runners."""

    task_dsl: dict[str, Any]
    base_dir: Path
    metadata: dict[str, Any]


def load_financebench_task(
    financebench_root: str | Path,
    *,
    case_id: str,
) -> FinanceBenchTask:
    """Load one FinanceBench case as a CLOVER document task DSL."""

    root = Path(financebench_root).expanduser().resolve()
    row = _row_by_case_id(root).get(case_id)
    if row is None:
        raise ValueError(f"FinanceBench case id not found: {case_id}")
    doc_info = _document_info_by_name(root).get(row["doc_name"], {})
    pdf_path = find_pdf_path(root, str(row["doc_name"]))
    task_dsl = task_dsl_from_row(row, pdf_path=pdf_path)
    metadata = {
        "dataset": "financebench",
        "dataset_id": "financebench",
        "case_id": row["financebench_id"],
        "expected_answer": row.get("answer"),
        "case": copy.deepcopy(row),
        "document_info": copy.deepcopy(doc_info),
        "pdf_path": str(pdf_path),
    }
    return FinanceBenchTask(task_dsl=task_dsl, base_dir=pdf_path.parent, metadata=metadata)


def task_dsl_from_row(row: dict[str, Any], *, pdf_path: Path) -> dict[str, Any]:
    """Build the document reasoning task DSL used by CLOVER examples."""

    return {
        "task_type": "document_reasoning",
        "question": row["question"],
        "sources": [
            {
                "id": 0,
                "type": "pdf",
                "file": str(pdf_path),
                "doc_name": row.get("doc_name"),
            }
        ],
        "answer": {
            "name": "answer",
            "type": "string",
        },
    }


def list_financebench_cases(financebench_root: str | Path) -> list[dict[str, Any]]:
    """Return all open-source FinanceBench cases in file order."""

    root = Path(financebench_root).expanduser().resolve()
    cases = []
    for index, row in enumerate(read_jsonl(root / QUESTIONS_PATH)):
        cases.append(
            {
                "dataset_id": "financebench",
                "case_id": row["financebench_id"],
                "case_index": index,
                "answer_type": "string",
                "question_reasoning": row.get("question_reasoning"),
                "question_type": row.get("question_type"),
                "doc_name": row.get("doc_name"),
                "company": row.get("company"),
            }
        )
    return cases


def select_cases(
    *,
    financebench_root: str | Path,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    sample_size: int | None = None,
    seed: int = 20260529,
    question_reasoning: str | None = None,
) -> list[dict[str, Any]]:
    """Select FinanceBench cases with Databench-style sampling semantics."""

    if max_cases == 0:
        return []
    cases = list_financebench_cases(financebench_root)
    if question_reasoning:
        cases = [
            case
            for case in cases
            if str(case.get("question_reasoning")) == question_reasoning
        ]
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        rng = random.Random(seed)
        cases = rng.sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def find_pdf_path(financebench_root: str | Path, doc_name: str) -> Path:
    """Resolve the PDF path for a FinanceBench document name."""

    pdf_dir = Path(financebench_root).expanduser().resolve() / PDFS_DIR
    exact = pdf_dir / f"{doc_name}.pdf"
    if exact.is_file():
        return exact
    matches = sorted(pdf_dir.glob(f"{doc_name}*.pdf"))
    if len(matches) == 1:
        return matches[0]
    lower_name = f"{doc_name}.pdf".lower()
    ci_matches = [path for path in pdf_dir.glob("*.pdf") if path.name.lower() == lower_name]
    if len(ci_matches) == 1:
        return ci_matches[0]
    raise FileNotFoundError(f"FinanceBench PDF not found for document: {doc_name}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"FinanceBench file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_by_case_id(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["financebench_id"]): row
        for row in read_jsonl(root / QUESTIONS_PATH)
    }


def _document_info_by_name(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["doc_name"]): row
        for row in read_jsonl(root / DOCUMENT_INFO_PATH)
    }
