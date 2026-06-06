from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.financebench.adapter import load_financebench_task
from benchmarks.financebench.download import (
    convert_financebench_rows,
    download_and_convert_financebench,
)
from benchmarks.financebench.eval import load_financebench_document_examples


class FinanceBenchDownloadConversionTest(unittest.TestCase):
    def test_converts_numerical_rows_to_dataset_and_examples_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "datasets" / "financebench"
            rows, infos = _write_fixture(source_root)

            summary = convert_financebench_rows(
                rows=rows,
                document_infos=infos,
                source_root=source_root,
                output_root=output_root,
            )

            index_rows = _read_jsonl(output_root / "index.jsonl")
            filtered_rows = _read_jsonl(
                output_root / "data" / "financebench_open_source.jsonl"
            )
            task_spec = _read_json(output_root / "case_1" / "task.json")
            metadata = _read_json(output_root / "case_1" / "metadata.json")
            examples = load_financebench_document_examples(examples_root=output_root)
            raw_task = load_financebench_task(output_root, case_id="financebench_id_1")
            case_document_is_symlink = (output_root / "case_1" / "document.pdf").is_symlink()

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["document_count"], 2)
        self.assertEqual(len(index_rows), 2)
        self.assertEqual(len(filtered_rows), 2)
        self.assertTrue(
            all(row["question_reasoning"] == "Numerical reasoning" for row in filtered_rows)
        )
        self.assertEqual(task_spec["task_type"], "document_reasoning")
        self.assertEqual(task_spec["sources"][0]["file"], "document.pdf")
        self.assertEqual(metadata["expected_answer"], "$10.00")
        self.assertEqual(metadata["source_dataset_dir"], "<source>")
        self.assertEqual(
            metadata["source_questions_file"],
            "<source>/data/financebench_open_source.jsonl",
        )
        self.assertEqual(
            metadata["source_pdf"],
            "../pdfs/DOC_2023_10K.pdf",
        )
        self.assertTrue(case_document_is_symlink)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].case_id, "financebench_id_1")
        self.assertEqual(
            examples[0].task_dsl["sources"][0]["file"],
            "../pdfs/DOC_2023_10K.pdf",
        )
        self.assertEqual(raw_task.task_dsl["task_type"], "document_reasoning")
        self.assertEqual(raw_task.metadata["expected_answer"], "$10.00")

    def test_refuses_to_overwrite_existing_output_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "datasets" / "financebench"
            rows, infos = _write_fixture(source_root)
            convert_financebench_rows(
                rows=rows,
                document_infos=infos,
                source_root=source_root,
                output_root=output_root,
            )

            with self.assertRaises(FileExistsError):
                convert_financebench_rows(
                    rows=rows,
                    document_infos=infos,
                    source_root=source_root,
                    output_root=output_root,
                )

    def test_summary_uses_portable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "datasets" / "financebench"
            rows, infos = _write_fixture(source_root)

            summary = convert_financebench_rows(
                rows=rows,
                document_infos=infos,
                source_root=source_root,
                output_root=output_root,
            )
            saved_summary = _read_json(output_root / "conversion_summary.json")

        self.assertEqual(summary["source_root"], "<source>")
        self.assertEqual(summary["output_root"], "<output>")
        self.assertEqual(summary["pdfs_dir"], "<output>/pdfs")
        self.assertEqual(saved_summary["cases_index"], "<output>/index.jsonl")
        self.assertNotIn(str(source_root), json.dumps(saved_summary))
        self.assertNotIn(str(output_root), json.dumps(saved_summary))

    def test_downloads_sources_then_converts_selected_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "datasets" / "financebench"
            rows, infos = _fixture_records()
            downloaded_urls: list[str] = []

            def fake_downloader(
                url: str,
                destination: Path,
                overwrite: bool,
                timeout_seconds: float,
            ) -> bool:
                downloaded_urls.append(url)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.name == "financebench_open_source.jsonl":
                    _write_jsonl(destination, rows)
                elif destination.name == "financebench_document_information.jsonl":
                    _write_jsonl(destination, infos)
                elif destination.suffix == ".pdf":
                    destination.write_bytes(b"%PDF-1.4\n")
                else:  # pragma: no cover - defensive guard for new source files.
                    raise AssertionError(f"Unexpected download destination: {destination}")
                return True

            summary = download_and_convert_financebench(
                source_root=source_root,
                output_root=output_root,
                overwrite=True,
                downloader=fake_downloader,
            )
            saved_summary = _read_json(output_root / "conversion_summary.json")
            questions_downloaded = (
                source_root / "data" / "financebench_open_source.jsonl"
            ).exists()
            pdf_2023_downloaded = (source_root / "pdfs" / "DOC_2023_10K.pdf").exists()
            pdf_2022_downloaded = (source_root / "pdfs" / "DOC_2022_10K.pdf").exists()
            unused_pdf_downloaded = (source_root / "pdfs" / "DOC_UNUSED_10K.pdf").exists()

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["download"]["github_repo"], "patronus-ai/financebench")
        self.assertEqual(summary["download"]["pdf_count"], 2)
        self.assertTrue(questions_downloaded)
        self.assertTrue(pdf_2023_downloaded)
        self.assertTrue(pdf_2022_downloaded)
        self.assertFalse(unused_pdf_downloaded)
        self.assertIn(
            "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/"
            "financebench_open_source.jsonl",
            downloaded_urls,
        )
        self.assertEqual(saved_summary["stage"], "financebench_download_and_conversion")


def _write_fixture(root: Path) -> tuple[list[dict], list[dict]]:
    (root / "pdfs").mkdir(parents=True)
    rows, infos = _fixture_records()
    for name in ("DOC_2022_10K", "DOC_2023_10K"):
        (root / "pdfs" / f"{name}.pdf").write_bytes(b"%PDF-1.4\n")
    return rows, infos


def _fixture_records() -> tuple[list[dict], list[dict]]:
    rows = [
        {
            "financebench_id": "financebench_id_1",
            "company": "ExampleCo",
            "doc_name": "DOC_2023_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Numerical reasoning",
            "question": "What is revenue?",
            "answer": "$10.00",
            "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [
                {
                    "doc_name": "DOC_2023_10K",
                    "evidence_page_num": 7,
                    "evidence_text": "Revenue was $10.",
                }
            ],
        },
        {
            "financebench_id": "financebench_id_2",
            "company": "ExampleCo",
            "doc_name": "DOC_2023_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Information extraction",
            "question": "What is cash?",
            "answer": "$3.00",
        },
        {
            "financebench_id": "financebench_id_3",
            "company": "ExampleCo",
            "doc_name": "DOC_2022_10K",
            "question_type": "domain-relevant",
            "question_reasoning": "Numerical reasoning",
            "question": "Did revenue grow?",
            "answer": "Yes, by $2.00.",
        },
        {
            "financebench_id": "financebench_id_4",
            "company": "ExampleCo",
            "doc_name": "DOC_UNUSED_10K",
            "question_type": "domain-relevant",
            "question_reasoning": "Information extraction",
            "question": "What is the unused item?",
            "answer": "$1.00.",
        },
    ]
    infos = [
        {
            "doc_name": "DOC_2022_10K",
            "company": "ExampleCo",
            "doc_type": "10k",
            "doc_period": 2022,
        },
        {
            "doc_name": "DOC_2023_10K",
            "company": "ExampleCo",
            "doc_type": "10k",
            "doc_period": 2023,
        },
        {
            "doc_name": "DOC_UNUSED_10K",
            "company": "ExampleCo",
            "doc_type": "10k",
            "doc_period": 2021,
        },
    ]
    return rows, infos


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
