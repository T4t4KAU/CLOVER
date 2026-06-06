from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clover.resource import prepare_physical_plan_resources


class ResourceProcessingTest(unittest.TestCase):
    def test_empty_resource_processing_is_noop_and_removed(self) -> None:
        physical_plan = {
            "task_type": "table_reasoning.query",
            "resources": [],
            "resource_processing": [],
            "nodes": [],
            "edges": [],
        }

        prepared = prepare_physical_plan_resources(physical_plan)

        self.assertNotIn("resource_processing", prepared)
        self.assertIn("resource_processing", physical_plan)
        self.assertEqual(prepared["nodes"], [])

    @unittest.skipUnless(
        importlib.util.find_spec("fitz") is not None,
        "PyMuPDF is required for PDF resource processing tests",
    )
    def test_prepares_pdf_chunks_and_rewrites_document_map_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "report.pdf"
            _write_pdf(
                pdf_path,
                [
                    "Revenue was 100. Operating income was 20. "
                    "The annual report includes detailed financial tables.",
                    "Revenue was 140. Operating income was 35. "
                    "The annual report includes more disclosures.",
                ],
            )
            physical_plan = _document_physical_plan(pdf_path)

            with patch.dict(
                "os.environ",
                {"CLOVER_RESOURCE_CACHE_ROOT": str(root / "resource_cache")},
            ):
                prepared = prepare_physical_plan_resources(physical_plan)
                first_resource_path_exists = Path(
                    prepared["resources"][0]["path"]
                ).is_file()

        self.assertNotIn("resource_processing", prepared)
        self.assertGreaterEqual(len(prepared["resources"]), 2)
        self.assertTrue(
            all(resource["type"] == "document_chunk" for resource in prepared["resources"])
        )
        self.assertEqual(
            prepared["map_groups"][0]["input"]["chunks"],
            [resource["id"] for resource in prepared["resources"]],
        )
        first_resource = prepared["resources"][0]
        self.assertTrue(first_resource["id"].startswith("document_1:chunk_"))
        self.assertEqual(first_resource["source"], "document_1")
        self.assertEqual(first_resource["format"], "text")
        self.assertTrue(first_resource["content_ref"].startswith("resource_cache://rc_"))
        self.assertTrue(first_resource_path_exists)
        self.assertIn("char_start", first_resource)
        self.assertIn("page_start", first_resource)


def _document_physical_plan(pdf_path: Path) -> dict:
    return {
        "task_type": "document_reasoning",
        "resources": [
            {
                "id": "document_1",
                "type": "document",
                "source_type": "pdf",
                "path": str(pdf_path),
                "format": "pdf",
                "schema": {
                    "format": "pdf",
                    "page_count": 2,
                    "page_indexing": "zero_based",
                    "text_extraction": {"extractor": "pymupdf"},
                    "chunking": {
                        "strategy": "sliding_window",
                        "unit": "char",
                        "size": 80,
                        "overlap": 10,
                        "preserve_page_spans": True,
                        "chunk_count": 2,
                    },
                },
            }
        ],
        "resource_processing": [
            {
                "id": "RP0",
                "op": "extract_text",
                "source": "document_1",
                "output": "document_1.text",
                "params": {
                    "extractor": "pymupdf",
                    "page_indexing": "zero_based",
                },
            },
            {
                "id": "RP1",
                "op": "chunk_text",
                "source": "document_1.text",
                "output": "document_1.chunks",
                "params": {
                    "strategy": "sliding_window",
                    "unit": "char",
                    "size": 80,
                    "overlap": 10,
                    "preserve_page_spans": True,
                },
            },
        ],
        "map_groups": [
            {
                "id": "G0",
                "op": "map",
                "input": {
                    "resource_view": "document_1.chunks",
                    "chunks": "all",
                },
                "params": {"local_instruction": "Extract revenue values."},
                "output": "G0",
                "output_type": "jsonl",
            }
        ],
        "edges": [],
    }


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    import fitz

    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(path)
    finally:
        document.close()


if __name__ == "__main__":
    unittest.main()
