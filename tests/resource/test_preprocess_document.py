from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clover.resource import preprocess_task_dsl


class DocumentPreprocessTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("fitz") is not None,
        "PyMuPDF is required for PDF preprocessing tests",
    )
    def test_preprocesses_pdf_document_to_local_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pdf(
                root / "document.pdf",
                [
                    "Revenue was 100. Operating income was 20.",
                    "Cash flow from operations was 30.",
                ],
            )
            task_dsl = {
                "task_type": "document_reasoning",
                "question": "What was the operating margin?",
                "sources": [
                    {
                        "id": 0,
                        "type": "pdf",
                        "file": "document.pdf",
                        "doc_name": "toy_annual_report",
                    }
                ],
                "answer": {"name": "answer", "type": "string"},
            }

            with patch.dict(
                "os.environ",
                {"CLOVER_RESOURCE_CACHE_ROOT": str(root / "resource_cache")},
            ):
                result = preprocess_task_dsl(task_dsl, base_dir=root)

            local_source = result["local_dsl"]["sources"][0]
            remote_source = result["remote_dsl"]["sources"][0]
            context = result["context"]
            cache = context["resource_cache"]["document_1"]["chunks"]
            chunk_map = context["chunk_map"]["document_1"]
            chunks_text = Path(cache["artifacts"]["chunks"]["path"]).read_text(
                encoding="utf-8"
            )

        self.assertEqual(local_source["id"], "document_1")
        self.assertEqual(local_source["type"], "document")
        self.assertEqual(local_source["source_type"], "pdf")
        self.assertEqual(local_source["format"], "pdf")
        self.assertEqual(local_source["doc_name"], "toy_annual_report")
        self.assertEqual(local_source["schema"]["page_count"], 2)
        self.assertEqual(local_source["schema"]["page_indexing"], "zero_based")
        self.assertTrue(
            local_source["schema"]["text_extraction"]["text_layer_available"]
        )
        self.assertFalse(local_source["schema"]["text_extraction"]["ocr_required"])
        self.assertEqual(local_source["schema"]["chunking"]["strategy"], "sliding_window")
        self.assertEqual(local_source["schema"]["chunking"]["unit"], "char")
        self.assertEqual(local_source["schema"]["chunking"]["size"], 3000)
        self.assertEqual(local_source["schema"]["chunking"]["overlap"], 20)
        self.assertEqual(local_source["schema"]["chunking"]["chunk_count"], 1)

        self.assertEqual(remote_source["id"], "document_1")
        self.assertEqual(remote_source["type"], "document")
        self.assertEqual(remote_source["format"], "pdf")
        self.assertEqual(remote_source["doc_name"], "toy_annual_report")
        self.assertNotIn("path", remote_source)
        self.assertNotIn("file", remote_source)
        remote_payload = json.dumps(result["remote_dsl"])
        self.assertNotIn(str(local_source["path"]), remote_payload)
        self.assertNotIn("Revenue was 100", remote_payload)

        self.assertEqual(chunk_map["page_indexing"], "zero_based")
        self.assertEqual(chunk_map["default_strategy"], "sliding_window")
        self.assertEqual(chunk_map["unit"], "char")
        self.assertEqual(chunk_map["size"], 3000)
        self.assertEqual(chunk_map["overlap"], 20)
        self.assertEqual(len(chunk_map["chunks"]), 1)
        self.assertEqual(chunk_map["chunks"][0]["chunk_id"], "chunk_0")
        self.assertEqual(chunk_map["chunks"][0]["page_start"], 0)
        self.assertEqual(chunk_map["chunks"][0]["page_end"], 1)
        self.assertEqual(chunk_map["chunks"][0]["char_start"], 0)
        self.assertTrue(
            chunk_map["chunks"][0]["content_ref"].startswith("resource_cache://rc_")
        )
        self.assertTrue(cache["entry_dir"].startswith(str(Path(tmpdir).resolve())))
        self.assertIn("Revenue was 100", chunks_text)


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
