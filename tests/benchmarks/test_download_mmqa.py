from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.mmqa.download import download_and_convert_mmqa


class MMQADownloadTest(unittest.TestCase):
    def test_downloads_drive_release_then_converts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_root = root / "mmqa"
            downloaded: list[tuple[str, str]] = []

            def fake_downloader(file_id: str, destination: Path) -> None:
                downloaded.append((file_id, destination.name))
                split = (
                    "three_table"
                    if destination.name == "Synthesized_three_table.json"
                    else "two_table"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps([_mmqa_case(split)], ensure_ascii=False),
                    encoding="utf-8",
                )

            summary = download_and_convert_mmqa(
                source_root=source_root,
                output_root=output_root,
                splits=("two_table", "three_table"),
                overwrite=True,
                downloader=fake_downloader,
            )

            cases_paths = sorted(output_root.glob("*/*/cases.jsonl"))
            first_record = json.loads(
                cases_paths[0].read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(
            [name for _, name in downloaded],
            ["Synthesized_two_table.json", "Synthesized_three_table.json"],
        )
        self.assertEqual(summary["stage"], "mmqa_google_drive_download_and_conversion")
        self.assertEqual(summary["download"]["source"], "google_drive")
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(len(cases_paths), 2)
        self.assertIn(first_record["split"], {"two_table", "three_table"})
        self.assertEqual(first_record["table_count"], 2)


def _mmqa_case(split: str) -> dict[str, object]:
    return {
        "id_": 1 if split == "two_table" else 2,
        "Question": "Which artist has the winning album?",
        "table_names": ["albums", "artists"],
        "tables": [
            {
                "table_columns": ["album_id", "album", "artist_id"],
                "table_content": [["a1", "Blue", "p1"]],
            },
            {
                "table_columns": ["artist_id", "artist"],
                "table_content": [["p1", "Ada"]],
            },
        ],
        "foreign_keys": [["albums.artist_id", "artists.artist_id"]],
        "primary_keys": ["albums.album_id", "artists.artist_id"],
        "answer": "Ada",
        "SQL": (
            "SELECT artists.artist FROM albums JOIN artists "
            "ON albums.artist_id = artists.artist_id"
        ),
    }


if __name__ == "__main__":
    unittest.main()
