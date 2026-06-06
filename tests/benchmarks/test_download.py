from __future__ import annotations

import unittest

from benchmarks.download import build_arg_parser, _normalize_datasets


class BenchmarkDownloadScriptTest(unittest.TestCase):
    def test_dataset_selection_defaults_to_all(self) -> None:
        self.assertEqual(
            _normalize_datasets(("all",)),
            {"databench", "tablebench", "financebench"},
        )

    def test_dataset_selection_accepts_csv_values(self) -> None:
        self.assertEqual(
            _normalize_datasets(("databench,tablebench",)),
            {"databench", "tablebench"},
        )

    def test_tablebench_visualization_is_excluded_by_default(self) -> None:
        args = build_arg_parser().parse_args(["--dataset", "tablebench"])
        self.assertFalse(args.include_tablebench_visualization)

    def test_dataset_source_accepts_modelscope(self) -> None:
        args = build_arg_parser().parse_args(
            ["--dataset", "tablebench", "--dataset-source", "modelscope"]
        )
        self.assertEqual(args.dataset_source, "modelscope")


if __name__ == "__main__":
    unittest.main()
