"""Download benchmark datasets and convert them into CLOVER's local layouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ["PYTHONWARNINGS"] = "ignore"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path = [
    path
    for path in sys.path
    if Path(path or os.getcwd()).resolve() != SCRIPT_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
loaded_warnings = sys.modules.get("warnings")
loaded_warnings_file = getattr(loaded_warnings, "__file__", None)
if loaded_warnings_file and Path(loaded_warnings_file).resolve() == SCRIPT_DIR / "warnings.py":
    del sys.modules["warnings"]
import warnings as _warnings

_warnings.simplefilter("ignore")

from benchmarks.databench.download import (
    DEFAULT_CONFIG_NAME as DATABENCH_DEFAULT_CONFIG,
    DEFAULT_DATASET_SOURCE,
    DEFAULT_REPO_ID as DATABENCH_DEFAULT_REPO,
    DEFAULT_SPLITS as DATABENCH_DEFAULT_SPLITS,
    DEFAULT_TABLE_KIND as DATABENCH_DEFAULT_TABLE_KIND,
    DATASET_SOURCES,
    download_and_convert_databench,
)
from benchmarks.financebench.download import (
    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS as FINANCEBENCH_DEFAULT_TIMEOUT,
    DEFAULT_GITHUB_REF as FINANCEBENCH_DEFAULT_GITHUB_REF,
    DEFAULT_GITHUB_REPO as FINANCEBENCH_DEFAULT_GITHUB_REPO,
    DEFAULT_PDF_MODE as FINANCEBENCH_DEFAULT_PDF_MODE,
    DEFAULT_QUESTION_REASONING as FINANCEBENCH_DEFAULT_QUESTION_REASONING,
    PDF_MODES,
    download_and_convert_financebench,
)
from benchmarks.tablebench.download import (
    DEFAULT_CONFIG_NAME as TABLEBENCH_DEFAULT_CONFIG,
    DEFAULT_REPO_ID as TABLEBENCH_DEFAULT_REPO,
    DEFAULT_SPLITS as TABLEBENCH_DEFAULT_SPLITS,
    download_and_convert_tablebench,
)
from benchmarks.tablefact.download import (
    DEFAULT_SOURCE_ROOT as TABLEFACT_DEFAULT_SOURCE_ROOT,
    DEFAULT_SPLITS as TABLEFACT_DEFAULT_SPLITS,
    convert_tablefact_release,
)
from benchmarks.wikitq.download import (
    DEFAULT_SOURCE_ROOT as WIKITQ_DEFAULT_SOURCE_ROOT,
    DEFAULT_SPLIT as WIKITQ_DEFAULT_SPLIT,
    download_and_convert_wikitq,
)

DEFAULT_DATASETS_ROOT = Path("datasets")


def download_and_convert_benchmarks(
    *,
    datasets: Sequence[str],
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    dataset_source: str = DEFAULT_DATASET_SOURCE,
    modelscope_cache_dir: str | Path | None = None,
    overwrite: bool = False,
    download_overwrite: bool = False,
    databench_repo_id: str = DATABENCH_DEFAULT_REPO,
    databench_config_name: str = DATABENCH_DEFAULT_CONFIG,
    databench_splits: Sequence[str] = DATABENCH_DEFAULT_SPLITS,
    databench_table_kind: str = DATABENCH_DEFAULT_TABLE_KIND,
    databench_dataset_ids: Sequence[str] | None = None,
    databench_limit_datasets: int | None = None,
    tablebench_repo_id: str = TABLEBENCH_DEFAULT_REPO,
    tablebench_config_name: str = TABLEBENCH_DEFAULT_CONFIG,
    tablebench_splits: Sequence[str] = TABLEBENCH_DEFAULT_SPLITS,
    tablebench_case_ids: Sequence[str] | None = None,
    tablebench_qtypes: Sequence[str] | None = None,
    tablebench_qsubtypes: Sequence[str] | None = None,
    tablebench_limit_cases: int | None = None,
    tablebench_include_visualization: bool = False,
    wikitq_source_root: str | Path = WIKITQ_DEFAULT_SOURCE_ROOT,
    wikitq_split: str = WIKITQ_DEFAULT_SPLIT,
    wikitq_case_ids: Sequence[str] | None = None,
    wikitq_limit_cases: int | None = None,
    tablefact_source_root: str | Path = TABLEFACT_DEFAULT_SOURCE_ROOT,
    tablefact_splits: Sequence[str] = TABLEFACT_DEFAULT_SPLITS,
    tablefact_case_ids: Sequence[str] | None = None,
    tablefact_limit_cases: int | None = None,
    financebench_download: bool = True,
    financebench_github_repo: str = FINANCEBENCH_DEFAULT_GITHUB_REPO,
    financebench_github_ref: str = FINANCEBENCH_DEFAULT_GITHUB_REF,
    financebench_download_timeout_seconds: float = FINANCEBENCH_DEFAULT_TIMEOUT,
    financebench_question_reasoning: str | None = FINANCEBENCH_DEFAULT_QUESTION_REASONING,
    financebench_case_ids: Sequence[str] | None = None,
    financebench_limit_cases: int | None = None,
    financebench_pdf_mode: str = FINANCEBENCH_DEFAULT_PDF_MODE,
) -> dict[str, Any]:
    """Download selected datasets and convert them to CLOVER-ready folders."""

    selected = _normalize_datasets(datasets)
    root = _resolve_path(datasets_root)
    root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    if "databench" in selected:
        summaries["databench"] = download_and_convert_databench(
            output_root=root / "databench",
            repo_id=databench_repo_id,
            config_name=databench_config_name,
            splits=tuple(databench_splits),
            table_kind=databench_table_kind,
            dataset_source=dataset_source,
            modelscope_cache_dir=(
                Path(modelscope_cache_dir) / "databench"
                if modelscope_cache_dir is not None
                else None
            ),
            dataset_ids=databench_dataset_ids,
            limit_datasets=databench_limit_datasets,
            overwrite=overwrite,
        )

    if "tablebench" in selected:
        summaries["tablebench"] = download_and_convert_tablebench(
            output_root=root / "tablebench",
            source_root=root / "tablebench_source",
            repo_id=tablebench_repo_id,
            config_name=tablebench_config_name,
            splits=tuple(tablebench_splits),
            dataset_source=dataset_source,
            modelscope_cache_dir=(
                Path(modelscope_cache_dir) / "tablebench"
                if modelscope_cache_dir is not None
                else None
            ),
            case_ids=tablebench_case_ids,
            qtypes=tablebench_qtypes,
            qsubtypes=tablebench_qsubtypes,
            include_visualization=tablebench_include_visualization,
            limit_cases=tablebench_limit_cases,
            overwrite=overwrite,
            download_overwrite=download_overwrite,
        )

    if "wikitq" in selected:
        summaries["wikitq"] = download_and_convert_wikitq(
            source_root=wikitq_source_root,
            output_root=root / "wikitq",
            split=wikitq_split,
            case_ids=wikitq_case_ids,
            limit_cases=wikitq_limit_cases,
            overwrite=overwrite,
        )

    if "tablefact" in selected:
        summaries["tablefact"] = convert_tablefact_release(
            source_root=tablefact_source_root,
            output_root=root / "tablefact",
            splits=tuple(tablefact_splits),
            case_ids=tablefact_case_ids,
            limit_cases=tablefact_limit_cases,
            overwrite=overwrite,
        )

    if "financebench" in selected:
        summaries["financebench"] = download_and_convert_financebench(
            output_root=root / "financebench",
            source_root=root / "financebench_source",
            download=financebench_download,
            github_repo=financebench_github_repo,
            github_ref=financebench_github_ref,
            download_overwrite=download_overwrite,
            download_timeout_seconds=financebench_download_timeout_seconds,
            question_reasoning=financebench_question_reasoning,
            case_ids=financebench_case_ids,
            limit_cases=financebench_limit_cases,
            pdf_mode=financebench_pdf_mode,
            overwrite=overwrite,
        )

    summary = {
        "stage": "benchmark_dataset_download",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets_root": _portable_path(root),
        "dataset_source": dataset_source,
        "datasets": sorted(selected),
        "outputs": {
            name: summary.get("output_root")
            for name, summary in summaries.items()
            if isinstance(summary, dict)
        },
        "summaries": summaries,
    }
    _write_json(root / "download_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download benchmark datasets and convert them into CLOVER's "
            "local evaluation layout. TableBench visualization cases are "
            "excluded by default."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="{databench,tablebench,wikitq,tablefact,financebench,all}",
        help=(
            "Dataset to prepare. Repeat for multiple datasets. Defaults to "
            "databench, tablebench, and financebench. Choose wikitq or "
            "tablefact explicitly when a local source root is available."
        ),
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Root directory for converted datasets, default: datasets.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=DATASET_SOURCES,
        default=DEFAULT_DATASET_SOURCE,
        help=(
            "Dataset hub for DataBench/TableBench downloads. FinanceBench "
            "continues to use its GitHub source."
        ),
    )
    parser.add_argument(
        "--modelscope-cache-dir",
        default=None,
        help=(
            "Optional local_dir root passed to ModelScope "
            "dataset_snapshot_download."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--download-overwrite",
        action="store_true",
        help="Redownload cached raw files where the dataset downloader supports it.",
    )

    databench = parser.add_argument_group("DataBench")
    databench.add_argument("--databench-repo-id", default=DATABENCH_DEFAULT_REPO)
    databench.add_argument("--databench-config-name", default=DATABENCH_DEFAULT_CONFIG)
    databench.add_argument(
        "--databench-split",
        dest="databench_splits",
        action="append",
        default=None,
    )
    databench.add_argument(
        "--databench-table-kind",
        choices=("all", "sample"),
        default=DATABENCH_DEFAULT_TABLE_KIND,
    )
    databench.add_argument(
        "--databench-dataset-id",
        action="append",
        default=None,
        help="DataBench dataset id to include. Repeat or comma-separate.",
    )
    databench.add_argument("--databench-limit-datasets", type=int, default=None)

    tablebench = parser.add_argument_group("TableBench")
    tablebench.add_argument("--tablebench-repo-id", default=TABLEBENCH_DEFAULT_REPO)
    tablebench.add_argument("--tablebench-config-name", default=TABLEBENCH_DEFAULT_CONFIG)
    tablebench.add_argument(
        "--tablebench-split",
        dest="tablebench_splits",
        action="append",
        default=None,
    )
    tablebench.add_argument(
        "--tablebench-case-id",
        action="append",
        default=None,
        help="TableBench case id to include. Repeat or comma-separate.",
    )
    tablebench.add_argument(
        "--tablebench-qtype",
        action="append",
        default=None,
        help="TableBench qtype to include. Repeat or comma-separate.",
    )
    tablebench.add_argument(
        "--tablebench-qsubtype",
        action="append",
        default=None,
        help="TableBench qsubtype to include. Repeat or comma-separate.",
    )
    tablebench.add_argument("--tablebench-limit-cases", type=int, default=None)
    tablebench.add_argument(
        "--include-tablebench-visualization",
        action="store_true",
        help=(
            "Include TableBench Visualization cases. Default is false so the "
            "converted dataset contains only non-visual cases."
        ),
    )

    wikitq = parser.add_argument_group("WikiTableQuestions")
    wikitq.add_argument("--wikitq-source-root", default=str(WIKITQ_DEFAULT_SOURCE_ROOT))
    wikitq.add_argument("--wikitq-split", default=WIKITQ_DEFAULT_SPLIT)
    wikitq.add_argument(
        "--wikitq-case-id",
        action="append",
        default=None,
        help="WikiTQ case id to include. Repeat or comma-separate.",
    )
    wikitq.add_argument("--wikitq-limit-cases", type=int, default=None)

    tablefact = parser.add_argument_group("TableFact / TabFact")
    tablefact.add_argument(
        "--tablefact-source-root",
        "--tabfact-source-root",
        dest="tablefact_source_root",
        default=str(TABLEFACT_DEFAULT_SOURCE_ROOT),
    )
    tablefact.add_argument(
        "--tablefact-split",
        "--tabfact-split",
        dest="tablefact_splits",
        action="append",
        default=None,
    )
    tablefact.add_argument(
        "--tablefact-case-id",
        "--tabfact-case-id",
        dest="tablefact_case_id",
        action="append",
        default=None,
        help="TableFact case id to include. Repeat or comma-separate.",
    )
    tablefact.add_argument(
        "--tablefact-limit-cases",
        "--tabfact-limit-cases",
        dest="tablefact_limit_cases",
        type=int,
        default=None,
    )

    financebench = parser.add_argument_group("FinanceBench")
    financebench.add_argument(
        "--financebench-skip-download",
        dest="financebench_download",
        action="store_false",
        help="Use an existing FinanceBench source cache instead of downloading.",
    )
    financebench.set_defaults(financebench_download=True)
    financebench.add_argument(
        "--financebench-github-repo",
        default=FINANCEBENCH_DEFAULT_GITHUB_REPO,
    )
    financebench.add_argument(
        "--financebench-github-ref",
        default=FINANCEBENCH_DEFAULT_GITHUB_REF,
    )
    financebench.add_argument(
        "--financebench-download-timeout-seconds",
        type=float,
        default=FINANCEBENCH_DEFAULT_TIMEOUT,
    )
    financebench.add_argument(
        "--financebench-question-reasoning",
        default=FINANCEBENCH_DEFAULT_QUESTION_REASONING,
        help="Use an empty string to include all FinanceBench reasoning types.",
    )
    financebench.add_argument(
        "--financebench-case-id",
        action="append",
        default=None,
        help="FinanceBench case id to include. Repeat or comma-separate.",
    )
    financebench.add_argument("--financebench-limit-cases", type=int, default=None)
    financebench.add_argument(
        "--financebench-pdf-mode",
        choices=PDF_MODES,
        default=FINANCEBENCH_DEFAULT_PDF_MODE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = download_and_convert_benchmarks(
        datasets=args.dataset or ("all",),
        datasets_root=args.datasets_root,
        dataset_source=args.dataset_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        overwrite=args.overwrite,
        download_overwrite=args.download_overwrite,
        databench_repo_id=args.databench_repo_id,
        databench_config_name=args.databench_config_name,
        databench_splits=tuple(args.databench_splits or DATABENCH_DEFAULT_SPLITS),
        databench_table_kind=args.databench_table_kind,
        databench_dataset_ids=_expand_csv_values(args.databench_dataset_id),
        databench_limit_datasets=args.databench_limit_datasets,
        tablebench_repo_id=args.tablebench_repo_id,
        tablebench_config_name=args.tablebench_config_name,
        tablebench_splits=tuple(args.tablebench_splits or TABLEBENCH_DEFAULT_SPLITS),
        tablebench_case_ids=_expand_csv_values(args.tablebench_case_id),
        tablebench_qtypes=_expand_csv_values(args.tablebench_qtype),
        tablebench_qsubtypes=_expand_csv_values(args.tablebench_qsubtype),
        tablebench_limit_cases=args.tablebench_limit_cases,
        tablebench_include_visualization=args.include_tablebench_visualization,
        wikitq_source_root=args.wikitq_source_root,
        wikitq_split=args.wikitq_split,
        wikitq_case_ids=_expand_csv_values(args.wikitq_case_id),
        wikitq_limit_cases=args.wikitq_limit_cases,
        tablefact_source_root=args.tablefact_source_root,
        tablefact_splits=tuple(args.tablefact_splits or TABLEFACT_DEFAULT_SPLITS),
        tablefact_case_ids=_expand_csv_values(args.tablefact_case_id),
        tablefact_limit_cases=args.tablefact_limit_cases,
        financebench_download=args.financebench_download,
        financebench_github_repo=args.financebench_github_repo,
        financebench_github_ref=args.financebench_github_ref,
        financebench_download_timeout_seconds=(
            args.financebench_download_timeout_seconds
        ),
        financebench_question_reasoning=(
            args.financebench_question_reasoning or None
        ),
        financebench_case_ids=_expand_csv_values(args.financebench_case_id),
        financebench_limit_cases=args.financebench_limit_cases,
        financebench_pdf_mode=args.financebench_pdf_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _normalize_datasets(datasets: Sequence[str]) -> set[str]:
    expanded = set(_expand_csv_values(datasets))
    if not expanded or "all" in expanded:
        expanded = {"databench", "tablebench", "financebench"}
    unknown = sorted(
        expanded
        - {"databench", "tablebench", "wikitq", "tablefact", "tabfact", "financebench"}
    )
    if unknown:
        raise ValueError(f"Unknown dataset selection: {unknown}")
    if "tabfact" in expanded:
        expanded.remove("tabfact")
        expanded.add("tablefact")
    return expanded


def _expand_csv_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    return expanded


def _resolve_path(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = REPO_ROOT / selected
    return selected.resolve()


def _portable_path(path: Path, *, base: Path = REPO_ROOT) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return (Path("<external>") / resolved.name).as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
