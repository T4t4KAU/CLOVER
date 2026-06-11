"""Unified benchmark entry point for CLOVER.

This module only owns command-line parsing, shared progress display, optional
energy profiling, and run-summary normalization. Dataset-specific evaluation
logic stays in the DataBench, TableBench, and FinanceBench modules.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

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
shadow_warning_paths = {
    SCRIPT_DIR / "warnings.py",
    REPO_ROOT / "warnings.py",
}
if loaded_warnings_file and Path(loaded_warnings_file).resolve() in shadow_warning_paths:
    del sys.modules["warnings"]
warnings_import_path = sys.path[:]
sys.path = [
    path
    for path in sys.path
    if Path(path or os.getcwd()).resolve() not in {SCRIPT_DIR, REPO_ROOT}
]
import warnings as _warnings
sys.path = warnings_import_path

_warnings.simplefilter("ignore")

from benchmarks.energy import EnergyProfiler
from benchmarks.warnings import suppress_benchmark_warnings
from clover.config import load_model_config, load_optional_model_config
from clover.executor.slm_dispatcher import (
    DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    DEFAULT_MAX_PENDING_SLM_SEQUENCES,
)
from clover.supervisor.client import generate_remote_text


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "runs"
DEFAULT_REMOTE_LLM_CONFIG = REPO_ROOT / "model_config" / "remote_llm_config.json"
DEFAULT_SLM_CONFIG_PATH = REPO_ROOT / "model_config" / "local_slm_config.json"
DEFAULT_FINANCEBENCH_EXAMPLES_ROOT = REPO_ROOT / "datasets" / "financebench"
DEFAULT_STATIC_TOOL_RUN_DIR = (
    REPO_ROOT / "benchmark" / "runs" / "databench_full_static_merged_20260527"
)
DEFAULT_EDGE_ELECTRICITY_PRICE_USD_PER_KWH = 0.1883
DEFAULT_MAX_CONTEXT_CHARS = 500_000
DEFAULT_REMOTE_BATCH_SIZE = 64
DEFAULT_REMOTE_CONCURRENCY = 64
DEFAULT_MAX_PARALLEL_EXECUTION_UNITS = 64
DEFAULT_EVAL_WORKERS = 64
EVAL_MODE_IN_CONTEXT = "inContext"
SUPPORTED_EVAL_MODES = frozenset(
    {"closedBook", "inContext", "inContext_reverse", "oracle", "oracle_reverse"}
)
TABLEBENCH_INSTRUCTION_TYPES = frozenset({"DP", "TCoT", "PoT"})
SLM_SCHEDULER_TPTT = "tptt"
SLM_SCHEDULER_CHOICES = frozenset({SLM_SCHEDULER_TPTT, "fifo"})
REMOTE_LLM_ENV_PREFIXES = (
    "CLOVER_REMOTE_LLM",
    "CLOVER_LLM",
    "REMOTE_LLM",
    "LLM",
)
LOCAL_SLM_ENV_PREFIXES = (
    "CLOVER_LOCAL_SLM",
    "CLOVER_SLM",
    "LOCAL_SLM",
    "SLM",
)
SYNTHESIZE_LLM_ENV_PREFIXES = (
    "CLOVER_SYNTHESIZE_LLM",
    "CLOVER_SYNTHESIS_LLM",
    "SYNTHESIZE_LLM",
    "SYNTHESIS_LLM",
)
SAMPLING_CONFIG_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "reasoning_effort",
)
MODEL_API_CHECK_PROMPT = "Return exactly OK."
DEFAULT_MODEL_API_CHECK_TIMEOUT_SECONDS = 30.0
CLOVER_BANNER = r"""
   ________    ____ _    ____________
  / ____/ /   / __ \ |  / / ____/ __ \
 / /   / /   / / / / | / / __/ / /_/ /
/ /___/ /___/ /_/ /| |/ / /___/ _, _/
\____/_____/\____/ |___/_____/_/ |_|
""".strip("\n")


def main() -> None:
    args = _parse_args()
    _validate_modes(args)
    run_name = args.run_name or _default_run_name(args)
    output_dir = (args.output_root / run_name).expanduser().resolve()

    with suppress_benchmark_warnings():
        with EnergyProfiler(
            enabled=args.energy_profile,
            sample_ms=args.energy_sample_ms,
            baseline_seconds=args.energy_baseline_seconds,
            password_env=args.energy_password_env,
        ) as energy_profiler:
            summary = _run_selected_mode(args=args, output_dir=output_dir)

    summary = _normalize_summary(
        summary,
        output_dir=output_dir,
        energy_summary=energy_profiler.summary,
        electricity_price_usd_per_kwh=args.edge_electricity_price_usd_per_kwh,
    )
    _write_summary_if_possible(summary, output_dir=output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_selected_mode(
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    databench_root = args.databench_root.expanduser().resolve()
    tablebench_root = args.tablebench_root.expanduser().resolve()
    financebench_root = args.financebench_root.expanduser().resolve()
    case_ids = set(args.case_id or [])
    qtypes = set(args.qtype or [])
    qsubtypes = set(args.qsubtype or [])
    run_name = output_dir.name
    progress_factory = None if args.no_progress else EvalProgressBar

    if args.financebench_eval:
        from benchmarks.financebench.eval import run_financebench_document_eval

        remote_config = _load_remote_config(args)
        local_slm_config = _load_local_config(args)
        print_eval_startup_banner(
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            workflow="document_reasoning",
            max_parallel_execution_units=args.max_parallel_execution_units,
            max_parallel_slm_node_jobs=args.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=args.max_parallel_slm_sequences,
            max_pending_slm_sequences=args.max_pending_slm_sequences,
            slm_scheduler=args.slm_scheduler,
        )
        preflight_model_api_checks(
            args=args,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
        )
        return run_financebench_document_eval(
            examples_root=args.financebench_examples_root.expanduser().resolve(),
            output_dir=output_dir,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            max_cases=args.max_cases,
            case_ids=case_ids,
            sample_size=args.sample_size,
            seed=args.seed,
            max_retries=args.max_retries,
            max_parallel_execution_units=args.max_parallel_execution_units,
            max_parallel_slm_node_jobs=args.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=args.max_parallel_slm_sequences,
            max_pending_slm_sequences=args.max_pending_slm_sequences,
            max_workers=args.max_workers,
            node_timeout_seconds=args.node_timeout_seconds,
            overwrite=args.overwrite,
            remote_cost_model=args.remote_cost_model,
            progress_factory=progress_factory,
        )

    if args.financebench_remote_only_baseline:
        from benchmarks.financebench.remote_baseline import (
            run_financebench_remote_only_baseline,
        )

        remote_config = _load_remote_config(args)
        print_eval_startup_banner(
            remote_config=remote_config,
            local_slm_config=None,
            workflow=f"remote_only_financebench_{args.financebench_eval_mode}",
        )
        preflight_model_api_checks(
            args=args,
            remote_config=remote_config,
            local_slm_config=None,
        )
        return run_financebench_remote_only_baseline(
            financebench_root=financebench_root,
            output_dir=output_dir,
            remote_config=remote_config,
            max_cases=args.max_cases,
            case_ids=case_ids,
            sample_size=args.sample_size,
            seed=args.seed,
            question_reasoning=args.question_reasoning,
            eval_mode=args.financebench_eval_mode,
            max_context_chars=args.max_context_chars,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            remote_cost_model=args.remote_cost_model,
            progress_factory=progress_factory,
        )

    if args.tablebench_eval:
        from benchmarks.tablebench.eval import run_tablebench_eval

        remote_config = _load_remote_config(args)
        synthesize_config = _load_synthesize_config(args)
        local_slm_config = _load_local_config(args)
        print_eval_startup_banner(
            remote_config=remote_config,
            synthesize_config=synthesize_config,
            local_slm_config=local_slm_config,
            workflow="table_reasoning.tablebench",
            remote_batch_size=args.remote_batch_size,
            remote_concurrency=args.remote_concurrency,
            max_parallel_execution_units=args.max_parallel_execution_units,
            max_parallel_slm_node_jobs=args.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=args.max_parallel_slm_sequences,
            max_pending_slm_sequences=args.max_pending_slm_sequences,
            slm_scheduler=args.slm_scheduler,
        )
        preflight_model_api_checks(
            args=args,
            remote_config=remote_config,
            synthesize_config=synthesize_config,
            local_slm_config=local_slm_config,
        )
        return run_tablebench_eval(
            tablebench_root=tablebench_root,
            output_dir=output_dir,
            remote_config=remote_config,
            synthesize_config=synthesize_config,
            local_slm_config=local_slm_config,
            max_cases=args.max_cases,
            case_ids=case_ids,
            dataset_id=args.dataset_id,
            qtypes=qtypes,
            qsubtypes=qsubtypes,
            include_visualization=args.include_visualization,
            sample_size=args.sample_size,
            seed=args.seed,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            validation_mode=args.validation_mode,
            remote_batch_size=args.remote_batch_size,
            remote_concurrency=args.remote_concurrency,
            max_parallel_execution_units=args.max_parallel_execution_units,
            max_parallel_slm_node_jobs=args.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=args.max_parallel_slm_sequences,
            max_pending_slm_sequences=args.max_pending_slm_sequences,
            eval_batch_size=args.eval_batch_size,
            profile_baseline=args.profile_baseline,
            remote_cost_model=args.remote_cost_model,
            overwrite=args.overwrite,
            progress_factory=progress_factory,
        )

    if args.tablebench_remote_only_baseline:
        from benchmarks.tablebench.remote_baseline import (
            run_tablebench_remote_only_baseline,
        )

        remote_config = _load_remote_config(args)
        print_eval_startup_banner(
            remote_config=remote_config,
            local_slm_config=None,
            workflow=f"remote_only_tablebench_{args.tablebench_instruction_type}",
        )
        preflight_model_api_checks(
            args=args,
            remote_config=remote_config,
            local_slm_config=None,
        )
        return run_tablebench_remote_only_baseline(
            tablebench_root=tablebench_root,
            output_dir=output_dir,
            remote_config=remote_config,
            max_cases=args.max_cases,
            case_ids=case_ids,
            dataset_id=args.dataset_id,
            qtypes=qtypes,
            qsubtypes=qsubtypes,
            include_visualization=args.include_visualization,
            sample_size=args.sample_size,
            seed=args.seed,
            max_workers=args.max_workers,
            instruction_type=args.tablebench_instruction_type,
            execution_timeout_seconds=args.execution_timeout_seconds,
            overwrite=args.overwrite,
            remote_cost_model=args.remote_cost_model,
            progress_factory=progress_factory,
        )

    if args.databench_eval:
        from benchmarks.databench.eval import run_databench_eval

        remote_config = _load_remote_config(args)
        local_slm_config = _load_local_config(args)
        print_eval_startup_banner(
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            workflow="table_reasoning.databench",
            remote_batch_size=args.remote_batch_size,
            remote_concurrency=args.remote_concurrency,
            max_parallel_execution_units=args.max_parallel_execution_units,
            max_parallel_slm_node_jobs=args.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=args.max_parallel_slm_sequences,
            max_pending_slm_sequences=args.max_pending_slm_sequences,
            slm_scheduler=args.slm_scheduler,
        )
        preflight_model_api_checks(
            args=args,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
        )
        return run_databench_eval(
            databench_root=databench_root,
            output_dir=output_dir,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            max_cases=args.max_cases,
            case_ids=case_ids,
            dataset_id=args.dataset_id,
            sample_size=args.sample_size,
            seed=args.seed,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            validation_mode=args.validation_mode,
            remote_batch_size=args.remote_batch_size,
            remote_concurrency=args.remote_concurrency,
            max_parallel_execution_units=args.max_parallel_execution_units,
            max_parallel_slm_node_jobs=args.max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=args.max_parallel_slm_sequences,
            max_pending_slm_sequences=args.max_pending_slm_sequences,
            eval_batch_size=args.eval_batch_size,
            profile_baseline=args.profile_baseline,
            remote_cost_model=args.remote_cost_model,
            overwrite=args.overwrite,
            progress_factory=progress_factory,
            preprocess_progress_factory=None if args.no_progress else EvalPreprocessProgressBar,
        )

    if args.remote_only_baseline:
        from benchmarks.databench.remote_baseline import run_databench_remote_only_baseline

        remote_config = _load_remote_config(args)
        print_eval_startup_banner(
            remote_config=remote_config,
            local_slm_config=None,
            workflow="remote_only_databench_code",
        )
        preflight_model_api_checks(
            args=args,
            remote_config=remote_config,
            local_slm_config=None,
        )
        return run_databench_remote_only_baseline(
            databench_root=databench_root,
            output_dir=output_dir,
            remote_config=remote_config,
            max_cases=args.max_cases,
            case_ids=case_ids,
            dataset_id=args.dataset_id,
            sample_size=args.sample_size,
            seed=args.seed,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            execution_timeout_seconds=args.execution_timeout_seconds,
            remote_cost_model=args.remote_cost_model,
            progress_factory=progress_factory,
        )

    if args.static_tool_eval:
        from benchmarks.databench.static_tool_eval import run_static_tool_eval

        return run_static_tool_eval(
            run_dir=args.run_dir.expanduser().resolve(),
            databench_root=databench_root,
            output_dir=_static_tool_output_dir(args, output_dir=output_dir),
            max_cases=args.max_cases,
            case_ids=case_ids,
            progress_every=args.progress_every,
        )

    if args.all_databench_tables:
        from benchmarks.databench.adapter import run_all_databench_tables

        return run_all_databench_tables(
            databench_root=databench_root,
            case_index=args.case_index,
            output_root=args.output_root.expanduser().resolve(),
            run_name=run_name,
        )

    from benchmarks.databench.adapter import first_databench_dataset, run_databench_case

    dataset_id = args.dataset_id or first_databench_dataset(databench_root)
    first_case_id = next(iter(case_ids), None)
    return run_databench_case(
        databench_root=databench_root,
        dataset_id=dataset_id,
        case_id=first_case_id,
        case_index=args.case_index,
        output_root=args.output_root.expanduser().resolve(),
        run_name=run_name,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.eval",
        description="Run CLOVER benchmark stages.",
    )
    parser.add_argument("--databench-root", type=Path, default=REPO_ROOT / "datasets" / "databench")
    parser.add_argument(
        "--tablebench-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "tablebench",
    )
    parser.add_argument(
        "--financebench-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "financebench",
    )
    parser.add_argument(
        "--financebench-examples-root",
        type=Path,
        default=DEFAULT_FINANCEBENCH_EXAMPLES_ROOT,
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--qtype", action="append", default=[])
    parser.add_argument("--qsubtype", action="append", default=[])
    parser.add_argument("--include-visualization", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)

    modes = parser.add_argument_group("modes")
    modes.add_argument("--eval", dest="databench_eval", action="store_true")
    modes.add_argument("--remote-only-baseline", action="store_true")
    modes.add_argument("--tablebench-eval", action="store_true")
    modes.add_argument("--tablebench-remote-only-baseline", action="store_true")
    modes.add_argument("--financebench-eval", action="store_true")
    modes.add_argument("--financebench-remote-only-baseline", action="store_true")
    modes.add_argument("--static-tool-eval", action="store_true")
    modes.add_argument("--all-databench-tables", action="store_true")

    parser.add_argument("--remote-llm-config", type=Path, default=_default_remote_llm_config_path())
    parser.add_argument("--synthesize-llm-config", type=Path, default=None)
    parser.add_argument("--local-slm-config", type=Path, default=_default_local_slm_config_path())
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_STATIC_TOOL_RUN_DIR)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_EVAL_WORKERS)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--validation-mode", choices=("none", "remote_supervisor"), default="none")
    parser.add_argument("--remote-batch-size", type=int, default=DEFAULT_REMOTE_BATCH_SIZE)
    parser.add_argument("--remote-concurrency", type=int, default=DEFAULT_REMOTE_CONCURRENCY)
    parser.add_argument(
        "--slm-scheduler",
        choices=tuple(sorted(SLM_SCHEDULER_CHOICES)),
        default=os.environ.get("CLOVER_SLM_SCHEDULER", SLM_SCHEDULER_TPTT),
    )
    parser.add_argument(
        "--max-parallel-execution-units",
        type=int,
        default=DEFAULT_MAX_PARALLEL_EXECUTION_UNITS,
    )
    parser.add_argument(
        "--max-parallel-slm-node-jobs",
        type=int,
        default=DEFAULT_MAX_PARALLEL_SLM_NODE_JOBS,
    )
    parser.add_argument(
        "--max-parallel-slm-sequences",
        type=int,
        default=DEFAULT_MAX_PARALLEL_SLM_SEQUENCES,
    )
    parser.add_argument(
        "--max-pending-slm-sequences",
        type=int,
        default=DEFAULT_MAX_PENDING_SLM_SEQUENCES,
    )
    parser.add_argument("--max-tptt-leaf-sequences-per-tree", type=int, default=None)
    parser.add_argument("--tptt-coalesce-ms", type=float, default=None)
    parser.add_argument("--tptt-prefix-tokens", type=int, default=None)
    parser.add_argument("--node-timeout-seconds", type=float, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--profile-baseline", action="store_true")
    parser.add_argument("--remote-cost-model", default=os.environ.get("CLOVER_REMOTE_COST_MODEL"))
    parser.add_argument("--execution-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--tablebench-instruction-type",
        choices=tuple(sorted(TABLEBENCH_INSTRUCTION_TYPES)),
        default="DP",
    )
    parser.add_argument(
        "--financebench-eval-mode",
        choices=tuple(sorted(SUPPORTED_EVAL_MODES)),
        default=EVAL_MODE_IN_CONTEXT,
    )
    parser.add_argument("--question-reasoning", default=None)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-model-api-check",
        action="store_true",
        default=_env_truthy("CLOVER_SKIP_MODEL_API_CHECK"),
    )
    parser.add_argument(
        "--model-api-check-timeout",
        type=float,
        default=float(
            os.environ.get(
                "CLOVER_MODEL_API_CHECK_TIMEOUT",
                DEFAULT_MODEL_API_CHECK_TIMEOUT_SECONDS,
            )
        ),
    )

    parser.add_argument("--energy-profile", action="store_true")
    parser.add_argument("--energy-sample-ms", type=int, default=500)
    parser.add_argument("--energy-baseline-seconds", type=float, default=0.0)
    parser.add_argument("--energy-password-env", default="CLOVER_POWERMETRICS_PASSWORD")
    parser.add_argument(
        "--edge-electricity-price-usd-per-kwh",
        type=float,
        default=DEFAULT_EDGE_ELECTRICITY_PRICE_USD_PER_KWH,
    )
    return parser.parse_args()


def _validate_modes(args: argparse.Namespace) -> None:
    selected = [
        args.databench_eval,
        args.remote_only_baseline,
        args.tablebench_eval,
        args.tablebench_remote_only_baseline,
        args.financebench_eval,
        args.financebench_remote_only_baseline,
        args.static_tool_eval,
        args.all_databench_tables,
    ]
    if sum(bool(item) for item in selected) > 1:
        raise SystemExit("Choose only one benchmark mode flag.")


def _default_run_name(args: argparse.Namespace) -> str:
    if args.tablebench_eval:
        return "tablebench_eval"
    if args.tablebench_remote_only_baseline:
        return f"tablebench_remote_only_baseline_{args.tablebench_instruction_type.lower()}"
    if args.financebench_eval:
        return "financebench_document_eval"
    if args.financebench_remote_only_baseline:
        return "financebench_remote_only_baseline"
    if args.remote_only_baseline:
        return "databench_remote_only_baseline"
    if args.static_tool_eval:
        return "databench_static_tool_eval"
    if args.databench_eval:
        return "databench_eval"
    if args.all_databench_tables:
        return "databench_all_tables"
    return "databench_preprocess"


def _load_remote_config(args: argparse.Namespace) -> dict[str, Any]:
    return load_model_config(
        _resolve_path(args.remote_llm_config),
        env_prefixes=REMOTE_LLM_ENV_PREFIXES,
    )


def _load_synthesize_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.synthesize_llm_config is None:
        return None
    return load_model_config(
        _resolve_path(args.synthesize_llm_config),
        env_prefixes=SYNTHESIZE_LLM_ENV_PREFIXES,
    )


def _load_local_config(args: argparse.Namespace) -> dict[str, Any] | None:
    config = load_optional_model_config(
        _resolve_path(args.local_slm_config),
        env_prefixes=LOCAL_SLM_ENV_PREFIXES,
    )
    if config is None:
        return None
    selected = dict(config)
    selected["slm_scheduler"] = args.slm_scheduler
    if args.max_tptt_leaf_sequences_per_tree is not None:
        selected["max_tptt_leaf_sequences_per_tree"] = (
            args.max_tptt_leaf_sequences_per_tree
        )
    if args.tptt_coalesce_ms is not None:
        selected["tptt_coalesce_ms"] = args.tptt_coalesce_ms
    if args.tptt_prefix_tokens is not None:
        selected["tptt_prefix_tokens"] = args.tptt_prefix_tokens
    return selected


def preflight_model_api_checks(
    *,
    args: argparse.Namespace,
    remote_config: dict[str, Any] | None,
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
) -> None:
    """Fail early when configured model endpoints cannot serve a tiny request."""

    if getattr(args, "skip_model_api_check", False):
        print("Model API check: skipped", file=sys.stderr, flush=True)
        return

    checks: list[tuple[str, dict[str, Any]]] = []
    if remote_config is not None:
        checks.append(("Remote LLM", remote_config))
    if synthesize_config is not None:
        checks.append(("Synthesize LLM", synthesize_config))
    if local_slm_config is not None:
        checks.append(("Local SLM", local_slm_config))
    if not checks:
        return

    timeout_seconds = float(
        getattr(args, "model_api_check_timeout", DEFAULT_MODEL_API_CHECK_TIMEOUT_SECONDS)
        or DEFAULT_MODEL_API_CHECK_TIMEOUT_SECONDS
    )
    print("Model API check:", file=sys.stderr)
    failures: list[str] = []
    for label, config in checks:
        check_config = _model_api_check_config(config, timeout_seconds=timeout_seconds)
        try:
            result = generate_remote_text(
                prompt=MODEL_API_CHECK_PROMPT,
                remote_config=check_config,
            )
        except Exception as exc:  # noqa: BLE001 - preflight should report provider errors directly.
            detail = f"{type(exc).__name__}: {exc}"
            failures.append(f"{label}: {detail}")
            print(
                f"  {label}: failed ({_model_config_ref(config)}) {detail}",
                file=sys.stderr,
                flush=True,
            )
            continue
        text = (result.text or "").strip()
        suffix = f", response={text[:24]!r}" if text else ""
        print(
            f"  {label}: ok ({_model_config_ref(config)}{suffix})",
            file=sys.stderr,
            flush=True,
        )
    if failures:
        raise SystemExit(
            "Model API connectivity check failed. "
            "Fix model configs/API keys/server status, or pass --skip-model-api-check. "
            + "; ".join(failures)
        )
    print(file=sys.stderr, flush=True)


def _model_api_check_config(
    config: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    selected = dict(config)
    selected["timeout"] = _bounded_positive_float(
        selected.get("timeout"),
        fallback=timeout_seconds,
        upper_bound=timeout_seconds,
    )
    selected["max_retries"] = 0
    api_type = selected.get("api_type", "responses")
    if api_type == "responses":
        selected["max_output_tokens"] = _bounded_positive_int(
            selected.get("max_output_tokens"),
            fallback=8,
            upper_bound=8,
        )
    else:
        selected["max_tokens"] = _bounded_positive_int(
            selected.get("max_tokens", selected.get("max_output_tokens")),
            fallback=8,
            upper_bound=8,
        )
    return selected


def _model_config_ref(config: dict[str, Any]) -> str:
    provider = config.get("provider") or "unknown"
    model = config.get("model") or "unknown-model"
    base_url = config.get("base_url")
    if base_url:
        return f"{provider}/{model} @ {base_url}"
    return f"{provider}/{model}"


def _bounded_positive_float(
    value: Any,
    *,
    fallback: float,
    upper_bound: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    if parsed <= 0:
        parsed = fallback
    return min(parsed, upper_bound)


def _bounded_positive_int(
    value: Any,
    *,
    fallback: int,
    upper_bound: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    if parsed <= 0:
        parsed = fallback
    return min(parsed, upper_bound)


def _static_tool_output_dir(args: argparse.Namespace, *, output_dir: Path) -> Path:
    source_dir = args.run_dir.expanduser().resolve()
    if args.in_place:
        return source_dir
    if output_dir == source_dir:
        raise ValueError(
            "Static tool eval output directory matches --run-dir. "
            "Use --in-place to update the existing run directory."
        )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    return output_dir


def print_eval_startup_banner(
    *,
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    workflow: str = "unknown",
    remote_batch_size: int | None = None,
    remote_concurrency: int | None = None,
    max_parallel_execution_units: int | None = None,
    max_parallel_slm_node_jobs: int | None = None,
    max_parallel_slm_sequences: int | None = None,
    max_pending_slm_sequences: int | None = None,
    slm_scheduler: str | None = None,
) -> None:
    """Print the shared eval startup header without exposing secrets."""

    print(CLOVER_BANNER, file=sys.stderr)
    print(file=sys.stderr)
    print("Workflow:", file=sys.stderr)
    print(f"  task: {workflow}", file=sys.stderr)
    if remote_batch_size is not None:
        print(f"  remote_batch_size: {remote_batch_size}", file=sys.stderr)
    if remote_concurrency is not None:
        print(f"  remote_concurrency: {remote_concurrency}", file=sys.stderr)
    if max_parallel_execution_units is not None:
        print(f"  max_parallel_execution_units: {max_parallel_execution_units}", file=sys.stderr)
    if max_parallel_slm_node_jobs is not None:
        print(f"  max_parallel_slm_node_jobs: {max_parallel_slm_node_jobs}", file=sys.stderr)
    if max_parallel_slm_sequences is not None:
        print(f"  max_parallel_slm_sequences: {max_parallel_slm_sequences}", file=sys.stderr)
    if max_pending_slm_sequences is not None:
        print(f"  max_pending_slm_sequences: {max_pending_slm_sequences}", file=sys.stderr)
    if slm_scheduler is not None:
        print(f"  slm_scheduler: {slm_scheduler}", file=sys.stderr)
    print("Remote LLM:", file=sys.stderr)
    for line in _model_config_lines(remote_config):
        print(f"  {line}", file=sys.stderr)
    if synthesize_config is not None:
        print("Synthesize LLM:", file=sys.stderr)
        for line in _model_config_lines(synthesize_config):
            print(f"  {line}", file=sys.stderr)
    print("Local SLM:", file=sys.stderr)
    if local_slm_config is None:
        print("  not configured", file=sys.stderr)
    else:
        for line in _model_config_lines(local_slm_config):
            print(f"  {line}", file=sys.stderr)
    print(file=sys.stderr, flush=True)


class EvalProgressBar:
    """Shared stderr progress bar for benchmark case records."""

    def __init__(self, total: int, *, width: int = 28) -> None:
        self.total = total
        self.width = width
        self._last_len = 0
        self._color = "NO_COLOR" not in os.environ

    def update(self, records: list[dict[str, Any]]) -> None:
        if self.total <= 0:
            return
        completed = len(records)
        correct = sum(1 for record in records if record.get("answer_correct"))
        mismatches = sum(
            1
            for record in records
            if record.get("runtime_ok") and not record.get("answer_correct")
        )
        failures = sum(1 for record in records if not record.get("runtime_ok"))
        retry_cases = sum(1 for record in records if record.get("retry_count", 0) > 0)
        filled = int(self.width * completed / self.total) if self.total else self.width
        percent = _safe_divide(completed, self.total) or 0.0
        acc = _safe_divide(correct, completed) or 0.0
        plain_text = (
            f"[{'#' * filled}{'-' * (self.width - filled)}] "
            f"{completed}/{self.total} {percent:6.1%} "
            f"correct={correct} mismatch={mismatches} fail={failures} "
            f"acc={acc:.3f} retry={retry_cases}"
        )
        text = (
            f"\r{self._bar(filled)} "
            f"{self._paint(f'{completed}/{self.total}', '1;37')} "
            f"{self._paint(f'{percent:6.1%}', '1;36')} "
            f"{self._paint(f'correct={correct}', '1;32')} "
            f"{self._paint(f'mismatch={mismatches}', '1;33')} "
            f"{self._paint(f'fail={failures}', '1;31')} "
            f"acc={acc:.3f} "
            f"{self._paint(f'retry={retry_cases}', '1;35')}"
        )
        padding = " " * max(0, self._last_len - len(plain_text))
        print(text + padding, file=sys.stderr, end="", flush=True)
        self._last_len = len(plain_text)

    def close(self) -> None:
        print(file=sys.stderr, flush=True)

    def _bar(self, filled: int) -> str:
        done = "#" * filled
        remaining = "-" * max(0, self.width - filled)
        return f"[{self._paint(done, '1;32')}{self._paint(remaining, '2;37')}]"

    def _paint(self, text: str, code: str) -> str:
        if not self._color or not text:
            return text
        return f"\033[{code}m{text}\033[0m"


class EvalPreprocessProgressBar:
    """Separate stderr progress bar for DataBench preprocessing groups."""

    def __init__(self, total: int, *, width: int = 28) -> None:
        self.total = total
        self.width = width
        self._last_len = 0
        self._color = "NO_COLOR" not in os.environ

    def update(
        self,
        completed: int,
        *,
        prepared_cases: int = 0,
        failed_cases: int = 0,
    ) -> None:
        if self.total <= 0:
            return
        filled = int(self.width * completed / self.total) if self.total else self.width
        percent = _safe_divide(completed, self.total) or 0.0
        plain_text = (
            f"Preprocess: [{'#' * filled}{'-' * (self.width - filled)}] "
            f"{completed}/{self.total} {percent:6.1%} "
            f"cases={prepared_cases} fail={failed_cases}"
        )
        text = (
            f"\r{self._paint('Preprocess:', '1;37')} {self._bar(filled)} "
            f"{self._paint(f'{completed}/{self.total}', '1;37')} "
            f"{self._paint(f'{percent:6.1%}', '1;36')} "
            f"cases={prepared_cases} "
            f"{self._paint(f'fail={failed_cases}', '1;31')}"
        )
        padding = " " * max(0, self._last_len - len(plain_text))
        print(text + padding, file=sys.stderr, end="", flush=True)
        self._last_len = len(plain_text)

    def close(self) -> None:
        print(file=sys.stderr, flush=True)

    def _bar(self, filled: int) -> str:
        done = "#" * filled
        remaining = "-" * max(0, self.width - filled)
        return f"[{self._paint(done, '1;32')}{self._paint(remaining, '2;37')}]"

    def _paint(self, text: str, code: str) -> str:
        if not self._color or not text:
            return text
        return f"\033[{code}m{text}\033[0m"


def _normalize_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    energy_summary: dict[str, Any] | None,
    electricity_price_usd_per_kwh: float,
) -> dict[str, Any]:
    edge_cost = _edge_energy_cost_estimate(
        energy_summary,
        electricity_price_usd_per_kwh=electricity_price_usd_per_kwh,
    )
    remote_cost_total = _remote_cost_total(summary)
    total_cost = None
    if remote_cost_total is not None and edge_cost["cost_usd"] is not None:
        total_cost = remote_cost_total + edge_cost["cost_usd"]

    summary["edge_energy_profile"] = energy_summary
    summary["edge_energy_cost_estimate"] = edge_cost
    summary["total_economic_cost_estimate"] = {
        "currency": "USD",
        "cloud_api_cost_usd": remote_cost_total,
        "edge_energy_cost_usd": edge_cost["cost_usd"],
        "total_usd": _round_usd(total_cost) if total_cost is not None else None,
    }
    summary["measurement_summary"] = {
        "stage": summary.get("stage"),
        "workflow": summary.get("workflow"),
        "run_name": summary.get("run_name") or output_dir.name,
        "total_cases": summary.get("total_cases"),
        "accuracy_on_all_cases": summary.get("accuracy_on_all_cases"),
        "correct": summary.get("correct"),
        "wall_clock_seconds": summary.get("elapsed_seconds"),
        "remote_calls": summary.get("remote_calls"),
        "local_slm_calls": summary.get("local_slm_calls"),
        "remote_tokens": _token_total(summary.get("remote_token_usage")),
        "local_slm_tokens": _token_total(summary.get("local_slm_token_usage")),
        "builder_agent_tokens": _token_total(summary.get("builder_agent_token_usage")),
        "cloud_api_cost_usd": remote_cost_total,
        "edge_energy_joules": edge_cost["energy_joules"],
        "edge_energy_cost_usd": edge_cost["cost_usd"],
        "total_economic_cost_usd": _round_usd(total_cost) if total_cost is not None else None,
        "slm_scheduler": _summary_slm_scheduler(summary),
        "parallel_workers": summary.get("parallel_workers"),
    }
    return summary


def _edge_energy_cost_estimate(
    energy_summary: dict[str, Any] | None,
    *,
    electricity_price_usd_per_kwh: float,
) -> dict[str, Any]:
    joules, source = _energy_joules(energy_summary)
    cost = None
    if joules is not None:
        cost = max(0.0, joules) * electricity_price_usd_per_kwh / 3_600_000.0
    return {
        "currency": "USD",
        "electricity_price_usd_per_kwh": electricity_price_usd_per_kwh,
        "energy_joules": joules,
        "energy_source": source,
        "cost_usd": _round_usd(cost) if cost is not None else None,
    }


def _energy_joules(energy_summary: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not isinstance(energy_summary, dict) or not energy_summary.get("enabled"):
        return None, None
    baseline_adjusted = energy_summary.get("baseline_adjusted")
    if isinstance(baseline_adjusted, dict) and baseline_adjusted.get("total_joules") is not None:
        return float(baseline_adjusted["total_joules"]), "baseline_adjusted.total_joules"
    gross = energy_summary.get("gross")
    if isinstance(gross, dict) and gross.get("total_joules") is not None:
        return float(gross["total_joules"]), "gross.total_joules"
    return None, None


def _remote_cost_total(summary: dict[str, Any]) -> float | None:
    cost = summary.get("remote_cost_estimate")
    if not isinstance(cost, dict):
        return None
    cost_usd = cost.get("cost_usd")
    if not isinstance(cost_usd, dict) or cost_usd.get("total") is None:
        return None
    return float(cost_usd["total"])


def _token_total(token_usage: Any) -> int | None:
    if not isinstance(token_usage, dict):
        return None
    value = token_usage.get("total_tokens")
    return int(value) if value is not None else None


def _summary_slm_scheduler(summary: dict[str, Any]) -> str | None:
    scheduler = summary.get("slm_scheduler")
    if scheduler is not None:
        return str(scheduler)
    local_slm = summary.get("local_slm")
    if isinstance(local_slm, dict) and local_slm.get("slm_scheduler") is not None:
        return str(local_slm["slm_scheduler"])
    return None


def _write_summary_if_possible(summary: dict[str, Any], *, output_dir: Path) -> None:
    summary_path = output_dir / "run_summary.json"
    if not output_dir.exists():
        return
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _default_remote_llm_config_path() -> Path:
    return _env_path(
        "CLOVER_REMOTE_LLM_CONFIG",
        "CLOVER_LLM_CONFIG",
        "LLM_CONFIG",
        default=DEFAULT_REMOTE_LLM_CONFIG,
    )


def _default_local_slm_config_path() -> Path:
    return _env_path(
        "CLOVER_LOCAL_SLM_CONFIG",
        "CLOVER_SLM_CONFIG",
        "SLM_CONFIG",
        default=DEFAULT_SLM_CONFIG_PATH,
    )


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(default)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _model_config_lines(config: dict[str, Any]) -> list[str]:
    sampling = _sampling_config(config)
    return [
        f"provider: {config.get('provider', 'unknown')}",
        f"model: {config.get('model', 'unknown')}",
        f"max_tokens: {_max_tokens(config)}",
        f"sampling: {_format_kv(sampling) if sampling else 'default'}",
    ]


def _max_tokens(config: dict[str, Any]) -> Any:
    return config.get("max_tokens", config.get("max_output_tokens", "default"))


def _sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    sampling = {
        key: config[key]
        for key in SAMPLING_CONFIG_KEYS
        if key in config and config[key] is not None
    }
    extra_body = config.get("extra_body")
    if isinstance(extra_body, dict):
        sampling.update(
            {
                key: extra_body[key]
                for key in SAMPLING_CONFIG_KEYS
                if key in extra_body and extra_body[key] is not None
            }
        )
    return sampling


def _format_kv(payload: dict[str, Any]) -> str:
    return ", ".join(f"{key}={payload[key]}" for key in sorted(payload))


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _round_usd(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 8)


if __name__ == "__main__":
    main()
