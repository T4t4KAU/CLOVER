"""Lean command-line entry point for CLOVER's table benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmarks.utils import build_brief_summary, format_brief_summary
from benchmarks.warnings import suppress_benchmark_warnings
from clover.config import load_model_config, load_optional_model_config
from clover.supervisor.client import generate_remote_text


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "runs"
DEFAULT_REMOTE_CONFIG = REPO_ROOT / "model_config" / "deepseek_remote_llm_config.json"
DEFAULT_LOCAL_CONFIG = REPO_ROOT / "model_config" / "local_slm_config.json"
MODEL_API_CHECK_PROMPT = "Return exactly OK."


class EvalProgressBar:
    """Compact stderr progress display shared by the three evaluators."""

    def __init__(self, total: int, *, width: int = 24) -> None:
        self.total = total
        self.width = width
        self._last_length = 0
        if total > 0:
            self.update([])

    def update(self, records: list[dict[str, Any]]) -> None:
        if self.total <= 0:
            return
        completed = len(records)
        correct = sum(bool(record.get("answer_correct")) for record in records)
        failures = sum(not bool(record.get("runtime_ok")) for record in records)
        filled = min(self.width, int(self.width * completed / self.total))
        accuracy = correct / completed if completed else 0.0
        text = (
            f"\r[{'#' * filled}{'-' * (self.width - filled)}] "
            f"{completed}/{self.total} correct={correct} fail={failures} "
            f"acc={accuracy:.3f}"
        )
        padding = " " * max(0, self._last_length - len(text))
        print(text + padding, file=sys.stderr, end="", flush=True)
        self._last_length = len(text)

    def close(self) -> None:
        if self.total > 0:
            print(file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CLOVER on TableBench, WikiTableQuestions, MMQA, or TableFact."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--tablebench-eval", action="store_true")
    modes.add_argument("--wikitq-eval", action="store_true")
    modes.add_argument("--mmqa-eval", action="store_true")
    modes.add_argument("--tablefact-eval", "--tabfact-eval", action="store_true")

    parser.add_argument(
        "--tablebench-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "tablebench",
    )
    parser.add_argument(
        "--wikitq-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "wikitq",
    )
    parser.add_argument(
        "--tablefact-root",
        "--tabfact-root",
        dest="tablefact_root",
        type=Path,
        default=REPO_ROOT / "datasets" / "tablefact",
    )
    parser.add_argument(
        "--mmqa-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "mmqa",
    )
    parser.add_argument("--wikitq-split", default=None)
    parser.add_argument("--tablefact-split", "--tabfact-split", default="test")
    parser.add_argument("--mmqa-split", default=None)
    parser.add_argument(
        "--tablefact-subset",
        "--tabfact-subset",
        choices=("simple", "complex", "small"),
        default=None,
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--qtype", action="append", default=[])
    parser.add_argument("--qsubtype", action="append", default=[])
    parser.add_argument("--include-visualization", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260528)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument(
        "--remote-llm-config",
        type=Path,
        default=_env_path(
            "CLOVER_REMOTE_LLM_CONFIG",
            "CLOVER_LLM_CONFIG",
            default=DEFAULT_REMOTE_CONFIG,
        ),
    )
    parser.add_argument("--synthesize-llm-config", type=Path, default=None)
    parser.add_argument(
        "--local-slm-config",
        type=Path,
        default=_env_path(
            "CLOVER_LOCAL_SLM_CONFIG",
            "CLOVER_SLM_CONFIG",
            default=DEFAULT_LOCAL_CONFIG,
        ),
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--validation-mode",
        choices=("none", "remote_supervisor"),
        default="none",
    )
    parser.add_argument("--remote-batch-size", type=int, default=1)
    parser.add_argument("--remote-concurrency", type=int, default=8)
    parser.add_argument("--slm-scheduler", choices=("fifo", "tptt"), default="tptt")
    parser.add_argument("--max-parallel-execution-units", type=int, default=8)
    parser.add_argument("--max-parallel-slm-node-jobs", type=int, default=32)
    parser.add_argument("--max-parallel-slm-sequences", type=int, default=32)
    parser.add_argument("--max-pending-slm-sequences", type=int, default=64)
    parser.add_argument("--max-tptt-leaf-sequences-per-tree", type=int, default=None)
    parser.add_argument("--tptt-coalesce-ms", type=float, default=None)
    parser.add_argument("--tptt-prefix-tokens", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--profile-baseline", action="store_true")
    parser.add_argument("--remote-cost-model", default=None)
    parser.add_argument("--skip-model-api-check", action="store_true")
    parser.add_argument("--model-api-check-timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.remote_batch_size != 1:
        print(
            "Table query batching is disabled; overriding --remote-batch-size to 1.",
            file=sys.stderr,
        )
        args.remote_batch_size = 1
    dataset = _selected_dataset(args)
    run_name = args.run_name or f"{dataset}_eval"
    output_dir = (args.output_root / run_name).expanduser().resolve()

    remote_config = load_model_config(args.remote_llm_config.expanduser().resolve())
    synthesize_config = (
        load_model_config(args.synthesize_llm_config.expanduser().resolve())
        if args.synthesize_llm_config is not None
        else None
    )
    local_config = load_optional_model_config(
        args.local_slm_config.expanduser().resolve()
    )
    if local_config is None:
        raise SystemExit("A local SLM config is required for CLOVER table evaluation.")
    local_config = _configure_local_model(local_config, args)

    _print_startup(
        dataset=dataset,
        remote_config=remote_config,
        synthesize_config=synthesize_config,
        local_config=local_config,
        output_dir=output_dir,
    )
    preflight_model_api_checks(
        args=args,
        remote_config=remote_config,
        synthesize_config=synthesize_config,
        local_slm_config=local_config,
    )

    common = {
        "output_dir": output_dir,
        "remote_config": remote_config,
        "synthesize_config": synthesize_config,
        "local_slm_config": local_config,
        "max_cases": args.max_cases,
        "case_ids": set(args.case_id),
        "dataset_id": args.dataset_id,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "validation_mode": args.validation_mode,
        "remote_batch_size": args.remote_batch_size,
        "remote_concurrency": args.remote_concurrency,
        "max_parallel_execution_units": args.max_parallel_execution_units,
        "max_parallel_slm_node_jobs": args.max_parallel_slm_node_jobs,
        "max_parallel_slm_sequences": args.max_parallel_slm_sequences,
        "max_pending_slm_sequences": args.max_pending_slm_sequences,
        "eval_batch_size": args.eval_batch_size,
        "profile_baseline": args.profile_baseline,
        "remote_cost_model": args.remote_cost_model,
        "overwrite": args.overwrite,
        "progress_factory": None if args.no_progress else EvalProgressBar,
    }

    with suppress_benchmark_warnings():
        if dataset == "tablebench":
            from benchmarks.tablebench.eval import run_tablebench_eval

            summary = run_tablebench_eval(
                tablebench_root=args.tablebench_root.expanduser().resolve(),
                qtypes=set(args.qtype),
                qsubtypes=set(args.qsubtype),
                include_visualization=args.include_visualization,
                **common,
            )
        elif dataset == "wikitq":
            from benchmarks.wikitq.eval import run_wikitq_eval

            summary = run_wikitq_eval(
                wikitq_root=args.wikitq_root.expanduser().resolve(),
                split=args.wikitq_split,
                **common,
            )
        elif dataset == "mmqa":
            from benchmarks.mmqa.eval import run_mmqa_eval

            summary = run_mmqa_eval(
                mmqa_root=args.mmqa_root.expanduser().resolve(),
                split=args.mmqa_split,
                **common,
            )
        else:
            from benchmarks.tablefact.eval import run_tablefact_eval

            summary = run_tablefact_eval(
                tablefact_root=args.tablefact_root.expanduser().resolve(),
                split=args.tablefact_split,
                subset=args.tablefact_subset,
                **common,
            )

    brief = build_brief_summary(summary)
    print(format_brief_summary(brief))
    return 0


def _selected_dataset(args: argparse.Namespace) -> str:
    if args.tablebench_eval:
        return "tablebench"
    if args.wikitq_eval:
        return "wikitq"
    if args.mmqa_eval:
        return "mmqa"
    return "tablefact"


def _configure_local_model(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    selected = dict(config)
    selected["slm_scheduler"] = args.slm_scheduler
    overrides = {
        "max_tptt_leaf_sequences_per_tree": args.max_tptt_leaf_sequences_per_tree,
        "tptt_coalesce_ms": args.tptt_coalesce_ms,
        "tptt_prefix_tokens": args.tptt_prefix_tokens,
    }
    for key, value in overrides.items():
        if value is not None:
            selected[key] = value
    return selected


def preflight_model_api_checks(
    *,
    args: argparse.Namespace,
    remote_config: dict[str, Any] | None,
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
) -> None:
    """Fail before evaluation when one of the configured endpoints is unavailable."""

    if getattr(args, "skip_model_api_check", False):
        print("Model API check: skipped", file=sys.stderr)
        return

    timeout = float(getattr(args, "model_api_check_timeout", 30.0) or 30.0)
    checks = [
        ("remote", remote_config),
        ("synthesize", synthesize_config),
        ("local", local_slm_config),
    ]
    failures = []
    for label, config in checks:
        if config is None:
            continue
        try:
            generate_remote_text(
                prompt=MODEL_API_CHECK_PROMPT,
                remote_config=_model_api_check_config(config, timeout=timeout),
            )
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        print(f"Model API check: {label} ok", file=sys.stderr)
    if failures:
        raise SystemExit(
            "Model API connectivity check failed. " + "; ".join(failures)
        )


def _model_api_check_config(
    config: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    selected = dict(config)
    try:
        configured_timeout = float(selected.get("timeout") or timeout)
    except (TypeError, ValueError):
        configured_timeout = timeout
    selected["timeout"] = min(max(configured_timeout, 0.001), timeout)
    selected["max_retries"] = 0
    if selected.get("api_type", "responses") == "responses":
        selected["max_output_tokens"] = 8
    else:
        selected["max_tokens"] = 8
    return selected


def _print_startup(
    *,
    dataset: str,
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None,
    local_config: dict[str, Any],
    output_dir: Path,
) -> None:
    print("CLOVER table evaluation", file=sys.stderr)
    print(f"  dataset: {dataset}", file=sys.stderr)
    print(f"  output: {output_dir}", file=sys.stderr)
    print(f"  remote: {_model_ref(remote_config)}", file=sys.stderr)
    if synthesize_config is not None:
        print(f"  synthesize: {_model_ref(synthesize_config)}", file=sys.stderr)
    print(f"  local: {_model_ref(local_config)}", file=sys.stderr)


def _model_ref(config: dict[str, Any]) -> str:
    return (
        f"{config.get('provider', 'unknown')}/"
        f"{config.get('model', 'unknown')} @ "
        f"{config.get('base_url', 'default endpoint')}"
    )


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return default


if __name__ == "__main__":
    raise SystemExit(main())
