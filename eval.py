from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from clover.config import load_model_config, load_optional_model_config
from clover.local_slm import DEFAULT_SLM_CONFIG_PATH
from benchmarks.databench.adapter import (
    first_databench_dataset,
    run_all_databench_tables,
    run_databench_case,
)
from benchmarks.databench.eval import run_databench_eval
from benchmarks.databench.static_tool_eval import run_static_tool_eval
from benchmarks.warnings import suppress_benchmark_warnings


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_NAME = "databench_remote_agent_loop"
DEFAULT_EVAL_RUN_NAME = "databench_eval"
DEFAULT_STATIC_TOOL_RUN_NAME = "databench_static_tool_eval"
DEFAULT_STATIC_TOOL_RUN_DIR = (
    REPO_ROOT / "benchmark" / "runs" / "databench_full_static_merged_20260527"
)
DEFAULT_REMOTE_LLM_CONFIG = REPO_ROOT / "config" / "doubao_remote_llm_config.json"
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
SAMPLING_CONFIG_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "reasoning_effort",
)
CLOVER_BANNER = r"""
   ________    ____ _    ____________
  / ____/ /   / __ \ |  / / ____/ __ \
 / /   / /   / / / / | / / __/ / /_/ /
/ /___/ /___/ /_/ /| |/ / /___/ _, _/
\____/_____/\____/ |___/_____/_/ |_|
""".strip("\n")


def main() -> None:
    args = _parse_args()
    databench_root = args.databench_root.resolve()
    run_name = args.run_name or (
        DEFAULT_STATIC_TOOL_RUN_NAME
        if args.static_tool_eval
        else DEFAULT_EVAL_RUN_NAME
        if args.databench_eval
        else DEFAULT_RUN_NAME
    )

    with suppress_benchmark_warnings():
        if args.databench_eval:
            remote_config = load_model_config(
                _resolve_path(args.remote_llm_config),
                env_prefixes=REMOTE_LLM_ENV_PREFIXES,
            )
            local_slm_config = load_optional_model_config(
                _resolve_path(args.local_slm_config),
                env_prefixes=LOCAL_SLM_ENV_PREFIXES,
            )
            print_eval_startup_banner(
                remote_config=remote_config,
                local_slm_config=local_slm_config,
                workflow="table_reasoning",
                remote_batch_size=args.remote_batch_size,
                local_batch_size=args.local_batch_size,
            )
            summary = run_databench_eval(
                databench_root=databench_root,
                output_dir=(args.output_root / run_name).resolve(),
                remote_config=remote_config,
                local_slm_config=local_slm_config,
                max_cases=args.max_cases,
                case_ids={args.case_id} if args.case_id else set(),
                dataset_id=args.dataset_id,
                sample_size=args.sample_size,
                seed=args.seed,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                remote_batch_size=args.remote_batch_size,
                local_batch_size=args.local_batch_size,
                eval_batch_size=args.eval_batch_size,
                profile_baseline=args.profile_baseline,
                overwrite=args.overwrite,
                progress_factory=None if args.no_progress else EvalProgressBar,
                preprocess_progress_factory=None
                if args.no_progress
                else EvalPreprocessProgressBar,
            )
        elif args.static_tool_eval:
            summary = run_static_tool_eval(
                run_dir=args.run_dir.resolve(),
                databench_root=databench_root,
                output_dir=_static_tool_output_dir(args, run_name),
                max_cases=args.max_cases,
                case_ids={args.case_id} if args.case_id else set(),
                progress_every=args.progress_every,
            )
        elif args.all_databench_tables:
            summary = run_all_databench_tables(
                databench_root=databench_root,
                case_index=args.case_index,
                output_root=args.output_root.resolve(),
                run_name=run_name,
            )
        else:
            dataset_id = args.dataset_id or first_databench_dataset(databench_root)
            summary = run_databench_case(
                databench_root=databench_root,
                dataset_id=dataset_id,
                case_id=args.case_id,
                case_index=args.case_index,
                output_root=args.output_root.resolve(),
                run_name=run_name,
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLOVER evaluation stages.")
    parser.add_argument(
        "--databench-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "databench",
        help="Path to the Databench dataset root.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Databench dataset id. Defaults to the first dataset directory.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Databench case id. If omitted, --case-index is used.",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Databench case index used when --case-id is omitted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "benchmark" / "runs",
        help="Directory where run outputs are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run directory name under --output-root.",
    )
    parser.add_argument(
        "--all-databench-tables",
        action="store_true",
        help="Run one case for every Databench table directory.",
    )
    parser.add_argument(
        "--static-tool-eval",
        action="store_true",
        help="Execute existing Databench physical plans with static tools.",
    )
    parser.add_argument(
        "--eval",
        dest="databench_eval",
        action="store_true",
        help="Run full Databench evaluation from initial task DSL through Reporter.",
    )
    parser.add_argument(
        "--remote-llm-config",
        type=Path,
        default=_default_remote_llm_config_path(),
        help="Remote LLM config used by --eval.",
    )
    parser.add_argument(
        "--local-slm-config",
        type=Path,
        default=_default_local_slm_config_path(),
        help="Local SLM config used by NodeAgent loops in --eval.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_STATIC_TOOL_RUN_DIR,
        help="Run directory containing cases/*/physical_plan.json for --static-tool-eval.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of cases for evaluation runs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional random Databench sample size for --eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260528,
        help="Random seed used with --sample-size.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel worker count for --eval.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Reporter retry limit for --eval.",
    )
    parser.add_argument(
        "--remote-batch-size",
        type=int,
        default=16,
        help="Maximum questions per Remote LLM batch in table reasoning eval.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=4,
        help="Maximum same-table DAGs merged into one local execution in table reasoning eval.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation-side ingest batch size metadata for table reasoning eval.",
    )
    parser.add_argument(
        "--profile-baseline",
        action="store_true",
        help="Also run one-by-one local execution profiling.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress interval for --static-tool-eval; use 0 to disable.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar for --eval.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="For --static-tool-eval, write summary files back into --run-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="For --static-tool-eval, replace an existing output run directory.",
    )
    return parser.parse_args()


def _static_tool_output_dir(args: argparse.Namespace, run_name: str) -> Path:
    source_dir = args.run_dir.resolve()
    if args.in_place:
        return source_dir
    output_dir = (args.output_root / run_name).resolve()
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
    local_slm_config: dict[str, Any] | None,
    workflow: str | None = None,
    remote_batch_size: int | None = None,
    local_batch_size: int | None = None,
) -> None:
    """Print the shared eval startup header without exposing secrets."""

    print(CLOVER_BANNER, file=sys.stderr)
    print(file=sys.stderr)
    if workflow is not None:
        print("Workflow:", file=sys.stderr)
        print(f"  task: {workflow}", file=sys.stderr)
        if remote_batch_size is not None:
            print(f"  remote_batch_size: {remote_batch_size}", file=sys.stderr)
        if local_batch_size is not None:
            print(f"  local_batch_size: {local_batch_size}", file=sys.stderr)
    print("Remote LLM:", file=sys.stderr)
    for line in _model_config_lines(remote_config):
        print(f"  {line}", file=sys.stderr)
    print("Local SLM:", file=sys.stderr)
    if local_slm_config is None:
        print("  not configured", file=sys.stderr)
    else:
        for line in _model_config_lines(local_slm_config):
            print(f"  {line}", file=sys.stderr)
    print(file=sys.stderr, flush=True)


class EvalProgressBar:
    """Shared stderr progress bar for dataset evaluation records."""

    def __init__(self, total: int, *, width: int = 28) -> None:
        self.total = total
        self.width = width
        self._last_len = 0
        self._color = "NO_COLOR" not in os.environ

    def update(self, records: list[dict[str, Any]]) -> None:
        completed = len(records)
        correct = sum(1 for record in records if record.get("answer_correct"))
        mismatches = sum(
            1
            for record in records
            if record.get("runtime_ok") and not record.get("answer_correct")
        )
        retry_cases = sum(1 for record in records if record.get("retry_count", 0) > 0)
        failures = sum(1 for record in records if not record.get("runtime_ok"))
        filled = int(self.width * completed / self.total) if self.total else self.width
        percent = _safe_divide(completed, self.total) or 0.0
        bar = self._bar(filled)
        plain_text = (
            f"[{'#' * filled}{'-' * (self.width - filled)}] "
            f"{completed}/{self.total} {percent:6.1%} "
            f"correct={correct} mismatch={mismatches} fail={failures} "
            f"acc={(_safe_divide(correct, completed) or 0.0):.3f} retry={retry_cases}"
        )
        text = (
            f"\r{bar} "
            f"{self._paint(f'{completed}/{self.total}', '1;37')} "
            f"{self._paint(f'{percent:6.1%}', '1;36')} "
            f"{self._paint(f'correct={correct}', '1;32')} "
            f"{self._paint(f'mismatch={mismatches}', '1;33')} "
            f"{self._paint(f'fail={failures}', '1;31')} "
            f"acc={(_safe_divide(correct, completed) or 0.0):.3f} "
            f"{self._paint(f'retry={retry_cases}', '1;35')}"
        )
        padding = " " * max(0, self._last_len - len(plain_text))
        print(text + padding, file=sys.stderr, end="", flush=True)
        self._last_len = len(plain_text)

    def close(self) -> None:
        print(file=sys.stderr, flush=True)

    def _bar(self, filled: int) -> str:
        done = "#" * filled
        remaining = "-" * (self.width - filled)
        return f"[{self._paint(done, '1;32')}{self._paint(remaining, '2;37')}]"

    def _paint(self, text: str, code: str) -> str:
        if not self._color or not text:
            return text
        return f"\033[{code}m{text}\033[0m"


class EvalPreprocessProgressBar:
    """Separate stderr progress bar for local eval preprocessing."""

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
        remaining = "-" * (self.width - filled)
        return f"[{self._paint(done, '1;32')}{self._paint(remaining, '2;37')}]"

    def _paint(self, text: str, code: str) -> str:
        if not self._color or not text:
            return text
        return f"\033[{code}m{text}\033[0m"


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


def _model_config_lines(config: dict[str, Any]) -> list[str]:
    sampling = _sampling_config(config)
    return [
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


if __name__ == "__main__":
    main()
