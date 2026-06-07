"""Probe local SLM construction of minimal TableBench task DSLs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tablebench.adapter import iter_tablebench_dataset_dirs, read_cases
from clover.resource.dsl_builder import build_table_task_dsl_with_builder_agent


DEFAULT_TABLEBENCH_ROOT = REPO_ROOT / "datasets" / "tablebench"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "runs" / "dsl_builder_probe"
DEFAULT_LOCAL_SLM_CONFIG = REPO_ROOT / "model_config" / "local_slm_config.json"
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    tablebench_root = args.tablebench_root.expanduser().resolve()
    output_dir = (args.output_root / args.run_name).expanduser().resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    slm_config = _load_slm_config(args)
    selected_cases = _select_cases(
        tablebench_root=tablebench_root,
        max_cases=args.max_cases,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    records_path = output_dir / "dsl_builder_records.jsonl"
    token_usage = _empty_usage()
    answer_type_matches = 0
    fallback_count = 0
    json_successes = 0
    build_successes = 0

    with records_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(selected_cases, start=1):
            result = build_table_task_dsl_with_builder_agent(
                question=case["question"],
                table_path=case["table_path"],
                source_file="table.csv",
                slm_config=slm_config,
                max_preview_rows=args.max_preview_rows,
                max_columns=args.max_columns,
            )
            usage = _extract_token_usage(result.response_payload)
            _add_usage(token_usage, usage)
            built_answer_type = result.task_dsl["answer"]["type"]
            expected_answer_type = str(case.get("answer_type") or "")
            answer_type_match = built_answer_type == expected_answer_type
            if answer_type_match:
                answer_type_matches += 1
            if result.fallback_used:
                fallback_count += 1
            if result.parsed_output:
                json_successes += 1
            if result.parsed_output:
                build_successes += 1
            record = {
                "sample_index": index - 1,
                "builder_mode": result.builder_mode,
                "dataset_id": case["dataset_id"],
                "case_id": case["case_id"],
                "question": case["question"],
                "expected_answer_type": expected_answer_type,
                "built_answer_type": built_answer_type,
                "answer_type_match": answer_type_match,
                "fallback_used": result.fallback_used,
                "hints": result.task_dsl.get("hints", {}),
                "parsed_output": result.parsed_output,
                "raw_output": result.raw_output,
                "tool_call": result.tool_call,
                "diagnostics": result.diagnostics,
                "token_usage": usage,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not args.no_progress:
                print(
                    f"{index}/{len(selected_cases)} "
                    f"answer_type_acc={answer_type_matches / index:.3f} "
                    f"fallback={fallback_count}",
                    flush=True,
                )

    total = len(selected_cases)
    summary = {
        "run_name": args.run_name,
        "stage": "tablebench_dsl_builder_probe",
        "builder_mode": "builder_agent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tablebench_root": str(tablebench_root),
        "sample_size": total,
        "requested_sample_size": args.sample_size,
        "max_cases": args.max_cases,
        "seed": args.seed,
        "local_slm": _slm_summary(slm_config),
        "answer_type_matches": answer_type_matches,
        "answer_type_accuracy": answer_type_matches / total if total else 0.0,
        "build_successes": build_successes,
        "build_success_rate": build_successes / total if total else 0.0,
        "json_successes": json_successes,
        "json_success_rate": json_successes / total if total else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total if total else 0.0,
        "token_usage": token_usage,
        "elapsed_seconds": time.perf_counter() - started,
        "records": str(records_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe raw question/table to DSL construction with local SLM."
    )
    parser.add_argument("--tablebench-root", type=Path, default=DEFAULT_TABLEBENCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-name",
        default=f"dsl_builder_probe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--local-slm-config", type=Path, default=DEFAULT_LOCAL_SLM_CONFIG)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-preview-rows", type=int, default=2)
    parser.add_argument("--max-columns", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_slm_config(args: argparse.Namespace) -> dict[str, Any]:
    from clover.config import load_model_config
    from clover.executor.local_slm import LOCAL_SLM_ENV_PREFIXES

    return load_model_config(
        args.local_slm_config.expanduser().resolve(),
        env_prefixes=LOCAL_SLM_ENV_PREFIXES,
    )


def _slm_summary(slm_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if slm_config is None:
        return None
    return {
        "provider": slm_config.get("provider"),
        "api_type": slm_config.get("api_type"),
        "base_url": slm_config.get("base_url"),
        "model": slm_config.get("model"),
    }


def _extract_token_usage(payload: dict[str, Any]) -> dict[str, int]:
    if not payload:
        return _empty_usage()
    from clover.supervisor.client import extract_token_usage

    return extract_token_usage(payload)


def _select_cases(
    *,
    tablebench_root: Path,
    max_cases: int | None,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for dataset_dir in iter_tablebench_dataset_dirs(tablebench_root):
        cases_path = dataset_dir / "cases.jsonl"
        if not cases_path.is_file():
            continue
        for case_index, case in enumerate(read_cases(cases_path)):
            cases.append(
                {
                    "dataset_id": dataset_dir.name,
                    "case_id": case["case_id"],
                    "case_index": case_index,
                    "question": case["question"],
                    "answer_type": case.get("type"),
                    "table_path": dataset_dir / "table.csv",
                }
            )
    if sample_size is not None:
        rng = random.Random(seed)
        cases = rng.sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_KEYS}


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in TOKEN_KEYS:
        total[key] += int(usage.get(key, 0) or 0)


if __name__ == "__main__":
    main()
