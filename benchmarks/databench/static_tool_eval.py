from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any

from clover.executor import ExecutionPlanBuilder, execute_execution_plan
from benchmarks.databench.adapter import load_databench_task
from benchmarks.warnings import suppress_benchmark_warnings


REPO_ROOT = Path(__file__).resolve().parents[2]


class TracedExecutionError(RuntimeError):
    def __init__(self, failing_node: dict[str, Any] | None, original: Exception) -> None:
        super().__init__(str(original))
        self.failing_node = failing_node
        self.original = original


def main() -> None:
    args = parse_args()
    output_dir = _prepare_output_dir(args)
    with suppress_benchmark_warnings():
        summary = run_static_tool_eval(
            run_dir=args.run_dir.resolve(),
            databench_root=args.databench_root.resolve(),
            output_dir=output_dir,
            max_cases=args.max_cases,
            case_ids=set(args.case_id or []),
            progress_every=args.progress_every,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_static_tool_eval(
    *,
    run_dir: Path,
    databench_root: Path,
    output_dir: Path,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    with suppress_benchmark_warnings():
        started = time.perf_counter()
        case_dirs = list(iter_case_dirs(run_dir, case_ids=case_ids, max_cases=max_cases))
        table_cache: dict[str, Any] = {}

        records = []
        for index, case_dir in enumerate(case_dirs):
            if progress_every and index and index % progress_every == 0:
                print(f"processed {index}/{len(case_dirs)}", file=sys.stderr, flush=True)
            records.append(evaluate_case(case_dir, databench_root, index, table_cache))

        output_dir.mkdir(parents=True, exist_ok=True)
        cases_index = output_dir / "cases_index.jsonl"
        execution_failures = output_dir / "execution_failure_cases.jsonl"
        mismatch_cases = output_dir / "answer_mismatch_cases.jsonl"
        mismatch_case_ids = output_dir / "answer_mismatch_case_ids.txt"
        mismatch_summary_path = output_dir / "answer_mismatch_cases_summary.json"

        write_jsonl(cases_index, records)
        write_jsonl(
            execution_failures,
            [record for record in records if not record["execution_ok"]],
        )
        mismatch_records = [
            answer_mismatch_record(record)
            for record in records
            if record["execution_ok"] and not record["answer_correct_relaxed"]
        ]
        write_jsonl(mismatch_cases, mismatch_records)
        mismatch_case_ids.write_text(
            "".join(f"{record['case_id']}\n" for record in mismatch_records),
            encoding="utf-8",
        )

        summary = build_summary(
            records=records,
            run_dir=run_dir,
            output_dir=output_dir,
            elapsed_seconds=time.perf_counter() - started,
            cases_index=cases_index,
            execution_failures=execution_failures,
            mismatch_cases=mismatch_cases,
            mismatch_case_ids=mismatch_case_ids,
            mismatch_summary_path=mismatch_summary_path,
        )
        write_json(output_dir / "run_summary.json", summary)
        write_json(mismatch_summary_path, build_answer_mismatch_summary(mismatch_records))
        return summary


def evaluate_case(
    case_dir: Path,
    databench_root: Path,
    sample_index: int,
    table_cache: dict[str, Any],
) -> dict[str, Any]:
    case_id = case_dir.name
    dataset_id = case_id.rsplit("_", 1)[0]
    task = load_databench_task(databench_root, dataset_id=dataset_id, case_id=case_id)
    case = task.metadata["case"]
    plan_path = case_dir / "physical_plan.json"
    answer_type = case.get("type") or task.task_dsl["answer"]["type"]
    expected_raw = task.metadata.get("expected_answer")
    expected_normalized = normalize_answer(expected_raw, answer_type)

    base_record = {
        "sample_index": sample_index,
        "dataset_id": dataset_id,
        "case_id": case_id,
        "answer_type": answer_type,
        "question": case.get("question") or task.task_dsl.get("question"),
        "physical_plan": display_path(plan_path),
        "parse_ok": plan_path.is_file(),
        "execution_ok": False,
        "answer_correct_relaxed": False,
        "expected_raw": expected_raw,
        "expected_normalized": expected_normalized,
        "actual_preview": None,
        "actual_normalized_preview": None,
        "execution_elapsed_ms": None,
        "fast_path_hits": 0,
        "fast_path_misses": 0,
        "error": None,
        "failing_node": None,
        "case_dir": display_path(case_dir),
        "files": case_files(case_dir),
    }
    if not plan_path.is_file():
        base_record["error"] = {
            "type": "FileNotFoundError",
            "message": f"Missing physical plan: {plan_path}",
        }
        return base_record

    plan = read_json(plan_path)
    base_record["ops"] = [node.get("op") for node in plan.get("nodes", [])]
    try:
        execution_result = _execute_plan(plan, table_cache=table_cache)
    except Exception as exc:  # noqa: BLE001 - benchmark runner should capture all failures.
        base_record["error"] = format_error(exc)
        return base_record
    base_record["execution_elapsed_ms"] = execution_result.elapsed_ms
    base_record["fast_path_hits"] = execution_result.fast_path_hits
    base_record["fast_path_misses"] = execution_result.fast_path_misses
    if not execution_result.ok:
        base_record["failing_node"] = execution_result.failing_node
        base_record["error"] = execution_result.error
        return base_record

    outputs = execution_result.outputs

    actual = outputs.get("answer")
    if actual is None and outputs:
        actual = next(reversed(outputs.values()))
    actual_normalized = normalize_answer(actual, answer_type)
    base_record.update(
        {
            "execution_ok": True,
            "actual_preview": preview(actual),
            "actual_normalized_preview": preview(actual_normalized),
            "answer_correct_relaxed": answers_equal_relaxed(
                expected_normalized,
                actual_normalized,
                answer_type,
            ),
        }
    )
    return base_record


def execute_plan_with_trace(
    plan: dict[str, Any],
    table_cache: dict[str, Any],
) -> dict[str, Any]:
    result = _execute_plan(plan, table_cache=table_cache)
    if not result.ok:
        raise TracedExecutionError(
            result.failing_node,
            RuntimeError(result.error["message"] if result.error else "execution failed"),
        )
    return dict(result.outputs)


def _execute_plan(
    plan: dict[str, Any],
    *,
    table_cache: dict[str, Any],
):
    return execute_execution_plan(
        ExecutionPlanBuilder.default().build(plan),
        collector_context=plan,
        table_cache=table_cache,
    )


def build_summary(
    *,
    records: list[dict[str, Any]],
    run_dir: Path,
    output_dir: Path,
    elapsed_seconds: float,
    cases_index: Path,
    execution_failures: Path,
    mismatch_cases: Path,
    mismatch_case_ids: Path,
    mismatch_summary_path: Path,
) -> dict[str, Any]:
    total = len(records)
    execution_successes = sum(record["execution_ok"] for record in records)
    relaxed_correct = sum(record["answer_correct_relaxed"] for record in records)
    failed_ops = Counter(
        record["failing_node"]["op"]
        for record in records
        if record.get("failing_node") and not record["execution_ok"]
    )
    error_types = Counter(
        record["error"]["type"]
        for record in records
        if record.get("error")
    )
    ops_seen = Counter(
        op
        for record in records
        for op in record.get("ops", [])
        if op is not None
    )
    answer_types = Counter(record["answer_type"] for record in records)
    relaxed_failures_by_type = Counter(
        record["answer_type"]
        for record in records
        if not record["answer_correct_relaxed"]
    )
    fast_path_hits = sum(record.get("fast_path_hits", 0) for record in records)
    fast_path_misses = sum(record.get("fast_path_misses", 0) for record in records)
    executed_node_count = fast_path_hits + fast_path_misses

    return {
        "run_name": output_dir.name,
        "stage": "databench_static_tool_execution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": display_path(run_dir),
        "total_cases": total,
        "physical_plan_available": sum(record["parse_ok"] for record in records),
        "physical_plan_unavailable": sum(not record["parse_ok"] for record in records),
        "execution_successes": execution_successes,
        "execution_failures": total - execution_successes,
        "answer_correct_relaxed": relaxed_correct,
        "answer_incorrect_relaxed": total - relaxed_correct,
        "relaxed_accuracy_on_executed": safe_divide(relaxed_correct, execution_successes),
        "relaxed_accuracy_on_all_cases": safe_divide(relaxed_correct, total),
        "elapsed_seconds": elapsed_seconds,
        "fast_path_hits": fast_path_hits,
        "fast_path_misses": fast_path_misses,
        "fast_path_hit_rate": safe_divide(fast_path_hits, executed_node_count),
        "answer_types": dict(sorted(answer_types.items())),
        "ops_seen": dict(sorted(ops_seen.items())),
        "failed_ops": dict(sorted(failed_ops.items())),
        "execution_error_types": dict(sorted(error_types.items())),
        "relaxed_failures_by_type": dict(sorted(relaxed_failures_by_type.items())),
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "execution_failure_cases": display_path(execution_failures),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "answer_mismatch_case_ids": display_path(mismatch_case_ids),
        "answer_mismatch_cases_summary": display_path(mismatch_summary_path),
    }


def build_answer_mismatch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(record["answer_type"] for record in records)
    by_dataset = Counter(record["dataset_id"] for record in records)
    return {
        "total": len(records),
        "by_answer_type": dict(sorted(by_type.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "cases": records,
    }


def answer_mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "answer_type": record["answer_type"],
        "question": record["question"],
        "expected": record["expected_normalized"],
        "actual": record["actual_normalized_preview"],
        "ops": record.get("ops", []),
        "remote_output": record["files"].get("remote_output"),
        "physical_plan": record["physical_plan"],
    }


def normalize_answer(value: Any, answer_type: str) -> Any:
    parsed = parse_literal_value(value)
    normalized_type = str(answer_type).lower()
    if normalized_type.startswith("list"):
        if parsed is None:
            return []
        items = parsed if isinstance(parsed, list) else [parsed]
        element_type = "number" if "number" in normalized_type else "category"
        return [normalize_answer(item, element_type) for item in items]
    if normalized_type == "boolean":
        return to_bool(parsed)
    if normalized_type in {"number", "integer", "int", "float"}:
        return to_number(parsed)
    if parsed is None:
        return None
    return str(parsed)


def parse_literal_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def to_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def answers_equal_relaxed(expected: Any, actual: Any, answer_type: str) -> bool:
    normalized_type = str(answer_type).lower()
    if normalized_type.startswith("list"):
        if not isinstance(expected, list) or not isinstance(actual, list):
            return False
        if len(expected) != len(actual):
            return False
        return all(
            answers_equal_relaxed(left, right, "number" if "number" in normalized_type else "category")
            for left, right in zip(expected, actual)
        )
    if "number" in normalized_type:
        return numbers_equal(expected, actual, rel_tol=1e-6, abs_tol=1e-6)
    if normalized_type == "boolean":
        return expected is actual
    if expected is None or actual is None:
        return expected is actual
    return str(expected).strip().lower() == str(actual).strip().lower()


def numbers_equal(expected: Any, actual: Any, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            return False
        if len(expected) != len(actual):
            return False
        return all(numbers_equal(left, right, rel_tol, abs_tol) for left, right in zip(expected, actual))
    if expected is None or actual is None:
        return expected is actual
    try:
        return math.isclose(float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False


def iter_case_dirs(
    run_dir: Path,
    *,
    case_ids: set[str] | None = None,
    max_cases: int | None = None,
) -> list[Path]:
    cases_root = run_dir / "cases"
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Missing cases directory: {cases_root}")
    selected = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if case_ids and case_dir.name not in case_ids:
            continue
        selected.append(case_dir)
        if max_cases is not None and len(selected) >= max_cases:
            break
    return selected


def case_files(case_dir: Path) -> dict[str, str | None]:
    names = [
        "prompt.md",
        "remote_output.txt",
        "remote_response.json",
        "parsed_sql.json",
        "logic_dag.json",
        "physical_plan.json",
        "task_dsl.json",
        "local_dsl.json",
        "remote_dsl.json",
        "context.json",
    ]
    return {
        path.stem if path.suffix != ".md" else path.name: display_path(path) if path.is_file() else None
        for path in (case_dir / name for name in names)
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (date, datetime, datetime_time)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def format_error(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:],
    }


def preview(value: Any, max_length: int = 240) -> str:
    text = repr(json_ready(value))
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.in_place:
        return args.run_dir.resolve()
    output_dir = (args.output_root / args.run_name).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute existing Databench physical plans with table_reasoning static tools "
            "and report answer accuracy."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "benchmark" / "runs" / "databench_full_static_merged_20260527",
        help="Run directory containing cases/*/physical_plan.json.",
    )
    parser.add_argument(
        "--databench-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "databench",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "benchmark" / "runs",
    )
    parser.add_argument("--run-name", default="databench_static_tool_eval")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write summary/index files back into --run-dir instead of creating a new run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory when not using --in-place.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
