"""Remote-only DataBench baseline using the standard code-prompt setup."""

from __future__ import annotations

import ast
import copy
import json
import multiprocessing as mp
import re
import shutil
import textwrap
import time
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.costing import estimate_openai_text_cost
from clover.supervisor import extract_token_usage, generate_remote_text
from benchmarks.databench.adapter import load_databench_task, write_json
from benchmarks.databench.eval import select_cases
from benchmarks.databench.static_tool_eval import (
    answers_equal_relaxed,
    display_path,
    format_error,
    json_ready,
    normalize_answer,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.warnings import suppress_benchmark_warnings


PROMPT_MODE_CODE_DTYPE = "code_prompt_2_databench"
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 20.0
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class RemoteBaselineCase:
    sampled_case: dict[str, Any]
    sample_index: int
    case_dir: Path


def run_databench_remote_only_baseline(
    *,
    databench_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    sample_size: int | None = None,
    seed: int = 20260528,
    max_workers: int | None = None,
    overwrite: bool = False,
    execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    remote_cost_model: str | None = None,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run DataBench with one Remote LLM code-generation call per case."""

    with suppress_benchmark_warnings():
        started = time.perf_counter()
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_cases = select_cases(
            databench_root=databench_root,
            max_cases=max_cases,
            case_ids=case_ids or set(),
            dataset_id=dataset_id,
            sample_size=sample_size,
            seed=seed,
        )
        worker_count = _resolve_worker_count(
            max_workers=max_workers,
            remote_config=remote_config,
            case_count=len(selected_cases),
        )
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            records = _run_remote_baseline_cases(
                databench_root=databench_root,
                output_dir=output_dir,
                selected_cases=selected_cases,
                remote_config=remote_config,
                worker_count=worker_count,
                execution_timeout_seconds=execution_timeout_seconds,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        records.sort(key=lambda item: item["sample_index"])

        cases_index = output_dir / "cases_index.jsonl"
        mismatch_cases = output_dir / "answer_mismatch_cases.jsonl"
        failure_cases = output_dir / "failure_cases.jsonl"
        write_jsonl(cases_index, records)
        write_jsonl(
            mismatch_cases,
            [
                _mismatch_record(record)
                for record in records
                if record.get("runtime_ok") and not record.get("answer_correct")
            ],
        )
        write_jsonl(
            failure_cases,
            [record for record in records if not record.get("runtime_ok")],
        )

        summary = _build_summary(
            records=records,
            output_dir=output_dir,
            remote_config=remote_config,
            selected_cases=selected_cases,
            elapsed_seconds=time.perf_counter() - started,
            worker_count=worker_count,
            seed=seed,
            sample_size=sample_size,
            cases_index=cases_index,
            mismatch_cases=mismatch_cases,
            failure_cases=failure_cases,
            execution_timeout_seconds=execution_timeout_seconds,
            remote_cost_model=remote_cost_model,
        )
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_remote_baseline_cases(
    *,
    databench_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
    worker_count: int,
    execution_timeout_seconds: float,
    progress_bar: Any | None = None,
) -> list[dict[str, Any]]:
    if not selected_cases:
        return []

    records: list[dict[str, Any]] = []
    max_workers = max(1, min(worker_count, len(selected_cases)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_remote_baseline_case,
                databench_root=databench_root,
                case=RemoteBaselineCase(
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    case_dir=output_dir / "cases" / sampled_case["case_id"],
                ),
                remote_config=copy.deepcopy(remote_config),
                execution_timeout_seconds=execution_timeout_seconds,
            ): RemoteBaselineCase(
                sampled_case=sampled_case,
                sample_index=sample_index,
                case_dir=output_dir / "cases" / sampled_case["case_id"],
            )
            for sample_index, sampled_case in enumerate(selected_cases)
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate benchmark cases.
                record = _failed_case_record(
                    sampled_case=case.sampled_case,
                    sample_index=case.sample_index,
                    case_dir=case.case_dir,
                    exc=exc,
                )
            records.append(record)
            if progress_bar is not None:
                progress_bar.update(records)
    return records


def run_remote_baseline_case(
    *,
    databench_root: Path,
    case: RemoteBaselineCase,
    remote_config: dict[str, Any],
    execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one DataBench case through Remote LLM code generation plus Python execution."""

    case.case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    task = load_databench_task(
        databench_root=databench_root,
        dataset_id=case.sampled_case["dataset_id"],
        case_id=case.sampled_case["case_id"],
    )
    answer_type = task.metadata["case"].get("type") or task.task_dsl["answer"]["type"]
    expected_raw = task.metadata.get("expected_answer")
    expected_normalized = normalize_answer(expected_raw, answer_type)
    source = _single_task_source(task.task_dsl)
    table_path = _source_path(task.base_dir, source)
    table_profile = _table_profile(table_path)
    prompt = render_remote_code_prompt(
        question=task.task_dsl["question"],
        answer_type=answer_type,
        table_profile=table_profile,
    )

    remote_started = time.perf_counter()
    llm_result = generate_remote_text(
        prompt=prompt,
        remote_config=remote_config,
    )
    remote_elapsed = time.perf_counter() - remote_started
    usage = extract_token_usage(llm_result.response_payload)
    remote_output = llm_result.text

    parse_ok = False
    generated_code: str | None = None
    parse_error: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    try:
        generated_code = generated_answer_function_source(remote_output)
        parse_ok = True
    except Exception as exc:  # noqa: BLE001 - report format errors.
        parse_error = format_error(exc)

    execution_started = time.perf_counter()
    if generated_code is not None:
        execution_result = execute_generated_answer_function(
            generated_code,
            table_path=table_path,
            timeout_seconds=execution_timeout_seconds,
        )
    execution_elapsed = time.perf_counter() - execution_started
    actual = execution_result.get("answer") if execution_result else None
    actual_normalized = normalize_answer(actual, answer_type)
    runtime_ok = bool(parse_ok and execution_result and execution_result.get("ok"))
    error = None
    if not parse_ok:
        error = parse_error
    elif execution_result and not execution_result.get("ok"):
        error = execution_result.get("error")

    write_json(case.case_dir / "task_dsl.json", task.task_dsl)
    write_json(case.case_dir / "table_profile.json", table_profile)
    (case.case_dir / "prompt.py").write_text(prompt + "\n", encoding="utf-8")
    write_json(case.case_dir / "remote_response.json", llm_result.response_payload)
    (case.case_dir / "remote_output.py").write_text(
        remote_output + "\n",
        encoding="utf-8",
    )
    if generated_code is not None:
        (case.case_dir / "generated_code.py").write_text(
            generated_code + "\n",
            encoding="utf-8",
        )
    if execution_result is not None:
        write_json(case.case_dir / "execution_result.json", execution_result)
    if error is not None:
        write_json(case.case_dir / "error.json", error)

    record = {
        "sample_index": case.sample_index,
        "dataset_id": case.sampled_case["dataset_id"],
        "case_id": case.sampled_case["case_id"],
        "case_index": case.sampled_case.get("case_index"),
        "answer_type": answer_type,
        "question": task.task_dsl["question"],
        "expected_raw": expected_raw,
        "expected_normalized": json_ready(expected_normalized),
        "response_id": llm_result.response_id,
        "response_status": llm_result.response_status,
        "api_type": llm_result.api_type,
        "parse_ok": parse_ok,
        "runtime_ok": runtime_ok,
        "answer_correct": bool(
            runtime_ok
            and answers_equal_relaxed(
                expected_normalized,
                actual_normalized,
                answer_type,
            )
        ),
        "final_answer": json_ready(actual),
        "final_answer_preview": preview(actual),
        "final_answer_normalized": json_ready(actual_normalized),
        "remote_elapsed_seconds": remote_elapsed,
        "execution_elapsed_seconds": execution_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
        "remote_token_usage": usage,
        "error": error,
        "case_dir": display_path(case.case_dir),
        "files": {
            "task_dsl": display_path(case.case_dir / "task_dsl.json"),
            "table_profile": display_path(case.case_dir / "table_profile.json"),
            "prompt": display_path(case.case_dir / "prompt.py"),
            "remote_output": display_path(case.case_dir / "remote_output.py"),
            "remote_response": display_path(case.case_dir / "remote_response.json"),
            "generated_code": display_path(case.case_dir / "generated_code.py")
            if generated_code is not None
            else None,
            "execution_result": display_path(case.case_dir / "execution_result.json")
            if execution_result is not None
            else None,
            "error": display_path(case.case_dir / "error.json")
            if error is not None
            else None,
        },
    }
    write_json(case.case_dir / "case_result.json", record)
    return record


def render_remote_code_prompt(
    *,
    question: str,
    answer_type: str,
    table_profile: dict[str, Any],
) -> str:
    """Render the DataBench-style code prompt with dtypes."""

    dtypes = _databench_dtype_block(table_profile["dtypes"])
    columns = json.dumps(table_profile["columns"], ensure_ascii=False)
    return_type = _python_return_type(answer_type)
    return (
        "You are an assistant tasked with completing the answer function. "
        "Return only Python code.\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        f"def answer(df) -> {return_type}:\n"
        "    '''The df dtypes are:\n"
        f"{textwrap.indent(dtypes, '    ')}\n"
        f"    Returns: {question} {_answer_type_suffix(answer_type)}\n"
        "    '''\n"
        f"    df.columns = {columns}\n"
    )


def generated_answer_function_source(remote_output: str) -> str:
    """Normalize Remote LLM code output into a complete answer(df) function."""

    code = _extract_python_code(remote_output)
    try:
        module = ast.parse(code)
    except SyntaxError:
        module = None
    if module is not None and _contains_answer_function(module):
        return code.strip()

    body = textwrap.dedent(code).strip()
    if not body:
        raise ValueError("Remote output did not contain Python code")
    if _looks_like_expression(body):
        body = f"return {body}"
    source = "def answer(df):\n" + textwrap.indent(body, "    ")
    try:
        ast.parse(source)
    except IndentationError:
        # Some chat models return a function body with a stray function-level
        # indent on only the return line. As a fallback, flatten leading
        # whitespace and let Python validate the normalized function.
        body = "\n".join(line.lstrip() for line in body.splitlines())
        source = "def answer(df):\n" + textwrap.indent(body, "    ")
        ast.parse(source)
    return source


def execute_generated_answer_function(
    source: str,
    *,
    table_path: Path,
    timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute generated code in a short-lived process and return a JSON-safe result."""

    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_execute_generated_answer_function_worker,
        args=(str(table_path), source, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(2)
        return {
            "ok": False,
            "answer": None,
            "error": {
                "type": "TimeoutError",
                "message": f"Generated code exceeded {timeout_seconds:.1f}s timeout",
            },
        }
    if queue.empty():
        return {
            "ok": False,
            "answer": None,
            "error": {
                "type": "RuntimeError",
                "message": f"Generated code process exited with code {process.exitcode}",
            },
        }
    return queue.get()


def _execute_generated_answer_function_worker(
    table_path: str,
    source: str,
    queue: mp.Queue,
) -> None:
    try:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(table_path)
        namespace: dict[str, Any] = {"np": np, "pd": pd}
        exec(source, namespace, namespace)  # noqa: S102 - benchmark executes model code.
        answer_fn = namespace.get("answer")
        if not callable(answer_fn):
            raise ValueError("Generated code did not define callable answer(df)")
        answer = answer_fn(df)
        queue.put(
            {
                "ok": True,
                "answer": _runtime_json_ready(answer),
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - serialize model-code failures.
        queue.put(
            {
                "ok": False,
                "answer": None,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                    )[-6:],
                },
            }
        )


def _extract_python_code(remote_output: str) -> str:
    text = remote_output.strip()
    if not text:
        raise ValueError("Remote output is empty")
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text


def _contains_answer_function(module: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "answer"
        for node in module.body
    )


def _looks_like_expression(body: str) -> bool:
    if "\n" in body:
        return False
    try:
        ast.parse(body, mode="eval")
    except SyntaxError:
        return False
    return True


def _runtime_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _runtime_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_runtime_json_ready(item) for item in value]
    if hasattr(value, "to_dict") and value.__class__.__name__ == "DataFrame":
        return _runtime_json_ready(value.to_dict(orient="records"))
    if hasattr(value, "tolist"):
        try:
            return _runtime_json_ready(value.tolist())
        except TypeError:
            pass
    return json_ready(value)


def _table_profile(table_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(table_path)
    return {
        "path": str(table_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
    }


def _single_task_source(task_dsl: dict[str, Any]) -> dict[str, Any]:
    sources = task_dsl.get("sources", [])
    if len(sources) != 1 or not isinstance(sources[0], dict):
        raise ValueError("DataBench remote baseline requires one table source")
    return sources[0]


def _source_path(base_dir: Path, source: dict[str, Any]) -> Path:
    raw_path = source.get("path") or source.get("file")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("DataBench source must include path or file")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _answer_type_suffix(answer_type: str) -> str:
    normalized = str(answer_type).lower()
    if normalized == "boolean":
        return "Return a Python bool: True or False."
    if "number" in normalized:
        if normalized.startswith("list"):
            return "Return a Python list of numbers."
        return "Return one Python int or float."
    if normalized.startswith("list"):
        return "Return a Python list of category values."
    return "Return one category value present in the dataset."


def _python_return_type(answer_type: str) -> str:
    normalized = str(answer_type).lower()
    if normalized == "boolean":
        return "bool"
    if "number" in normalized:
        return "list[float]" if normalized.startswith("list") else "float"
    if normalized.startswith("list"):
        return "list[str]"
    return "str"


def _databench_dtype_block(dtypes: dict[str, Any]) -> str:
    lines = ["{"]
    for column, dtype in dtypes.items():
        lines.append(f"  {column!r}: dtype({str(dtype)!r}),")
    lines.append("}")
    return "\n".join(lines)


def _resolve_worker_count(
    *,
    max_workers: int | None,
    remote_config: dict[str, Any],
    case_count: int,
) -> int:
    if case_count <= 0:
        return 1
    configured_workers = max_workers or remote_config.get("parallel_workers") or 5
    worker_count = int(configured_workers)
    if worker_count <= 0:
        raise ValueError("parallel worker count must be positive")
    return min(worker_count, case_count)


def _failed_case_record(
    *,
    sampled_case: dict[str, Any],
    sample_index: int,
    case_dir: Path,
    exc: Exception,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    error = format_error(exc)
    write_json(case_dir / "case_error.json", error)
    return {
        "sample_index": sample_index,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "question": None,
        "expected_raw": None,
        "expected_normalized": None,
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_normalized": None,
        "remote_elapsed_seconds": 0.0,
        "execution_elapsed_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "remote_token_usage": _empty_usage(),
        "error": error,
        "case_dir": display_path(case_dir),
    }


def _build_summary(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    elapsed_seconds: float,
    worker_count: int,
    seed: int,
    sample_size: int | None,
    cases_index: Path,
    mismatch_cases: Path,
    failure_cases: Path,
    execution_timeout_seconds: float,
    remote_cost_model: str | None = None,
) -> dict[str, Any]:
    total = len(records)
    runtime_successes = sum(1 for record in records if record.get("runtime_ok"))
    correct = sum(1 for record in records if record.get("answer_correct"))
    mismatches = sum(
        1
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    error_types = Counter(
        record["error"]["type"]
        for record in records
        if isinstance(record.get("error"), dict) and record.get("error")
    )
    answer_types = Counter(record.get("answer_type") for record in records)
    mismatches_by_type = Counter(
        record.get("answer_type")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    by_dataset = _dataset_summary(records)
    token_usage = _sum_token_usage(records)
    local_slm_token_usage = _empty_usage()
    remote_cost_estimate = estimate_openai_text_cost(
        token_usage,
        remote_config=remote_config,
        pricing_model=remote_cost_model,
    )
    remote_elapsed = sum(
        float(record.get("remote_elapsed_seconds", 0.0) or 0.0)
        for record in records
    )
    execution_elapsed = sum(
        float(record.get("execution_elapsed_seconds", 0.0) or 0.0)
        for record in records
    )
    return {
        "run_name": output_dir.name,
        "stage": "databench_remote_only_baseline",
        "workflow": "remote_only_databench_code",
        "prompt_mode": PROMPT_MODE_CODE_DTYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "parallel_workers": worker_count,
        "max_retries": 0,
        "validation_mode": "remote_only",
        "remote_batch_size": None,
        "remote_concurrency": worker_count,
        "max_parallel_execution_units": None,
        "eval_batch_size": None,
        "profile_baseline": None,
        "execution_timeout_seconds": execution_timeout_seconds,
        "remote_llm": _config_summary(remote_config),
        "local_slm": None,
        "total_cases": total,
        "remote_calls": total,
        "local_slm_calls": 0,
        "parse_successes": sum(1 for record in records if record.get("parse_ok")),
        "parse_failures": sum(1 for record in records if not record.get("parse_ok")),
        "runtime_successes": runtime_successes,
        "runtime_failures": total - runtime_successes,
        "correct": correct,
        "mismatches": mismatches,
        "failures": total - runtime_successes,
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "accuracy_on_all_cases": safe_divide(correct, total),
        "retry_cases": 0,
        "retry_case_ids": [],
        "total_retry_rounds": 0,
        "sql_repair_cases": 0,
        "sql_repair_case_ids": [],
        "initial_execution_failures": 0,
        "initial_execution_failure_case_ids": [],
        "answer_types": _counter_as_strings(answer_types),
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
        "error_types": dict(sorted(error_types.items())),
        "by_dataset": by_dataset,
        "remote_token_usage": token_usage,
        "local_slm_token_usage": local_slm_token_usage,
        "remote_cost_estimate": remote_cost_estimate,
        "system_profile": _baseline_system_profile(
            remote_calls=total,
            remote_token_usage=token_usage,
            local_slm_token_usage=local_slm_token_usage,
        ),
        "remote_elapsed_seconds_sum": remote_elapsed,
        "remote_elapsed_seconds_avg": safe_divide(remote_elapsed, total),
        "execution_elapsed_seconds_sum": execution_elapsed,
        "execution_elapsed_seconds_avg": safe_divide(execution_elapsed, total),
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
    }


def _dataset_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        datasets.setdefault(str(record.get("dataset_id")), []).append(record)
    summary = {}
    for dataset_id, dataset_records in sorted(datasets.items()):
        total = len(dataset_records)
        runtime_successes = sum(1 for record in dataset_records if record.get("runtime_ok"))
        correct = sum(1 for record in dataset_records if record.get("answer_correct"))
        summary[dataset_id] = {
            "total_cases": total,
            "runtime_successes": runtime_successes,
            "runtime_failures": total - runtime_successes,
            "correct": correct,
            "accuracy_on_all_cases": safe_divide(correct, total),
            "accuracy_on_successes": safe_divide(correct, runtime_successes),
            "remote_token_usage": _sum_token_usage(dataset_records),
        }
    return summary


def _sum_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        usage = record.get("remote_token_usage")
        if not isinstance(usage, dict):
            continue
        for key in TOKEN_KEYS:
            totals[key] += int(usage.get(key, 0) or 0)
    return {key: int(totals.get(key, 0)) for key in TOKEN_KEYS}


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_KEYS}


def _counter_as_strings(counter: Counter[Any]) -> dict[str, int]:
    return dict(sorted((str(key or "unknown"), int(value)) for key, value in counter.items()))


def _baseline_system_profile(
    *,
    remote_calls: int,
    remote_token_usage: dict[str, int],
    local_slm_token_usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "stages": {},
        "counters": {},
        "summary": {
            "remote_calls": remote_calls,
            "local_slm_calls": 0,
            "merged_plan_count": 0,
            "reused_nodes": 0,
            "parallel_system_instances": 0,
            "remote_token_usage": remote_token_usage,
            "local_slm_token_usage": local_slm_token_usage,
            "validation_mode": "remote_only",
        },
    }


def _config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
    }


def _mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "answer_type": record.get("answer_type"),
        "question": record.get("question"),
        "expected": record.get("expected_normalized"),
        "actual": record.get("final_answer_normalized"),
        "remote_output": record.get("files", {}).get("remote_output"),
        "generated_code": record.get("files", {}).get("generated_code"),
    }
