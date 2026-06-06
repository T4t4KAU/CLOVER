"""Remote-only TableBench baselines using TableBench instruction types."""

from __future__ import annotations

import copy
import csv
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.costing import estimate_openai_text_cost
from benchmarks.databench.static_tool_eval import (
    display_path,
    format_error,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.tablebench.adapter import (
    iter_tablebench_dataset_dirs,
    read_cases,
    write_json,
)
from benchmarks.tablebench.metrics import score_tablebench_answer, tablebench_metric_name
from benchmarks.warnings import suppress_benchmark_warnings
from clover.supervisor import extract_token_usage, generate_remote_text


TABLEBENCH_INSTRUCTION_TYPES = frozenset({"DP", "TCoT", "PoT"})
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class TableBenchRemoteCase:
    sampled_case: dict[str, Any]
    sample_index: int
    case_dir: Path


def run_tablebench_remote_only_baseline(
    *,
    tablebench_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    qtypes: set[str] | None = None,
    qsubtypes: set[str] | None = None,
    include_visualization: bool = False,
    sample_size: int | None = None,
    seed: int = 20260528,
    max_workers: int | None = None,
    instruction_type: str = "DP",
    execution_timeout_seconds: float = 20.0,
    overwrite: bool = False,
    remote_cost_model: str | None = None,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run an independent Remote-only TableBench baseline."""

    with suppress_benchmark_warnings():
        started = time.perf_counter()
        instruction_type = _normalize_instruction_type(instruction_type)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_cases = select_tablebench_cases(
            tablebench_root=tablebench_root,
            max_cases=max_cases,
            case_ids=case_ids or set(),
            dataset_id=dataset_id,
            qtypes=qtypes or set(),
            qsubtypes=qsubtypes or set(),
            include_visualization=include_visualization,
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
            records = _run_remote_cases(
                tablebench_root=tablebench_root,
                output_dir=output_dir,
                selected_cases=selected_cases,
                remote_config=remote_config,
                worker_count=worker_count,
                instruction_type=instruction_type,
                execution_timeout_seconds=execution_timeout_seconds,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        records.sort(key=lambda item: item["sample_index"])
        cases_index = output_dir / "cases_index.jsonl"
        predictions_path = output_dir / "predictions.jsonl"
        mismatch_cases = output_dir / "answer_mismatch_cases.jsonl"
        failure_cases = output_dir / "failure_cases.jsonl"
        write_jsonl(cases_index, [_case_index_record(record) for record in records])
        write_jsonl(predictions_path, [_prediction_record(record) for record in records])
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
            include_visualization=include_visualization,
            cases_index=cases_index,
            predictions_path=predictions_path,
            mismatch_cases=mismatch_cases,
            failure_cases=failure_cases,
            instruction_type=instruction_type,
            execution_timeout_seconds=execution_timeout_seconds,
            remote_cost_model=remote_cost_model,
        )
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_remote_cases(
    *,
    tablebench_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
    worker_count: int,
    instruction_type: str,
    execution_timeout_seconds: float,
    progress_bar: Any | None,
) -> list[dict[str, Any]]:
    if not selected_cases:
        return []
    records: list[dict[str, Any]] = []
    max_workers = max(1, min(worker_count, len(selected_cases)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_remote_baseline_case,
                tablebench_root=tablebench_root,
                case=TableBenchRemoteCase(
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    case_dir=output_dir / "cases" / sampled_case["case_id"],
                ),
                remote_config=copy.deepcopy(remote_config),
                instruction_type=instruction_type,
                execution_timeout_seconds=execution_timeout_seconds,
            ): TableBenchRemoteCase(
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
                    instruction_type=instruction_type,
                )
            records.append(record)
            if progress_bar is not None:
                progress_bar.update(records)
    return records


def run_remote_baseline_case(
    *,
    tablebench_root: Path,
    case: TableBenchRemoteCase,
    remote_config: dict[str, Any],
    instruction_type: str = "DP",
    execution_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    case.case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset_dir = tablebench_root / case.sampled_case["dataset_id"]
    table_path = dataset_dir / "table.csv"
    case_payload = _load_case_payload(dataset_dir, case.sampled_case["case_id"])
    table = tablebench_table_from_csv(table_path)
    prompt = render_tablebench_instruction_prompt(
        table=table,
        question=str(case_payload.get("question") or ""),
        instruction_type=instruction_type,
    )

    remote_started = time.perf_counter()
    llm_result = generate_remote_text(prompt=prompt, remote_config=remote_config)
    remote_elapsed = time.perf_counter() - remote_started
    usage = extract_token_usage(llm_result.response_payload)
    remote_output = llm_result.text
    parsed = parse_tablebench_prediction(
        remote_output,
        instruction_type=instruction_type,
        table_path=table_path,
        work_dir=case.case_dir,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    parsed_answer = parsed["parsed_prediction"]
    parse_ok = bool(str(parsed_answer or "").strip())
    score = score_tablebench_answer(
        expected=case_payload.get("answer"),
        actual=parsed_answer,
        qtype=case_payload.get("qtype"),
        qsubtype=case_payload.get("qsubtype"),
    )
    runtime_ok = parse_ok
    error = parsed.get("error")
    if not parse_ok:
        error = error or {"type": "PredictionParseError", "message": "Missing final answer"}

    write_json(case.case_dir / "case.json", case_payload)
    (case.case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    write_json(case.case_dir / "remote_response.json", llm_result.response_payload)
    (case.case_dir / "remote_output.txt").write_text(
        remote_output + "\n",
        encoding="utf-8",
    )
    if error is not None:
        write_json(case.case_dir / "error.json", error)
    if parsed.get("python_code"):
        (case.case_dir / "generated_code.py").write_text(
            str(parsed["python_code"]).rstrip() + "\n",
            encoding="utf-8",
        )
    if parsed.get("stdout") is not None:
        (case.case_dir / "program_stdout.txt").write_text(
            str(parsed["stdout"]),
            encoding="utf-8",
        )

    record = {
        "sample_index": case.sample_index,
        "dataset_id": case.sampled_case["dataset_id"],
        "case_id": case.sampled_case["case_id"],
        "case_index": case.sampled_case.get("case_index"),
        "answer_type": case_payload.get("type"),
        "qtype": case_payload.get("qtype"),
        "qsubtype": case_payload.get("qsubtype"),
        "question": case_payload.get("question"),
        "expected_raw": case_payload.get("answer"),
        "expected_standard_text": score.expected,
        "response_id": llm_result.response_id,
        "response_status": llm_result.response_status,
        "api_type": llm_result.api_type,
        "model_name": str(remote_config.get("model") or "remote"),
        "instruction_type": instruction_type,
        "instruction": prompt,
        "table": table,
        "remote_output": remote_output,
        "prompt_mode": f"tablebench_{instruction_type}",
        "parse_ok": parse_ok,
        "ecr_1": parsed.get("ecr_1"),
        "runtime_ok": runtime_ok,
        "answer_correct": bool(runtime_ok and score.correct),
        "tablebench_metric": score.metric,
        "tablebench_score": score.score if runtime_ok else 0.0,
        "final_answer": json_ready(parsed_answer),
        "final_answer_preview": preview(parsed_answer),
        "final_answer_standard_text": score.actual,
        "remote_elapsed_seconds": remote_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
        "remote_token_usage": usage,
        "error": error,
        "case_dir": display_path(case.case_dir),
        "files": {
            "case": display_path(case.case_dir / "case.json"),
            "prompt": display_path(case.case_dir / "prompt.txt"),
            "remote_output": display_path(case.case_dir / "remote_output.txt"),
            "remote_response": display_path(case.case_dir / "remote_response.json"),
            "generated_code": display_path(case.case_dir / "generated_code.py")
            if parsed.get("python_code")
            else None,
            "program_stdout": display_path(case.case_dir / "program_stdout.txt")
            if parsed.get("stdout") is not None
            else None,
            "error": display_path(case.case_dir / "error.json") if error else None,
        },
    }
    write_json(case.case_dir / "case_result.json", record)
    return record


def render_tablebench_instruction_prompt(
    *,
    table: dict[str, Any],
    question: str,
    instruction_type: str,
) -> str:
    instruction_type = _normalize_instruction_type(instruction_type)
    table_repr = repr(table)
    answer_format = (
        "The answer should follow the format below:\n"
        "[Answer Format]\n"
        "Final Answer: AnswerName1, AnswerName2...\n\n"
        "Ensure the final answer format is the last output line and can only be "
        'in the "Final Answer: AnswerName1, AnswerName2..." form, no other form. '
        'Ensure the "AnswerName" is a number or entity name, as short as possible, '
        "without any explanation.\n"
    )
    if instruction_type == "DP":
        return (
            "You are a table analyst. Your task is to answer questions based on "
            "the table content.\n\n\n"
            f"{answer_format}\n\n"
            "Give the final answer to the question directly without any explanation.\n\n"
            "Read the table below in JSON format:\n"
            "[TABLE] \n"
            f"{table_repr}\n\n"
            "Let's get start!\n"
            f"Question: {question.strip()}\n"
        )
    if instruction_type == "TCoT":
        return (
            "You are a table analyst. Your task is to answer questions based on "
            "the table content.\n\n\n"
            f"{answer_format}\n\n"
            "Think step by step over the table. Keep the reasoning concise, and "
            "make the final answer the last line in the exact Final Answer "
            "format.\n\n"
            "Read the table below in JSON format:\n"
            "[TABLE] \n"
            f"{table_repr}\n\n"
            "Let's get start!\n"
            f"Question: {question.strip()}\n"
        )
    return (
        "You are a table analyst. Your task is to answer questions based on "
        "the table content.\n\n\n"
        f"{answer_format}\n\n"
        "Write Python code to compute the answer. Return only one Python code "
        "block in the form ```python ... ```.\n"
        "A CSV file named table.csv will be available in the working directory. "
        "You may use pandas. The code must print the final result in the exact "
        "Final Answer format.\n\n"
        "Read the table below in JSON format:\n"
        "[TABLE] \n"
        f"{table_repr}\n\n"
        "Let's get start!\n"
        f"Question: {question.strip()}\n"
    )


def parse_tablebench_prediction(
    prediction: str,
    *,
    instruction_type: str = "DP",
    table_path: Path | None = None,
    work_dir: Path | None = None,
    execution_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    instruction_type = _normalize_instruction_type(instruction_type)
    if instruction_type in {"DP", "TCoT"}:
        parsed_prediction = parse_dp_prediction(prediction)
        return {
            "parsed_prediction": parsed_prediction,
            "ecr_1": None,
            "python_code": None,
            "stdout": None,
            "error": None if parsed_prediction else _parse_error("Missing Final Answer"),
        }
    return parse_pot_prediction(
        prediction,
        table_path=table_path,
        work_dir=work_dir,
        execution_timeout_seconds=execution_timeout_seconds,
    )


def parse_dp_prediction(prediction: str) -> str:
    text = str(prediction or "")
    matches = re.findall(r"Final Answer:\s*(.+)", text)
    return matches[-1].strip() if matches else ""


def parse_pot_prediction(
    prediction: str,
    *,
    table_path: Path | None,
    work_dir: Path | None,
    execution_timeout_seconds: float,
) -> dict[str, Any]:
    python_code = parse_python_code(prediction)
    if not python_code:
        return {
            "parsed_prediction": "",
            "ecr_1": False,
            "python_code": None,
            "stdout": "",
            "error": _parse_error("Missing Python code block"),
        }
    if table_path is None or work_dir is None:
        return {
            "parsed_prediction": "",
            "ecr_1": False,
            "python_code": python_code,
            "stdout": "",
            "error": {
                "type": "PoTExecutionSetupError",
                "message": "table_path and work_dir are required for PoT parsing",
            },
        }
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(table_path, work_dir / "table.csv")
    source = surround_pycode_with_main(python_code)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=work_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=execution_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "parsed_prediction": "",
            "ecr_1": False,
            "python_code": python_code,
            "stdout": exc.stdout or "",
            "error": {
                "type": "PoTExecutionTimeout",
                "message": f"Generated code exceeded {execution_timeout_seconds:.1f}s",
            },
        }
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    ecr_1 = completed.returncode == 0
    parsed_prediction = parse_code_output_prediction(stdout) if stdout else ""
    if not parsed_prediction and stdout:
        parsed_prediction = stdout.strip()
    error = None
    if not ecr_1:
        error = {
            "type": "PoTExecutionError",
            "message": f"Generated code exited with code {completed.returncode}",
            "stderr_tail": stderr[-2000:],
        }
    elif not parsed_prediction:
        error = _parse_error("Generated code did not print Final Answer")
    return {
        "parsed_prediction": parsed_prediction,
        "ecr_1": ecr_1,
        "python_code": python_code,
        "stdout": stdout,
        "error": error,
    }


def parse_python_code(prediction: str) -> str:
    matches = re.findall(r"```python\n(.*?)```", str(prediction or ""), flags=re.S)
    return matches[-1].strip() if matches else ""


def parse_code_output_prediction(prediction: str) -> str:
    match = re.search(r"Final Answer: (.+)", str(prediction or ""))
    return match.group(1).strip() if match else ""


def surround_pycode_with_main(pycode: str) -> str:
    lines = ["if __name__ == '__main__':"]
    for line in pycode.strip().splitlines():
        lines.append(f"    {line}")
    return "\n".join(lines) + "\n"


def _parse_error(message: str) -> dict[str, str]:
    return {"type": "PredictionParseError", "message": message}


def select_tablebench_cases(
    *,
    tablebench_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    qtypes: set[str],
    qsubtypes: set[str],
    include_visualization: bool,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_cases == 0:
        return []
    cases = list_tablebench_cases(tablebench_root)
    if not include_visualization:
        cases = [case for case in cases if case.get("qtype") != "Visualization"]
    if dataset_id is not None:
        cases = [case for case in cases if case["dataset_id"] == dataset_id]
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if qtypes:
        cases = [case for case in cases if str(case.get("qtype") or "") in qtypes]
    if qsubtypes:
        cases = [case for case in cases if str(case.get("qsubtype") or "") in qsubtypes]
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        import random

        rng = random.Random(seed)
        cases = rng.sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def list_tablebench_cases(tablebench_root: Path) -> list[dict[str, Any]]:
    cases = []
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
                    "answer_type": case.get("type"),
                    "qtype": case.get("qtype"),
                    "qsubtype": case.get("qsubtype"),
                }
            )
    return cases


def _load_case_payload(dataset_dir: Path, case_id: str) -> dict[str, Any]:
    for case in read_cases(dataset_dir / "cases.jsonl"):
        if case.get("case_id") == case_id:
            return case
    raise ValueError(f"Case id not found in {dataset_dir}: {case_id}")


def tablebench_table_from_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"columns": [], "data": []}
    columns = [str(cell).strip() for cell in rows[0]]
    data = [[_table_json_cell(cell) for cell in row] for row in rows[1:]]
    return {"columns": columns, "data": data}


def _table_json_cell(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


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
    include_visualization: bool,
    cases_index: Path,
    predictions_path: Path,
    mismatch_cases: Path,
    failure_cases: Path,
    instruction_type: str,
    execution_timeout_seconds: float,
    remote_cost_model: str | None,
) -> dict[str, Any]:
    total = len(records)
    runtime_successes = sum(1 for record in records if record.get("runtime_ok"))
    correct = sum(1 for record in records if record.get("answer_correct"))
    score_sum = sum(float(record.get("tablebench_score") or 0.0) for record in records)
    token_usage = _sum_token_usage(records)
    local_slm_token_usage = _empty_usage()
    remote_elapsed = sum(
        float(record.get("remote_elapsed_seconds", 0.0) or 0.0)
        for record in records
    )
    mismatches = sum(
        1
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    mismatches_by_type = Counter(
        str(record.get("answer_type") or "unknown")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    return {
        "run_name": output_dir.name,
        "stage": "tablebench_remote_only_baseline",
        "workflow": f"remote_only_tablebench_{instruction_type}",
        "prompt_mode": f"tablebench_{instruction_type}",
        "instruction_type": instruction_type,
        "execution_timeout_seconds": execution_timeout_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tablebench_standard": {
            "visualization_included": include_visualization,
            "visualization_default": "skipped",
            "metrics": {
                "FactChecking": "EM",
                "NumericalReasoning": "EM",
                "DataAnalysis.CorrelationAnalysis": "EM_with_error_10",
                "DataAnalysis.TrendForecasting": "EM_with_error_10",
                "DataAnalysis.StatisticalAnalysis": "EM_with_error_10",
                "DataAnalysis.ImpactAnalysis": "EM",
                "DataAnalysis.other": "ROUGE-L",
            },
        },
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
        "remote_llm": _config_summary(remote_config),
        "local_slm": None,
        "total_cases": total,
        "remote_calls": total,
        "local_slm_calls": 0,
        "ecr_1_successes": sum(
            1 for record in records if record.get("ecr_1") is True
        ),
        "ecr_1_failures": sum(
            1 for record in records if record.get("ecr_1") is False
        ),
        "parse_successes": sum(1 for record in records if record.get("parse_ok")),
        "parse_failures": sum(1 for record in records if not record.get("parse_ok")),
        "runtime_successes": runtime_successes,
        "runtime_failures": total - runtime_successes,
        "correct": correct,
        "mismatches": mismatches,
        "failures": total - runtime_successes,
        "accuracy_on_all_cases": safe_divide(correct, total),
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "retry_cases": 0,
        "retry_case_ids": [],
        "total_retry_rounds": 0,
        "sql_repair_cases": 0,
        "sql_repair_case_ids": [],
        "initial_execution_failures": 0,
        "initial_execution_failure_case_ids": [],
        "standard_score_average": safe_divide(score_sum, total),
        "scores_by_metric": _score_groups(records, "tablebench_metric"),
        "scores_by_qtype": _score_groups(records, "qtype"),
        "scores_by_qsubtype": _score_groups(records, "qsubtype"),
        "answer_types": _string_counter(records, "answer_type"),
        "mismatches_by_type": dict(sorted(mismatches_by_type.items())),
        "qtypes": _string_counter(records, "qtype"),
        "qsubtypes": _string_counter(records, "qsubtype"),
        "error_types": dict(
            sorted(
                Counter(
                    record["error"]["type"]
                    for record in records
                    if isinstance(record.get("error"), dict) and record.get("error")
                ).items()
            )
        ),
        "remote_token_usage": token_usage,
        "local_slm_token_usage": local_slm_token_usage,
        "remote_cost_estimate": estimate_openai_text_cost(
            token_usage,
            remote_config=remote_config,
            pricing_model=remote_cost_model,
        ),
        "system_profile": _baseline_system_profile(
            remote_calls=total,
            remote_token_usage=token_usage,
            local_slm_token_usage=local_slm_token_usage,
        ),
        "remote_elapsed_seconds_sum": remote_elapsed,
        "remote_elapsed_seconds_avg": safe_divide(remote_elapsed, total),
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "predictions": display_path(predictions_path),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
    }


def _score_groups(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(record)
    result = {}
    for group_key, group in sorted(groups.items()):
        total = len(group)
        runtime_successes = sum(1 for record in group if record.get("runtime_ok"))
        correct = sum(1 for record in group if record.get("answer_correct"))
        score_sum = sum(float(record.get("tablebench_score") or 0.0) for record in group)
        result[group_key] = {
            "total": total,
            "runtime_successes": runtime_successes,
            "correct": correct,
            "accuracy": safe_divide(correct, total),
            "score": safe_divide(score_sum, total),
        }
    return result


def _string_counter(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(record.get(key) or "unknown") for record in records).items())
    )


def _sum_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        usage = record.get("remote_token_usage")
        if not isinstance(usage, dict):
            continue
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            totals[key] += int(usage.get(key, 0) or 0)
    return {
        "input_tokens": int(totals.get("input_tokens", 0)),
        "cached_input_tokens": int(totals.get("cached_input_tokens", 0)),
        "output_tokens": int(totals.get("output_tokens", 0)),
        "reasoning_tokens": int(totals.get("reasoning_tokens", 0)),
        "total_tokens": int(totals.get("total_tokens", 0)),
    }


def _prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("case_id"),
        "case_id": record.get("case_id"),
        "dataset_id": record.get("dataset_id"),
        "qtype": record.get("qtype"),
        "qsubtype": record.get("qsubtype"),
        "table": record.get("table"),
        "question": record.get("question"),
        "answer": record.get("expected_raw"),
        "instruction_type": record.get("instruction_type"),
        "instruction": record.get("instruction"),
        "model_name": record.get("model_name"),
        "prediction": record.get("remote_output", record.get("final_answer_standard_text")),
        "parsed_result": {
            "parsed_prediction": record.get("final_answer_standard_text"),
            "Parse@1": record.get("parse_ok"),
            "ecr_1": record.get("ecr_1"),
        },
    }


def _case_index_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"table", "instruction", "remote_output"}
    }


def _mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "qtype": record.get("qtype"),
        "qsubtype": record.get("qsubtype"),
        "answer_type": record.get("answer_type"),
        "metric": record.get("tablebench_metric"),
        "score": record.get("tablebench_score"),
        "question": record.get("question"),
        "expected": record.get("expected_standard_text"),
        "actual": record.get("final_answer_standard_text"),
        "remote_output": record.get("files", {}).get("remote_output"),
    }


def _failed_case_record(
    *,
    sampled_case: dict[str, Any],
    sample_index: int,
    case_dir: Path,
    exc: Exception,
    instruction_type: str | None = None,
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
        "qtype": sampled_case.get("qtype"),
        "qsubtype": sampled_case.get("qsubtype"),
        "instruction_type": instruction_type,
        "tablebench_metric": tablebench_metric_name(
            sampled_case.get("qtype"),
            sampled_case.get("qsubtype"),
        ),
        "tablebench_score": 0.0,
        "question": None,
        "expected_raw": None,
        "expected_standard_text": None,
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_standard_text": None,
        "remote_elapsed_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "remote_token_usage": _empty_usage(),
        "error": error,
        "case_dir": display_path(case_dir),
    }


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_KEYS}


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


def _normalize_instruction_type(instruction_type: str) -> str:
    value = str(instruction_type or "DP").strip()
    lowered = value.lower()
    if lowered == "pot":
        value = "PoT"
    elif lowered == "tcot":
        value = "TCoT"
    else:
        value = value.upper()
    if value not in TABLEBENCH_INSTRUCTION_TYPES:
        choices = ", ".join(sorted(TABLEBENCH_INSTRUCTION_TYPES))
        raise ValueError(f"Unsupported TableBench instruction type {value!r}; use {choices}")
    return value


def _config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
    }
