"""Databench Supervisor SQL decomposition benchmark."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clover.optimizer import parse_remote_sql_to_logic_dag, parse_sql_response
from clover.optimizer import optimize_logic_dag_to_physical_plan
from clover.resource import preprocess_task_dsl
from clover.supervisor import create_remote_llm_client, generate_remote_text
from clover.supervisor import render_initial_task_prompt
from benchmarks.databench.adapter import (
    iter_databench_dataset_dirs,
    load_databench_task,
    read_cases,
    write_json,
)
from benchmarks.warnings import suppress_benchmark_warnings


def run_databench_remote_sql_sample(
    databench_root: Path,
    output_root: Path,
    run_name: str,
    remote_llm_config_path: Path,
    sample_size: int,
    seed: int,
    max_workers: int | None = None,
) -> dict[str, Any]:
    with suppress_benchmark_warnings():
        return _run_databench_remote_sql_sample(
            databench_root=databench_root,
            output_root=output_root,
            run_name=run_name,
            remote_llm_config_path=remote_llm_config_path,
            sample_size=sample_size,
            seed=seed,
            max_workers=max_workers,
        )


def _run_databench_remote_sql_sample(
    databench_root: Path,
    output_root: Path,
    run_name: str,
    remote_llm_config_path: Path,
    sample_size: int,
    seed: int,
    max_workers: int | None = None,
) -> dict[str, Any]:
    run_dir = output_root / run_name
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    remote_config = _read_json(remote_llm_config_path)

    all_cases = list_databench_cases(databench_root)
    sampled_cases = sample_cases(all_cases, sample_size=sample_size, seed=seed)
    worker_count = _resolve_worker_count(
        max_workers=max_workers,
        remote_config=remote_config,
        case_count=len(sampled_cases),
    )

    case_records = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for sample_index, sampled_case in enumerate(sampled_cases):
            case_dir = cases_dir / sampled_case["case_id"]
            future = executor.submit(
                run_remote_sql_case,
                databench_root=databench_root,
                sampled_case=sampled_case,
                case_dir=case_dir,
                client=None,
                remote_config=remote_config,
                sample_index=sample_index,
            )
            futures[future] = (sample_index, sampled_case, case_dir)

        for future in as_completed(futures):
            sample_index, sampled_case, case_dir = futures[future]
            try:
                case_records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - keep the sample running.
                case_record = _failed_case_record(
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    case_dir=case_dir,
                    exc=exc,
                )
                case_records.append(case_record)

    case_records.sort(key=lambda item: item["sample_index"])
    index_path = run_dir / "cases_index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_file:
        for case_record in case_records:
            index_file.write(json.dumps(case_record, ensure_ascii=False) + "\n")

    parse_successes = sum(1 for item in case_records if item["parse_ok"])
    summary = {
        "run_name": run_name,
        "stage": "supervisor_decompose_sql_parse",
        "dataset": "databench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(sampled_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "total_available_cases": len(all_cases),
        "parallel_workers": worker_count,
        "remote_llm": {
            "provider": remote_config.get("provider"),
            "api_type": remote_config.get("api_type"),
            "base_url": remote_config.get("base_url"),
            "model": remote_config.get("model"),
        },
        "parse_successes": parse_successes,
        "parse_failures": len(case_records) - parse_successes,
        "run_dir": str(run_dir),
        "cases_index": str(index_path),
        "cases": case_records,
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


def run_remote_sql_case(
    databench_root: Path,
    sampled_case: dict[str, Any],
    case_dir: Path,
    client: Any | None,
    remote_config: dict[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = create_remote_llm_client(remote_config)

    task = load_databench_task(
        databench_root=databench_root,
        dataset_id=sampled_case["dataset_id"],
        case_id=sampled_case["case_id"],
    )
    preprocess_result = preprocess_task_dsl(task.task_dsl, base_dir=task.base_dir)
    prompt = render_initial_task_prompt(preprocess_result["remote_dsl"])

    llm_result = generate_remote_text(
        prompt=prompt,
        remote_config=remote_config,
        client=client,
    )
    response_payload = llm_result.response_payload
    remote_output = llm_result.text

    parse_ok = False
    parsed_sql: dict[str, Any] | None = None
    logic_dag: dict[str, Any] | None = None
    physical_plan: dict[str, Any] | None = None
    parse_error: dict[str, Any] | None = None
    try:
        parsed = parse_sql_response(remote_output, preprocess_result["remote_dsl"])
        parsed_sql = {
            "sql": parsed.sql,
            "source_ids": list(parsed.source_ids),
        }
        logic_dag = parse_remote_sql_to_logic_dag(
            remote_output,
            preprocess_result["remote_dsl"],
        )
        physical_plan = optimize_logic_dag_to_physical_plan(
            logic_dag=logic_dag,
            context=preprocess_result["context"],
            local_dsl=preprocess_result["local_dsl"],
        )
        parse_ok = True
    except Exception as exc:  # noqa: BLE001 - stored for benchmark diagnosis.
        parse_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    write_json(case_dir / "task_dsl.json", task.task_dsl)
    write_json(case_dir / "local_dsl.json", preprocess_result["local_dsl"])
    write_json(case_dir / "remote_dsl.json", preprocess_result["remote_dsl"])
    write_json(case_dir / "context.json", preprocess_result["context"])
    (case_dir / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
    write_json(case_dir / "remote_response.json", response_payload)
    (case_dir / "remote_output.txt").write_text(remote_output + "\n", encoding="utf-8")
    if parsed_sql is not None:
        write_json(case_dir / "parsed_sql.json", parsed_sql)
    if logic_dag is not None:
        write_json(case_dir / "logic_dag.json", logic_dag)
    if physical_plan is not None:
        write_json(case_dir / "physical_plan.json", physical_plan)
    if parse_error is not None:
        write_json(case_dir / "parse_error.json", parse_error)

    case_record = {
        "sample_index": sample_index,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "answer_type": task.task_dsl["answer"]["type"],
        "question": task.task_dsl["question"],
        "response_id": llm_result.response_id,
        "response_status": llm_result.response_status,
        "api_type": llm_result.api_type,
        "parse_ok": parse_ok,
        "case_dir": str(case_dir),
        "files": {
            "prompt": str(case_dir / "prompt.md"),
            "remote_output": str(case_dir / "remote_output.txt"),
            "remote_response": str(case_dir / "remote_response.json"),
            "parsed_sql": str(case_dir / "parsed_sql.json")
            if parsed_sql is not None
            else None,
            "logic_dag": str(case_dir / "logic_dag.json")
            if logic_dag is not None
            else None,
            "physical_plan": str(case_dir / "physical_plan.json")
            if physical_plan is not None
            else None,
            "parse_error": str(case_dir / "parse_error.json")
            if parse_error is not None
            else None,
        },
        "ops": _logic_dag_ops(logic_dag),
        "error": parse_error,
    }
    return case_record


def _logic_dag_ops(logic_dag: dict[str, Any] | None) -> list[str]:
    if not isinstance(logic_dag, dict):
        return []
    nodes = logic_dag.get("nodes")
    if isinstance(nodes, list):
        return [
            node["op"]
            for node in nodes
            if isinstance(node, dict) and "op" in node
        ]
    ops: list[str] = []
    for query_plan in logic_dag.get("query_plans", []):
        if not isinstance(query_plan, dict):
            continue
        ops.extend(_logic_dag_ops(query_plan))
    return ops


def _failed_case_record(
    sampled_case: dict[str, Any],
    sample_index: int,
    case_dir: Path,
    exc: Exception,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    error = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    write_json(case_dir / "case_error.json", error)
    return {
        "sample_index": sample_index,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "answer_type": sampled_case.get("answer_type"),
        "question": None,
        "response_id": None,
        "response_status": None,
        "parse_ok": False,
        "case_dir": str(case_dir),
        "files": {
            "prompt": None,
            "remote_output": None,
            "remote_response": None,
            "parsed_sql": None,
            "logic_dag": None,
            "physical_plan": None,
            "parse_error": str(case_dir / "case_error.json"),
        },
        "ops": [],
        "error": error,
    }


def list_databench_cases(databench_root: Path) -> list[dict[str, Any]]:
    cases = []
    for dataset_dir in iter_databench_dataset_dirs(databench_root):
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
                }
            )
    return cases


def sample_cases(
    cases: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    if sample_size >= len(cases):
        return list(cases)
    return rng.sample(cases, sample_size)


def _resolve_worker_count(
    max_workers: int | None,
    remote_config: dict[str, Any],
    case_count: int,
) -> int:
    configured_workers = max_workers or remote_config.get("parallel_workers")
    if configured_workers is None:
        configured_workers = min(5, case_count)
    worker_count = int(configured_workers)
    if worker_count <= 0:
        raise ValueError("parallel worker count must be positive")
    return min(worker_count, case_count)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
