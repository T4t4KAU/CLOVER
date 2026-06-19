#!/usr/bin/env python3
"""Replay mixed table/document local SLM sequences through one dispatcher."""

from __future__ import annotations

import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    item
    for item in sys.path
    if Path(item or os.getcwd()).resolve() != SCRIPT_DIR
]
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
loaded_warnings = sys.modules.get("warnings")
if loaded_warnings is not None:
    warnings_file = getattr(loaded_warnings, "__file__", "")
    if warnings_file and Path(warnings_file).resolve().parent == SCRIPT_DIR:
        del sys.modules["warnings"]

import argparse
import copy
import hashlib
import json
import random
import re
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from unittest.mock import patch

from benchmarks.energy import EnergyProfiler
from benchmarks.financebench.eval import load_financebench_document_examples
from clover.config import load_model_config, resolve_model_config_env
from clover.executor.agents.template_tree import (
    DOCUMENT_WORKER_LEAF_KEY,
    render_document_worker_prompt,
)
from clover.executor.local_slm import create_slm_client
from clover.executor.sandbox.document_reasoning import _with_question_context
from clover.executor.slm_dispatcher import (
    LocalSlmSequenceDispatcher,
    LocalSlmSequenceRequest,
)
from clover.executor.token_count import count_tokens
from clover.optimizer import (
    optimize_logic_dag_to_physical_plan,
    parse_remote_document_code_to_logic_dag,
)
from clover.resource import prepare_physical_plan_resources
from clover.runtime import DocumentReasoningCaseSpec, build_document_task_items
from clover.supervisor import SupervisorAgent, extract_token_usage


DEFAULT_TABLE_PROMPTS = (
    REPO_ROOT
    / "benchmark"
    / "runs"
    / "local_slm_replay"
    / "tablebench_256case_agent_prompts_current_template.json"
)
DEFAULT_FINANCEBENCH_ROOT = REPO_ROOT / "datasets" / "financebench"
DEFAULT_REMOTE_CONFIG = REPO_ROOT / "model_config" / "doubao_remote_llm_config.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "runs" / "local_slm_replay"
DEFAULT_LOCAL_MODEL = "/home/hwx/Documents/models/Qwen3-1.7B"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    table_sequences = _load_table_sequences(
        args.table_prompts,
        count=args.table_count,
        tokenizer_name=args.local_model,
    )
    document_payload = _load_or_build_document_sequences(args, run_id=run_id)
    document_sequences = document_payload["sequences"]
    workload = table_sequences + document_sequences
    rng = random.Random(args.seed)
    rng.shuffle(workload)

    local_config_base = _local_slm_config(args)
    mode_results = []
    for mode in _parse_modes(args.modes):
        mode_results.append(
            _run_mode(
                workload,
                mode=mode,
                args=args,
                local_config_base=local_config_base,
            )
        )

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "workload": {
            "total_sequences": len(workload),
            "table_sequences": len(table_sequences),
            "document_sequences": len(document_sequences),
            "document_cases": document_payload["summary"].get("cases_built", 0),
            "document_failures": len(document_payload.get("failures", [])),
        },
        "config": {
            "local_model": args.local_model,
            "local_base_url": args.local_base_url,
            "max_parallel_sequences": args.max_parallel_sequences,
            "max_pending_sequences": args.max_pending_sequences,
            "max_tptt_leaf_sequences_per_tree": args.max_tptt_leaf_sequences_per_tree,
            "tptt_coalesce_ms": args.tptt_coalesce_ms,
            "max_tokens": args.max_tokens,
        },
        "document_prompt_cache": document_payload["cache_path"],
        "modes": mode_results,
    }

    output_path = args.output_dir / f"mixed_10doc_50table_real_qwen17b_{run_id}.json"
    _write_json(output_path, summary)
    print(json.dumps(_printable_summary(summary, output_path), ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 10 FinanceBench document worker prompts and 50 TableBench "
            "evidence prompts, then replay them through FIFO/TPTT local SLM scheduling."
        )
    )
    parser.add_argument("--table-prompts", type=Path, default=DEFAULT_TABLE_PROMPTS)
    parser.add_argument("--financebench-root", type=Path, default=DEFAULT_FINANCEBENCH_ROOT)
    parser.add_argument("--remote-config", type=Path, default=DEFAULT_REMOTE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--table-count", type=int, default=50)
    parser.add_argument("--document-count", type=int, default=10)
    parser.add_argument("--document-prompt-cache", type=Path, default=None)
    parser.add_argument("--document-prompts-per-case", type=int, default=1)
    parser.add_argument("--document-candidate-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--modes", default="fifo,tptt")
    parser.add_argument("--local-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--max-parallel-sequences", type=int, default=8)
    parser.add_argument("--max-pending-sequences", type=int, default=256)
    parser.add_argument("--max-tptt-leaf-sequences-per-tree", type=int, default=64)
    parser.add_argument("--tptt-coalesce-ms", type=float, default=40.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--energy-sample-ms", type=int, default=500)
    return parser.parse_args()


def _load_table_sequences(
    path: Path,
    *,
    count: int,
    tokenizer_name: str,
) -> list[dict[str, Any]]:
    payload = _read_json(path)
    prompts = payload.get("prompts", [])
    if len(prompts) < count:
        raise ValueError(f"Need {count} table prompts, found {len(prompts)} in {path}")
    sequences = []
    for index, item in enumerate(prompts[:count]):
        prompt = str(item["prompt"])
        sequences.append(
            {
                "kind": "table",
                "sequence_id": str(item.get("sequence_id") or f"table_{index + 1}"),
                "prompt": prompt,
                "leaf_key": list(item["leaf_key"]),
                "prompt_kind": str(
                    item.get("prompt_kind") or "table_reasoning_agent_loop"
                ),
                "prompt_tokens": int(
                    item.get("prompt_tokens")
                    or count_tokens(prompt, tokenizer_name=tokenizer_name)
                ),
                "prompt_chars": len(prompt),
                "dataset_id": item.get("dataset_id"),
                "case_id": item.get("case_id"),
                "question": item.get("question"),
                "table_path": item.get("table_path"),
            }
        )
    return sequences


def _load_or_build_document_sequences(
    args: argparse.Namespace,
    *,
    run_id: str,
) -> dict[str, Any]:
    cache_path = (
        args.document_prompt_cache
        or args.output_dir / f"mixed_document_prompts_{args.document_count}case_{run_id}.json"
    )
    if cache_path.is_file():
        payload = _read_json(cache_path)
        payload["cache_path"] = str(cache_path)
        return payload

    remote_config = resolve_model_config_env(load_model_config(args.remote_config))
    supervisor = SupervisorAgent(remote_config=remote_config)
    examples = load_financebench_document_examples(
        examples_root=args.financebench_root,
        max_cases=args.document_candidate_cases,
    )
    sequences: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for example in examples:
        if len({item["case_id"] for item in sequences}) >= args.document_count:
            break
        try:
            case_sequences = _build_document_case_sequences(
                example,
                supervisor=supervisor,
                tokenizer_name=args.local_model,
                prompts_per_case=args.document_prompts_per_case,
            )
        except Exception as exc:  # noqa: BLE001 - keep sampling until target count.
            failures.append(
                {
                    "case_id": example.case_id,
                    "example_id": example.example_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not case_sequences:
            failures.append(
                {
                    "case_id": example.case_id,
                    "example_id": example.example_id,
                    "error": "no document worker prompt generated",
                }
            )
            continue
        sequences.extend(case_sequences)

    built_cases = len({item["case_id"] for item in sequences})
    if built_cases < args.document_count:
        raise RuntimeError(
            f"Only built {built_cases}/{args.document_count} document cases; "
            f"failures={len(failures)}"
        )
    payload = {
        "summary": {
            "cases_built": built_cases,
            "sequences": len(sequences),
            "candidate_cases": len(examples),
            "prompts_per_case": args.document_prompts_per_case,
        },
        "sequences": sequences,
        "failures": failures,
        "cache_path": str(cache_path),
    }
    _write_json(cache_path, payload)
    return payload


def _build_document_case_sequences(
    example: Any,
    *,
    supervisor: SupervisorAgent,
    tokenizer_name: str,
    prompts_per_case: int,
) -> list[dict[str, Any]]:
    task_items = build_document_task_items(
        [
            DocumentReasoningCaseSpec(
                case_id=example.case_id,
                task_dsl=example.task_dsl,
                base_dir=example.base_dir,
                metadata=copy.deepcopy(example.metadata),
                answer_key=f"answer_doc_{example.sample_index + 1}",
            )
        ]
    )
    task = next(iter(task_items.values()))
    step = supervisor.decompose(task_dsl=task.remote_dsl)
    logic_dag = parse_remote_document_code_to_logic_dag(step.command or "", task.remote_dsl)
    physical_plan = optimize_logic_dag_to_physical_plan(
        logic_dag=logic_dag,
        context=task.context,
        local_dsl=task.local_dsl,
    )
    physical_plan = prepare_physical_plan_resources(physical_plan)
    prompts = _render_document_plan_prompts(
        physical_plan,
        question=task.question,
        tokenizer_name=tokenizer_name,
        case_id=example.case_id,
        example_id=example.example_id,
        sample_index=example.sample_index,
    )
    if prompts_per_case > 0:
        prompts = prompts[:prompts_per_case]
    for item in prompts:
        item["remote_decompose_usage"] = extract_token_usage(step.response_payload)
        item["remote_response_id"] = step.response_id
    return prompts


def _render_document_plan_prompts(
    physical_plan: dict[str, Any],
    *,
    question: str,
    tokenizer_name: str,
    case_id: str,
    example_id: str,
    sample_index: int,
) -> list[dict[str, Any]]:
    resources = {resource["id"]: resource for resource in physical_plan.get("resources", [])}
    sequences: list[dict[str, Any]] = []
    ordinal = 0
    for group in physical_plan.get("map_groups", []):
        params = group.get("params", {})
        chunk_ids = group.get("input", {}).get("chunks", [])
        if not isinstance(chunk_ids, list):
            continue
        for chunk_id in chunk_ids:
            resource = resources.get(chunk_id)
            if resource is None:
                continue
            chunk_record = _load_chunk_record(resource)
            instruction = _with_question_context(
                str(params.get("local_instruction") or "").strip(),
                question=question,
            )
            advice = str(params.get("advice") or params.get("local_guidance") or "")
            prompt = render_document_worker_prompt(
                chunk_text=str(chunk_record.get("text") or ""),
                local_instruction=instruction,
                advice=advice,
            )
            ordinal += 1
            sequence_id = f"financebench_{case_id}__{group.get('id', 'map')}__{chunk_id}"
            sequences.append(
                {
                    "kind": "document",
                    "sequence_id": sequence_id,
                    "prompt": prompt,
                    "leaf_key": list(DOCUMENT_WORKER_LEAF_KEY),
                    "prompt_kind": "document_worker",
                    "prompt_tokens": count_tokens(prompt, tokenizer_name=tokenizer_name),
                    "prompt_chars": len(prompt),
                    "case_id": case_id,
                    "example_id": example_id,
                    "sample_index": sample_index,
                    "question": question,
                    "group_id": group.get("id"),
                    "chunk_id": chunk_id,
                    "chunk_chars": len(str(chunk_record.get("text") or "")),
                    "ordinal": ordinal,
                }
            )
    return sequences


def _load_chunk_record(resource: dict[str, Any]) -> dict[str, Any]:
    path = Path(resource["path"])
    item_id = str(resource["item_id"])
    for record in _read_jsonl(path):
        if str(record.get("chunk_id")) == item_id:
            return record
    raise ValueError(f"Chunk {item_id} not found in {path}")


def _local_slm_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "provider": "vllm_local",
        "api_type": "chat_completions",
        "api_key": "EMPTY",
        "base_url": args.local_base_url,
        "model": args.local_model,
        "tokenizer": args.local_model,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }


def _run_mode(
    workload: list[dict[str, Any]],
    *,
    mode: str,
    args: argparse.Namespace,
    local_config_base: dict[str, Any],
) -> dict[str, Any]:
    local_config = dict(local_config_base)
    local_config["slm_scheduler"] = mode
    client = create_slm_client(local_config)
    dispatcher = LocalSlmSequenceDispatcher(
        slm_config=local_config,
        client=client,
        max_parallel_sequences=args.max_parallel_sequences,
        max_pending_sequences=args.max_pending_sequences,
        slm_scheduler=mode,
        max_tptt_leaf_sequences_per_tree=args.max_tptt_leaf_sequences_per_tree,
        tptt_coalesce_ms=args.tptt_coalesce_ms,
    )
    start_order: list[dict[str, Any]] = []
    prompt_to_ids: dict[str, deque[str]] = {}
    metadata_by_id = {item["sequence_id"]: _sequence_metadata(item) for item in workload}
    for item in workload:
        prompt_to_ids.setdefault(item["prompt"], deque()).append(item["sequence_id"])
    lock = threading.Lock()
    mode_started = time.perf_counter()
    metrics_before = _scrape_vllm_metrics(args.metrics_url)

    from clover.executor import slm_dispatcher as slm_dispatcher_module

    real_generate = slm_dispatcher_module.generate_slm_text

    def recording_generate(prompt: str, **kwargs: Any) -> Any:
        with lock:
            sequence_id = _pop_sequence_id(prompt_to_ids, prompt)
            record = dict(metadata_by_id.get(sequence_id, {}))
            record["started_offset_ms"] = (time.perf_counter() - mode_started) * 1000.0
            start_order.append(record)
        return real_generate(prompt, **kwargs)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        with EnergyProfiler(
            enabled=args.energy,
            sample_ms=args.energy_sample_ms,
        ) as energy:
            with patch(
                "clover.executor.slm_dispatcher.generate_slm_text",
                side_effect=recording_generate,
            ):
                with ThreadPoolExecutor(max_workers=len(workload)) as pool:
                    future_to_item = {
                        pool.submit(
                            dispatcher.generate,
                            _sequence_request(item, local_config=local_config),
                        ): item
                        for item in workload
                    }
                    for future in as_completed(future_to_item):
                        item = future_to_item[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            errors.append(
                                {
                                    **_sequence_metadata(item),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue
                        results.append(_result_record(result))
        energy_summary = energy.summary
    finally:
        dispatcher.close()

    wall_seconds = time.perf_counter() - mode_started
    metrics_after = _scrape_vllm_metrics(args.metrics_url)
    trace = [result["trace"] for result in results]
    summary = {
        "mode": mode,
        "completed": len(results),
        "errors": len(errors),
        "wall_seconds": wall_seconds,
        "throughput_seq_per_s": len(results) / wall_seconds if wall_seconds > 0 else 0.0,
        "scheduling": _scheduling_metrics(start_order),
        "latency": _latency_metrics(trace),
        "usage": _usage_metrics(results),
        "energy": energy_summary,
        "vllm_metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "delta": _metrics_delta(metrics_before, metrics_after),
        },
        "start_order": start_order,
        "errors_detail": errors,
        "results_preview": results[:10],
    }
    return summary


def _sequence_request(
    item: dict[str, Any],
    *,
    local_config: dict[str, Any],
) -> LocalSlmSequenceRequest:
    return LocalSlmSequenceRequest(
        prompt=item["prompt"],
        leaf_key=tuple(item["leaf_key"]),
        prompt_kind=str(item["prompt_kind"]),
        sequence_id=str(item["sequence_id"]),
        job_id=f"{item['kind']}_slm",
        iteration=1,
        prompt_len=int(item.get("prompt_tokens") or 0),
        slm_config=local_config,
        metadata={
            "kind": item["kind"],
            "case_id": item.get("case_id"),
            "prompt_hash": _hash_prompt(item["prompt"]),
        },
    )


def _sequence_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_id": item["sequence_id"],
        "kind": item["kind"],
        "prompt_kind": item["prompt_kind"],
        "leaf_key": item["leaf_key"],
        "prompt_tokens": item.get("prompt_tokens"),
        "prompt_chars": item.get("prompt_chars"),
        "case_id": item.get("case_id"),
        "dataset_id": item.get("dataset_id"),
        "prompt_hash": _hash_prompt(item["prompt"]),
    }


def _result_record(result: Any) -> dict[str, Any]:
    usage = extract_token_usage(result.response_payload)
    return {
        "sequence_id": result.sequence_id,
        "kind": result.request.metadata.get("kind"),
        "trace": result.trace_metadata(),
        "usage": usage,
        "output_chars": len(result.text),
        "output_preview": result.text[:300],
    }


def _scheduling_metrics(start_order: list[dict[str, Any]]) -> dict[str, Any]:
    if not start_order:
        return {}
    kinds = [str(item.get("kind")) for item in start_order]
    leaves = [tuple(item.get("leaf_key") or []) for item in start_order]
    prompt_tokens = [int(item.get("prompt_tokens") or 0) for item in start_order]
    return {
        "started": len(start_order),
        "type_switches": _switch_count(kinds),
        "kind_runs": _run_count(kinds),
        "longest_kind_run": _longest_run(kinds),
        "same_leaf_adjacent_ratio": _same_adjacent_ratio(leaves),
        "prompt_tokens_in_start_order": _series(prompt_tokens),
        "first_20_kinds": kinds[:20],
        "first_20_prompt_tokens": prompt_tokens[:20],
    }


def _latency_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    queue_wait = [float(trace.get("queue_wait_ms") or 0.0) for trace in traces]
    inference = [float(trace.get("inference_ms") or 0.0) for trace in traces]
    prompt_len = [int(trace.get("prompt_len") or 0) for trace in traces]
    return {
        "queue_wait_ms": _series(queue_wait),
        "inference_ms": _series(inference),
        "prompt_len": _series(prompt_len),
    }


def _usage_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for result in results:
        for key, value in result.get("usage", {}).items():
            totals[key] = totals.get(key, 0) + int(value or 0)
    return totals


def _scrape_vllm_metrics(metrics_url: str) -> dict[str, float]:
    if not metrics_url:
        return {}
    try:
        import httpx

        response = httpx.get(metrics_url, timeout=5.0)
        response.raise_for_status()
    except Exception:
        return {}
    selected: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)", line)
        if not match:
            continue
        name = match.group(1)
        normalized = name.replace(":", "_")
        if not _is_selected_vllm_metric(normalized):
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        selected[normalized] = selected.get(normalized, 0.0) + value
    return selected


def _is_selected_vllm_metric(name: str) -> bool:
    needles = (
        "prefix_cache",
        "kv_cache",
        "prompt_tokens",
        "request_prompt_tokens",
        "time_to_first_token",
        "time_per_output_token",
        "e2e_request_latency",
    )
    return name.startswith("vllm_") and any(needle in name for needle in needles)


def _metrics_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(keys)
        if after.get(key, 0.0) - before.get(key, 0.0)
    }


def _series(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "avg": statistics.fmean(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _switch_count(values: list[str]) -> int:
    return sum(1 for prev, cur in zip(values, values[1:]) if prev != cur)


def _run_count(values: list[str]) -> int:
    if not values:
        return 0
    return 1 + _switch_count(values)


def _longest_run(values: list[str]) -> int:
    longest = current = 0
    previous = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def _same_adjacent_ratio(values: list[tuple[Any, ...]]) -> float:
    if len(values) <= 1:
        return 1.0
    same = sum(1 for prev, cur in zip(values, values[1:]) if prev == cur)
    return same / (len(values) - 1)


def _pop_sequence_id(prompt_to_ids: dict[str, deque[str]], prompt: str) -> str:
    queue = prompt_to_ids.get(prompt)
    if queue:
        return queue.popleft()
    return f"unmapped_{_hash_prompt(prompt)}"


def _parse_modes(raw: str) -> list[str]:
    modes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not modes:
        raise ValueError("--modes must include at least one scheduler")
    for mode in modes:
        if mode not in {"fifo", "tptt"}:
            raise ValueError(f"Unsupported scheduler mode: {mode}")
    return modes


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> Any:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _printable_summary(summary: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "output_path": str(output_path),
        "workload": summary["workload"],
        "modes": [
            {
                "mode": item["mode"],
                "completed": item["completed"],
                "errors": item["errors"],
                "wall_seconds": round(item["wall_seconds"], 3),
                "throughput_seq_per_s": round(item["throughput_seq_per_s"], 3),
                "scheduling": item["scheduling"],
                "latency": item["latency"],
                "energy_status": (item.get("energy") or {}).get("status"),
                "energy_total_joules": (
                    ((item.get("energy") or {}).get("gross") or {}).get("total_joules")
                ),
                "vllm_metrics_delta": item.get("vllm_metrics", {}).get("delta", {}),
            }
            for item in summary["modes"]
        ],
    }


if __name__ == "__main__":
    main()
