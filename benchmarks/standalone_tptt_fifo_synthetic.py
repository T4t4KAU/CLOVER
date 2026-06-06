#!/usr/bin/env python3
"""Standalone FIFO/TPTT synthetic prefix-cache benchmark.

This script deliberately avoids CLOVER runtime/executor scheduling. It builds
synthetic local-SLM sequences, orders them with a tiny FIFO or TPTT-like policy,
sends them to an OpenAI-compatible vLLM server, and reads real prefix-cache
metrics from /metrics.
"""

from __future__ import annotations

import os
import sys
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
import hashlib
import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

from benchmarks.energy import EnergyProfiler


DEFAULT_MODEL = "/home/hwx/Documents/models/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "runs" / "standalone_tptt_fifo"


@dataclass(frozen=True)
class Sequence:
    sequence_id: str
    group_id: str
    prompt: str
    prompt_tokens: int
    group_prefix_tokens: int
    prompt_hash: str


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    sequences = build_sequences(args, tokenizer)
    ordered = order_sequences(sequences, mode=args.mode)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    metrics_before = scrape_metrics(args.metrics_url)
    with EnergyProfiler(
        enabled=args.energy,
        sample_ms=args.energy_sample_ms,
    ) as energy:
        started = time.perf_counter()
        results = run_requests(ordered, args=args, client=client)
        wall_seconds = time.perf_counter() - started
    energy_summary = energy.summary
    metrics_after = scrape_metrics(args.metrics_url)
    delta = metrics_delta(metrics_before, metrics_after)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": args.model,
        "base_url": args.base_url,
        "metrics_url": args.metrics_url,
        "config": {
            "groups": args.groups,
            "repeats": args.repeats,
            "sequences": len(sequences),
            "group_prefix_target_tokens": args.group_prefix_tokens,
            "unique_suffix_target_tokens": args.unique_suffix_tokens,
            "max_tokens": args.max_tokens,
            "max_workers": args.max_workers,
            "seed": args.seed,
        },
        "wall_seconds": wall_seconds,
        "throughput_seq_per_s": len(results) / wall_seconds if wall_seconds else 0.0,
        "workload": workload_summary(sequences, ordered),
        "latency": latency_summary(results),
        "usage": usage_summary(results),
        "energy": energy_summary,
        "vllm_metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "delta": delta,
            "prefix_cache_hit_ratio": _safe_div(
                delta.get("vllm_prefix_cache_hits_total", 0.0),
                delta.get("vllm_prefix_cache_queries_total", 0.0),
            ),
            "prompt_tokens_cached_ratio": _safe_div(
                delta.get("vllm_prompt_tokens_cached_total", 0.0),
                delta.get("vllm_prompt_tokens_total", 0.0),
            ),
        },
        "results_preview": results[:10],
    }
    output_path = (
        args.output_dir
        / f"synthetic_{args.mode}_g{args.groups}_r{args.repeats}"
        / f"{args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    write_json(output_path, summary)
    print(
        json.dumps(
            printable_summary(summary, output_path),
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone synthetic FIFO/TPTT prefix-cache benchmark."
    )
    parser.add_argument("--mode", choices=("fifo", "tptt"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--group-prefix-tokens", type=int, default=768)
    parser.add_argument("--unique-suffix-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--energy-sample-ms", type=int, default=500)
    return parser.parse_args()


def build_sequences(args: argparse.Namespace, tokenizer: Any) -> list[Sequence]:
    rng = random.Random(args.seed)
    global_prefix = (
        "You are a synthetic local SLM worker. Return a terse JSON object.\n"
        "The following long context is intentionally designed for prefix-cache "
        "testing.\n"
    )
    sequences: list[Sequence] = []
    group_prefixes: dict[str, tuple[str, int]] = {}
    for group_index in range(args.groups):
        group_id = f"G{group_index:03d}"
        group_prefix = make_text_with_min_tokens(
            tokenizer,
            seed_text=(
                f"\n# STATIC PREFIX FOR {group_id}\n"
                f"cache_group={group_id}; stable_section=begin; "
            ),
            filler=f"stable-{group_id}-ledger-column-value ",
            target_tokens=args.group_prefix_tokens,
        )
        group_prefixes[group_id] = (
            group_prefix,
            token_count(tokenizer, global_prefix + group_prefix),
        )

    # FIFO input is round-robin by construction. TPTT will reorder it.
    for repeat_index in range(args.repeats):
        group_order = list(range(args.groups))
        if args.mode == "fifo":
            # Keep deterministic but avoid accidentally sorted by text length.
            rng.shuffle(group_order)
        for group_index in group_order:
            group_id = f"G{group_index:03d}"
            group_prefix, group_prefix_tokens = group_prefixes[group_id]
            suffix = make_text_with_min_tokens(
                tokenizer,
                seed_text=(
                    f"\n# DYNAMIC LEAF PAYLOAD\n"
                    f"sequence={group_id}-R{repeat_index:03d}; "
                    f"nonce={rng.randrange(10**12):012d}; "
                ),
                filler=f"payload-{group_id}-{repeat_index}-unique-field ",
                target_tokens=args.unique_suffix_tokens,
            )
            prompt = (
                global_prefix
                + group_prefix
                + suffix
                + "\nReturn JSON only: {\"ok\": true, \"group\": \""
                + group_id
                + "\"}."
            )
            sequence_id = f"{group_id}_R{repeat_index:03d}"
            sequences.append(
                Sequence(
                    sequence_id=sequence_id,
                    group_id=group_id,
                    prompt=prompt,
                    prompt_tokens=token_count(tokenizer, prompt),
                    group_prefix_tokens=group_prefix_tokens,
                    prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                )
            )
    return sequences


def make_text_with_min_tokens(
    tokenizer: Any,
    *,
    seed_text: str,
    filler: str,
    target_tokens: int,
) -> str:
    text = seed_text
    while token_count(tokenizer, text) < target_tokens:
        text += filler
    return text


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def order_sequences(sequences: list[Sequence], *, mode: str) -> list[Sequence]:
    if mode == "fifo":
        return list(sequences)
    if mode == "tptt":
        return sorted(
            sequences,
            key=lambda item: (item.group_id, item.prompt_tokens, item.sequence_id),
        )
    raise ValueError(f"Unsupported mode: {mode}")


def run_requests(
    ordered: list[Sequence],
    *,
    args: argparse.Namespace,
    client: OpenAI,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    if args.max_workers <= 1:
        return [
            run_one(sequence, args=args, client=client, started=started)
            for sequence in ordered
        ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(run_one, sequence, args=args, client=client, started=started)
            for sequence in ordered
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["submit_index"])
    return results


def run_one(
    sequence: Sequence,
    *,
    args: argparse.Namespace,
    client: OpenAI,
    started: float,
) -> dict[str, Any]:
    submit_index = int(sequence.sequence_id.split("_R")[-1]) * args.groups + int(
        sequence.group_id[1:]
    )
    request_started = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": sequence.prompt}],
        temperature=0,
        max_tokens=args.max_tokens,
    )
    elapsed_ms = (time.perf_counter() - request_started) * 1000.0
    usage = response.usage.model_dump(mode="json") if response.usage else {}
    return {
        "submit_index": submit_index,
        "sequence_id": sequence.sequence_id,
        "group_id": sequence.group_id,
        "prompt_tokens": sequence.prompt_tokens,
        "group_prefix_tokens": sequence.group_prefix_tokens,
        "prompt_hash": sequence.prompt_hash,
        "started_offset_ms": (request_started - started) * 1000.0,
        "elapsed_ms": elapsed_ms,
        "usage": usage,
        "cached_tokens": cached_tokens_from_usage(usage),
    }


def cached_tokens_from_usage(usage: dict[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        value = details.get("cached_tokens")
        if isinstance(value, int):
            return value
    return 0


def scrape_metrics(metrics_url: str) -> dict[str, float]:
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
        match = re.match(
            r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)",
            line,
        )
        if not match:
            continue
        name = match.group(1).replace(":", "_")
        if not (
            name.startswith("vllm_")
            and any(
                needle in name
                for needle in (
                    "prefix_cache",
                    "prompt_tokens",
                    "time_to_first_token",
                    "e2e_request_latency",
                )
            )
        ):
            continue
        selected[name] = selected.get(name, 0.0) + float(match.group(2))
    return selected


def metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    delta = {}
    for key in sorted(set(before) | set(after)):
        value = after.get(key, 0.0) - before.get(key, 0.0)
        if value:
            delta[key] = value
    return delta


def workload_summary(sequences: list[Sequence], ordered: list[Sequence]) -> dict[str, Any]:
    prompt_tokens = [item.prompt_tokens for item in sequences]
    groups = [item.group_id for item in ordered]
    return {
        "sequence_count": len(sequences),
        "group_count": len(set(item.group_id for item in sequences)),
        "prompt_tokens": series(prompt_tokens),
        "first_32_groups": groups[:32],
        "group_switches": sum(
            1 for prev, cur in zip(groups, groups[1:]) if prev != cur
        ),
        "longest_group_run": longest_run(groups),
    }


def latency_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"elapsed_ms": series([item["elapsed_ms"] for item in results])}


def usage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }
    per_request_cached = []
    for item in results:
        usage = item.get("usage") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        totals["cached_tokens"] += int(item.get("cached_tokens") or 0)
        per_request_cached.append(int(item.get("cached_tokens") or 0))
    totals["cached_ratio"] = _safe_div(totals["cached_tokens"], totals["prompt_tokens"])
    return {
        **totals,
        "per_request_cached_tokens": series(per_request_cached),
    }


def series(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "avg": statistics.fmean(ordered),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "max": ordered[-1],
    }


def percentile(ordered: list[float], pct: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def longest_run(values: list[str]) -> int:
    longest = current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def printable_summary(summary: dict[str, Any], output_path: Path) -> dict[str, Any]:
    delta = summary["vllm_metrics"]["delta"]
    hits = delta.get("vllm_prefix_cache_hits_total", 0.0)
    queries = delta.get("vllm_prefix_cache_queries_total", 0.0)
    return {
        "output_path": str(output_path),
        "mode": summary["mode"],
        "wall_seconds": round(summary["wall_seconds"], 3),
        "throughput_seq_per_s": round(summary["throughput_seq_per_s"], 3),
        "workload": summary["workload"],
        "latency": summary["latency"],
        "usage": summary["usage"],
        "prefix_cache_hits": hits,
        "prefix_cache_queries": queries,
        "prefix_cache_hit_ratio": _safe_div(hits, queries),
        "energy": printable_energy(summary.get("energy")),
    }


def printable_energy(energy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(energy, dict):
        return {}
    gross = energy.get("gross") if isinstance(energy.get("gross"), dict) else {}
    return {
        "status": energy.get("status"),
        "backend": energy.get("backend"),
        "total_joules": gross.get("total_joules"),
        "cpu_joules": gross.get("cpu_joules"),
        "gpu_joules": gross.get("gpu_joules"),
        "total_avg_watts": gross.get("total_avg_watts"),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
