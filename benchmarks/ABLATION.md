# CLOVER Ablation Suite

The three dataset experiments are independent. Each can run on either a fixed-size subset (default 100 cases) or the full dataset:

- `bash benchmarks/run_ablation_suite.sh wikitq`: WikiTQ.
- `bash benchmarks/run_ablation_suite.sh tablebench`: TableBench, restricted to `FactChecking` and `NumericalReasoning`.
- `bash benchmarks/run_ablation_suite.sh tablefact`: TableFact, restricted to `FactChecking` (the `simple` and `complex` subtypes).

Sampling relies only on dataset metadata, question text, answer type, table shape, and cell representation. It never reads model predictions, execution traces, or correctness labels. Default manifests live in `benchmarks/ablation_cases/`, and all eleven variants reuse the same case set.

The suite supports three selection policies:

- `representative`: original task-type stratified subset, used to report overall ACC/cost trends.
- `edge_opportunity`: Edge-mechanism-enriched subset, used to test whether local semantic processing improves ACC, reduces Cloud calls, and how much node repair and terminal review each contribute.
- `full_eval`: the full eligible dataset (TableBench 493 / WikiTQ 4344 / TableFact 1998), used for the final reported ablation numbers.

For the final paper, use `full_eval` so the ablation ACC matches the main results table. The 100-case subsets remain useful for quick iteration and the `mechanism-focused, outcome-blind` analysis.

For WikiTQ, 85 cases come from field selection, value normalization, short-list assembly, and small candidate sets, while 15 are deterministic control questions. For TableBench, 70 cases come from local semantic opportunities and 30 are deterministic control questions. For TableFact, 70 cases come from field selection, value normalization, and small candidate sets, while 30 are deterministic control questions. Each manifest's `.summary.json` records the sampling basis explicitly and marks `uses_model_predictions=false` and `uses_answer_correctness=false`.

In the paper, refer to the 100-case subset explicitly as the "mechanism-focused, outcome-blind subset". Do not treat ACC on this subset as an unbiased population ACC over the entire benchmark. If space permits, also report the `representative` subset as a robustness control.

## Running

```bash
conda activate clover

# 100-case subset (default, for quick iteration):
bash benchmarks/run_ablation_suite.sh wikitq /path/to/edge-model
bash benchmarks/run_ablation_suite.sh tablebench /path/to/edge-model
bash benchmarks/run_ablation_suite.sh tablefact /path/to/edge-model

# Full dataset (for final reported numbers):
CLOVER_ABLATION_FULL_EVAL=true bash benchmarks/run_ablation_suite.sh wikitq /path/to/edge-model
CLOVER_ABLATION_FULL_EVAL=true bash benchmarks/run_ablation_suite.sh tablebench /path/to/edge-model
CLOVER_ABLATION_FULL_EVAL=true bash benchmarks/run_ablation_suite.sh tablefact /path/to/edge-model
```

The suite runs the following eleven variants:

1. `full`: Full CLOVER.
2. `all_edge`: Disable Static Fast Path and route every statically executable node to the Edge Agent.
3. `no_edge`: Disable Edge Agent, node repair, node review, and terminal Edge Review; keep all other static and Cloud capabilities aligned with Full.
4. `static`: Disable only node-level Edge Repair; keep terminal Edge Review.
5. `no_contract`: Disable Edge Agent output contract verification.
6. `end_review`: Disable node-level Edge Repair and Node Review; keep only terminal Edge Review.
7. `one_shot`: Keep Cloud final synthesis but forbid Cloud from emitting follow-up SQL/DAG actions.
8. `cloud_finalize`: Disable static and terminal Edge finalization; route everything to Cloud synthesis.
9. `static_only`: Disable the entire Edge family and Cloud Replan/Synthesis, leaving one Cloud planning pass plus Static Fast Path/Finalization. Measures the upper bound of Static execution alone; cases that Static cannot finalize are expected to fail.
10. `no_static`: Disable Static Fast Path and Static Finalization, keeping the full Edge family (Agent/Repair/Review) plus Cloud Synthesis as the terminal fallback. Tests whether Edge can independently finalize when Static is unavailable.
11. `no_closure_checker`: Keep all Full CLOVER routing/finalization components enabled but disable the Observable Closure Checker, measuring whether looser static finalization helps or hurts.

In `all_edge`, Edge output flows directly into the downstream DAG; static execution results serve only as a shadow reference for agreement-rate computation and never replace Edge output on disagreement. The summary report additionally reports Accuracy, Cloud calls, Edge calls, Edge tokens, total runtime, runtime failures, and the agreement rate between Edge output and the static reference.

`no_edge` is the unconfounded control for verifying whether Edge truly substitutes for Cloud. It keeps static finalization, Cloud synthesis, and Cloud replan, so the Cloud-call delta against Full can be attributed to whether the Edge path is enabled.

`static_only` and `no_edge` are complementary: `no_edge` keeps Cloud Replan/Synthesis as fallback, while `static_only` further disables them, leaving only Static. Comparing the two separates the contributions of Static and Cloud fallback. `no_static` and `all_edge` are complementary: `all_edge` only disables Static Fast Path while keeping Static Finalization, whereas `no_static` further disables Static Finalization to test whether Edge can independently complete terminal finalization.

Full CLOVER enables proactive local semantic review by default:

```text
CLOVER_EDGE_REVIEW_PROACTIVE=true
```

Static execution triggers only when evidence is closed and size-bounded, including single-row multi-field selection, candidate selection over at most five rows, simple boolean combinations, short-list assembly, and value normalization for percentages, units, quotes, labels, or trailing parentheses. Edge output must reference fact IDs and replay through deterministic operations; on review failure, execution falls back to the original static or Cloud path. List questions involving superlatives, ranking, counting, or cross-row comparison do not enter proactive Edge assembly and remain on the deterministic or Cloud path.

To reduce ordering bias, the eleven variants are reproducibly shuffled using the seed by default; the actual order is recorded in `variant_order.txt`. A fixed order can be specified via:

```bash
CLOVER_ABLATION_VARIANT_ORDER=full,all_edge,no_edge,static,no_contract,end_review,one_shot,cloud_finalize,static_only,no_static,no_closure_checker
```

Each experiment starts the vLLM server only once. Before each variant, a local-model warm-up is performed and excluded from the measured time. Results are written to:

```text
benchmark/runs/<dataset>_ablation_<timestamp>/
```

The `sanity_check.json` in that directory validates the fixed case set, fine-grained feature flags, the replan count for `w/o Cloud Replan`, and the terminal path for Cloud Finalization. After the eleven variants finish, the script prints the summary table to the terminal and generates:

```text
ablation_summary.md
ablation_summary.csv
ablation_summary.json
```

The table includes:

- Accuracy, percentage-point delta vs Full CLOVER, and the exact McNemar paired test.
- Regression/recovery case counts vs Full.
- Node Edge runs/successes/steps, node reviews, and contract rejections.
- Terminal Edge calls/hits/escalations, proactive semantic opportunities/hits, and Cloud replan counts.
- Final answer sources, Cloud/local model tokens, estimated cost, and runtime.
- The direct Edge-to-Cloud substitution effect between `Full CLOVER` and `w/o Edge Agent`, including Cloud calls, synthesis, replan, tokens, cost, and per-query call delta.
- The terminal/proactive semantic review contribution of `End-only Review` vs `w/o Edge Agent`, and the node-repair contribution of `Full CLOVER` vs `End-only Review`.
- Per-variant regression/recovery case IDs vs Full, retry-case accuracy, and runtime error types.

Additionally, the suite generates:

```text
ablation_case_diagnostics.jsonl
ablation_discordant_cases.csv
```

The former records each case's correctness, answer source, retry, and error type across the eleven variants; the latter keeps only paired cases where Full and a variant disagree, making it easy to inspect why Contract Verification, Cloud Replan, or Observable Closure Checking produced reverse gains.

If the eleven runs are already complete and only the summary needs to be regenerated without rerunning:

```bash
python -m benchmarks.summarize_ablation_suite \
  --suite-root benchmark/runs/wikitq_ablation_<timestamp> \
  --dataset wikitq
```

## Regenerating the Fixed Subset

```bash
CLOVER_ABLATION_REGENERATE_MANIFEST=1 \
bash benchmarks/run_ablation_suite.sh wikitq /path/to/edge-model
```

Default parameters:

```text
CLOVER_ABLATION_SIZE=100
CLOVER_ABLATION_SEED=20260619
CLOVER_ABLATION_SELECTION_POLICY=edge_opportunity
CLOVER_ABLATION_FULL_EVAL=false
```

Run the representative control subset:

```bash
CLOVER_ABLATION_SELECTION_POLICY=representative \
bash benchmarks/run_ablation_suite.sh wikitq /path/to/edge-model
```

The policies use different manifest filenames and never overwrite each other:

```text
wikitq_representative_100_seed20260619.jsonl
wikitq_edge_opportunity_100_seed20260619.jsonl
wikitq_full_eval.jsonl
tablebench_representative_100_seed20260619.jsonl
tablebench_edge_opportunity_100_seed20260619.jsonl
tablebench_full_eval.jsonl
tablefact_representative_100_seed20260619.jsonl
tablefact_edge_opportunity_100_seed20260619.jsonl
tablefact_full_eval.jsonl
mmqa_representative_100_seed20260619.jsonl
mmqa_edge_opportunity_100_seed20260619.jsonl
mmqa_full_eval.jsonl
```

TableBench download/conversion is also fixed to keep only the two reasoning types. To regenerate from previously converted data:

```bash
CLOVER_DATASET_OVERWRITE=1 \
bash benchmarks/download_datasets.sh --dataset tablebench --overwrite
```

TableFact download/conversion defaults to the `test` split only (the `simple` and `complex` subtypes). To regenerate:

```bash
CLOVER_DATASET_OVERWRITE=1 \
bash benchmarks/download_datasets.sh --dataset tablefact --overwrite
```

MMQA download/conversion fetches the public Google Drive multi-table release
(`Synthesized_two_table.json` and `Synthesized_three_table.json`) and writes the
nested `split/table-set/cases.jsonl` layout used by the evaluator. If the
network requires a proxy, set the standard `https_proxy`/`HTTPS_PROXY`
environment variables before running:

```bash
https_proxy=127.0.0.1:7890 HTTPS_PROXY=127.0.0.1:7890 \
CLOVER_DATASET_OVERWRITE=1 \
bash benchmarks/download_datasets.sh --dataset mmqa --overwrite
```

MMQA ablations use the same runner:

```bash
MMQA_SPLIT=two_table \
bash benchmarks/run_ablation_suite.sh mmqa /path/to/edge-model
```

## Cost-Accuracy Pareto (P0-2)

Aggregate cost-accuracy pairs across CLOVER, baselines, and ablation variants for Pareto analysis:

```bash
python -m benchmarks.summarize_cost_accuracy \
  --dataset wikitq \
  --ablation-suite benchmark/runs/wikitq_ablation_<timestamp> \
  --clover-run benchmark/runs/wikitq_clover_<timestamp> \
  --pure-cot-run benchmark/runs/wikitq_pure_cot_<timestamp> \
  --external-baselines benchmarks/external_baselines.json \
  --output-dir benchmark/runs/pareto_wikitq
```

Output: `cost_accuracy_pareto.{csv,md,json}` with per-query and per-1K-query cost, accuracy, Cloud/Edge calls, and total tokens.

External baselines JSON format:

```json
{
  "baselines": [
    {
      "method": "ReAcTable",
      "method_key": "reactable",
      "dataset": "wikitq",
      "total_cases": 4344,
      "correct_cases": 3500,
      "estimated_cost_usd": 12.5,
      "remote_calls": 8688,
      "remote_tokens": 25000000
    }
  ]
}
```

## Edge Model Scale Sweep (P0-3)

Measure how Edge model size affects accuracy, cost, and Edge-only finalization. Runs `full` and `no_static` variants across multiple Edge models on the full datasets:

```bash
# Edit USER_EDGE_MODELS at the top of the script, then:
bash benchmarks/run_edge_model_sweep.sh

# Or override via environment:
CLOVER_EDGE_MODELS=/models/Qwen2.5-3B-Instruct:/models/Qwen2.5-7B-Instruct:/models/Qwen3-4B-Instruct \
bash benchmarks/run_edge_model_sweep.sh
```

Output: `edge_model_sweep.{csv,md,json}` with accuracy, cost, Cloud/Edge calls, model calls (latency proxy), and tokens per query for each (Edge model, dataset, variant) combination.

The `full` variant shows how Edge scale affects overall CLOVER performance. The `no_static` variant shows whether larger Edge models can independently finalize when Static is unavailable.
