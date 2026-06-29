# CLOVER Ablation Suite

Current ablation reporting is restricted to TableBench. MMQA should be used for
ordinary ACC/regression evaluation only, not for ablation conclusions, unless a
future experiment explicitly re-opens that scope.

Terminology: `Global` denotes the planner/synthesizer/replan model role. It may be backed by a cloud API or by a local vLLM endpoint. Legacy config and counter names such as `remote_llm`, `remote_calls`, or `cloud_replan_calls` are retained for backward compatibility, but paper-facing reports should describe this role as Global rather than Cloud.

The TableBench experiment can run on either a fixed-size subset (default 100 cases) or the full dataset:

- `bash benchmarks/run_ablation_suite.sh tablebench`: TableBench, restricted to `FactChecking` and `NumericalReasoning`.

Sampling relies only on dataset metadata, question text, answer type, table shape, and cell representation. It never reads model predictions, execution traces, or correctness labels. Default manifests live in `benchmarks/ablation_cases/`, and all variants reuse the same case set.

The suite supports three selection policies:

- `representative`: original task-type stratified subset, used to report overall ACC/cost trends.
- `edge_opportunity`: Edge-mechanism-enriched subset, used to test whether local semantic processing improves ACC, reduces Global calls, and how much node repair and terminal review each contribute.
- `full_eval`: the full eligible TableBench dataset (493 cases), used for the final reported ablation numbers.

For the final paper, use `full_eval` so the ablation ACC matches the main results table. The 100-case subsets remain useful for quick iteration and the `mechanism-focused, outcome-blind` analysis.

For TableBench, 70 cases come from local semantic opportunities and 30 are deterministic control questions. Each manifest's `.summary.json` records the sampling basis explicitly and marks `uses_model_predictions=false` and `uses_answer_correctness=false`.

In the paper, refer to the 100-case subset explicitly as the "mechanism-focused, outcome-blind subset". Do not treat ACC on this subset as an unbiased population ACC over the entire benchmark. If space permits, also report the `representative` subset as a robustness control.

## Running

```bash
conda activate clover

# 100-case subset (default, for quick iteration):
bash benchmarks/run_ablation_suite.sh tablebench /path/to/edge-model

# Full dataset (for final reported numbers):
CLOVER_ABLATION_FULL_EVAL=true bash benchmarks/run_ablation_suite.sh tablebench /path/to/edge-model
```

The default paper ablation runs the compact mechanism set:

1. `full`: Full CLOVER.
2. `no_static`: Disable Static Fast Path and Static Finalization, keeping the full Edge family plus Global Synthesis as the terminal fallback.
3. `static_only`: Disable the entire Edge family and Global Replan/Synthesis, leaving one Global planning pass plus Static Fast Path/Finalization.
4. `no_retry`: Disable retry paths: node-level Edge Repair, Global Replan, and supervisor retry rounds (`max_retries=0`).

Exploratory variants remain available for appendix/debug runs:

1. `full`: Full CLOVER.
2. `all_edge`: Disable Static Fast Path and route every statically executable node to the Edge Agent.
3. `no_edge`: Disable Edge Agent, node repair, node review, and terminal Edge Review; keep all other static and Global capabilities aligned with Full.
4. `static`: Disable only node-level Edge Repair; keep terminal Edge Review.
5. `no_contract`: Disable Edge Agent output contract verification.
6. `end_review`: Disable node-level Edge Repair and Node Review; keep only terminal Edge Review.
7. `one_shot`: Keep Global final synthesis but forbid Global model from emitting follow-up SQL/DAG actions. This is a narrower legacy probe; use `no_retry` for the paper retry-mechanism ablation.
8. `no_retry`: Disable node-level Edge Repair, Global Replan, and supervisor retry rounds.
9. `cloud_finalize`: Disable static and terminal Edge finalization; route everything to Global synthesis.
10. `static_only`: Disable the entire Edge family and Global Replan/Synthesis, leaving one Global planning pass plus Static Fast Path/Finalization. Measures the upper bound of Static execution alone; cases that Static cannot finalize are expected to fail.
11. `no_static`: Disable Static Fast Path and Static Finalization, keeping the full Edge family (Agent/Repair/Review) plus Global Synthesis as the terminal fallback. Tests whether Edge can independently finalize when Static is unavailable.
12. `no_closure_checker`: Keep all Full CLOVER routing/finalization components enabled but disable the Observable Closure Checker, measuring whether looser static finalization helps or hurts.

In `all_edge`, Edge output flows directly into the downstream DAG; static execution results serve only as a shadow reference for agreement-rate computation and never replace Edge output on disagreement. The summary report additionally reports Accuracy, Global calls, Edge calls, Edge tokens, total runtime, runtime failures, and the agreement rate between Edge output and the static reference.

`no_edge` is the unconfounded control for verifying whether Edge truly substitutes for the Global model. It keeps static finalization, Global synthesis, and Global replan, so the Global-call delta against Full can be attributed to whether the Edge path is enabled.

`static_only` and `no_edge` are complementary: `no_edge` keeps Global Replan/Synthesis as fallback, while `static_only` further disables them, leaving only Static. Comparing the two separates the contributions of Static and Global fallback. `no_static` and `all_edge` are complementary: `all_edge` only disables Static Fast Path while keeping Static Finalization, whereas `no_static` further disables Static Finalization to test whether Edge can independently complete terminal finalization.

Full CLOVER enables proactive local semantic review by default:

```text
CLOVER_EDGE_REVIEW_PROACTIVE=true
```

Static execution triggers only when evidence is closed and size-bounded, including single-row multi-field selection, candidate selection over at most five rows, simple boolean combinations, short-list assembly, and value normalization for percentages, units, quotes, labels, or trailing parentheses. Edge output must reference fact IDs and replay through deterministic operations; on review failure, execution falls back to the original static or Global path. List questions involving superlatives, ranking, counting, or cross-row comparison do not enter proactive Edge assembly and remain on the deterministic or Global path.

The default `USER_VARIANT_ORDER` is the compact paper set:

```bash
full,no_static,static_only,no_retry
```

For exploratory runs, a fixed order can be specified via:

```bash
CLOVER_ABLATION_VARIANT_ORDER=full,all_edge,no_edge,static,no_contract,end_review,one_shot,no_retry,cloud_finalize,static_only,no_static,no_closure_checker
```

Each experiment starts the vLLM server only once. Before each variant, a local-model warm-up is performed and excluded from the measured time. Results are written to:

```text
benchmark/runs/<dataset>_ablation_<timestamp>/
```

The `sanity_check.json` in that directory validates the fixed case set, fine-grained feature flags, disabled retry activity for `w/o Retry`, and the terminal path for Global Finalization. After the variants finish, the script prints the summary table to the terminal and generates:

```text
ablation_summary.md
ablation_summary.csv
ablation_summary.json
```

The table includes:

- Accuracy, percentage-point delta vs Full CLOVER, and the exact McNemar paired test.
- Regression/recovery case counts vs Full.
- Node Edge runs/successes/steps, node reviews, and contract rejections.
- Terminal Edge calls/hits/escalations, proactive semantic opportunities/hits, and Global replan counts.
- Final answer sources, Global/Edge model tokens, estimated cost, and runtime.
- The direct Edge-to-Global substitution effect between `Full CLOVER` and `w/o Edge Agent`, including Global calls, synthesis, replan, tokens, cost, and per-query call delta.
- The terminal/proactive semantic review contribution of `End-only Review` vs `w/o Edge Agent`, and the node-repair contribution of `Full CLOVER` vs `End-only Review`.
- Per-variant regression/recovery case IDs vs Full, retry-case accuracy, and runtime error types.

Additionally, the suite generates:

```text
ablation_case_diagnostics.jsonl
ablation_discordant_cases.csv
```

The former records each case's correctness, answer source, retry, and error type across variants; the latter keeps only paired cases where Full and a variant disagree, making it easy to inspect why Static execution, Static-only execution, retry, or other mechanisms produced reverse gains.

If the runs are already complete and only the summary needs to be regenerated without rerunning:

```bash
python -m benchmarks.summarize_ablation_suite \
  --suite-root benchmark/runs/tablebench_ablation_<timestamp> \
  --dataset tablebench
```

## Latest full TableBench ablation (relaxed Contract Gate)

Run: `/root/autodl-tmp/CLOVER/benchmark/runs/tablebench_full_ablation_relaxed_contract_qwen25_14b_20260629_092745`

### Experimental setup

- Dataset: full TableBench evaluation split used by the current benchmark runner, 491 evaluated cases.
- Model: `Qwen2.5-14B-Instruct` for both planner/synthesizer and local Edge roles.
- Serving: local vLLM, single shared server because both Edge roles use the same model.
- Decoding: temperature `0.0`; max model length `8192`; max generation tokens `2048`.
- Validation: `remote_supervisor`; supervisor retry budget `3` for Full CLOVER.
- Contract Gate: relaxed at execution time. Structural output normalization remains enabled so invalid, non-serializable, or empty outputs can still trigger generic repair.
- Reported variants: Full CLOVER, `w/o Static`, `Static-Only`, and `w/o Retry`.

`w/o Contract Verification` was run once as a sanity check and changed ACC by only `-0.61 pp` (`326/491`, `66.40%`), so it is omitted from the official ablation table.

### Results

| Experiment | Correct | ACC | Δ vs Full | Input Tokens | Output Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full CLOVER | 329/491 | 67.01% | +0.00 pp | 2,598,629 | 70,058 | 2,668,687 |
| w/o Static | 168/491 | 34.22% | -32.79 pp | 7,313,721 | 192,589 | 7,506,310 |
| Static-Only | 271/491 | 55.19% | -11.81 pp | 1,922,676 | 35,513 | 1,958,189 |
| w/o Retry | 312/491 | 63.54% | -3.46 pp | 2,223,980 | 45,228 | 2,269,208 |

### Conclusion

Static execution/finalization is the dominant contributor. Removing it drops ACC by `32.79 pp` and increases total token usage from `2.67M` to `7.51M`, showing that deterministic static paths are both more accurate and much cheaper than forcing the model to handle those operations.

Static-only execution is strong but incomplete: it reaches `55.19%` ACC with the lowest token usage among the main variants, but still trails Full CLOVER by `11.81 pp`. This supports the design choice that static execution should be the backbone, while model-based Edge/Global recovery remains necessary for cases that static evidence cannot close.

Retry is useful but secondary. Disabling retry reduces ACC by `3.46 pp` while also reducing token usage, so retry should remain enabled for final accuracy runs and can be disabled only for speed/cost-oriented debugging.

## Regenerating the Fixed Subset

```bash
CLOVER_ABLATION_REGENERATE_MANIFEST=1 \
bash benchmarks/run_ablation_suite.sh tablebench /path/to/edge-model
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
bash benchmarks/run_ablation_suite.sh tablebench /path/to/edge-model
```

The policies use different manifest filenames and never overwrite each other:

```text
tablebench_representative_100_seed20260619.jsonl
tablebench_edge_opportunity_100_seed20260619.jsonl
tablebench_full_eval.jsonl
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

## Cost-Accuracy Pareto (P0-2)

Aggregate cost-accuracy pairs across CLOVER, baselines, and ablation variants for Pareto analysis:

```bash
python -m benchmarks.summarize_cost_accuracy \
  --dataset tablebench \
  --ablation-suite benchmark/runs/tablebench_ablation_<timestamp> \
  --clover-run benchmark/runs/tablebench_clover_<timestamp> \
  --pure-cot-run benchmark/runs/tablebench_pure_cot_<timestamp> \
  --external-baselines benchmarks/external_baselines.json \
  --output-dir benchmark/runs/pareto_tablebench
```

Output: `cost_accuracy_pareto.{csv,md,json}` with per-query and per-1K-query cost, accuracy, Global/Edge calls, and total tokens.

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

Output: `edge_model_sweep.{csv,md,json}` with accuracy, cost, Global/Edge calls, model calls (latency proxy), and tokens per query for each (Edge model, dataset, variant) combination.

The `full` variant shows how Edge scale affects overall CLOVER performance. The `no_static` variant shows whether larger Edge models can independently finalize when Static is unavailable.
