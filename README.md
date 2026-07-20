<p align="center">
  <img src="assets/clover-logo.svg" alt="CLOVER" width="520">
</p>

# CLOVER

CLOVER is a context-routed framework for accurate and token-efficient tabular
reasoning with locally deployable open-weight language models. It compiles a
generated SQL plan into an explicit execution graph, runs supported relational
and arithmetic operations with a deterministic executor, and validates every
intermediate result.

When execution fails, CLOVER permits local repair only when the evidence,
allowed modification, and validation procedure are confined to the failed
node. Failures requiring changes to a source, dependency, join, or broader plan
are routed to global replanning. The local and global roles differ in context
and authority, not model size; the experiments use the same locally deployed
model checkpoint for both roles, without cloud APIs.

## Highlights

- Checked SQL-to-execution-graph compilation.
- Deterministic execution for supported table operations.
- Conservative node-local repair based on observable closure.
- Global replanning for structural or cross-node failures.
- One representation for single-table and multi-table questions.
- Local deployment with Qwen2.5-14B-Instruct or Qwen2.5-32B-Instruct.

## Setup

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Using Conda:

```bash
conda create -n clover python=3.10
conda activate clover
pip install -r requirements.txt
```

ModelScope dataset downloads are optional:

```bash
pip install '.[modelscope]'
```

## Model configuration

The main experiment launcher starts a local OpenAI-compatible vLLM service and
generates the runtime configuration automatically. By default, the planning
and local-repair roles share one model service and checkpoint:

```bash
CLOVER_EDGE1_MODEL_PATH=/models/Qwen2.5-14B-Instruct \
CLOVER_EDGE1_GPUS=0 \
bash benchmarks/run_vllm_eval_clover.sh tablebench --max-cases 50
```

Some internal configuration names retain `edge`, `cloud`, `remote`, or `SLM`
for backward compatibility. They identify runtime roles or legacy options and
do not imply cloud deployment, separate machines, or different model sizes.
For the paper setting, point both semantic roles to the same local checkpoint.

## Prepare benchmark datasets

Download and convert TableBench, WikiTQ, TabFact, and MMQA:

```bash
bash benchmarks/download_datasets.sh
```

Prepare one dataset only:

```bash
bash benchmarks/download_datasets.sh \
  --dataset tablebench \
  --datasets-root datasets
```

TableBench evaluation uses its non-visual reasoning cases. WikiTQ uses the
`pristine-unseen-tables` split, TabFact uses `small-test`, and MMQA provides the
dedicated two-table and three-table evaluations.

## Run CLOVER evaluation

Use the same launcher for all four datasets:

```bash
bash benchmarks/run_vllm_eval_clover.sh tablebench
bash benchmarks/run_vllm_eval_clover.sh wikitq
bash benchmarks/run_vllm_eval_clover.sh tablefact
bash benchmarks/run_vllm_eval_mmqa.sh two_table
```

Pass `--max-cases N` for a prefix smoke test or `--sample-size N` for a sampled
run. Outputs are written below `benchmark/runs` unless overridden by the
launcher settings.

## Ablations

The ablation suite evaluates the routing and recovery components of CLOVER:

```bash
bash benchmarks/run_ablation_suite.sh
```

See [benchmarks/ABLATION.md](benchmarks/ABLATION.md) for variant definitions and
report-generation commands.

## Test

Run the full unit test suite:

```bash
./run_tests.sh
```

Run selected modules:

```bash
./run_tests.sh tests.benchmarks.test_eval_runtime_grouping
```

## License

CLOVER is released under the MIT License. See [LICENSE](LICENSE).
