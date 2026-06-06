<p align="center">
  <img src="assets/clover-logo.svg" alt="CLOVER" width="520">
</p>

# CLOVER

CLOVER is a cost-efficient cloud-edge collaborative multi-agent system for data reasoning. It separates global reasoning, local planning, local execution, and result review into coordinated agents so that expensive cloud reasoning is used only where it is most valuable.

## Highlights

- Cloud-edge collaborative agent architecture.
- Remote global reasoning with local task execution.
- Modular workflow stages for planning, execution, reporting, and retry.
- Extensible design for multiple data reasoning task types.

## Setup

Using uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Using conda:

```bash
conda create -n clover python=3.10
conda activate clover
pip install -r requirements.txt
```

ModelScope dataset downloads are optional. Install the extra before using
`CLOVER_DATASET_SOURCE=modelscope`:

```bash
pip install '.[modelscope]'
```

Configure models with JSON files. Public templates are committed under
`model_config/`, but real API keys should stay in environment variables.

Remote LLM config, for example `model_config/remote_llm_config.json`:

```json
{
  "provider": "openai",
  "api_type": "responses",
  "api_key_env": "OPENAI_API_KEY",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-5.2",
  "timeout": 180,
  "max_retries": 2,
  "max_output_tokens": 12000,
  "temperature": 0
}
```

DeepSeek Remote LLM config, for example `model_config/deepseek_remote_llm_config.json`:

```json
{
  "provider": "deepseek",
  "api_type": "chat_completions",
  "api_key_env": "DEEPSEEK_API_KEY",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-pro",
  "timeout": 180,
  "max_retries": 2,
  "max_tokens": 12000,
  "temperature": 0
}
```

Qwen2.5-Coder-7B-Instruct Local SLM config, for example `model_config/local_slm_config.json`:

```json
{
  "provider": "local",
  "api_type": "chat_completions",
  "api_key": "EMPTY",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "timeout": 1800,
  "max_retries": 2,
  "max_tokens": 4096,
  "temperature": 0
}
```

Start a local OpenAI-compatible SLM server with either MLX or vLLM:

```bash
bash benchmarks/start_mlx_openai_server.sh
# or
bash benchmarks/start_vllm_openai_server.sh
```

Both scripts default to `Qwen/Qwen2.5-Coder-7B-Instruct` at `http://127.0.0.1:8000/v1`,
matching `model_config/local_slm_config.json`. Override the model with
`CLOVER_LOCAL_MODEL`, `CLOVER_MLX_MODEL`, or `CLOVER_VLLM_MODEL`.

More design notes are available in `docs/architecture.md`,
`docs/runtime_design.md`, `docs/benchmarking.md`, and
`docs/model_configs.md`. Prompt templates are indexed in `docs/prompts.md`.

Then set the API keys and config paths:

```bash
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export CLOVER_REMOTE_LLM_CONFIG=model_config/remote_llm_config.json
export CLOVER_LOCAL_SLM_CONFIG=model_config/local_slm_config.json
```

## Prepare Benchmark Datasets

Download and convert the supported benchmark datasets into CLOVER's local eval
layout:

```bash
python -m benchmarks.download --datasets-root datasets
```

By default this prepares DataBench, non-visual TableBench, and the numerical
FinanceBench subset. To prepare only TableBench:

```bash
python -m benchmarks.download --dataset tablebench --datasets-root datasets
```

TableBench visualization/chart-generation cases are excluded by default because
CLOVER's TableBench eval targets non-visual table reasoning.

## Run Benchmark

Benchmark outputs are written under `benchmark/runs`. The shell wrappers read
the model configs from `CLOVER_REMOTE_LLM_CONFIG` and `CLOVER_LOCAL_SLM_CONFIG`
by default, and any extra CLI arguments are forwarded to `python -m benchmarks.eval`.
Local execution concurrency is split into `--max-parallel-execution-units`,
`--max-parallel-slm-node-jobs`, `--max-parallel-slm-sequences`, and
`--max-pending-slm-sequences`. Use `--slm-scheduler fifo` to disable TPTT for
an ablation run, or set `CLOVER_SLM_SCHEDULER=fifo` in the shell wrappers.

### DataBench

Run CLOVER on a small DataBench smoke test:

```bash
bash benchmarks/run_databench_eval.sh \
  --databench-root datasets/databench \
  --run-name databench_bench_smoke \
  --max-cases 50
```

Run the DataBench Remote LLM baseline on the same split:

```bash
bash benchmarks/run_databench_remote_baseline.sh \
  --databench-root datasets/databench \
  --run-name databench_remote_baseline_smoke \
  --max-cases 50
```

For the full DataBench run, remove `--max-cases`:

```bash
bash benchmarks/run_databench_eval.sh \
  --databench-root datasets/databench \
  --run-name databench_bench
```

### TableBench

Run CLOVER on non-visual TableBench cases:

```bash
bash benchmarks/run_tablebench_eval.sh \
  --tablebench-root datasets/tablebench \
  --run-name tablebench_clover_smoke \
  --max-cases 50
```

Run the TableBench Remote LLM baseline:

```bash
bash benchmarks/run_tablebench_remote_baseline.sh \
  --tablebench-root datasets/tablebench \
  --run-name tablebench_remote_baseline_smoke \
  --max-cases 50
```

The TableBench baseline uses direct prediction (`DP`) by default. To run
table chain-of-thought or program-of-thought baselines:

```bash
CLOVER_TABLEBENCH_INSTRUCTION_TYPE=TCoT \
  bash benchmarks/run_tablebench_remote_baseline.sh \
    --tablebench-root datasets/tablebench \
    --run-name tablebench_tcot_baseline_smoke \
    --max-cases 50

CLOVER_TABLEBENCH_INSTRUCTION_TYPE=PoT \
  bash benchmarks/run_tablebench_remote_baseline.sh \
    --tablebench-root datasets/tablebench \
    --run-name tablebench_pot_baseline_smoke \
    --max-cases 50
```

## Test

Run the unit test suite from the repository root:

```bash
./run_tests.sh
```

Run one or more specific test modules:

```bash
./run_tests.sh tests.benchmarks.test_eval_runtime_grouping
```

## License

CLOVER is released under the MIT License. See [LICENSE](LICENSE).
