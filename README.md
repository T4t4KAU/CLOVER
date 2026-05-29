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

Configure models with JSON files.

Remote LLM config, for example `config/openai_remote_llm_config.json`:

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

DeepSeek Remote LLM config, for example `config/deepseek_remote_llm_config.json`:

```json
{
  "provider": "deepseek",
  "api_type": "chat_completions",
  "api_key_env": "DEEPSEEK_API_KEY",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "timeout": 180,
  "max_retries": 2,
  "max_tokens": 12000,
  "temperature": 0
}
```

Local SLM config, for example `config/local_slm_config.json`:

```json
{
  "provider": "local",
  "api_type": "chat_completions",
  "api_key": "EMPTY",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "qwen3.6-27b",
  "timeout": 1800,
  "max_retries": 2,
  "max_tokens": 4096,
  "temperature": 0
}
```

Then set the API keys and config paths:

```bash
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export CLOVER_REMOTE_LLM_CONFIG=config/openai_remote_llm_config.json
export CLOVER_LOCAL_SLM_CONFIG=config/local_slm_config.json
```

## Prepare DataBench

Download full DataBench tables from HuggingFace and convert them into CLOVER's local eval layout:

```bash
python -m benchmarks.databench.download \
  --output-root datasets/databench \
  --repo-id cardiffnlp/databench \
  --config-name qa \
  --split train \
  --table-kind all
```

For a lightweight local copy, use the 20-row sample tables:

```bash
python -m benchmarks.databench.download \
  --output-root datasets/databench_sample \
  --table-kind sample
```

Sample-table conversion uses DataBench's `sample_answer` field as the expected answer.
Add `--overwrite` only when replacing an existing converted dataset.

## Run DataBench Benchmark

Run a small benchmark smoke test from the repository root:

```bash
bash benchmarks/run_databench_eval.sh \
  --databench-root datasets/databench \
  --run-name databench_bench_smoke \
  --max-cases 50
```

For the full DataBench benchmark, remove `--max-cases`:

```bash
bash benchmarks/run_databench_eval.sh \
  --databench-root datasets/databench \
  --run-name databench_bench
```

Results are written under `benchmark/runs`.

## Run Examples

The table reasoning examples contain 10 lightweight Databench cases under `examples/table_reasoning`.

Run from the repository root:

```bash
bash examples/table_reasoning/run_eval.sh
```

Results are written to:

```text
examples/table_reasoning/runs/eval
```

## License

CLOVER is released under the MIT License. See [LICENSE](LICENSE).
