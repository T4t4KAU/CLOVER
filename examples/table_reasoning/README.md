# Table Reasoning Examples

Run from the repository root:

```bash
bash examples/table_reasoning/run_eval.sh
```

The script builds a temporary Databench-compatible input directory from `case_1` to `case_10`, then calls the root `eval.py` end to end.

Results are written to:

```text
examples/table_reasoning/runs/eval
```

Model configs and batch sizes can be overridden with the same environment variables used by the main eval scripts, for example:

```bash
CLOVER_REMOTE_LLM_CONFIG=config/remote_llm_config.json \
CLOVER_LOCAL_SLM_CONFIG=config/local_slm_config.json \
CLOVER_REMOTE_BATCH_SIZE=8 \
CLOVER_LOCAL_BATCH_SIZE=8 \
bash examples/table_reasoning/run_eval.sh
```
