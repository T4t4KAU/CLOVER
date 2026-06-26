#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MMQA convenience wrapper for the CLOVER local-vLLM evaluator.
#
# Usage:
#   bash benchmarks/run_vllm_eval_mmqa.sh [two_table|three_table] [eval options...]
#
# Examples:
#   bash benchmarks/run_vllm_eval_mmqa.sh two_table --sample-size 100
#   bash benchmarks/run_vllm_eval_mmqa.sh three_table --max-cases 200
#   MMQA_SPLIT=two_table bash benchmarks/run_vllm_eval_mmqa.sh
#
# Model/server settings are inherited from benchmarks/run_vllm_eval_clover.sh:
#   CLOVER_EDGE_MODEL_PATH=/root/autodl-tmp/models/Qwen3.6-27B
#   CLOVER_EDGE1_GPUS=0
#   CLOVER_EDGE1_PORT=8000
#   CLOVER_EDGE1_MAX_MODEL_LEN=8192
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "two_table" || "${1:-}" == "three_table" ]]; then
  export MMQA_SPLIT="$1"
  shift
fi

exec bash "${SCRIPT_DIR}/run_vllm_eval_clover.sh" mmqa "$@"
