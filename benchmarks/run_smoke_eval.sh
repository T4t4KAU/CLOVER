#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_CASES="${CLOVER_SMOKE_MAX_CASES:-1}"
TARGETS="${CLOVER_SMOKE_TARGETS:-databench,tablebench}"
RUN_PREFIX="${CLOVER_SMOKE_RUN_PREFIX:-smoke_$(date +%Y%m%d_%H%M%S)}"

COMMON_ARGS=(
  --max-cases "${MAX_CASES}"
  --overwrite
  --no-progress
)

IFS=',' read -r -a TARGET_ITEMS <<< "${TARGETS}"
for target in "${TARGET_ITEMS[@]}"; do
  target="${target#"${target%%[![:space:]]*}"}"
  target="${target%"${target##*[![:space:]]}"}"
  case "${target}" in
    databench)
      "${SCRIPT_DIR}/run_databench_eval.sh" \
        --run-name "${RUN_PREFIX}_databench" \
        "${COMMON_ARGS[@]}" \
        "$@"
      ;;
    tablebench)
      "${SCRIPT_DIR}/run_tablebench_eval.sh" \
        --run-name "${RUN_PREFIX}_tablebench" \
        "${COMMON_ARGS[@]}" \
        "$@"
      ;;
    financebench)
      "${SCRIPT_DIR}/run_financebench_eval.sh" \
        --run-name "${RUN_PREFIX}_financebench" \
        "${COMMON_ARGS[@]}" \
        "$@"
      ;;
    "")
      ;;
    *)
      echo "Unknown smoke target: ${target}" >&2
      echo "Use CLOVER_SMOKE_TARGETS=databench,tablebench,financebench" >&2
      exit 2
      ;;
  esac
done
