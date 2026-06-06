#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$#" -gt 0 ]]; then
  exec "${PYTHON_CMD[@]}" -m unittest "$@"
fi

exec "${PYTHON_CMD[@]}" -m unittest discover -s tests -t . -v
