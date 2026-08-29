#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${RESULTS_DIR:?Set RESULTS_DIR to one model/method result directory}"
python "${PROJECT_DIR}/run/longbench/eval.py" \
  --results_dir "${RESULTS_DIR}" \
  --method "${METHOD:-compilerkv}"
