#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
cd "$ROOT"

# 本脚本只做后评估，不启动任何新模型训练。若主流水线仍在运行则退出，
# 由调用方稍后重试，避免常驻轮询进程。
test -x "$PYTHON_BIN"
if [[ ! -s data/results/global_taskloss_summary_v1.json ]]; then
  echo "taskloss main pipeline is not complete" >&2
  exit 3
fi

bash scripts/decision_aware/run_cloud_global_taskloss_mixed_eval_v1.sh
