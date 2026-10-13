#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
WORKERS="${WORKERS:-8}"
SMOKE="${SMOKE:-0}"
cd "$ROOT"

run_market_task() {
  local market="$1"
  local config="$2"
  local task="$3"
  local -a smoke_args=()
  if [[ "$SMOKE" == "1" ]]; then smoke_args=(--smoke); fi
  local models="seasonal,xgboost,xgboost_absolute,xgboost_pseudohuber"
  if [[ "$task" != "rt" ]]; then models="${models},lear"; fi
  "$PYTHON_BIN" scripts/decision_aware/run_global_baselines.py \
    --config "$config" --market "$market" --task "$task" --phase inner \
    --models "$models" --workers "$WORKERS" "${smoke_args[@]}"
}

for task in da rt_at_da rt; do
  run_market_task CAISO configs/decision_aware/global_caiso_prediction_contract_v1.yaml "$task"
  run_market_task NYISO configs/decision_aware/global_nyiso_prediction_contract_v1.yaml "$task"
  run_market_task PJM configs/decision_aware/global_pjm_prediction_contract_v1.yaml "$task"
done
