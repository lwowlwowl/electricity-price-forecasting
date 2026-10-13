#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/electricity-price-forecasting
PYTHON_BIN="/root/miniconda3/bin/python3.12"
SOURCE="data/checkpoints/global_ercot_huber_f1_stable_v2/da_inner_seed0/candidate_epoch_010.pt"
WAIT_PID="${1:-}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 20
  done
fi

run_one() {
  local config="$1"
  "${PYTHON_BIN}" scripts/decision_aware/train_global_forecaster.py \
    --config "${config}" \
    --task da \
    --phase inner \
    --initialize-from "${SOURCE}" \
    --smoke \
    --device cuda
  "${PYTHON_BIN}" scripts/decision_aware/train_global_forecaster.py \
    --config "${config}" \
    --task da \
    --phase inner \
    --initialize-from "${SOURCE}" \
    --device cuda
}

run_one configs/decision_aware/global_ercot_mae_f1_head_v1.yaml
run_one configs/decision_aware/global_ercot_mae_f1_decoder_v1.yaml
