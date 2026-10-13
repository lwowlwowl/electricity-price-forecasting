#!/usr/bin/env bash
set -euo pipefail

cd /root/electricity-price-forecasting
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/decision_aware/train_global_forecaster.py \
  --config configs/decision_aware/global_ercot_mae_f1_v1.yaml \
  --task da \
  --phase inner \
  --initialize-from data/checkpoints/global_ercot_huber_f1_stable_v2/da_inner_seed0/candidate_epoch_010.pt
