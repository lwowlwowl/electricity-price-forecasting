#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/electricity-price-forecasting
PYTHON_BIN="/root/miniconda3/bin/python3.12"
WAIT_PID="${1:-}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 20
  done
fi

RUN_DIRS="data/checkpoints/global_ercot_mae_f1_v1/da_inner_seed0,data/checkpoints/global_ercot_mae_f1_head_v1/da_inner_seed0,data/checkpoints/global_ercot_mae_f1_decoder_v1/da_inner_seed0"
FAMILIES="daopt0,daopt1,daopt2,daopt3"
TRIPLETS="da_daopt0__rt_at_da_xgboost_robust__rt_xgboost_robust,da_daopt1__rt_at_da_xgboost_robust__rt_xgboost_robust,da_daopt2__rt_at_da_xgboost_robust__rt_xgboost_robust,da_daopt3__rt_at_da_xgboost_robust__rt_xgboost_robust"

"${PYTHON_BIN}" scripts/decision_aware/prepare_global_da_candidates.py \
  --run-dirs "${RUN_DIRS}" \
  --top-k 4

"${PYTHON_BIN}" scripts/decision_aware/export_global_f0_f1_predictions.py \
  --config configs/decision_aware/global_ercot_mae_f1_v1.yaml \
  --families "${FAMILIES}" \
  --candidate-manifest data/results/global_da_robust_candidate_manifest.json \
  --tasks da \
  --device cuda \
  --batch-size 64 \
  --output-dir data/results/global_da_robust_predictions

"${PYTHON_BIN}" scripts/decision_aware/evaluate_global_continuous_mpc.py \
  --config configs/decision_aware/global_ercot_mae_f1_v1.yaml \
  --prediction-dir data/results/global_da_robust_predictions \
  --families "${FAMILIES}" \
  --cross-baselines xgboost_robust \
  --triplets "${TRIPLETS}" \
  --workers 4 \
  --executor process \
  --status inner_validation_da_robust_optimization \
  --output data/results/global_da_robust_continuous_mpc_v1.json
