#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
CONFIG="configs/decision_aware/global_ercot_huber_f1_stable_v2.yaml"
PREDICTION_DIR="data/results/global_da_multiseed_predictions"
TEMPLATE_DIR="data/results/global_da_robust_predictions"
RESULT_DIR="data/results/cloud_global_da_multiseed"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_multiseed_manifest.py
"$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
  --config "$CONFIG" \
  --candidate-manifest data/results/global_da_multiseed_candidate_manifest.json \
  --families globalseed1,globalseed2 --tasks da --device cuda \
  --output-dir "$PREDICTION_DIR"

for seed in 1 2; do
  family="globalseed${seed}"
  blend_dir="data/results/global_da_seed${seed}_antiblend_predictions"
  output="data/results/global_da_seed${seed}_antiblend_continuous_mpc_v1.json"
  "$PYTHON_BIN" scripts/decision_aware/prepare_global_da_blends.py \
    --source-family "$family" --source-dir "$PREDICTION_DIR" \
    --template-family daopt2 --template-dir "$TEMPLATE_DIR" \
    --weights=-0.20 --output-dir "$blend_dir" \
    --output-manifest "data/results/global_da_seed${seed}_antiblend_manifest.json"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config "$CONFIG" --prediction-dir "$blend_dir" \
    --families blendgm20 --cross-baselines xgboost_robust \
    --triplets da_blendgm20__rt_at_da_xgboost_robust__rt_xgboost_robust \
    --workers 12 --executor process \
    --status inner_validation_da_antiblend_multiseed \
    --output "$output"
done

# seed0使用已冻结的daopt2，与seed1/2做等权模型平均；
# 然后使用在2023已冻结的-20% Global残差权重。
ensemble_source="data/results/global_da_multiseed_ensemble_predictions"
"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_seed_ensemble.py \
  --archives "data/results/global_da_robust_predictions/daopt2_da_inner_seed0.npz,${PREDICTION_DIR}/globalseed1_da_inner_seed0.npz,${PREDICTION_DIR}/globalseed2_da_inner_seed0.npz" \
  --family globalensemble --output-dir "$ensemble_source"
"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_blends.py \
  --source-family globalensemble --source-dir "$ensemble_source" \
  --template-family daopt2 --template-dir "$TEMPLATE_DIR" \
  --weights=-0.20 --output-dir data/results/global_da_ensemble_antiblend_predictions \
  --output-manifest data/results/global_da_ensemble_antiblend_manifest.json
"$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
  --config "$CONFIG" \
  --prediction-dir data/results/global_da_ensemble_antiblend_predictions \
  --families blendgm20 --cross-baselines xgboost_robust \
  --triplets da_blendgm20__rt_at_da_xgboost_robust__rt_xgboost_robust \
  --workers 12 --executor process \
  --status inner_validation_da_antiblend_multiseed_ensemble \
  --output data/results/global_da_ensemble_antiblend_continuous_mpc_v1.json

printf '%s\tmultiseed_evaluation\tcomplete\n' "$(date --iso-8601=seconds)"
