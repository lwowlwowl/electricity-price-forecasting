#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
CONFIG="configs/decision_aware/global_ercot_huber_f1_stable_v2.yaml"
CHECKPOINT_ROOT="data/checkpoints/global_ercot_huber_f1_stable_v2"
RESULT_DIR="data/results/cloud_global_da_outer_validation"
PREDICTION_DIR="data/results/global_da_outer_multiseed_predictions"
TEMPLATE_DIR="data/results/global_outer_prediction_templates"
ENSEMBLE_DIR="data/results/global_da_outer_median_predictions"
BLEND_DIR="data/results/global_da_outer_median_antiblend_predictions"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 候选已在2023内部验证冻结：F1、seed 0/1/2中位数、Global权重-0.15。
# 2024外层验证期间不搜索结构、seed、聚合方式或融合权重。
for seed in 0 1 2; do
  run_dir="$CHECKPOINT_ROOT/da_outer_seed${seed}"
  log="$RESULT_DIR/da_outer_seed${seed}.log"
  if [[ -s "$run_dir/summary.json" ]]; then
    printf '%s\tseed%s\talready_complete\n' "$(date --iso-8601=seconds)" "$seed"
    continue
  fi
  resume_args=()
  if [[ -s "$run_dir/latest.pt" ]]; then
    resume_args=(--resume)
  fi
  printf '%s\tseed%s\tstarting\n' "$(date --iso-8601=seconds)" "$seed"
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$CONFIG" --task da --phase outer --seed "$seed" \
    --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"
  test -s "$run_dir/summary.json"
  test -s "$run_dir/best.pt"
  printf '%s\tseed%s\tcomplete\n' "$(date --iso-8601=seconds)" "$seed"
done

# 三任务基线使用2020--2023训练、2024验证；稳健损失类型保持2023已冻结值。
"$PYTHON_BIN" scripts/decision_aware/run_global_baselines.py \
  --config "$CONFIG" --task da --phase outer \
  --models xgboost_absolute --workers 6
"$PYTHON_BIN" scripts/decision_aware/run_global_baselines.py \
  --config "$CONFIG" --task rt_at_da --phase outer \
  --models xgboost_pseudohuber --workers 6
"$PYTHON_BIN" scripts/decision_aware/run_global_baselines.py \
  --config "$CONFIG" --task rt --phase outer \
  --models xgboost_absolute --workers 6

"$PYTHON_BIN" scripts/decision_aware/export_global_prediction_templates.py \
  --config "$CONFIG" --phase outer --tasks rt_at_da,rt \
  --family outertemplate --output-dir "$TEMPLATE_DIR"

"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_multiseed_manifest.py \
  --checkpoint-root "$CHECKPOINT_ROOT" --phase outer --seeds 0,1,2 \
  --output data/results/global_da_outer_multiseed_candidate_manifest.json
"$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
  --config "$CONFIG" --phase outer \
  --candidate-manifest data/results/global_da_outer_multiseed_candidate_manifest.json \
  --families globalseed0,globalseed1,globalseed2 --tasks da --device cuda \
  --output-dir "$PREDICTION_DIR"

"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_seed_ensemble.py \
  --archives "$PREDICTION_DIR/globalseed0_da_outer_seed0.npz,$PREDICTION_DIR/globalseed1_da_outer_seed0.npz,$PREDICTION_DIR/globalseed2_da_outer_seed0.npz" \
  --family globalmedian --aggregation median --phase outer \
  --output-dir "$ENSEMBLE_DIR"
"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_blends.py \
  --source-family globalmedian --source-dir "$ENSEMBLE_DIR" \
  --template-family outertemplate --template-dir "$TEMPLATE_DIR" \
  --weights=-0.15 --phase outer \
  --baseline-dir data/results/global_baselines_v1/da_outer \
  --output-dir "$BLEND_DIR" \
  --output-manifest data/results/global_da_outer_median_antiblend_manifest.json

"$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
  --config "$CONFIG" --phase outer --prediction-dir "$BLEND_DIR" \
  --families blendgm15 --cross-baselines xgboost_robust \
  --triplets 'da_blendgm15__rt_at_da_xgboost_robust__rt_xgboost_robust,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust' \
  --workers 12 --executor process \
  --status outer_validation_frozen_da_median_antiblend \
  --output data/results/global_da_outer_median_antiblend_continuous_mpc_v1.json

"$PYTHON_BIN" scripts/decision_aware/summarize_global_da_outer_validation.py
printf '%s\touter_validation\tcomplete\n' "$(date --iso-8601=seconds)"
