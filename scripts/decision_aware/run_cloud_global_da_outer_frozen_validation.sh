#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
CONFIG="configs/decision_aware/global_ercot_huber_f1_outer_frozen_v1.yaml"
INNER_ROOT="data/checkpoints/global_ercot_huber_f1_stable_v2"
CHECKPOINT_ROOT="data/checkpoints/global_ercot_huber_f1_outer_frozen_v1"
RESULT_DIR="data/results/cloud_global_da_outer_frozen_validation"
PREDICTION_DIR="data/results/global_da_outer_frozen_multiseed_predictions"
TEMPLATE_DIR="data/results/global_outer_prediction_templates"
ENSEMBLE_DIR="data/results/global_da_outer_frozen_median_predictions"
BLEND_DIR="data/results/global_da_outer_frozen_median_antiblend_predictions"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 严格外层训练：每个seed的最终epoch和学习率轨迹只取自2023 inner run。
# 2024在最终epoch只评估一次，不参与早停、调度或checkpoint选择。
for seed in 0 1 2; do
  run_dir="$CHECKPOINT_ROOT/da_outer_seed${seed}"
  log="$RESULT_DIR/da_outer_seed${seed}.log"
  plan="$INNER_ROOT/da_inner_seed${seed}/latest.pt"
  test -s "$plan"
  if [[ -s "$run_dir/summary.json" ]]; then
    printf '%s\tseed%s\talready_complete\n' "$(date --iso-8601=seconds)" "$seed"
    continue
  fi
  resume_args=()
  if [[ -s "$run_dir/latest.pt" ]]; then
    resume_args=(--resume)
  fi
  printf '%s\tseed%s\tstarting_frozen_plan\n' "$(date --iso-8601=seconds)" "$seed"
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$CONFIG" --task da --phase outer --seed "$seed" --device cuda \
    --frozen-training-plan-from "$plan" "${resume_args[@]}" 2>&1 | tee -a "$log"
  test -s "$run_dir/summary.json"
  test -s "$run_dir/best.pt"
  printf '%s\tseed%s\tcomplete_frozen_plan\n' "$(date --iso-8601=seconds)" "$seed"
done

"$PYTHON_BIN" scripts/decision_aware/audit_global_outer_frozen_training.py

# XGBoost和RT模板已在本地按2020--2023/2024合同生成并上传。
# 云端只核对完整性，不重复训练，也不要求安装joblib/xgboost。
test -s data/results/global_baselines_v1/da_outer/summary.json
test -s data/results/global_baselines_v1/rt_at_da_outer/summary.json
test -s data/results/global_baselines_v1/rt_outer/summary.json
test "$(find data/results/global_baselines_v1/da_outer -maxdepth 1 -name 'xgboost_absolute_*_seed0.npz' | wc -l)" -eq 15
test "$(find data/results/global_baselines_v1/rt_at_da_outer -maxdepth 1 -name 'xgboost_pseudohuber_*_seed0.npz' | wc -l)" -eq 15
test "$(find data/results/global_baselines_v1/rt_outer -maxdepth 1 -name 'xgboost_absolute_*_seed0.npz' | wc -l)" -eq 15
test -s "$TEMPLATE_DIR/outertemplate_rt_at_da_outer_seed0.npz"
test -s "$TEMPLATE_DIR/outertemplate_rt_outer_seed0.npz"

"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_multiseed_manifest.py \
  --checkpoint-root "$CHECKPOINT_ROOT" --phase outer --seeds 0,1,2 \
  --output data/results/global_da_outer_frozen_multiseed_candidate_manifest.json
"$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
  --config "$CONFIG" --phase outer \
  --candidate-manifest data/results/global_da_outer_frozen_multiseed_candidate_manifest.json \
  --families globalseed0,globalseed1,globalseed2 \
  --tasks da --device cuda --output-dir "$PREDICTION_DIR"

"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_seed_ensemble.py \
  --archives "$PREDICTION_DIR/globalseed0_da_outer_seed0.npz,$PREDICTION_DIR/globalseed1_da_outer_seed0.npz,$PREDICTION_DIR/globalseed2_da_outer_seed0.npz" \
  --family globalfrozenmedian --aggregation median --phase outer \
  --output-dir "$ENSEMBLE_DIR"
"$PYTHON_BIN" scripts/decision_aware/prepare_global_da_blends.py \
  --source-family globalfrozenmedian --source-dir "$ENSEMBLE_DIR" \
  --template-family outertemplate --template-dir "$TEMPLATE_DIR" \
  --weights=-0.15 --phase outer \
  --baseline-dir data/results/global_baselines_v1/da_outer \
  --output-dir "$BLEND_DIR" \
  --output-manifest data/results/global_da_outer_frozen_median_antiblend_manifest.json

"$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
  --config "$CONFIG" --phase outer --prediction-dir "$BLEND_DIR" \
  --families blendgm15 --cross-baselines xgboost_robust \
  --triplets 'da_blendgm15__rt_at_da_xgboost_robust__rt_xgboost_robust,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust' \
  --workers 12 --executor process \
  --status outer_validation_frozen_training_plan \
  --output data/results/global_da_outer_frozen_median_antiblend_continuous_mpc_v1.json

"$PYTHON_BIN" scripts/decision_aware/summarize_global_da_outer_validation.py \
  --input data/results/global_da_outer_frozen_median_antiblend_continuous_mpc_v1.json \
  --output data/results/global_da_outer_frozen_validation_conclusion_v1.json \
  --training-protocol frozen_inner_plan
printf '%s\touter_frozen_validation\tcomplete\n' "$(date --iso-8601=seconds)"
