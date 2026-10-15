#!/usr/bin/env bash
# 当前结构 vs 全F0 vs 全F1的ERCOT留出折消融；2025保持锁定。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

RESULT_DIR="data/results/cloud_universal_ercot_ablation_v1"
PREDICTION_DIR="data/results/universal_lomo_ercot_ablation_predictions_v1"
PROFIT_DIR="data/results/universal_lomo_ercot_ablation_profit_predictions_v1"
mkdir -p "$RESULT_DIR" "$PREDICTION_DIR" "$PROFIT_DIR"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
export PYTHONUNBUFFERED=1

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

preflight() {
  test -x "$PYTHON_BIN"
  for task in da rt_at_da rt; do
    test -s "data/results/universal_lomo_${task}_v1/ercot_${task}_v1.npz"
  done
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
import torch
sys.path.insert(0, 'src')
from decision_aware.config import PilotConfig

root = Path.cwd()
current_mae = PilotConfig.from_yaml(root / 'configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml')
all_f0 = PilotConfig.from_yaml(root / 'configs/decision_aware/universal_lomo_holdout_ercot_ablation_all_f0_mae_v1.yaml')
current_rt = PilotConfig.from_yaml(root / 'configs/decision_aware/universal_lomo_holdout_ercot_rt_at_da_v1.yaml')
all_f1 = PilotConfig.from_yaml(root / 'configs/decision_aware/universal_lomo_holdout_ercot_ablation_all_f1_rt_at_da_v1.yaml')
assert (current_mae.da_fusion_mode, current_mae.rt_fusion_mode) == ('F1', 'F1')
assert (all_f0.da_fusion_mode, all_f0.rt_fusion_mode) == ('F0', 'F0')
assert current_rt.rt_at_da_fusion_mode == 'F0'
assert all_f1.rt_at_da_fusion_mode == 'F1'
assert all_f0.global_training_loss == current_mae.global_training_loss == 'mae'
assert all_f1.global_training_loss == current_rt.global_training_loss == 'pseudo_huber'
assert torch.cuda.is_available(), 'CUDA unavailable'
print({'gpu': torch.cuda.get_device_name(0), 'new_trainings': 3, 'test_2025': False})
PY
}

train_and_evaluate() {
  local family="$1"
  local task="$2"
  local config="$3"
  local checkpoint_dir="$4"
  local best="$checkpoint_dir/${task}_inner_seed0/best.pt"
  local summary="$checkpoint_dir/${task}_inner_seed0/summary.json"
  local output="$PREDICTION_DIR/${family}_${task}_inner_seed0.npz"
  local log="$RESULT_DIR/${family}_${task}.log"
  local -a resume_args=()
  if [[ -s "$summary" && -s "$best" ]]; then
    record_status "${family}_${task}" training_already_complete
  else
    if [[ -s "$checkpoint_dir/${task}_inner_seed0/latest.pt" ]]; then
      resume_args=(--resume)
    fi
    record_status "${family}_${task}" training_starting
    "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
      --config "$config" --task "$task" --phase inner --seed 0 \
      --device cuda "${resume_args[@]}" 2>&1 | tee "$log"
    test -s "$summary" && test -s "$best"
    record_status "${family}_${task}" training_complete
  fi
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "${family}_${task}" holdout_already_complete
  else
    record_status "${family}_${task}" holdout_starting
    "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
      --checkpoint "$best" --task "$task" --holdout-market ERCOT \
      --output "$output" --device cuda --batch-size 64 2>&1 | tee -a "$log"
    record_status "${family}_${task}" holdout_complete
  fi
}

prepare_family() {
  local family="$1"
  local da_source="$2"
  local rtda_source="$3"
  local rt_source="$4"
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase inner --family "$family" --seed 0 --output-root "$PROFIT_DIR" \
    --source-da "$da_source" --source-rt-at-da "$rtda_source" \
    --source-rt "$rt_source"
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi

  train_and_evaluate all_f0 da \
    configs/decision_aware/universal_lomo_holdout_ercot_ablation_all_f0_mae_v1.yaml \
    data/checkpoints/universal_lomo_ablation_ercot_all_f0_v1
  train_and_evaluate all_f0 rt \
    configs/decision_aware/universal_lomo_holdout_ercot_ablation_all_f0_mae_v1.yaml \
    data/checkpoints/universal_lomo_ablation_ercot_all_f0_v1
  train_and_evaluate all_f1 rt_at_da \
    configs/decision_aware/universal_lomo_holdout_ercot_ablation_all_f1_rt_at_da_v1.yaml \
    data/checkpoints/universal_lomo_ablation_ercot_all_f1_v1

  record_status settlement preparing
  current_da=data/results/universal_lomo_da_v1/ercot_da_v1.npz
  current_rtda=data/results/universal_lomo_rt_at_da_v1/ercot_rt_at_da_v1.npz
  current_rt=data/results/universal_lomo_rt_v1/ercot_rt_v1.npz
  prepare_family current "$current_da" "$current_rtda" "$current_rt"
  prepare_family all_f0 "$PREDICTION_DIR/all_f0_da_inner_seed0.npz" \
    "$current_rtda" "$PREDICTION_DIR/all_f0_rt_inner_seed0.npz"
  prepare_family all_f1 "$current_da" \
    "$PREDICTION_DIR/all_f1_rt_at_da_inner_seed0.npz" "$current_rt"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PROFIT_DIR" --families current,all_f0,all_f1 \
    --phase inner --include-baselines xgboost_robust --workers 8 --executor process \
    --status universal_ercot_structure_ablation_seed0 \
    --output data/results/universal_lomo_ercot_ablation_continuous_mpc_v1.json \
    2>&1 | tee "$RESULT_DIR/settlement.log"
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
