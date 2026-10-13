#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
RESULT_DIR="data/results/cloud_global_joint_spread_grid_v1"
CHECKPOINT_ROOT="data/checkpoints/global_joint_spread_grid_v1"
PREDICTION_DIR="data/results/global_lag168_predictions_v1"
MANIFEST="data/results/global_joint_spread_grid_manifest_v1.json"
OUTPUT="data/results/global_joint_spread_grid_continuous_mpc_v1.json"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

timestamp() { date --iso-8601=seconds; }
record_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$RESULT_DIR/status.tsv"
}

run_candidate() {
  local family="$1"
  local spread_weight="$2"
  local direction_weight="$3"
  local profit_weight="$4"
  local lr="$5"
  local result="$RESULT_DIR/${family}_train.json"
  if [[ -s "$CHECKPOINT_ROOT/$family/candidate_manifest.json" && -s "$result" ]]; then
    record_status "$family" "already_complete"
    return
  fi
  record_status "$family" "training"
  "$PYTHON_BIN" scripts/decision_aware/train_global_joint_continuous_zo.py \
    --family "$family" --output-dir "$CHECKPOINT_ROOT" --result "$result" \
    --epochs 2 --episode-days 2 --episodes-per-node 2 --max-nodes 15 \
    --directions 1 --epsilon-spread 10 --epsilon-rt 15 \
    --profit-weight "$profit_weight" --proxy-scale 1000 \
    --prediction-scale 25 --spread-weight "$spread_weight" \
    --direction-weight "$direction_weight" --direction-threshold 6 \
    --direction-temperature 25 --lr "$lr" --scope price_head --device cuda
  record_status "$family" "complete"
}

{
  run_candidate jointspread_a 0.05 0.05 0.003 0.00002
  run_candidate jointspread_b 0.05 0.20 0.003 0.00002
  run_candidate jointspread_c 0.10 0.10 0.003 0.00010
  run_candidate jointspread_d 0.05 0.10 0.030 0.00002

  record_status "grid" "preparing_manifest"
  "$PYTHON_BIN" scripts/decision_aware/prepare_global_joint_grid_manifest.py \
    --family-dirs \
    "$CHECKPOINT_ROOT/jointspread_a,$CHECKPOINT_ROOT/jointspread_b,$CHECKPOINT_ROOT/jointspread_c,$CHECKPOINT_ROOT/jointspread_d" \
    --output "$MANIFEST"

  record_status "grid" "exporting_predictions"
  "$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --candidate-manifest "$MANIFEST" \
    --families jointspread_a,jointspread_b,jointspread_c,jointspread_d \
    --tasks da,rt_at_da,rt --phase inner --device cuda --batch-size 64 \
    --output-dir "$PREDICTION_DIR"

  TRIPLETS="da_jointspread_a__rt_at_da_jointspread_a__rt_jointspread_a"
  TRIPLETS+=" ,da_jointspread_b__rt_at_da_jointspread_b__rt_jointspread_b"
  TRIPLETS+=" ,da_jointspread_c__rt_at_da_jointspread_c__rt_jointspread_c"
  TRIPLETS+=" ,da_jointspread_d__rt_at_da_jointspread_d__rt_jointspread_d"
  TRIPLETS+=" ,da_seasonal__rt_at_da_seasonal__rt_stable"
  TRIPLETS+=" ,da_stable__rt_at_da_stable__rt_stable"
  TRIPLETS+=" ,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"

  record_status "grid" "running_full_settlement"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PREDICTION_DIR" \
    --families jointspread_a,jointspread_b,jointspread_c,jointspread_d,stable \
    --phase inner --workers 30 --executor process \
    --cross-baselines seasonal,xgboost_robust \
    --triplets "${TRIPLETS// /}" \
    --status inner_validation_candidate_not_final --output "$OUTPUT"
  test -s "$OUTPUT"
  record_status "all" "complete"
} 2>&1 | tee -a "$RESULT_DIR/runner.log"
