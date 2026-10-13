#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
TRAIN_PID="${TRAIN_PID:?TRAIN_PID must identify run_cloud_global_lag168_v1.sh}"
RESULT_DIR="data/results/cloud_global_lag168_v1"
STATUS_FILE="$RESULT_DIR/status.tsv"
LOG_FILE="$RESULT_DIR/evaluation_runner.log"
MANIFEST="data/results/global_lag168_candidate_manifest_v1.json"
PREDICTION_DIR="data/results/global_lag168_predictions_v1"
OUTPUT="data/results/global_lag168_continuous_mpc_v1.json"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

timestamp() { date --iso-8601=seconds; }
record_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"
}

{
  record_status "evaluation" "waiting_for_training"
  while ! grep -q $'^.*\tall\tcomplete$' "$STATUS_FILE"; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      record_status "evaluation" "training_incomplete"
      exit 1
    fi
    sleep 30
  done

  record_status "evaluation" "preparing_candidates"
  "$PYTHON_BIN" scripts/decision_aware/prepare_global_lag168_candidates.py \
    --keep 3 --output "$MANIFEST" --prediction-dir "$PREDICTION_DIR"

  record_status "evaluation" "exporting_da"
  "$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
    --config configs/decision_aware/global_ercot_lag168_da_mae_v1.yaml \
    --candidate-manifest "$MANIFEST" \
    --families lagda0,lagda1,lagda2 --tasks da --phase inner \
    --device cuda --batch-size 64 --output-dir "$PREDICTION_DIR"

  record_status "evaluation" "exporting_rt_at_da"
  "$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
    --config configs/decision_aware/global_ercot_lag168_rt_at_da_pseudohuber_v1.yaml \
    --candidate-manifest "$MANIFEST" \
    --families lagrtda0,lagrtda1,lagrtda2 --tasks rt_at_da --phase inner \
    --device cuda --batch-size 64 --output-dir "$PREDICTION_DIR"

  TRIPLETS="da_lagda0__rt_at_da_lagrtda0__rt_stable"
  TRIPLETS+=" ,da_lagda0__rt_at_da_lagrtda1__rt_stable"
  TRIPLETS+=" ,da_lagda0__rt_at_da_lagrtda2__rt_stable"
  TRIPLETS+=" ,da_lagda1__rt_at_da_lagrtda0__rt_stable"
  TRIPLETS+=" ,da_lagda1__rt_at_da_lagrtda1__rt_stable"
  TRIPLETS+=" ,da_lagda1__rt_at_da_lagrtda2__rt_stable"
  TRIPLETS+=" ,da_lagda2__rt_at_da_lagrtda0__rt_stable"
  TRIPLETS+=" ,da_lagda2__rt_at_da_lagrtda1__rt_stable"
  TRIPLETS+=" ,da_lagda2__rt_at_da_lagrtda2__rt_stable"
  # seasonal的DA/RT-at-DA就是未经训练的168小时原始锚点。
  # 把它与每个训练残差候选交叉，避免漏掉“少训练反而更赚钱”的情况。
  TRIPLETS+=" ,da_seasonal__rt_at_da_lagrtda0__rt_stable"
  TRIPLETS+=" ,da_seasonal__rt_at_da_lagrtda1__rt_stable"
  TRIPLETS+=" ,da_seasonal__rt_at_da_lagrtda2__rt_stable"
  TRIPLETS+=" ,da_lagda0__rt_at_da_seasonal__rt_stable"
  TRIPLETS+=" ,da_lagda1__rt_at_da_seasonal__rt_stable"
  TRIPLETS+=" ,da_lagda2__rt_at_da_seasonal__rt_stable"
  TRIPLETS+=" ,da_seasonal__rt_at_da_seasonal__rt_stable"
  TRIPLETS+=" ,da_stable__rt_at_da_stable__rt_stable"
  TRIPLETS+=" ,da_seasonal__rt_at_da_seasonal__rt_seasonal"
  TRIPLETS+=" ,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"

  record_status "evaluation" "running_full_settlement"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PREDICTION_DIR" \
    --families lagda0,lagda1,lagda2,lagrtda0,lagrtda1,lagrtda2,stable \
    --phase inner --workers 30 --executor process \
    --cross-baselines seasonal,xgboost_robust \
    --triplets "${TRIPLETS// /}" \
    --status inner_validation_candidate_not_final \
    --output "$OUTPUT"
  test -s "$OUTPUT"
  record_status "evaluation" "complete"
} 2>&1 | tee -a "$LOG_FILE"
