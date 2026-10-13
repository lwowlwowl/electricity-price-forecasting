#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
RESULT_DIR="data/results/cloud_global_spread_gates_v1"
PREDICTION_DIR="data/results/global_lag168_predictions_v1"
OUTPUT="data/results/global_spread_gate_continuous_mpc_v1.json"
FAMILIES="gate_blend50,gate_blend75,gate_blend90,gate_agree,gate_agreeblend,gate_extreme50,gate_clip50,gate_extreme100,gate_clip100,gate_extreme200,gate_clip200"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1
timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$RESULT_DIR/status.tsv"; }

{
  record_status "gates" "preparing"
  "$PYTHON_BIN" scripts/decision_aware/prepare_global_spread_gates.py \
    --prediction-dir "$PREDICTION_DIR"
  TRIPLETS=""
  IFS=',' read -ra ITEMS <<< "$FAMILIES"
  for family in "${ITEMS[@]}"; do
    label="da_${family}__rt_at_da_${family}__rt_${family}"
    TRIPLETS="${TRIPLETS:+$TRIPLETS,}$label"
  done
  TRIPLETS+=",da_seasonal__rt_at_da_seasonal__rt_stable"
  TRIPLETS+=",da_stable__rt_at_da_stable__rt_stable"
  TRIPLETS+=",da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
  record_status "gates" "running_full_settlement"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PREDICTION_DIR" --families "$FAMILIES,stable" \
    --phase inner --workers 30 --executor process \
    --cross-baselines seasonal,xgboost_robust --triplets "$TRIPLETS" \
    --status inner_validation_candidate_not_final --output "$OUTPUT"
  test -s "$OUTPUT"
  record_status "all" "complete"
} 2>&1 | tee -a "$RESULT_DIR/runner.log"
