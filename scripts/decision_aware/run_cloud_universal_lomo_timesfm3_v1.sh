#!/usr/bin/env bash
# Frozen TimesFM-3: four markets x three tasks, then ERCOT same-contract settlement.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
RESULT_DIR="data/results/cloud_universal_lomo_timesfm3_v1"
PREDICTION_DIR="data/results/universal_lomo_timesfm3_v1"
PROFIT_DIR="data/results/universal_lomo_timesfm3_profit_predictions_v1"
mkdir -p "$RESULT_DIR" "$PREDICTION_DIR" "$PROFIT_DIR"
exec 9>"$RESULT_DIR/run.lock"
flock -n 9 || { echo 'Frozen TimesFM-3 pipeline already running'; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
record_status() { printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "$STATUS_FILE"; }
trap 'record_status pipeline "failed_line_${LINENO}"' ERR

config_for() {
  local market="$1" task="$2" suffix="mae"
  if [[ "$task" == "rt_at_da" ]]; then suffix="rt_at_da"; fi
  printf 'configs/decision_aware/universal_lomo_holdout_%s_%s_v1.yaml' "$market" "$suffix"
}

run_one() {
  local market_lower="$1" market_upper="$2" task="$3"
  local config output label
  config="$(config_for "$market_lower" "$task")"
  output="$PREDICTION_DIR/${market_lower}_${task}_v1.npz"
  label="${market_lower}_${task}"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "$label" already_complete
    return
  fi
  record_status "$label" starting
  "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm3.py \
    --config "$config" --task "$task" --holdout-market "$market_upper" \
    --evaluation-stage inner --output "$output" \
    2>&1 | tee "$RESULT_DIR/${label}.log"
  test -s "$output" && test -s "${output%.npz}.json"
  record_status "$label" complete
}

{
  record_status preflight starting
  for task in da rt_at_da rt; do
    "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm3.py \
      --config "$(config_for ercot "$task")" --task "$task" \
      --holdout-market ERCOT --preflight-only
  done
  record_status preflight complete
  if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then exit 0; fi

  for market_lower in caiso ercot nyiso pjm; do
    market_upper="${market_lower^^}"
    for task in da rt_at_da rt; do
      run_one "$market_lower" "$market_upper" "$task"
    done
  done

  record_status settlement preparing
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase inner --family timesfm3_frozen --seed 0 --output-root "$PROFIT_DIR" \
    --source-da "$PREDICTION_DIR/ercot_da_v1.npz" \
    --source-rt-at-da "$PREDICTION_DIR/ercot_rt_at_da_v1.npz" \
    --source-rt "$PREDICTION_DIR/ercot_rt_v1.npz"
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase inner --family universal --seed 0 --output-root "$PROFIT_DIR" \
    --source-da data/results/universal_lomo_da_v1/ercot_da_v1.npz \
    --source-rt-at-da data/results/universal_lomo_rt_at_da_v1/ercot_rt_at_da_v1.npz \
    --source-rt data/results/universal_lomo_rt_v1/ercot_rt_v1.npz
  triplets="da_timesfm3_frozen__rt_at_da_timesfm3_frozen__rt_timesfm3_frozen,da_universal__rt_at_da_universal__rt_universal,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PROFIT_DIR" --families timesfm3_frozen,universal \
    --phase inner --include-baselines xgboost_robust --triplets "$triplets" \
    --workers 8 --executor process --status frozen_timesfm3_same_contract \
    --output data/results/universal_lomo_timesfm3_ercot_continuous_mpc_v1.json \
    2>&1 | tee "$RESULT_DIR/settlement.log"
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"

