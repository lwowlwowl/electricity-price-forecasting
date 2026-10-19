#!/usr/bin/env bash
# 2025后续实验：仅评价冻结的尾部加权DA，复用原RT-at-DA/滚动RT与结算合同。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

CONFIG="configs/decision_aware/allmarket_final_da_tailprice_v2.yaml"
CHECKPOINT_ROOT="data/checkpoints/allmarket_final_da_tailprice_v2"
CURRENT_ARCHIVE="data/results/allmarket_locked_2025_complete_v2_20261017.tar.gz"
CURRENT_PRED_DIR="data/results/allmarket_locked_2025_predictions_v1"
CURRENT_ENSEMBLE_DIR="data/results/allmarket_locked_2025_ensemble_v1"
CURRENT_SETTLEMENT="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json"
CANDIDATE_PRED_DIR="data/results/allmarket_tailprice_followup_2025_predictions_v2"
CANDIDATE_ENSEMBLE_DIR="data/results/allmarket_tailprice_followup_2025_ensemble_v2"
CANDIDATE_PROFIT_DIR="data/results/allmarket_tailprice_followup_2025_profit_v2"
CANDIDATE_SETTLEMENT="data/results/allmarket_tailprice_followup_2025_continuous_mpc_v2.json"
SUMMARY="data/results/allmarket_tailprice_followup_2025_summary_v2.json"
RESULT_DIR="data/results/cloud_allmarket_tailprice_followup_2025_v2"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
ARCHIVE="data/results/allmarket_tailprice_followup_2025_v2_completed.tar.gz"
MANIFEST="data/results/allmarket_tailprice_followup_2025_v2_completed.manifest.sha256"
mkdir -p "$RESULT_DIR" "$CANDIDATE_PRED_DIR" "$CANDIDATE_ENSEMBLE_DIR" "$CANDIDATE_PROFIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU_MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }
checkpoint_for() { echo "$CHECKPOINT_ROOT/da_final_seed${1}/best.pt"; }
start_gpu_monitor() {
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 2 > "$RESULT_DIR/gpu_usage.csv" &
  GPU_MONITOR_PID="$!"
}
stop_gpu_monitor() {
  if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=""
}
trap stop_gpu_monitor EXIT
trap 'record_status pipeline "failed_line_${LINENO}"' ERR

restore_current_outputs_if_needed() {
  if [[ -s "$CURRENT_ENSEMBLE_DIR/ercot_da_ensemble.npz" \
        && -s "$CURRENT_PRED_DIR/ercot_rt_seed0.npz" ]]; then return; fi
  test -s "$CURRENT_ARCHIVE"
  tar -xzf "$CURRENT_ARCHIVE" \
    data/results/allmarket_locked_2025_predictions_v1 \
    data/results/allmarket_locked_2025_ensemble_v1
}

preflight() {
  test -x "$PYTHON_BIN" && test -s "$CONFIG" && test -s "$CURRENT_SETTLEMENT"
  for seed in 0 1 2; do test -s "$(checkpoint_for "$seed")"; done
  restore_current_outputs_if_needed
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print({
    'gpu': torch.cuda.get_device_name(0),
    'new_inferences': 12,
    'new_trainings': 0,
    'selection_allowed': False,
    'original_2025_results_were_known_before_followup': True,
    'unchanged': ['RT-at-DA', 'rolling RT', 'LP/MPC', 'penalty', 'degradation'],
})
PY
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_critical_event_weighting.py \
    tests/test_allmarket_final_2025_pipeline.py
}

evaluate_da() {
  local market="$1" seed="$2"
  local output="$CANDIDATE_PRED_DIR/${market,,}_da_seed${seed}.npz"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "candidate_${market}_da_seed${seed}" already_complete
    return
  fi
  record_status "candidate_${market}_da_seed${seed}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
    --checkpoint "$(checkpoint_for "$seed")" --task da --holdout-market "$market" \
    --evaluation-stage locked_2025 --output "$output" --device cuda --batch-size 64 \
    2>&1 | tee "$RESULT_DIR/${market,,}_da_seed${seed}.log"
  record_status "candidate_${market}_da_seed${seed}" complete
}

attach_frozen_rt_legs() {
  local market seed task source target
  for market in CAISO ERCOT NYISO PJM; do
    for task in rt_at_da rt; do
      for seed in 0 1 2; do
        source="$CURRENT_PRED_DIR/${market,,}_${task}_seed${seed}.npz"
        target="$CANDIDATE_PRED_DIR/${market,,}_${task}_seed${seed}.npz"
        test -s "$source"
        cp -f "$source" "$target"
      done
    done
  done
}

settle_and_summarize() {
  attach_frozen_rt_legs
  "$PYTHON_BIN" scripts/decision_aware/prepare_allmarket_locked_2025.py \
    --input-dir "$CANDIDATE_PRED_DIR" --ensemble-dir "$CANDIDATE_ENSEMBLE_DIR" \
    --profit-dir "$CANDIDATE_PROFIT_DIR"
  local families="seed0,seed1,seed2,ensemble"
  local triplets="da_seed0__rt_at_da_seed0__rt_seed0,da_seed1__rt_at_da_seed1__rt_seed1,da_seed2__rt_at_da_seed2__rt_seed2,da_ensemble__rt_at_da_ensemble__rt_ensemble"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config "$CONFIG" --prediction-dir "$CANDIDATE_PROFIT_DIR" \
    --families "$families" --phase final --triplets "$triplets" \
    --no-selection-allowed --workers 8 --executor process \
    --status post_original_2025_tailprice_followup_no_selection \
    --output "$CANDIDATE_SETTLEMENT" 2>&1 | tee "$RESULT_DIR/settlement.log"
  "$PYTHON_BIN" scripts/decision_aware/summarize_allmarket_tailprice_followup_2025.py \
    --current-ensemble-dir "$CURRENT_ENSEMBLE_DIR" \
    --candidate-ensemble-dir "$CANDIDATE_ENSEMBLE_DIR" \
    --current-settlement "$CURRENT_SETTLEMENT" \
    --candidate-settlement "$CANDIDATE_SETTLEMENT" --output "$SUMMARY" \
    2>&1 | tee "$RESULT_DIR/summary.log"
}

package_results() {
  local paths=("$CONFIG" "$CHECKPOINT_ROOT" "$RESULT_DIR" "$CANDIDATE_PRED_DIR" \
    "$CANDIDATE_ENSEMBLE_DIR" "$CANDIDATE_PROFIT_DIR" "$CANDIDATE_SETTLEMENT" "$SUMMARY")
  find "${paths[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  tar -czf "$ARCHIVE" "${paths[@]}" "$MANIFEST"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  start_gpu_monitor
  for market in CAISO ERCOT NYISO PJM; do
    for seed in 0 1 2; do evaluate_da "$market" "$seed"; done
  done
  record_status settlement starting
  settle_and_summarize
  record_status settlement complete
  stop_gpu_monitor
  record_status package starting
  package_results
  record_status package complete
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
