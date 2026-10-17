#!/usr/bin/env bash
# ERCOT留出折的DA单因素尾部价格加权；不读取2024或2025。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

CONFIG="configs/decision_aware/universal_lomo_holdout_ercot_da_tailprice_v1.yaml"
CHECKPOINT_ROOT="data/checkpoints/universal_lomo_holdout_ercot_da_tailprice_v1"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/da_inner_seed0"
PREDICTION_DIR="data/results/universal_lomo_ercot_tailprice_da_predictions_v1"
PROFIT_DIR="data/results/universal_lomo_ercot_tailprice_da_profit_predictions_v1"
SUMMARY_DIR="data/results/universal_lomo_ercot_tailprice_da_summary_v1"
RESULT_DIR="data/results/cloud_universal_ercot_tailprice_da_v1"
SETTLEMENT="data/results/universal_lomo_ercot_tailprice_da_continuous_mpc_v1.json"
ARCHIVE="data/results/universal_lomo_ercot_tailprice_da_completed_20261016.tar.gz"
MANIFEST="data/results/universal_lomo_ercot_tailprice_da_completed_20261016.manifest.sha256"
mkdir -p "$PREDICTION_DIR" "$PROFIT_DIR" "$SUMMARY_DIR" "$RESULT_DIR"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
export PYTHONUNBUFFERED=1
GPU_MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

start_gpu_monitor() {
  nvidia-smi \
    --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
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

preflight() {
  test -x "$PYTHON_BIN"
  test -s "$CONFIG"
  test -s data/results/universal_lomo_da_v1/ercot_da_v1.npz
  test -s data/results/universal_lomo_rt_at_da_v1/ercot_rt_at_da_v1.npz
  test -s data/results/universal_lomo_rt_v1/ercot_rt_v1.npz
  for task in da rt_at_da rt; do
    test -s "data/results/global_f0_f1_stable_v2_predictions/c0_${task}_inner_seed0.npz"
  done
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0, 'src')
from decision_aware.config import PilotConfig

root = Path.cwd()
baseline_path = root / 'configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml'
candidate_path = root / 'configs/decision_aware/universal_lomo_holdout_ercot_da_tailprice_v1.yaml'
baseline_raw = yaml.safe_load(baseline_path.read_text(encoding='utf-8'))
candidate_raw = yaml.safe_load(candidate_path.read_text(encoding='utf-8'))
event_keys = {key for key in candidate_raw if key.startswith('global_event_')}
allowed = event_keys | {'global_use_critical_event_weighting', 'checkpoint_dir'}
differences = {
    key for key in set(baseline_raw) | set(candidate_raw)
    if baseline_raw.get(key) != candidate_raw.get(key)
}
assert differences <= allowed, sorted(differences - allowed)
cfg = PilotConfig.from_yaml(candidate_path)
assert tuple(cfg.global_training_markets) == ('CAISO', 'NYISO', 'PJM')
assert cfg.market == 'ERCOT' and cfg.seed == 0
assert cfg.global_use_entity_embeddings is False
assert cfg.global_num_nodes == 0 and cfg.global_num_markets == 0
assert cfg.da_fusion_mode == 'F1' and cfg.global_training_loss == 'mae'
assert cfg.global_use_critical_event_weighting is True
assert cfg.global_event_price_lower_quantile == 0.01
assert cfg.global_event_price_upper_quantile == 0.99
assert cfg.global_event_price_weight == 1.0
assert cfg.global_event_spread_weight == 0.0
assert cfg.global_event_sign_reversal_weight == 0.0
assert cfg.global_event_rt_first_step_weight == 0.0
assert cfg.global_event_max_weight == 2.0
assert torch.cuda.is_available(), 'CUDA unavailable'
print({
    'gpu': torch.cuda.get_device_name(0),
    'new_trainings': 1,
    'task': 'DA only',
    'test_2024_used_for_selection': False,
    'test_2025_read': False,
})
PY
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_critical_event_weighting.py \
    tests/test_universal_tailprice_da_pipeline.py
}

train_and_predict() {
  local best="$CHECKPOINT_DIR/best.pt"
  local summary="$CHECKPOINT_DIR/summary.json"
  local latest="$CHECKPOINT_DIR/latest.pt"
  local output="$PREDICTION_DIR/tail_price_da_da_inner_seed0.npz"
  local log="$RESULT_DIR/train_da.log"
  local -a resume_args=()
  if [[ -s "$summary" && -s "$best" ]]; then
    record_status da_tail_price training_already_complete
  else
    if [[ -s "$latest" ]]; then resume_args=(--resume); fi
    record_status da_tail_price training_starting
    "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
      --config "$CONFIG" --task da --phase inner --seed 0 \
      --device cuda "${resume_args[@]}" 2>&1 | tee "$log"
    test -s "$summary" && test -s "$best"
    record_status da_tail_price training_complete
  fi
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status da_tail_price holdout_already_complete
  else
    record_status da_tail_price holdout_starting
    "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
      --checkpoint "$best" --task da --holdout-market ERCOT \
      --output "$output" --device cuda --batch-size 64 2>&1 | tee -a "$log"
    test -s "$output" && test -s "${output%.npz}.json"
    record_status da_tail_price holdout_complete
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

settle_and_summarize() {
  local current_da="data/results/universal_lomo_da_v1/ercot_da_v1.npz"
  local current_rtda="data/results/universal_lomo_rt_at_da_v1/ercot_rt_at_da_v1.npz"
  local current_rt="data/results/universal_lomo_rt_v1/ercot_rt_v1.npz"
  local candidate_da="$PREDICTION_DIR/tail_price_da_da_inner_seed0.npz"
  prepare_family current "$current_da" "$current_rtda" "$current_rt"
  prepare_family tail_price_da "$candidate_da" "$current_rtda" "$current_rt"
  local triplets="da_current__rt_at_da_current__rt_current,da_tail_price_da__rt_at_da_current__rt_current,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
  record_status settlement starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PROFIT_DIR" --families current,tail_price_da \
    --phase inner --include-baselines xgboost_robust --triplets "$triplets" \
    --workers 8 --executor process \
    --status universal_ercot_da_tailprice_single_factor_seed0 \
    --output "$SETTLEMENT" 2>&1 | tee "$RESULT_DIR/settlement.log"
  record_status settlement complete
  record_status summary starting
  "$PYTHON_BIN" scripts/decision_aware/summarize_universal_ercot_tailprice_da.py \
    --config "$CONFIG" --candidate-da "$candidate_da" \
    --settlement "$SETTLEMENT" --output-dir "$SUMMARY_DIR" \
    2>&1 | tee "$RESULT_DIR/summary.log"
  record_status summary complete
}

package_results() {
  record_status package starting
  find "$RESULT_DIR" "$PREDICTION_DIR" "$PROFIT_DIR" "$SUMMARY_DIR" \
    "$CHECKPOINT_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  sha256sum "$CONFIG" "$SETTLEMENT" >> "$MANIFEST"
  tar -czf "$ARCHIVE" \
    "$CONFIG" "$RESULT_DIR" "$PREDICTION_DIR" "$PROFIT_DIR" \
    "$SUMMARY_DIR" "$CHECKPOINT_DIR" "$SETTLEMENT" "$MANIFEST"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
  record_status package complete
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  start_gpu_monitor
  train_and_predict
  settle_and_summarize
  stop_gpu_monitor
  package_results
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
