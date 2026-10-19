#!/usr/bin/env bash
# 仅重训DA三seed：四市场2020—2024，尾部价格权重为冻结单因素改动。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

CONFIG="configs/decision_aware/allmarket_final_da_tailprice_v2.yaml"
BASE_CONFIG="configs/decision_aware/allmarket_final_da_mae_v1.yaml"
CHECKPOINT_ROOT="data/checkpoints/allmarket_final_da_tailprice_v2"
RESULT_DIR="data/results/cloud_allmarket_final_da_tailprice_v2"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
ARCHIVE="data/results/allmarket_final_da_tailprice_v2_completed.tar.gz"
MANIFEST="data/results/allmarket_final_da_tailprice_v2_completed.manifest.sha256"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1
GPU_MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }
plan_for() {
  echo "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed${1}/best.pt"
}
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

preflight() {
  test -x "$PYTHON_BIN" && test -s "$CONFIG" && test -s "$BASE_CONFIG"
  for seed in 0 1 2; do test -s "$(plan_for "$seed")"; done
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import torch
import yaml
base = yaml.safe_load(Path('configs/decision_aware/allmarket_final_da_mae_v1.yaml').read_text(encoding='utf-8'))
candidate = yaml.safe_load(Path('configs/decision_aware/allmarket_final_da_tailprice_v2.yaml').read_text(encoding='utf-8'))
allowed = {
    'checkpoint_dir', 'global_use_critical_event_weighting',
    'global_event_price_lower_quantile', 'global_event_price_upper_quantile',
    'global_event_spread_quantile', 'global_event_price_weight',
    'global_event_spread_weight', 'global_event_sign_reversal_weight',
    'global_event_rt_first_step_weight', 'global_event_max_weight',
}
differences = {key for key in set(base) | set(candidate) if base.get(key) != candidate.get(key)}
assert differences <= allowed, sorted(differences - allowed)
assert candidate['global_training_markets'] == ['CAISO', 'ERCOT', 'NYISO', 'PJM']
assert candidate['global_use_critical_event_weighting'] is True
assert candidate['global_event_price_lower_quantile'] == 0.01
assert candidate['global_event_price_upper_quantile'] == 0.99
assert candidate['global_event_price_weight'] == 1.0
assert candidate['global_event_max_weight'] == 2.0
assert candidate['global_event_spread_weight'] == 0.0
assert candidate['global_event_sign_reversal_weight'] == 0.0
assert candidate['global_event_rt_first_step_weight'] == 0.0
assert torch.cuda.is_available(), 'CUDA unavailable'
print({
    'gpu': torch.cuda.get_device_name(0),
    'new_trainings': 3,
    'task': 'DA only',
    'training_years': '2020-2024',
    'candidate_2025_read_during_training': False,
    'frozen_change': 'lowest/highest 1% DA training targets receive total weight 2',
})
PY
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_critical_event_weighting.py \
    tests/test_train_global_forecaster_selection.py
}

train_one() {
  local seed="$1" output_dir="$CHECKPOINT_ROOT/da_final_seed${seed}"
  local best="$output_dir/best.pt" latest="$output_dir/latest.pt"
  local log="$RESULT_DIR/train_da_seed${seed}.log"
  if [[ -s "$best" && -s "$output_dir/summary.json" ]]; then
    record_status "da_seed${seed}" training_already_complete
    return
  fi
  local -a resume_args=()
  if [[ -s "$latest" ]]; then resume_args=(--resume); fi
  record_status "da_seed${seed}" training_starting
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$CONFIG" --task da --phase final --seed "$seed" \
    --frozen-training-plan-from "$(plan_for "$seed")" --device cuda \
    "${resume_args[@]}" 2>&1 | tee "$log"
  test -s "$best" && test -s "$output_dir/summary.json"
  record_status "da_seed${seed}" training_complete
}

package_results() {
  record_status package starting
  find "$RESULT_DIR" "$CHECKPOINT_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  sha256sum "$CONFIG" "$BASE_CONFIG" >> "$MANIFEST"
  tar -czf "$ARCHIVE" "$CONFIG" "$RESULT_DIR" "$CHECKPOINT_ROOT" "$MANIFEST"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
  record_status package complete
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  start_gpu_monitor
  for seed in 0 1 2; do train_one "$seed"; done
  stop_gpu_monitor
  package_results
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
