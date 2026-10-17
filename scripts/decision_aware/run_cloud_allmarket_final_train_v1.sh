#!/usr/bin/env bash
# 四市场2020—2024最终训练。此脚本明确不读取或评价2025。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

RESULT_DIR="data/results/cloud_allmarket_final_train_v1"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
ARCHIVE="data/results/allmarket_final_train_completed_20261017.tar.gz"
MANIFEST="data/results/allmarket_final_train_completed_20261017.manifest.sha256"
PREREG="data/results/allmarket_final_2025_preregistration_v1/protocol.json"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1
GPU_MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

config_for() {
  case "$1" in
    da) echo "configs/decision_aware/allmarket_final_da_mae_v1.yaml" ;;
    rt_at_da) echo "configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml" ;;
    rt) echo "configs/decision_aware/allmarket_final_rt_mae_v1.yaml" ;;
    *) return 2 ;;
  esac
}

plan_for() {
  local task="$1" seed="$2" root
  if [[ "$task" == "rt_at_da" ]]; then
    root="data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1"
  else
    root="data/checkpoints/universal_lomo_holdout_ercot_mae_v1"
  fi
  echo "$root/${task}_inner_seed${seed}/best.pt"
}

checkpoint_root_for() {
  case "$1" in
    da) echo "data/checkpoints/allmarket_final_da_mae_v1" ;;
    rt_at_da) echo "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1" ;;
    rt) echo "data/checkpoints/allmarket_final_rt_mae_v1" ;;
  esac
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

preflight() {
  test -x "$PYTHON_BIN"
  test -s "$PREREG"
  "$PYTHON_BIN" scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
  for task in da rt_at_da rt; do
    test -s "$(config_for "$task")"
    for seed in 0 1 2; do test -s "$(plan_for "$task" "$seed")"; done
  done
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print({
    'gpu': torch.cuda.get_device_name(0),
    'new_trainings': 9,
    'training_years': '2020-2024',
    'locked_2025_read': False,
    'tail_weighting_promoted': False,
})
PY
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_allmarket_final_2025_pipeline.py \
    tests/test_train_global_forecaster_selection.py \
    tests/test_decision_aware_global_data.py
}

train_one() {
  local task="$1" seed="$2" config plan checkpoint_root output_dir best latest log label
  config="$(config_for "$task")"
  plan="$(plan_for "$task" "$seed")"
  checkpoint_root="$(checkpoint_root_for "$task")"
  output_dir="$checkpoint_root/${task}_final_seed${seed}"
  best="$output_dir/best.pt"
  latest="$output_dir/latest.pt"
  log="$RESULT_DIR/train_${task}_seed${seed}.log"
  label="${task}_seed${seed}"
  if [[ -s "$best" && -s "$output_dir/summary.json" ]]; then
    record_status "$label" training_already_complete
    return
  fi
  local -a resume_args=()
  if [[ -s "$latest" ]]; then resume_args=(--resume); fi
  record_status "$label" training_starting
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$config" --task "$task" --phase final --seed "$seed" \
    --frozen-training-plan-from "$plan" --device cuda "${resume_args[@]}" \
    2>&1 | tee "$log"
  test -s "$best" && test -s "$output_dir/summary.json"
  record_status "$label" training_complete
}

package_results() {
  record_status package starting
  local paths=(
    "$RESULT_DIR"
    "$PREREG"
    "${PREREG%.json}.sha256"
    "data/checkpoints/allmarket_final_da_mae_v1"
    "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1"
    "data/checkpoints/allmarket_final_rt_mae_v1"
  )
  find "${paths[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  tar -czf "$ARCHIVE" "${paths[@]}" "$MANIFEST"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
  record_status package complete
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  start_gpu_monitor
  for task in da rt_at_da rt; do
    for seed in 0 1 2; do train_one "$task" "$seed"; done
  done
  stop_gpu_monitor
  package_results
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
