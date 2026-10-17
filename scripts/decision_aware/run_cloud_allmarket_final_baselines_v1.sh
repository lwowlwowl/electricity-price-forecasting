#!/usr/bin/env bash
# 2025打开前训练最终基线。只读2020—2024；不执行任何locked_2025评价。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
TIMESFM25_PYTHON="${TIMESFM25_PYTHON:-$ROOT/external/timesfm25_lora/.venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

RESULT_DIR="data/results/cloud_allmarket_final_baselines_v1"
BASELINE_ROOT="data/checkpoints/allmarket_final_baselines_v1"
PROTOCOL="configs/decision_aware/allmarket_final_baselines_v1.yaml"
BASELINE_PREREG="data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json"
ARCHIVE="data/results/allmarket_final_baselines_completed_20261017.tar.gz"
MANIFEST="data/results/allmarket_final_baselines_completed_20261017.manifest.sha256"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
mkdir -p "$RESULT_DIR" "$BASELINE_ROOT/pooled_xgboost" "$BASELINE_ROOT/dlinear" "$BASELINE_ROOT/timesfm25"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME="$ROOT/hf_cache" HF_HUB_CACHE="$ROOT/hf_cache/hub" TRANSFORMERS_CACHE="$ROOT/hf_cache/hub"
export TIMESFM25_MODEL_PATH="${TIMESFM25_MODEL_PATH:-$ROOT/hf_cache/hub/models--google--timesfm-2.5-200m-transformers/snapshots/5a9806b9b291fad9233b5249d88263f1846304d3}"
GPU_MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }
trap 'record_status pipeline "failed_line_${LINENO}"' ERR

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

config_for() {
  case "$1" in
    da) echo "configs/decision_aware/allmarket_final_da_mae_v1.yaml" ;;
    rt_at_da) echo "configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml" ;;
    rt) echo "configs/decision_aware/allmarket_final_rt_mae_v1.yaml" ;;
    *) return 2 ;;
  esac
}

market_config_for() {
  local market="$1" task="$2"
  case "$market" in
    ERCOT)
      case "$task" in
        da) echo "configs/decision_aware/global_ercot_taskloss_da_mae_v1.yaml" ;;
        rt_at_da) echo "configs/decision_aware/global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml" ;;
        rt) echo "configs/decision_aware/global_ercot_taskloss_rt_mae_v1.yaml" ;;
      esac ;;
    CAISO) echo "configs/decision_aware/global_caiso_prediction_contract_v1.yaml" ;;
    NYISO) echo "configs/decision_aware/global_nyiso_prediction_contract_v1.yaml" ;;
    PJM) echo "configs/decision_aware/global_pjm_prediction_contract_v1.yaml" ;;
    *) return 2 ;;
  esac
}

local_model_for() {
  echo "xgboost_absolute"
}

preflight() {
  test -x "$PYTHON_BIN"
  test -x "$TIMESFM25_PYTHON"
  test -s "$PROTOCOL"
  test -s "$BASELINE_PREREG"
  test -s "$TIMESFM25_MODEL_PATH/config.json"
  test -s "$TIMESFM25_MODEL_PATH/model.safetensors"
  local available_kb
  available_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] && (( available_kb >= 20 * 1024 * 1024 )) || {
    echo "可用磁盘不足20 GiB，拒绝启动最终基线训练" >&2
    return 4
  }
  test ! -e data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json
  "$PYTHON_BIN" scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py --verify
  "$PYTHON_BIN" -m py_compile \
    scripts/decision_aware/run_allmarket_final_xgboost.py \
    scripts/decision_aware/run_allmarket_final_dlinear.py \
    scripts/decision_aware/run_global_baselines.py \
    scripts/decision_aware/audit_allmarket_final_baselines.py
  for task in da rt_at_da rt; do
    "$TIMESFM25_PYTHON" scripts/decision_aware/run_allmarket_final_timesfm25_lora.py \
      --mode train --data-config "$(config_for "$task")" --task "$task" \
      --adapter-dir "$BASELINE_ROOT/timesfm25/${task}_seed0" \
      --output "$RESULT_DIR/timesfm25_${task}_seed0.json" --preflight-only
  done
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print({'training_years': '2020-2024', 'locked_2025_read': False, 'gpu': torch.cuda.get_device_name(0)})
PY
}

train_timesfm25() {
  local task="$1" adapter="$BASELINE_ROOT/timesfm25/${task}_seed0" output="$RESULT_DIR/timesfm25_${task}_seed0.json"
  if [[ -s "$adapter/adapter_model.safetensors" && -s "$adapter/final_training_summary.json" ]]; then
    record_status "timesfm25_${task}" already_complete; return
  fi
  record_status "timesfm25_${task}" starting
  "$TIMESFM25_PYTHON" scripts/decision_aware/run_allmarket_final_timesfm25_lora.py \
    --mode train --data-config "$(config_for "$task")" --task "$task" \
    --adapter-dir "$adapter" --output "$output" --force 2>&1 | tee "$RESULT_DIR/timesfm25_${task}.log"
  test -s "$adapter/adapter_model.safetensors" && test -s "$adapter/final_training_summary.json"
  record_status "timesfm25_${task}" complete
}

train_pooled_xgboost() {
  local task="$1" checkpoint="$BASELINE_ROOT/pooled_xgboost/${task}_seed0.joblib"
  if [[ -s "$checkpoint" && -s "${checkpoint%.joblib}.json" ]]; then
    record_status "pooled_xgboost_${task}" already_complete; return
  fi
  record_status "pooled_xgboost_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_allmarket_final_xgboost.py \
    --mode train --config "$(config_for "$task")" --task "$task" \
    --checkpoint "$checkpoint" --seed 0 --device cpu --n-jobs 1 --output-jobs 1 --force \
    2>&1 | tee "$RESULT_DIR/pooled_xgboost_${task}.log"
  test -s "$checkpoint"
  record_status "pooled_xgboost_${task}" complete
}

train_dlinear() {
  local task="$1" checkpoint="$BASELINE_ROOT/dlinear/${task}_seed0.pt"
  if [[ -s "$checkpoint" && -s "${checkpoint%.pt}.json" ]]; then
    record_status "dlinear_${task}" already_complete; return
  fi
  record_status "dlinear_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_allmarket_final_dlinear.py \
    --mode train --config "$(config_for "$task")" --task "$task" \
    --checkpoint "$checkpoint" --device cuda --force 2>&1 | tee "$RESULT_DIR/dlinear_${task}.log"
  test -s "$checkpoint"
  record_status "dlinear_${task}" complete
}

train_local_xgboost() {
  local market="$1" task="$2" model config
  model="$(local_model_for "$task")"
  config="$(market_config_for "$market" "$task")"
  record_status "local_xgboost_${market}_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_global_baselines.py \
    --config "$config" --market "$market" --task "$task" --phase final \
    --models "$model" --seed 0 --workers 8 \
    2>&1 | tee "$RESULT_DIR/local_xgboost_${market,,}_${task}.log"
  record_status "local_xgboost_${market}_${task}" complete
}

package_results() {
  local paths=("$RESULT_DIR" "$BASELINE_ROOT" "$BASELINE_PREREG" "${BASELINE_PREREG%.json}.sha256" data/results/allmarket_final_baselines_v1/artifact_audit.json)
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
  for task in da rt_at_da rt; do train_timesfm25 "$task"; done
  for task in da rt_at_da rt; do train_pooled_xgboost "$task"; done
  for task in da rt_at_da rt; do train_dlinear "$task"; done
  for market in CAISO ERCOT NYISO PJM; do
    for task in da rt_at_da rt; do train_local_xgboost "$market" "$task"; done
  done
  stop_gpu_monitor
  "$PYTHON_BIN" scripts/decision_aware/audit_allmarket_final_baselines.py --require-complete
  record_status package starting; package_results; record_status package complete
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
