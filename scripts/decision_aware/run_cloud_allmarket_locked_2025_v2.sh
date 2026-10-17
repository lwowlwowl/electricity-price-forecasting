#!/usr/bin/env bash
# 全模型一次性2025锁定测试。所有训练产物齐全、两份预注册哈希通过后才打开2025。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
TIMESFM25_PYTHON="${TIMESFM25_PYTHON:-$ROOT/external/timesfm25_lora/.venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RESUME_LOCKED_RUN="${RESUME_LOCKED_RUN:-0}"
cd "$ROOT"

RESULT_DIR="data/results/cloud_allmarket_locked_2025_v2"
TRANSFORMER_DIR="data/results/allmarket_locked_2025_predictions_v1"
ENSEMBLE_DIR="data/results/allmarket_locked_2025_ensemble_v1"
BASELINE_PRED_DIR="data/results/allmarket_locked_2025_baselines_v1"
PROFIT_DIR="data/results/allmarket_locked_2025_all_profit_v2"
SETTLEMENT="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json"
SUMMARY="data/results/allmarket_locked_2025_summary_v2.json"
FINAL_AUDIT="$RESULT_DIR/final_artifact_audit.json"
PREREG="data/results/allmarket_final_2025_preregistration_v1/protocol.json"
BASELINE_PREREG="data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json"
OPEN_MARKER="data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json"
BASELINE_ROOT="data/checkpoints/allmarket_final_baselines_v1"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
ARCHIVE="data/results/allmarket_locked_2025_complete_v2_20261017.tar.gz"
MANIFEST="data/results/allmarket_locked_2025_complete_v2_20261017.manifest.sha256"
mkdir -p "$RESULT_DIR" "$TRANSFORMER_DIR" "$ENSEMBLE_DIR" "$BASELINE_PRED_DIR" "$PROFIT_DIR"
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

transformer_checkpoint() {
  local task="$1" seed="$2"
  case "$task" in
    da) echo "data/checkpoints/allmarket_final_da_mae_v1/da_final_seed${seed}/best.pt" ;;
    rt_at_da) echo "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1/rt_at_da_final_seed${seed}/best.pt" ;;
    rt) echo "data/checkpoints/allmarket_final_rt_mae_v1/rt_final_seed${seed}/best.pt" ;;
  esac
}

local_model_for() {
  echo "xgboost_absolute"
}

preflight() {
  test -x "$PYTHON_BIN" && test -x "$TIMESFM25_PYTHON"
  test -s "$TIMESFM25_MODEL_PATH/config.json"
  test -s "$TIMESFM25_MODEL_PATH/model.safetensors"
  local available_kb
  available_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] && (( available_kb >= 20 * 1024 * 1024 )) || {
    echo "可用磁盘不足20 GiB，拒绝打开2025" >&2
    return 4
  }
  "$PYTHON_BIN" scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
  "$PYTHON_BIN" scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py --verify
  "$PYTHON_BIN" scripts/decision_aware/audit_allmarket_final_baselines.py --require-complete
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print({'gpu': torch.cuda.get_device_name(0), 'locked_2025_selection_allowed': False})
PY
  for task in da rt_at_da rt; do
    for seed in 0 1 2; do test -s "$(transformer_checkpoint "$task" "$seed")"; done
    "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm3.py \
      --config "$(config_for "$task")" --task "$task" --holdout-market ERCOT \
      --evaluation-stage locked_2025 --preflight-only
    "$TIMESFM25_PYTHON" scripts/decision_aware/run_allmarket_final_timesfm25_lora.py \
      --mode locked_2025 --data-config "$(config_for "$task")" --task "$task" --market ERCOT \
      --adapter-dir "$BASELINE_ROOT/timesfm25/${task}_seed0" \
      --output "$BASELINE_PRED_DIR/preflight_${task}.npz" --preflight-only
  done
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_allmarket_final_2025_pipeline.py \
    tests/test_allmarket_final_baselines_pipeline.py
  if [[ -e "$OPEN_MARKER" && "$RESUME_LOCKED_RUN" != "1" ]]; then
    echo "2025已打开；仅允许RESUME_LOCKED_RUN=1恢复同一冻结运行" >&2; return 3
  fi
}

mark_opened_once() {
  if [[ -e "$OPEN_MARKER" ]]; then return; fi
  "$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
main = Path('data/results/allmarket_final_2025_preregistration_v1/protocol.json')
baseline = Path('data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json')
audit = Path('data/results/allmarket_final_baselines_v1/artifact_audit.json')
marker = main.with_name('LOCKED_2025_OPENED.json')
marker.write_text(json.dumps({
    'opened_at_utc': datetime.now(timezone.utc).isoformat(),
    'main_protocol_sha256': sha256(main.read_bytes()).hexdigest(),
    'baseline_protocol_sha256': sha256(baseline.read_bytes()).hexdigest(),
    'baseline_artifact_audit_sha256': sha256(audit.read_bytes()).hexdigest(),
    'selection_allowed': False,
    'model_changes_after_opening_allowed': False,
}, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

evaluate_transformer() {
  local market="$1" task="$2" seed="$3" output="$TRANSFORMER_DIR/${market,,}_${task}_seed${seed}.npz"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then record_status "transformer_${market}_${task}_seed${seed}" already_complete; return; fi
  record_status "transformer_${market}_${task}_seed${seed}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
    --checkpoint "$(transformer_checkpoint "$task" "$seed")" --task "$task" \
    --holdout-market "$market" --evaluation-stage locked_2025 --output "$output" \
    --device cuda --batch-size 64 2>&1 | tee "$RESULT_DIR/transformer_${market,,}_${task}_seed${seed}.log"
  record_status "transformer_${market}_${task}_seed${seed}" complete
}

evaluate_timesfm3() {
  local market="$1" task="$2" output="$BASELINE_PRED_DIR/timesfm3/${market,,}_${task}.npz"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then record_status "timesfm3_${market}_${task}" already_complete; return; fi
  mkdir -p "$(dirname "$output")"
  record_status "timesfm3_${market}_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm3.py \
    --config "$(config_for "$task")" --task "$task" --holdout-market "$market" \
    --evaluation-stage locked_2025 --output "$output" 2>&1 | tee "$RESULT_DIR/timesfm3_${market,,}_${task}.log"
  record_status "timesfm3_${market}_${task}" complete
}

evaluate_timesfm25() {
  local market="$1" task="$2" output="$BASELINE_PRED_DIR/timesfm25/${market,,}_${task}.npz"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then record_status "timesfm25_${market}_${task}" already_complete; return; fi
  mkdir -p "$(dirname "$output")"
  record_status "timesfm25_${market}_${task}" starting
  "$TIMESFM25_PYTHON" scripts/decision_aware/run_allmarket_final_timesfm25_lora.py \
    --mode locked_2025 --data-config "$(config_for "$task")" --task "$task" --market "$market" \
    --adapter-dir "$BASELINE_ROOT/timesfm25/${task}_seed0" --output "$output" \
    2>&1 | tee "$RESULT_DIR/timesfm25_${market,,}_${task}.log"
  record_status "timesfm25_${market}_${task}" complete
}

evaluate_pooled_xgboost() {
  local market="$1" task="$2" output="$BASELINE_PRED_DIR/pooled_xgboost/${market,,}_${task}.npz"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then record_status "pooled_xgboost_${market}_${task}" already_complete; return; fi
  mkdir -p "$(dirname "$output")"
  record_status "pooled_xgboost_${market}_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_allmarket_final_xgboost.py \
    --mode locked_2025 --config "$(config_for "$task")" --task "$task" --market "$market" \
    --checkpoint "$BASELINE_ROOT/pooled_xgboost/${task}_seed0.joblib" --output "$output" \
    2>&1 | tee "$RESULT_DIR/pooled_xgboost_${market,,}_${task}.log"
  record_status "pooled_xgboost_${market}_${task}" complete
}

evaluate_dlinear() {
  local market="$1" task="$2" output="$BASELINE_PRED_DIR/dlinear/${market,,}_${task}.npz"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then record_status "dlinear_${market}_${task}" already_complete; return; fi
  mkdir -p "$(dirname "$output")"
  record_status "dlinear_${market}_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_allmarket_final_dlinear.py \
    --mode locked_2025 --config "$(config_for "$task")" --task "$task" --market "$market" \
    --checkpoint "$BASELINE_ROOT/dlinear/${task}_seed0.pt" --output "$output" --device cuda \
    2>&1 | tee "$RESULT_DIR/dlinear_${market,,}_${task}.log"
  record_status "dlinear_${market}_${task}" complete
}

evaluate_local_baselines() {
  local market="$1" task="$2" models model
  model="$(local_model_for "$task")"
  if [[ "$task" == "rt" ]]; then models="$model,seasonal"; else models="$model,seasonal,lear"; fi
  record_status "local_baselines_${market}_${task}" starting
  "$PYTHON_BIN" scripts/decision_aware/run_global_baselines.py \
    --config "$(market_config_for "$market" "$task")" --market "$market" --task "$task" \
    --phase locked_2025 --models "$models" --seed 0 --workers 8 --evaluate-only \
    2>&1 | tee "$RESULT_DIR/local_baselines_${market,,}_${task}.log"
  record_status "local_baselines_${market}_${task}" complete
}

prepare_profit_family() {
  local family="$1" root="$2"
  "$PYTHON_BIN" scripts/decision_aware/prepare_locked_baseline_profit_archives.py \
    --family "$family" --source-da "$root/ercot_da.npz" \
    --source-rt-at-da "$root/ercot_rt_at_da.npz" --source-rt "$root/ercot_rt.npz" \
    --output-dir "$PROFIT_DIR"
}

prepare_and_settle() {
  "$PYTHON_BIN" scripts/decision_aware/prepare_allmarket_locked_2025.py \
    --input-dir "$TRANSFORMER_DIR" --ensemble-dir "$ENSEMBLE_DIR" --profit-dir "$PROFIT_DIR"
  prepare_profit_family timesfm3 "$BASELINE_PRED_DIR/timesfm3"
  prepare_profit_family timesfm25 "$BASELINE_PRED_DIR/timesfm25"
  prepare_profit_family pooled_xgboost "$BASELINE_PRED_DIR/pooled_xgboost"
  prepare_profit_family dlinear "$BASELINE_PRED_DIR/dlinear"
  local local_root="data/results/global_baselines_v1/rt_locked_2025"
  "$PYTHON_BIN" scripts/decision_aware/prepare_locked_baseline_profit_archives.py \
    --family per_node_xgboost \
    --source-da data/results/global_baselines_v1/da_locked_2025/xgboost_absolute_ercot_da_locked_2025.npz \
    --source-rt-at-da data/results/global_baselines_v1/rt_at_da_locked_2025/xgboost_absolute_ercot_rt_at_da_locked_2025.npz \
    --source-rt "$local_root/xgboost_absolute_ercot_rt_locked_2025.npz" --output-dir "$PROFIT_DIR"
  "$PYTHON_BIN" scripts/decision_aware/prepare_locked_baseline_profit_archives.py \
    --family seasonal \
    --source-da data/results/global_baselines_v1/da_locked_2025/seasonal_ercot_da_locked_2025.npz \
    --source-rt-at-da data/results/global_baselines_v1/rt_at_da_locked_2025/seasonal_ercot_rt_at_da_locked_2025.npz \
    --source-rt "$local_root/seasonal_ercot_rt_locked_2025.npz" --output-dir "$PROFIT_DIR"
  "$PYTHON_BIN" scripts/decision_aware/prepare_locked_baseline_profit_archives.py \
    --family oracle --perfect-oracle \
    --source-da "$BASELINE_PRED_DIR/pooled_xgboost/ercot_da.npz" \
    --source-rt-at-da "$BASELINE_PRED_DIR/pooled_xgboost/ercot_rt_at_da.npz" \
    --source-rt "$BASELINE_PRED_DIR/pooled_xgboost/ercot_rt.npz" --output-dir "$PROFIT_DIR"
  local families="seed0,seed1,seed2,ensemble,timesfm3,timesfm25,pooled_xgboost,per_node_xgboost,dlinear,seasonal,oracle"
  local triplets="da_seed0__rt_at_da_seed0__rt_seed0,da_seed1__rt_at_da_seed1__rt_seed1,da_seed2__rt_at_da_seed2__rt_seed2,da_ensemble__rt_at_da_ensemble__rt_ensemble,da_timesfm3__rt_at_da_timesfm3__rt_timesfm3,da_timesfm25__rt_at_da_timesfm25__rt_timesfm25,da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost,da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost,da_dlinear__rt_at_da_dlinear__rt_dlinear,da_seasonal__rt_at_da_seasonal__rt_seasonal,da_oracle__rt_at_da_oracle__rt_oracle"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml \
    --prediction-dir "$PROFIT_DIR" --families "$families" --phase final \
    --triplets "$triplets" --no-selection-allowed --workers 8 --executor process \
    --status locked_2025_final_no_selection --output "$SETTLEMENT" \
    2>&1 | tee "$RESULT_DIR/settlement.log"
  "$PYTHON_BIN" scripts/decision_aware/summarize_allmarket_locked_2025.py \
    --transformer-dir "$TRANSFORMER_DIR" --ensemble-dir "$ENSEMBLE_DIR" \
    --baseline-dir "$BASELINE_PRED_DIR" \
    --settlement "$SETTLEMENT" --output "$SUMMARY" \
    2>&1 | tee "$RESULT_DIR/summary.log"
}

package_results() {
  "$PYTHON_BIN" scripts/decision_aware/audit_allmarket_locked_2025_outputs.py \
    --transformer-dir "$TRANSFORMER_DIR" --ensemble-dir "$ENSEMBLE_DIR" \
    --baseline-dir "$BASELINE_PRED_DIR" --profit-dir "$PROFIT_DIR" \
    --settlement "$SETTLEMENT" --summary "$SUMMARY" --marker "$OPEN_MARKER" \
    --output "$FINAL_AUDIT"
  test -s "$FINAL_AUDIT"
  local paths=("$RESULT_DIR" "$TRANSFORMER_DIR" "$ENSEMBLE_DIR" "$BASELINE_PRED_DIR" "$PROFIT_DIR" "$SETTLEMENT" "$SUMMARY" "$OPEN_MARKER" "$PREREG" "$BASELINE_PREREG")
  find "${paths[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  tar -czf "$ARCHIVE" "${paths[@]}" "$MANIFEST"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
}

{
  record_status preflight starting; preflight; record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  mark_opened_once; record_status locked_2025 opened
  start_gpu_monitor
  for market in CAISO ERCOT NYISO PJM; do
    for task in da rt_at_da rt; do
      for seed in 0 1 2; do evaluate_transformer "$market" "$task" "$seed"; done
      evaluate_timesfm3 "$market" "$task"
      evaluate_timesfm25 "$market" "$task"
      evaluate_pooled_xgboost "$market" "$task"
      evaluate_dlinear "$market" "$task"
      evaluate_local_baselines "$market" "$task"
    done
  done
  record_status settlement starting; prepare_and_settle; record_status settlement complete
  stop_gpu_monitor
  record_status package starting; package_results; record_status package complete
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
