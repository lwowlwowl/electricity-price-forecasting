#!/usr/bin/env bash
# Official TimesFM-2.5 Transformers/PEFT LoRA: smoke, 4-fold seed-0, ERCOT settlement.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/external/timesfm25_lora/.venv/bin/python}"
RESULT_DIR="data/results/cloud_universal_lomo_timesfm25_lora_v1"
PREDICTION_DIR="data/results/universal_lomo_timesfm25_lora_v1"
CHECKPOINT_DIR="data/checkpoints/universal_lomo_timesfm25_lora_v1"
PROFIT_DIR="data/results/universal_lomo_timesfm25_lora_profit_predictions_v1"
SMOKE_DIR="data/results/timesfm25_lora_source_smoke_v1"
mkdir -p "$RESULT_DIR" "$PREDICTION_DIR" "$CHECKPOINT_DIR" "$PROFIT_DIR" "$SMOKE_DIR"

exec 9>"$RESULT_DIR/run.lock"
flock -n 9 || { echo 'TimesFM-2.5 LoRA pipeline already running'; exit 1; }

export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$ROOT/hf_cache/hub"
export TRANSFORMERS_CACHE="$ROOT/hf_cache/hub"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TIMESFM25_MODEL_PATH="$ROOT/hf_cache/hub/models--google--timesfm-2.5-200m-transformers/snapshots/5a9806b9b291fad9233b5249d88263f1846304d3"

STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "$STATUS_FILE"
}
trap 'record_status pipeline "failed_line_${LINENO}"' ERR

config_for() {
  local market="$1" task="$2" suffix="mae"
  if [[ "$task" == "rt_at_da" ]]; then suffix="rt_at_da"; fi
  printf 'configs/decision_aware/universal_lomo_holdout_%s_%s_v1.yaml' "$market" "$suffix"
}

run_one() {
  local market_lower="$1" market_upper="$2" task="$3" seed="${4:-0}"
  local label output adapter
  label="${market_lower}_${task}_seed${seed}"
  output="$PREDICTION_DIR/holdout_${market_lower}_${task}_inner_seed${seed}.npz"
  adapter="$CHECKPOINT_DIR/holdout_${market_lower}_${task}_inner_seed${seed}"
  if [[ -s "$output" && -s "${output%.npz}.json" && -s "$adapter/adapter_model.safetensors" ]]; then
    record_status "$label" already_complete
    return
  fi
  record_status "$label" starting
  "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm25_lora.py \
    --data-config "$(config_for "$market_lower" "$task")" \
    --task "$task" --holdout-market "$market_upper" --seed "$seed" \
    --adapter-dir "$adapter" --output "$output" \
    2>&1 | tee "$RESULT_DIR/${label}.log"
  test -s "$output" && test -s "${output%.npz}.json"
  test -s "$adapter/adapter_model.safetensors"
  record_status "$label" complete
}

{
  test -x "$PYTHON_BIN"
  test -s "$TIMESFM25_MODEL_PATH/config.json"
  test -s "$TIMESFM25_MODEL_PATH/model.safetensors"
  test -s data/results/cloud_timesfm25_lora_setup_v1/runtime.json
  record_status preflight starting
  "$PYTHON_BIN" scripts/decision_aware/preflight_timesfm25_lora.py --require-runtime
  for task in da rt_at_da rt; do
    "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm25_lora.py \
      --data-config "$(config_for ercot "$task")" --task "$task" \
      --holdout-market ERCOT --seed 0 \
      --adapter-dir "$CHECKPOINT_DIR/preflight_${task}" \
      --output "$PREDICTION_DIR/preflight_${task}.npz" --preflight-only
  done
  record_status preflight complete
  if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then exit 0; fi

  record_status smoke starting
  "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_timesfm25_lora.py \
    --data-config "$(config_for ercot da)" --task da \
    --holdout-market ERCOT --seed 0 --smoke --force \
    --adapter-dir "$SMOKE_DIR/adapter" --output "$SMOKE_DIR/ercot_da_smoke.npz" \
    2>&1 | tee "$RESULT_DIR/smoke.log"
  test -s "$SMOKE_DIR/ercot_da_smoke.npz"
  test -s "$SMOKE_DIR/adapter/adapter_model.safetensors"
  record_status smoke complete
  if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    record_status all smoke_only_complete
    exit 0
  fi

  # Phase 1 is the minimum four-fold seed-0 comparison. Extra seeds remain optional.
  for market_lower in caiso ercot nyiso pjm; do
    market_upper="${market_lower^^}"
    for task in da rt_at_da rt; do
      run_one "$market_lower" "$market_upper" "$task" 0
    done
  done

  record_status settlement preparing
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase inner --family timesfm25_lora --seed 0 --output-root "$PROFIT_DIR" \
    --source-da "$PREDICTION_DIR/holdout_ercot_da_inner_seed0.npz" \
    --source-rt-at-da "$PREDICTION_DIR/holdout_ercot_rt_at_da_inner_seed0.npz" \
    --source-rt "$PREDICTION_DIR/holdout_ercot_rt_inner_seed0.npz"
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase inner --family universal --seed 0 --output-root "$PROFIT_DIR" \
    --source-da data/results/universal_lomo_da_v1/ercot_da_v1.npz \
    --source-rt-at-da data/results/universal_lomo_rt_at_da_v1/ercot_rt_at_da_v1.npz \
    --source-rt data/results/universal_lomo_rt_v1/ercot_rt_v1.npz
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase inner --family timesfm3_frozen --seed 0 --output-root "$PROFIT_DIR" \
    --source-da data/results/universal_lomo_timesfm3_v1/ercot_da_v1.npz \
    --source-rt-at-da data/results/universal_lomo_timesfm3_v1/ercot_rt_at_da_v1.npz \
    --source-rt data/results/universal_lomo_timesfm3_v1/ercot_rt_v1.npz
  triplets="da_timesfm25_lora__rt_at_da_timesfm25_lora__rt_timesfm25_lora,da_timesfm3_frozen__rt_at_da_timesfm3_frozen__rt_timesfm3_frozen,da_universal__rt_at_da_universal__rt_universal,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$PROFIT_DIR" --families timesfm25_lora,timesfm3_frozen,universal \
    --phase inner --include-baselines xgboost_robust --triplets "$triplets" \
    --workers 8 --executor process --status timesfm25_lora_same_contract \
    --output data/results/universal_lomo_timesfm25_lora_ercot_continuous_mpc_v1.json \
    2>&1 | tee "$RESULT_DIR/settlement.log"
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
