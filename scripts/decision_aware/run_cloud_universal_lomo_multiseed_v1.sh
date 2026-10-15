#!/usr/bin/env bash
# 默认补完整ERCOT三任务的seed 1、2，并自动完成同seed/集成盈利结算。
# 完整运行可显式设置 HOLDOUTS="caiso ercot nyiso pjm" TASKS="da rt_at_da rt"。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
HOLDOUTS="${HOLDOUTS:-ercot}"
TASKS="${TASKS:-da rt_at_da rt}"
SEEDS="${SEEDS:-1 2}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

RESULT_DIR="data/results/cloud_universal_lomo_multiseed_v1"
PREDICTION_DIR="data/results/universal_lomo_multiseed_v1"
ENSEMBLE_DIR="$PREDICTION_DIR/ensemble"
PROFIT_DIR="data/results/universal_lomo_ercot_multiseed_profit_predictions_v1"
mkdir -p "$RESULT_DIR" "$PREDICTION_DIR" "$ENSEMBLE_DIR" "$PROFIT_DIR"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
export PYTHONUNBUFFERED=1

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

preflight() {
  test -x "$PYTHON_BIN"
  test -s scripts/decision_aware/summarize_universal_lomo_multiseed.py
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print({'gpu': torch.cuda.get_device_name(0), 'test_2025': False})
PY
}

run_one() {
  local holdout="$1"
  local task="$2"
  local seed="$3"
  local suffix="mae"
  if [[ "$task" == "rt_at_da" ]]; then suffix="rt_at_da"; fi
  local label="universal_lomo_holdout_${holdout}_${suffix}_v1"
  local config="configs/decision_aware/${label}.yaml"
  local checkpoint_dir="data/checkpoints/${label}/${task}_inner_seed${seed}"
  local best="$checkpoint_dir/best.pt"
  local summary="$checkpoint_dir/summary.json"
  local latest="$checkpoint_dir/latest.pt"
  local output="$PREDICTION_DIR/${holdout}_${task}_inner_seed${seed}.npz"
  local log="$RESULT_DIR/${holdout}_${task}_seed${seed}.log"
  local -a resume_args=()
  if [[ -s "$summary" && -s "$best" ]]; then
    record_status "${holdout}_${task}_seed${seed}" training_already_complete
  else
    if [[ -s "$latest" ]]; then resume_args=(--resume); fi
    record_status "${holdout}_${task}_seed${seed}" training_starting
    "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
      --config "$config" --task "$task" --phase inner --seed "$seed" \
      --device cuda "${resume_args[@]}" 2>&1 | tee "$log"
    test -s "$summary" && test -s "$best"
    record_status "${holdout}_${task}_seed${seed}" training_complete
  fi
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "${holdout}_${task}_seed${seed}" holdout_already_complete
  else
    record_status "${holdout}_${task}_seed${seed}" holdout_starting
    "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
      --checkpoint "$best" --task "$task" --holdout-market "${holdout^^}" \
      --output "$output" --device cuda --batch-size 64 2>&1 | tee -a "$log"
    record_status "${holdout}_${task}_seed${seed}" holdout_complete
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

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  for holdout in $HOLDOUTS; do
    for task in $TASKS; do
      for seed in $SEEDS; do
        run_one "$holdout" "$task" "$seed"
      done
    done
  done
  holdouts_csv="${HOLDOUTS// /,}"
  tasks_csv="${TASKS// /,}"
  "$PYTHON_BIN" scripts/decision_aware/summarize_universal_lomo_multiseed.py \
    --holdouts "$holdouts_csv" --tasks "$tasks_csv" --seeds 0,1,2 \
    --ensemble-output-root "$ENSEMBLE_DIR" \
    --output "$PREDICTION_DIR/summary.json" 2>&1 | tee "$RESULT_DIR/summary.log"
  if [[ "$holdouts_csv" == "ercot" && "$tasks_csv" == "da,rt_at_da,rt" ]]; then
    current_da=data/results/universal_lomo_da_v1/ercot_da_v1.npz
    current_rtda=data/results/universal_lomo_rt_at_da_v1/ercot_rt_at_da_v1.npz
    current_rt=data/results/universal_lomo_rt_v1/ercot_rt_v1.npz
    prepare_family seed0 "$current_da" "$current_rtda" "$current_rt"
    for seed in 1 2; do
      prepare_family "seed${seed}" \
        "$PREDICTION_DIR/ercot_da_inner_seed${seed}.npz" \
        "$PREDICTION_DIR/ercot_rt_at_da_inner_seed${seed}.npz" \
        "$PREDICTION_DIR/ercot_rt_inner_seed${seed}.npz"
    done
    prepare_family ensemble \
      "$ENSEMBLE_DIR/ercot_da_inner_ensemble.npz" \
      "$ENSEMBLE_DIR/ercot_rt_at_da_inner_ensemble.npz" \
      "$ENSEMBLE_DIR/ercot_rt_inner_ensemble.npz"
    triplets="da_seed0__rt_at_da_seed0__rt_seed0,da_seed1__rt_at_da_seed1__rt_seed1,da_seed2__rt_at_da_seed2__rt_seed2,da_ensemble__rt_at_da_ensemble__rt_ensemble,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
    "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
      --config configs/decision_aware/global_ercot_huber_v1.yaml \
      --prediction-dir "$PROFIT_DIR" --families seed0,seed1,seed2,ensemble \
      --phase inner --include-baselines xgboost_robust --triplets "$triplets" \
      --workers 8 --executor process --status universal_ercot_three_task_multiseed \
      --output data/results/universal_lomo_ercot_multiseed_continuous_mpc_v1.json \
      2>&1 | tee "$RESULT_DIR/settlement.log"
  fi
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
