#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
RESULT_ROOT="data/results/historical_dlinear_router_inputs_v1"
CHECKPOINT_ROOT="data/checkpoints/historical_dlinear_router_inputs_v1"
PROFIT_ROOT="data/results/historical_transformer_dlinear_profit_v1"
STATUS="$RESULT_ROOT/status.tsv"
mkdir -p "$RESULT_ROOT" "$CHECKPOINT_ROOT" "$PROFIT_ROOT"

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$STATUS"
}

config_for() {
  case "$1" in
    da) printf '%s' 'configs/decision_aware/allmarket_final_da_mae_v1.yaml' ;;
    rt_at_da) printf '%s' 'configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml' ;;
    rt) printf '%s' 'configs/decision_aware/allmarket_final_rt_mae_v1.yaml' ;;
    *) return 2 ;;
  esac
}

"$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print({"gpu": torch.cuda.get_device_name(0), "locked_2025_read": False})
PY

for phase in inner outer; do
  for task in da rt_at_da; do
    config="$(config_for "$task")"
    checkpoint="$CHECKPOINT_ROOT/${task}_${phase}_seed0.pt"
    prediction="$RESULT_ROOT/ercot_${task}_${phase}.npz"
    if [[ -s "$checkpoint" && -s "${checkpoint%.pt}.json" ]]; then
      record_status "train_${phase}_${task}" already_complete
    else
      record_status "train_${phase}_${task}" starting
      "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
        --mode train --phase "$phase" --config "$config" --task "$task" \
        --checkpoint "$checkpoint" --device cuda \
        2>&1 | tee "$RESULT_ROOT/train_${phase}_${task}.log"
      record_status "train_${phase}_${task}" complete
    fi
    if [[ -s "$prediction" && -s "${prediction%.npz}.json" ]]; then
      record_status "evaluate_${phase}_${task}" already_complete
    else
      record_status "evaluate_${phase}_${task}" starting
      "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
        --mode evaluate --phase "$phase" --config "$config" --task "$task" \
        --checkpoint "$checkpoint" --market ERCOT --output "$prediction" --device cuda \
        2>&1 | tee "$RESULT_ROOT/evaluate_${phase}_${task}.log"
      record_status "evaluate_${phase}_${task}" complete
    fi
  done

  if [[ "$phase" == inner ]]; then
    source_root="data/results/universal_lomo_ercot_profit_predictions_v1"
    selection_flag="--selection-allowed"
  else
    source_root="data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1"
    selection_flag="--no-selection-allowed"
  fi
  "$PYTHON_BIN" scripts/decision_aware/prepare_historical_baseline_profit_archives.py \
    --family dlinear --phase "$phase" \
    --source-da "$RESULT_ROOT/ercot_da_${phase}.npz" \
    --source-rt-at-da "$RESULT_ROOT/ercot_rt_at_da_${phase}.npz" \
    --source-rt "$source_root/universal_rt_${phase}_seed0.npz" \
    --output-dir "$PROFIT_ROOT"

  for task in da rt_at_da rt; do
    cp "$source_root/universal_${task}_${phase}_seed0.npz" \
      "$PROFIT_ROOT/universal_${task}_${phase}_seed0.npz"
  done
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml \
    --prediction-dir "$PROFIT_ROOT" --families universal,dlinear --phase "$phase" \
    --triplets "da_universal__rt_at_da_universal__rt_universal,da_universal__rt_at_da_dlinear__rt_universal,da_dlinear__rt_at_da_universal__rt_universal,da_dlinear__rt_at_da_dlinear__rt_universal" \
    "$selection_flag" --workers 8 --executor process \
    --status "historical_transformer_dlinear_${phase}" \
    --output "$RESULT_ROOT/cross_task_${phase}_mpc_v1.json" \
    2>&1 | tee "$RESULT_ROOT/settlement_${phase}.log"
  record_status "settlement_${phase}" complete
done

record_status pipeline complete
tar -czf "$RESULT_ROOT/completed_artifacts.tar.gz" \
  "$RESULT_ROOT" "$CHECKPOINT_ROOT" "$PROFIT_ROOT" \
  --exclude="$RESULT_ROOT/completed_artifacts.tar.gz"
