#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
RESULT_ROOT="data/results/crossmarket_regret_router_lomo_dlinear_v1"
CHECKPOINT_ROOT="data/checkpoints/crossmarket_regret_router_lomo_dlinear_v1"
STATUS="$RESULT_ROOT/status.tsv"
mkdir -p "$RESULT_ROOT" "$CHECKPOINT_ROOT"

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$STATUS"
}

config_for() {
  case "$1" in
    da) printf '%s' 'configs/decision_aware/allmarket_final_da_mae_v1.yaml' ;;
    rt_at_da) printf '%s' 'configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml' ;;
    *) return 2 ;;
  esac
}

"$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print({"gpu": torch.cuda.get_device_name(0), "locked_2025_read": False})
PY

for market in CAISO ERCOT NYISO PJM; do
  lower="${market,,}"
  for task in da rt_at_da; do
    config="$(config_for "$task")"
    checkpoint="$CHECKPOINT_ROOT/${lower}_${task}_inner_seed0.pt"
    prediction="$RESULT_ROOT/${lower}_${task}_inner.npz"
    if [[ -s "$checkpoint" && -s "${checkpoint%.pt}.json" ]]; then
      record_status "train_${lower}_${task}" already_complete
    else
      record_status "train_${lower}_${task}" starting
      "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
        --mode train --phase inner --config "$config" --task "$task" \
        --holdout-market "$market" --checkpoint "$checkpoint" --device cuda \
        2>&1 | tee "$RESULT_ROOT/train_${lower}_${task}.log"
      record_status "train_${lower}_${task}" complete
    fi
    if [[ -s "$prediction" && -s "${prediction%.npz}.json" ]]; then
      record_status "evaluate_${lower}_${task}" already_complete
    else
      record_status "evaluate_${lower}_${task}" starting
      "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
        --mode evaluate --phase inner --config "$config" --task "$task" \
        --holdout-market "$market" --checkpoint "$checkpoint" --market "$market" \
        --output "$prediction" --device cuda \
        2>&1 | tee "$RESULT_ROOT/evaluate_${lower}_${task}.log"
      record_status "evaluate_${lower}_${task}" complete
    fi
  done
done

record_status pipeline complete
tar -czf "$RESULT_ROOT/completed_artifacts.tar.gz" \
  "$RESULT_ROOT" "$CHECKPOINT_ROOT" \
  --exclude="$RESULT_ROOT/completed_artifacts.tar.gz"
