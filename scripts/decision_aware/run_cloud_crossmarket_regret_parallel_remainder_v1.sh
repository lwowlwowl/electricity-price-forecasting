#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
RESULT_ROOT="data/results/crossmarket_regret_router_lomo_dlinear_v1"
CHECKPOINT_ROOT="data/checkpoints/crossmarket_regret_router_lomo_dlinear_v1"
STATUS="$RESULT_ROOT/status.tsv"
mkdir -p "$RESULT_ROOT" "$CHECKPOINT_ROOT"

config_for() {
  case "$1" in
    da) printf '%s' 'configs/decision_aware/allmarket_final_da_mae_v1.yaml' ;;
    rt_at_da) printf '%s' 'configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml' ;;
    *) return 2 ;;
  esac
}

run_pair() {
  local market="$1"
  local task="$2"
  local lower="${market,,}"
  local config checkpoint prediction
  config="$(config_for "$task")"
  checkpoint="$CHECKPOINT_ROOT/${lower}_${task}_inner_seed0.pt"
  prediction="$RESULT_ROOT/${lower}_${task}_inner.npz"
  if [[ ! -s "$checkpoint" || ! -s "${checkpoint%.pt}.json" ]]; then
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "parallel_train_${lower}_${task}" starting >> "$STATUS"
    "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
      --mode train --phase inner --config "$config" --task "$task" \
      --holdout-market "$market" --checkpoint "$checkpoint" --device cuda \
      > "$RESULT_ROOT/parallel_train_${lower}_${task}.log" 2>&1
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "parallel_train_${lower}_${task}" complete >> "$STATUS"
  fi
  if [[ ! -s "$prediction" || ! -s "${prediction%.npz}.json" ]]; then
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "parallel_evaluate_${lower}_${task}" starting >> "$STATUS"
    "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
      --mode evaluate --phase inner --config "$config" --task "$task" \
      --holdout-market "$market" --checkpoint "$checkpoint" --market "$market" \
      --output "$prediction" --device cuda \
      > "$RESULT_ROOT/parallel_evaluate_${lower}_${task}.log" 2>&1
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "parallel_evaluate_${lower}_${task}" complete >> "$STATUS"
  fi
}

pids=()
for market in ERCOT NYISO PJM; do
  for task in da rt_at_da; do
    run_pair "$market" "$task" &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" parallel_remainder complete >> "$STATUS"
