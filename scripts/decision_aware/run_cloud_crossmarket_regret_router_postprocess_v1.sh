#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
DLINEAR_ROOT="data/results/crossmarket_regret_router_lomo_dlinear_v1"
INPUT_ROOT="data/results/crossmarket_regret_router_profit_inputs_v1"
BASE_ROOT="data/results/crossmarket_regret_router_base_settlement_v1"
CANDIDATE_ROOT="data/results/crossmarket_regret_router_candidates_v1"
SETTLEMENT_ROOT="data/results/crossmarket_regret_router_settlement_v1"
LOG_ROOT="data/results/crossmarket_regret_router_pipeline_v1"
STATUS="$LOG_ROOT/status.tsv"
mkdir -p "$BASE_ROOT" "$SETTLEMENT_ROOT" "$LOG_ROOT"

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$STATUS"
}

# Training is launched separately so a transient SSH disconnect cannot cancel it.
record_status wait_for_lomo_dlinear starting
deadline=$((SECONDS + 3600))
while true; do
  complete=1
  for market in caiso ercot nyiso pjm; do
    for task in da rt_at_da; do
      [[ -s "$DLINEAR_ROOT/${market}_${task}_inner.npz" ]] || complete=0
      [[ -s "$DLINEAR_ROOT/${market}_${task}_inner.json" ]] || complete=0
    done
  done
  [[ "$complete" -eq 1 ]] && break
  if (( SECONDS >= deadline )); then
    record_status wait_for_lomo_dlinear timeout
    exit 3
  fi
  sleep 15
done
record_status wait_for_lomo_dlinear complete

record_status prepare_inputs starting
"$PYTHON_BIN" scripts/decision_aware/prepare_crossmarket_regret_router_v1.py \
  > "$LOG_ROOT/prepare_inputs.log" 2>&1
record_status prepare_inputs complete

base_triplets="da_transformer__rt_at_da_transformer__rt_transformer,da_static_hybrid__rt_at_da_static_hybrid__rt_static_hybrid"
for market in CAISO ERCOT NYISO PJM; do
  lower="${market,,}"
  record_status "base_settlement_${lower}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml \
    --prediction-dir "$INPUT_ROOT/$lower" \
    --families transformer,static_hybrid --phase inner \
    --triplets "$base_triplets" --selection-allowed \
    --workers 24 --executor process \
    --status "strict_lomo_${lower}_2023_base_settlement" \
    --output "$BASE_ROOT/$lower.json" \
    > "$LOG_ROOT/base_settlement_${lower}.log" 2>&1
  record_status "base_settlement_${lower}" complete
done

record_status build_routes starting
"$PYTHON_BIN" scripts/decision_aware/build_crossmarket_regret_router_v1.py \
  > "$LOG_ROOT/build_routes.log" 2>&1
record_status build_routes complete

families="transformer,static_hybrid,economic_regret_tree,mae_regret_tree,economic_no_tail_tree"
triplets=""
for family in transformer static_hybrid economic_regret_tree mae_regret_tree economic_no_tail_tree; do
  label="da_${family}__rt_at_da_${family}__rt_${family}"
  [[ -z "$triplets" ]] && triplets="$label" || triplets="$triplets,$label"
done
for market in CAISO ERCOT NYISO PJM; do
  lower="${market,,}"
  record_status "candidate_settlement_${lower}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml \
    --prediction-dir "$CANDIDATE_ROOT/$lower" \
    --families "$families" --phase inner --triplets "$triplets" \
    --selection-allowed --workers 32 --executor process \
    --status "strict_lomo_${lower}_2023_regret_router" \
    --output "$SETTLEMENT_ROOT/$lower.json" \
    > "$LOG_ROOT/candidate_settlement_${lower}.log" 2>&1
  record_status "candidate_settlement_${lower}" complete
done

record_status summarize starting
"$PYTHON_BIN" scripts/decision_aware/summarize_crossmarket_regret_router_v1.py \
  > "$LOG_ROOT/summarize.log" 2>&1
record_status summarize complete

record_status bundle starting
tar --exclude="$LOG_ROOT/completed_artifacts.tar.gz" -czf "$LOG_ROOT/completed_artifacts.tar.gz" \
  "$DLINEAR_ROOT" "$INPUT_ROOT" "$BASE_ROOT" "$CANDIDATE_ROOT" \
  "$SETTLEMENT_ROOT" data/results/crossmarket_regret_router_summary_v1.json \
  "$LOG_ROOT" data/checkpoints/crossmarket_regret_router_lomo_dlinear_v1
record_status pipeline complete
