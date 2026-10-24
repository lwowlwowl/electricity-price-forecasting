#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
CHECKPOINT_ROOT="data/checkpoints/historical_dlinear_router_inputs_v1"
DLINEAR_ROOT="data/results/crossmarket_history_assisted_dlinear_v1"
INPUT_ROOT="data/results/crossmarket_history_assisted_profit_inputs_v1"
BASE_ROOT="data/results/crossmarket_history_assisted_base_settlement_v1"
CANDIDATE_ROOT="data/results/crossmarket_history_assisted_candidates_v1"
SETTLEMENT_ROOT="data/results/crossmarket_history_assisted_settlement_v1"
LOG_ROOT="data/results/crossmarket_history_assisted_pipeline_v1"
STATUS="$LOG_ROOT/status.tsv"
mkdir -p "$DLINEAR_ROOT" "$BASE_ROOT" "$SETTLEMENT_ROOT" "$LOG_ROOT"

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

for task in da rt_at_da; do
  [[ -s "$CHECKPOINT_ROOT/${task}_inner_seed0.pt" ]] || {
    echo "missing checkpoint: $CHECKPOINT_ROOT/${task}_inner_seed0.pt" >&2
    exit 2
  }
done

# Prediction export is independent by market/task and does not update model weights.
pids=()
for market in CAISO ERCOT NYISO PJM; do
  lower="${market,,}"
  for task in da rt_at_da; do
    (
      output="$DLINEAR_ROOT/${lower}_${task}_inner.npz"
      if [[ ! -s "$output" || ! -s "${output%.npz}.json" ]]; then
        record_status "evaluate_${lower}_${task}" starting
        "$PYTHON_BIN" scripts/decision_aware/run_allmarket_historical_dlinear.py \
          --mode evaluate --phase inner --config "$(config_for "$task")" --task "$task" \
          --checkpoint "$CHECKPOINT_ROOT/${task}_inner_seed0.pt" \
          --market "$market" --output "$output" --device cuda \
          > "$LOG_ROOT/evaluate_${lower}_${task}.log" 2>&1
        record_status "evaluate_${lower}_${task}" complete
      fi
    ) &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

record_status prepare_inputs starting
"$PYTHON_BIN" scripts/decision_aware/prepare_crossmarket_regret_router_v1.py \
  --dlinear-root "$DLINEAR_ROOT" --output-root "$INPUT_ROOT" \
  > "$LOG_ROOT/prepare_inputs.log" 2>&1
record_status prepare_inputs complete

base_triplets="da_transformer__rt_at_da_transformer__rt_transformer,da_static_hybrid__rt_at_da_static_hybrid__rt_static_hybrid"
for market in CAISO ERCOT NYISO PJM; do
  lower="${market,,}"
  record_status "base_settlement_${lower}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml \
    --prediction-dir "$INPUT_ROOT/$lower" --families transformer,static_hybrid \
    --phase inner --triplets "$base_triplets" --selection-allowed \
    --workers 24 --executor process --status "history_assisted_${lower}_2023_base" \
    --output "$BASE_ROOT/$lower.json" > "$LOG_ROOT/base_${lower}.log" 2>&1
  record_status "base_settlement_${lower}" complete
done

record_status build_routes starting
"$PYTHON_BIN" scripts/decision_aware/build_crossmarket_regret_router_v1.py \
  --input-root "$INPUT_ROOT" --settlement-root "$BASE_ROOT" --output-root "$CANDIDATE_ROOT" \
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
    --prediction-dir "$CANDIDATE_ROOT/$lower" --families "$families" \
    --phase inner --triplets "$triplets" --selection-allowed \
    --workers 32 --executor process --status "history_assisted_${lower}_2023_router" \
    --output "$SETTLEMENT_ROOT/$lower.json" > "$LOG_ROOT/candidate_${lower}.log" 2>&1
  record_status "candidate_settlement_${lower}" complete
done

record_status summarize starting
"$PYTHON_BIN" scripts/decision_aware/summarize_crossmarket_regret_router_v1.py \
  --candidate-root "$CANDIDATE_ROOT" --settlement-root "$SETTLEMENT_ROOT" \
  --history-assisted \
  --output data/results/crossmarket_history_assisted_summary_v1.json \
  > "$LOG_ROOT/summarize.log" 2>&1
record_status summarize complete

tar -czf "$LOG_ROOT/completed_artifacts.tar.gz" \
  --exclude="$LOG_ROOT/completed_artifacts.tar.gz" \
  "$DLINEAR_ROOT" "$INPUT_ROOT" "$BASE_ROOT" "$CANDIDATE_ROOT" "$SETTLEMENT_ROOT" \
  data/results/crossmarket_history_assisted_summary_v1.json "$LOG_ROOT"
record_status pipeline complete
