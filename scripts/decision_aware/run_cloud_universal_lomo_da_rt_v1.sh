#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
SEED="${SEED:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

cd "$ROOT"
RESULT_DIR="data/results/cloud_universal_lomo_da_rt_v1"
mkdir -p "$RESULT_DIR" data/results/universal_lomo_da_v1 data/results/universal_lomo_rt_v1
export PYTHONUNBUFFERED=1
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

preflight() {
  test -x "$PYTHON_BIN"
  test -s data/results/multimarket_global_data_audit.json
  test -s scripts/decision_aware/evaluate_universal_holdout.py
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import torch
import yaml

root = Path.cwd()
markets = {"CAISO", "ERCOT", "NYISO", "PJM"}
for holdout in sorted(markets):
    path = root / f"configs/decision_aware/universal_lomo_holdout_{holdout.lower()}_mae_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["global_use_entity_embeddings"] is False
    assert config["global_num_nodes"] == 0
    assert config["global_num_markets"] == 0
    assert config["global_normalization_mode"] == "context_robust"
    assert config["global_training_loss"] == "mae"
    assert set(config["global_training_markets"]) == markets - {holdout}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0)})
PY
}

run_fold() {
  local task="$1"
  local holdout_lower="$2"
  local holdout_upper="$3"
  local label="universal_lomo_holdout_${holdout_lower}_mae_v1"
  local config="configs/decision_aware/${label}.yaml"
  local run_name="${task}_inner_seed${SEED}"
  local checkpoint_dir="data/checkpoints/${label}/${run_name}"
  local summary="${checkpoint_dir}/summary.json"
  local latest="${checkpoint_dir}/latest.pt"
  local best="${checkpoint_dir}/best.pt"
  local log="${RESULT_DIR}/${label}_${task}.log"
  local output="data/results/universal_lomo_${task}_v1/${holdout_lower}_${task}_v1.npz"
  local -a resume_args=()
  if [[ -s "$summary" && -s "$best" ]]; then
    record_status "${label}_${task}" "training_already_complete"
  else
    if [[ -s "$latest" ]]; then
      resume_args=(--resume)
      record_status "${label}_${task}" "training_resuming"
    else
      record_status "${label}_${task}" "training_starting"
    fi
    "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
      --config "$config" --task "$task" --phase inner --seed "$SEED" \
      --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"
    test -s "$summary"
    test -s "$best"
    record_status "${label}_${task}" "training_complete"
  fi
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "${label}_${task}" "holdout_already_complete"
  else
    record_status "${label}_${task}" "holdout_starting"
    "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
      --checkpoint "$best" --task "$task" \
      --holdout-market "$holdout_upper" --output "$output" \
      --device cuda --batch-size 64 2>&1 | tee -a "$log"
    test -s "$output"
    test -s "${output%.npz}.json"
    record_status "${label}_${task}" "holdout_complete"
  fi
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi

  for task in da rt; do
    run_fold "$task" caiso CAISO
    run_fold "$task" ercot ERCOT
    run_fold "$task" nyiso NYISO
    run_fold "$task" pjm PJM
  done
  record_status all universal_lomo_da_rt_complete
} 2>&1 | tee -a "$RUN_LOG"
