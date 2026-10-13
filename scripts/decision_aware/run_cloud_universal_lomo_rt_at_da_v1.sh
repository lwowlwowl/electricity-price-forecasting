#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
SEED="${SEED:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

cd "$ROOT"
RESULT_DIR="data/results/cloud_universal_lomo_rt_at_da_v1"
HOLDOUT_DIR="data/results/universal_lomo_rt_at_da_v1"
mkdir -p "$RESULT_DIR" "$HOLDOUT_DIR"
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
for holdout in ("caiso", "ercot", "nyiso", "pjm"):
    path = root / f"configs/decision_aware/universal_lomo_holdout_{holdout}_rt_at_da_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["global_use_entity_embeddings"] is False
    assert config["global_num_nodes"] == 0
    assert config["global_num_markets"] == 0
    assert config["global_normalization_mode"] == "context_robust"
    assert holdout.upper() not in config["global_training_markets"]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0)})
PY
}

run_fold() {
  local holdout_lower="$1"
  local holdout_upper="$2"
  local label="universal_lomo_holdout_${holdout_lower}_rt_at_da_v1"
  local config="configs/decision_aware/${label}.yaml"
  local run_name="rt_at_da_inner_seed${SEED}"
  local checkpoint_dir="data/checkpoints/${label}/${run_name}"
  local summary="${checkpoint_dir}/summary.json"
  local latest="${checkpoint_dir}/latest.pt"
  local best="${checkpoint_dir}/best.pt"
  local log="${RESULT_DIR}/${label}.log"
  local output="${HOLDOUT_DIR}/${holdout_lower}_rt_at_da_v1.npz"
  local -a resume_args=()
  if [[ -s "$summary" && -s "$best" ]]; then
    record_status "$label" "training_already_complete"
  else
    if [[ -s "$latest" ]]; then
      resume_args=(--resume)
      record_status "$label" "training_resuming"
    else
      record_status "$label" "training_starting"
    fi
    "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
      --config "$config" --task rt_at_da --phase inner --seed "$SEED" \
      --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"
    test -s "$summary"
    test -s "$best"
    record_status "$label" "training_complete"
  fi
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "$label" "holdout_already_complete"
  else
    record_status "$label" "holdout_starting"
    "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
      --checkpoint "$best" --task rt_at_da \
      --holdout-market "$holdout_upper" --output "$output" \
      --device cuda --batch-size 64 2>&1 | tee -a "$log"
    test -s "$output"
    test -s "${output%.npz}.json"
    record_status "$label" "holdout_complete"
  fi
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi

  run_fold caiso CAISO
  run_fold ercot ERCOT
  run_fold nyiso NYISO
  run_fold pjm PJM
  record_status all universal_lomo_rt_at_da_complete
} 2>&1 | tee -a "$RUN_LOG"
