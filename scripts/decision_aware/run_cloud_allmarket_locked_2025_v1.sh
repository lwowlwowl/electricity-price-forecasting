#!/usr/bin/env bash
# 2025一次性锁定评价。只能在最终训练全部完成且预注册哈希一致后运行。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RESUME_LOCKED_RUN="${RESUME_LOCKED_RUN:-0}"
cd "$ROOT"

RESULT_DIR="data/results/cloud_allmarket_locked_2025_v1"
PRED_DIR="data/results/allmarket_locked_2025_predictions_v1"
ENSEMBLE_DIR="data/results/allmarket_locked_2025_ensemble_v1"
PROFIT_DIR="data/results/allmarket_locked_2025_ercot_profit_v1"
SETTLEMENT="data/results/allmarket_locked_2025_ercot_continuous_mpc_v1.json"
PREREG="data/results/allmarket_final_2025_preregistration_v1/protocol.json"
OPEN_MARKER="data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json"
ARCHIVE="data/results/allmarket_locked_2025_completed_20261017.tar.gz"
MANIFEST="data/results/allmarket_locked_2025_completed_20261017.manifest.sha256"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
mkdir -p "$RESULT_DIR" "$PRED_DIR" "$ENSEMBLE_DIR" "$PROFIT_DIR"
export PYTHONUNBUFFERED=1

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

config_for() {
  case "$1" in
    da) echo "configs/decision_aware/allmarket_final_da_mae_v1.yaml" ;;
    rt_at_da) echo "configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml" ;;
    rt) echo "configs/decision_aware/allmarket_final_rt_mae_v1.yaml" ;;
  esac
}

checkpoint_for() {
  local task="$1" seed="$2"
  case "$task" in
    da) echo "data/checkpoints/allmarket_final_da_mae_v1/da_final_seed${seed}/best.pt" ;;
    rt_at_da) echo "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1/rt_at_da_final_seed${seed}/best.pt" ;;
    rt) echo "data/checkpoints/allmarket_final_rt_mae_v1/rt_final_seed${seed}/best.pt" ;;
  esac
}

preflight() {
  test -x "$PYTHON_BIN"
  "$PYTHON_BIN" scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
  for task in da rt_at_da rt; do
    for seed in 0 1 2; do test -s "$(checkpoint_for "$task" "$seed")"; done
  done
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_allmarket_final_2025_pipeline.py
  if [[ -e "$OPEN_MARKER" && "$RESUME_LOCKED_RUN" != "1" ]]; then
    echo "2025已被打开；仅允许以RESUME_LOCKED_RUN=1恢复同一冻结协议" >&2
    return 3
  fi
}

mark_opened_once() {
  if [[ -e "$OPEN_MARKER" ]]; then return; fi
  "$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
path = Path('data/results/allmarket_final_2025_preregistration_v1/protocol.json')
marker = path.with_name('LOCKED_2025_OPENED.json')
marker.write_text(json.dumps({
    'opened_at_utc': datetime.now(timezone.utc).isoformat(),
    'protocol_sha256': sha256(path.read_bytes()).hexdigest(),
    'selection_allowed': False,
    'model_changes_after_opening_allowed': False,
}, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

evaluate_one() {
  local market="$1" task="$2" seed="$3" checkpoint output label
  checkpoint="$(checkpoint_for "$task" "$seed")"
  output="$PRED_DIR/${market,,}_${task}_seed${seed}.npz"
  label="${market}_${task}_seed${seed}"
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "$label" prediction_already_complete
    return
  fi
  record_status "$label" prediction_starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
    --checkpoint "$checkpoint" --task "$task" --holdout-market "$market" \
    --evaluation-stage locked_2025 --output "$output" --device cuda --batch-size 64 \
    2>&1 | tee "$RESULT_DIR/eval_${market,,}_${task}_seed${seed}.log"
  test -s "$output" && test -s "${output%.npz}.json"
  record_status "$label" prediction_complete
}

settle() {
  "$PYTHON_BIN" scripts/decision_aware/prepare_allmarket_locked_2025.py \
    --input-dir "$PRED_DIR" --ensemble-dir "$ENSEMBLE_DIR" --profit-dir "$PROFIT_DIR"
  local triplets="da_seed0__rt_at_da_seed0__rt_seed0,da_seed1__rt_at_da_seed1__rt_seed1,da_seed2__rt_at_da_seed2__rt_seed2,da_ensemble__rt_at_da_ensemble__rt_ensemble"
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml \
    --prediction-dir "$PROFIT_DIR" --families seed0,seed1,seed2,ensemble \
    --phase final --triplets "$triplets" --no-selection-allowed \
    --workers 8 --executor process --status locked_2025_final_no_selection \
    --output "$SETTLEMENT" 2>&1 | tee "$RESULT_DIR/settlement.log"
}

package_results() {
  local paths=("$RESULT_DIR" "$PRED_DIR" "$ENSEMBLE_DIR" "$PROFIT_DIR" "$SETTLEMENT" "$OPEN_MARKER" "$PREREG")
  find "${paths[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  tar -czf "$ARCHIVE" "${paths[@]}" "$MANIFEST"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  mark_opened_once
  record_status locked_2025 opened
  for market in CAISO ERCOT NYISO PJM; do
    for task in da rt_at_da rt; do
      for seed in 0 1 2; do evaluate_one "$market" "$task" "$seed"; done
    done
  done
  record_status settlement starting
  settle
  record_status settlement complete
  record_status package starting
  package_results
  record_status package complete
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
