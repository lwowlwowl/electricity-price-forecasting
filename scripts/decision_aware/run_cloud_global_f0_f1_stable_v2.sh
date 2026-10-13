#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
SEED="${SEED:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

cd "$ROOT"
RESULT_DIR="data/results/cloud_global_f0_f1_stable_v2"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"

timestamp() { date --iso-8601=seconds; }

record_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"
}

preflight() {
  test -x "$PYTHON_BIN"
  test -s data/markets/ERCOT/ERCOT_统一小时数据_20200101_20260601.parquet
  test -s data/results/multimarket_global_data_audit.json
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

root = Path.cwd()
audit = json.loads((root / "data/results/multimarket_global_data_audit.json").read_text(encoding="utf-8"))
record = audit["markets"]["ERCOT"]
source = root / Path(str(record["path"]).replace("\\", "/"))
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if digest != record["sha256"]:
    raise SystemExit(f"ERCOT SHA256 mismatch: {digest}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "ercot_sha256": digest,
}, ensure_ascii=False))
PY
}

run_one() {
  local label="$1"
  local config="$2"
  local task="$3"
  local run_name="${label}_${task}_inner_seed${SEED}"
  local checkpoint_dir="data/checkpoints/${label}/${task}_inner_seed${SEED}"
  local summary="${checkpoint_dir}/summary.json"
  local latest="${checkpoint_dir}/latest.pt"
  local log="${RESULT_DIR}/${run_name}.log"
  local -a resume_args=()

  if [[ -s "$summary" ]]; then
    record_status "$run_name" "already_complete"
    return 0
  fi
  if [[ -s "$latest" ]]; then
    resume_args=(--resume)
    record_status "$run_name" "resuming"
  else
    record_status "$run_name" "starting"
  fi

  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$config" --task "$task" --phase inner --seed "$SEED" \
    --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"

  test -s "$summary"
  test -s "${checkpoint_dir}/candidate_manifest.json"
  record_status "$run_name" "complete"
}

{
  echo "[$(timestamp)] preflight started"
  preflight
  echo "[$(timestamp)] preflight complete"
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi

  run_one global_ercot_huber_f0_stable_v2 configs/decision_aware/global_ercot_huber_f0_stable_v2.yaml da
  run_one global_ercot_huber_f0_stable_v2 configs/decision_aware/global_ercot_huber_f0_stable_v2.yaml rt_at_da
  run_one global_ercot_huber_f0_stable_v2 configs/decision_aware/global_ercot_huber_f0_stable_v2.yaml rt
  run_one global_ercot_huber_f1_stable_v2 configs/decision_aware/global_ercot_huber_f1_stable_v2.yaml da
  run_one global_ercot_huber_f1_stable_v2 configs/decision_aware/global_ercot_huber_f1_stable_v2.yaml rt_at_da
  run_one global_ercot_huber_f1_stable_v2 configs/decision_aware/global_ercot_huber_f1_stable_v2.yaml rt
  echo "[$(timestamp)] all stable F0/F1 runs complete"
} 2>&1 | tee -a "$RUN_LOG"
