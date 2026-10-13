#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
SEED="${SEED:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
TRAIN_RT_ANCHOR="${TRAIN_RT_ANCHOR:-0}"
RESULT_DIR="data/results/cloud_global_lag168_v1"

cd "$ROOT"
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
  local run_dir="data/checkpoints/${label}/${task}_inner_seed${SEED}"
  local log="$RESULT_DIR/${task}_inner_seed${SEED}.log"
  local -a resume_args=()

  if [[ -s "$run_dir/summary.json" ]]; then
    record_status "$task" "already_complete"
    return 0
  fi
  if [[ -s "$run_dir/latest.pt" ]]; then
    resume_args=(--resume)
    record_status "$task" "resuming"
  else
    record_status "$task" "starting"
  fi
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$config" --task "$task" --phase inner --seed "$SEED" \
    --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"
  test -s "$run_dir/summary.json"
  test -s "$run_dir/candidate_manifest.json"
  record_status "$task" "complete"
}

{
  record_status "preflight" "starting"
  preflight
  record_status "preflight" "complete"
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi

  run_one \
    global_ercot_lag168_da_mae_v1 \
    configs/decision_aware/global_ercot_lag168_da_mae_v1.yaml \
    da
  run_one \
    global_ercot_lag168_rt_at_da_pseudohuber_v1 \
    configs/decision_aware/global_ercot_lag168_rt_at_da_pseudohuber_v1.yaml \
    rt_at_da
  # 2023全年无训练诊断中，RT lag168锚点MAE=50.72，明显弱于现有Global RT。
  # 默认不烧GPU；只有显式设置TRAIN_RT_ANCHOR=1才运行该支线。
  if [[ "$TRAIN_RT_ANCHOR" == "1" ]]; then
    run_one \
      global_ercot_lag168_rt_mae_v1 \
      configs/decision_aware/global_ercot_lag168_rt_mae_v1.yaml \
      rt
  else
    record_status "rt" "skipped_by_frozen_diagnostic"
  fi
  record_status "all" "complete"
} 2>&1 | tee -a "$RUN_LOG"
