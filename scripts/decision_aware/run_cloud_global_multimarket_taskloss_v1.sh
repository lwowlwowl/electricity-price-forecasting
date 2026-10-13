#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 复用已配置的AutoDL环境；不创建环境、不安装依赖。
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
SEED="${SEED:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

cd "$ROOT"
RESULT_DIR="data/results/cloud_global_multimarket_taskloss_v1"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

preflight() {
  test -x "$PYTHON_BIN"
  test -s data/results/multimarket_global_data_audit.json
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

root = Path.cwd()
audit = json.loads((root / "data/results/multimarket_global_data_audit.json").read_text(encoding="utf-8"))
verified = {}
for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
    record = audit["markets"][market]
    source = root / Path(str(record["path"]).replace("\\", "/"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise SystemExit(f"{market} SHA256 mismatch: {digest}")
    verified[market] = digest
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(json.dumps({
    "python": str(Path(torch.__file__).parents[1]),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "source_sha256": verified,
}, ensure_ascii=False))
PY
}

run_one() {
  local label="$1"
  local config="$2"
  local task="$3"
  local run_name="${task}_inner_seed${SEED}"
  local checkpoint_dir="data/checkpoints/${label}/${run_name}"
  local summary="${checkpoint_dir}/summary.json"
  local latest="${checkpoint_dir}/latest.pt"
  local log="${RESULT_DIR}/${label}.log"
  local -a resume_args=()
  if [[ -s "$summary" ]]; then record_status "$label" "already_complete"; return 0; fi
  if [[ -s "$latest" ]]; then resume_args=(--resume); record_status "$label" "resuming";
  else record_status "$label" "starting"; fi
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$config" --task "$task" --phase inner --seed "$SEED" \
    --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"
  test -s "$summary"
  test -s "${checkpoint_dir}/candidate_manifest.json"
  record_status "$label" "complete"
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi

  run_one global_multimarket_da_mae_v1 \
    configs/decision_aware/global_multimarket_da_mae_v1.yaml da
  run_one global_multimarket_rt_at_da_pseudohuber_v1 \
    configs/decision_aware/global_multimarket_rt_at_da_pseudohuber_v1.yaml rt_at_da
  run_one global_multimarket_rt_mae_v1 \
    configs/decision_aware/global_multimarket_rt_mae_v1.yaml rt

  # 只产生价格预测checkpoint。各ISO收益排名必须等各自结算合同冻结后单独运行。
  record_status all prediction_training_complete
} 2>&1 | tee -a "$RUN_LOG"
