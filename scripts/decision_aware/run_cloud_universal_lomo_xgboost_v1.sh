#!/usr/bin/env bash
# 四折、三任务的等训练年份 pooled LOMO XGBoost。只用CPU；不读取2025。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
XGB_THREADS="${XGB_THREADS:-1}"
OUTPUT_JOBS="${OUTPUT_JOBS:-8}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

cd "$ROOT"
RESULT_DIR="data/results/cloud_universal_lomo_xgboost_v1"
mkdir -p "$RESULT_DIR" data/results/universal_lomo_xgboost_v1
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

preflight() {
  test -x "$PYTHON_BIN"
  test -s scripts/decision_aware/run_universal_lomo_xgboost.py
  test -s data/results/multimarket_global_data_audit.json
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
import xgboost
import yaml

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from decision_aware.config import PilotConfig

markets = {"CAISO", "ERCOT", "NYISO", "PJM"}
for holdout in sorted(markets):
    for suffix in ("mae", "rt_at_da"):
        path = root / f"configs/decision_aware/universal_lomo_holdout_{holdout.lower()}_{suffix}_v1.yaml"
        cfg = PilotConfig.from_yaml(path)
        assert cfg.global_use_entity_embeddings is False
        assert cfg.global_normalization_mode == "context_robust"
        assert set(cfg.global_training_markets) == markets - {holdout}
        assert cfg.global_samples_per_market == 16350
print({"xgboost": xgboost.__version__, "device": "cpu", "test_2025": False})
PY
}

run_one() {
  local holdout_lower="$1"
  local holdout_upper="$2"
  local task="$3"
  local suffix="mae"
  if [[ "$task" == "rt_at_da" ]]; then suffix="rt_at_da"; fi
  local config="configs/decision_aware/universal_lomo_holdout_${holdout_lower}_${suffix}_v1.yaml"
  local output="data/results/universal_lomo_xgboost_v1/holdout_${holdout_lower}_${task}_inner_seed0.npz"
  local label="${holdout_lower}_${task}"
  local log="$RESULT_DIR/${label}.log"
  if [[ -s "$output" && -s "${output%.npz}.json" && -s "${output%.npz}.joblib" ]]; then
    record_status "$label" already_complete
    return
  fi
  record_status "$label" starting
  "$PYTHON_BIN" scripts/decision_aware/run_universal_lomo_xgboost.py \
    --config "$config" --task "$task" --holdout-market "$holdout_upper" \
    --evaluation-stage inner --seed 0 --n-estimators 180 \
    --device cpu --n-jobs "$XGB_THREADS" --output-jobs "$OUTPUT_JOBS" \
    --output "$output" 2>&1 | tee "$log"
  test -s "$output"
  test -s "${output%.npz}.json"
  test -s "${output%.npz}.joblib"
  record_status "$label" complete
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  for holdout_lower in caiso ercot nyiso pjm; do
    holdout_upper="${holdout_lower^^}"
    for task in da rt_at_da rt; do
      run_one "$holdout_lower" "$holdout_upper" "$task"
    done
  done
  record_status all complete
} 2>&1 | tee -a "$RUN_LOG"
