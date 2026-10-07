#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${1:-/root/autodl-tmp/epf-venv/bin/python}"
CONFIG="configs/decision_aware/joint_decision_aware_v3_penalty_active_kappa57.yaml"

cd "$ROOT"
mkdir -p data/results data/checkpoints
export PYTHONUNBUFFERED=1

run_seed() {
  local seed="$1"
  local stem="joint_decision_aware_v3_penalty_active_kappa57_seed${seed}"
  local result="data/results/${stem}.json"
  local checkpoint_dir="data/checkpoints/${stem}"
  local best="${checkpoint_dir}/best_validation_revenue.pt"
  local latest="${checkpoint_dir}/latest.pt"
  local log="data/results/${stem}.cloud.log"

  if [[ -s "$result" ]]; then
    echo "[$(date --iso-8601=seconds)] seed ${seed}: result already exists; skip"
    return 0
  fi

  mkdir -p "$checkpoint_dir"
  local -a resume_args=()
  if [[ -s "$latest" ]]; then
    resume_args=(--resume-from "$latest")
    echo "[$(date --iso-8601=seconds)] seed ${seed}: resume from ${latest}"
  else
    echo "[$(date --iso-8601=seconds)] seed ${seed}: start from source checkpoints"
  fi

  "$PYTHON_BIN" scripts/decision_aware/train_joint_decision_aware.py \
    --config "$CONFIG" \
    --mode decision_aware \
    --seed "$seed" \
    --skip-report-only \
    --output "$result" \
    --checkpoint "$best" \
    --latest-checkpoint "$latest" \
    "${resume_args[@]}" \
    2>&1 | tee -a "$log"

  test -s "$result"
  echo "[$(date --iso-8601=seconds)] seed ${seed}: complete"
}

echo "[$(date --iso-8601=seconds)] cloud seed runner started"
run_seed 0
run_seed 1
run_seed 2
echo "[$(date --iso-8601=seconds)] all seeds complete"
