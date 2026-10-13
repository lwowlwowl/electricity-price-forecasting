#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3.12}"
CONFIG="configs/decision_aware/global_ercot_huber_f1_stable_v2.yaml"
RESULT_DIR="data/results/cloud_global_da_multiseed"

cd "$ROOT"
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1

if ! "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available()'; then
  echo "CUDA is not available" >&2
  exit 1
fi

for seed in 1 2; do
  run_dir="data/checkpoints/global_ercot_huber_f1_stable_v2/da_inner_seed${seed}"
  log="$RESULT_DIR/da_inner_seed${seed}.log"
  if [[ -s "$run_dir/summary.json" ]]; then
    printf '%s\tseed%s\talready_complete\n' "$(date --iso-8601=seconds)" "$seed"
    continue
  fi
  resume_args=()
  if [[ -s "$run_dir/latest.pt" ]]; then
    resume_args=(--resume)
  fi
  printf '%s\tseed%s\tstarting\n' "$(date --iso-8601=seconds)" "$seed"
  "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
    --config "$CONFIG" --task da --phase inner --seed "$seed" \
    --device cuda "${resume_args[@]}" 2>&1 | tee -a "$log"
  test -s "$run_dir/summary.json"
  test -s "$run_dir/candidate_manifest.json"
  printf '%s\tseed%s\tcomplete\n' "$(date --iso-8601=seconds)" "$seed"
done
