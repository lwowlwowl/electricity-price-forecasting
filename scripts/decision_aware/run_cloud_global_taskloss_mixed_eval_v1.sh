#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
cd "$ROOT"
test -x "$PYTHON_BIN"
test -s data/results/global_taskloss_candidate_manifest_v1.json
test -s data/results/global_f0_f1_stable_v2_candidate_manifest.json

"$PYTHON_BIN" scripts/decision_aware/prepare_global_taskloss_mixed_manifest.py
"$PYTHON_BIN" scripts/decision_aware/export_global_f0_f1_predictions.py \
  --config configs/decision_aware/global_ercot_taskloss_da_mae_v1.yaml \
  --candidate-manifest data/results/global_taskloss_mixed_manifest_v1.json \
  --families new0,new1,stable0 --tasks da,rt_at_da,rt --phase inner \
  --device cuda --batch-size 32 \
  --output-dir data/results/global_taskloss_mixed_predictions_v1
"$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
  --config configs/decision_aware/global_ercot_taskloss_da_mae_v1.yaml \
  --prediction-dir data/results/global_taskloss_mixed_predictions_v1 \
  --families new0,new1,stable0 \
  --include-baselines seasonal,xgboost_robust \
  --output data/results/global_taskloss_mixed_continuous_mpc_v1.json
"$PYTHON_BIN" scripts/decision_aware/summarize_global_taskloss_mixed_v1.py
