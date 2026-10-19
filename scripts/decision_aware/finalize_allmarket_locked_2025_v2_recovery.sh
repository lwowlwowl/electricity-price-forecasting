#!/usr/bin/env bash
# Package already-completed locked-2025 results after recovery-only summary generation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"

RESULT_DIR="data/results/cloud_allmarket_locked_2025_v2"
TRANSFORMER_DIR="data/results/allmarket_locked_2025_predictions_v1"
ENSEMBLE_DIR="data/results/allmarket_locked_2025_ensemble_v1"
BASELINE_PRED_DIR="data/results/allmarket_locked_2025_baselines_v1"
PROFIT_DIR="data/results/allmarket_locked_2025_all_profit_v2"
SETTLEMENT="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json"
SUMMARY="data/results/allmarket_locked_2025_summary_v2.json"
FINAL_AUDIT="$RESULT_DIR/final_artifact_audit.json"
PREREG="data/results/allmarket_final_2025_preregistration_v1/protocol.json"
BASELINE_PREREG="data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json"
OPEN_MARKER="data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json"
MANIFEST="data/results/allmarket_locked_2025_complete_v2_20261017.manifest.sha256"
ARCHIVE="data/results/allmarket_locked_2025_complete_v2_20261017.tar.gz"
STATUS_FILE="$RESULT_DIR/status.tsv"

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }

"$PYTHON_BIN" - "$FINAL_AUDIT" "$SUMMARY" "$SETTLEMENT" <<'PY'
import json
import sys
from pathlib import Path

audit, summary, settlement = map(Path, sys.argv[1:])
for path in (audit, summary, settlement):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing final artifact: {path}")
payload = json.loads(audit.read_text(encoding="utf-8"))
if payload.get("complete") is not True or payload.get("selection_allowed") is not False:
    raise SystemExit("final artifact audit is not complete/no-selection")
PY

record_status package starting
paths=(
  "$RESULT_DIR" "$TRANSFORMER_DIR" "$ENSEMBLE_DIR" "$BASELINE_PRED_DIR"
  "$PROFIT_DIR" "$SETTLEMENT" "$SUMMARY" "$OPEN_MARKER" "$PREREG"
  "$BASELINE_PREREG"
)
find "${paths[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
tar -czf "$ARCHIVE" "${paths[@]}" "$MANIFEST"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
record_status package complete
record_status all complete
