#!/usr/bin/env python
"""在最终2025结果打包前审计所有预测、结算、汇总与冻结门禁产物。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da", "rt")
SEEDS = (0, 1, 2)
BASELINE_SYSTEMS = ("timesfm3", "timesfm25", "pooled_xgboost", "dlinear")
PROFIT_FAMILIES = (
    "seed0", "seed1", "seed2", "ensemble", "timesfm3", "timesfm25",
    "pooled_xgboost", "per_node_xgboost", "dlinear", "seasonal", "oracle",
)


def _valid_npz(path: Path, required: tuple[str, ...]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"{path}缺少字段{missing}")


def _json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-dir", default="data/results/allmarket_locked_2025_predictions_v1")
    parser.add_argument("--ensemble-dir", default="data/results/allmarket_locked_2025_ensemble_v1")
    parser.add_argument("--baseline-dir", default="data/results/allmarket_locked_2025_baselines_v1")
    parser.add_argument("--profit-dir", default="data/results/allmarket_locked_2025_all_profit_v2")
    parser.add_argument("--settlement", default="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json")
    parser.add_argument("--summary", default="data/results/allmarket_locked_2025_summary_v2.json")
    parser.add_argument("--marker", default="data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json")
    parser.add_argument("--baseline-audit", default="data/results/allmarket_final_baselines_v1/artifact_audit.json")
    parser.add_argument("--output", default="data/results/cloud_allmarket_locked_2025_v2/final_artifact_audit.json")
    args = parser.parse_args()
    transformer_dir = ROOT / args.transformer_dir
    ensemble_dir = ROOT / args.ensemble_dir
    baseline_dir = ROOT / args.baseline_dir
    profit_dir = ROOT / args.profit_dir
    checked: list[str] = []

    required_prediction = ("nodes", "dates", "prediction", "target", "metadata")
    for market in MARKETS:
        lower = market.lower()
        for task in TASKS:
            for seed in SEEDS:
                path = transformer_dir / f"{lower}_{task}_seed{seed}.npz"
                _valid_npz(path, required_prediction); checked.append(path.as_posix())
            path = ensemble_dir / f"{lower}_{task}_ensemble.npz"
            _valid_npz(path, required_prediction); checked.append(path.as_posix())
            for system in BASELINE_SYSTEMS:
                path = baseline_dir / system / f"{lower}_{task}.npz"
                _valid_npz(path, required_prediction); checked.append(path.as_posix())

    for family in PROFIT_FAMILIES:
        for task in TASKS:
            path = profit_dir / f"{family}_{task}_final_seed0.npz"
            _valid_npz(path, ("nodes", "dates", "prediction", "target_rt", "metadata"))
            checked.append(path.as_posix())

    summary = _json(ROOT / args.summary)
    if summary.get("year") != 2025 or summary.get("selection_allowed") is not False:
        raise ValueError("最终汇总没有锁定为2025 report-only")
    if set(summary.get("markets", {})) != set(MARKETS):
        raise ValueError("最终汇总市场不完整")
    settlement = _json(ROOT / args.settlement)
    if len(settlement.get("systems", {})) != len(PROFIT_FAMILIES):
        raise ValueError("ERCOT结算系统数量不完整")
    marker = _json(ROOT / args.marker)
    for key in ("main_protocol_sha256", "baseline_protocol_sha256", "baseline_artifact_audit_sha256"):
        if not marker.get(key):
            raise ValueError(f"2025开启标记缺少{key}")
    baseline_audit = _json(ROOT / args.baseline_audit)
    if baseline_audit.get("complete") is not True or baseline_audit.get("missing_count") != 0:
        raise ValueError("最终基线产物审计未通过")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "allmarket_locked_2025_final_artifact_audit_v1",
        "complete": True,
        "selection_allowed": False,
        "checked_prediction_and_profit_archives": len(checked),
        "settlement_systems": len(settlement["systems"]),
        "markets": list(MARKETS),
        "tasks": list(TASKS),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "locked_2025_final_artifact_audit_complete", **report}))


if __name__ == "__main__":
    main()
