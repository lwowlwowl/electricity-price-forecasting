#!/usr/bin/env python
"""冻结/核验最终基线与统一2025入口的静态协议。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "configs/decision_aware/allmarket_final_baselines_v1.yaml",
    "src/decision_aware/dlinear.py",
    "src/decision_aware/baselines_global.py",
    "scripts/decision_aware/run_allmarket_final_xgboost.py",
    "scripts/decision_aware/run_allmarket_final_timesfm25_lora.py",
    "scripts/decision_aware/run_allmarket_final_dlinear.py",
    "scripts/decision_aware/run_global_baselines.py",
    "scripts/decision_aware/run_universal_lomo_timesfm3.py",
    "scripts/decision_aware/prepare_locked_baseline_profit_archives.py",
    "scripts/decision_aware/summarize_allmarket_locked_2025.py",
    "scripts/decision_aware/audit_allmarket_final_baselines.py",
    "scripts/decision_aware/audit_allmarket_locked_2025_outputs.py",
    "scripts/decision_aware/run_cloud_allmarket_final_baselines_v1.sh",
    "scripts/decision_aware/run_local_allmarket_final_xgboost_v1.ps1",
    "scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh",
)
PARENT = "data/results/allmarket_final_2025_preregistration_v1/protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> dict:
    config = yaml.safe_load((ROOT / FILES[0]).read_text(encoding="utf-8"))
    if config.get("selection_uses_2025") is not False:
        raise ValueError("最终基线协议必须禁止使用2025选模")
    if config.get("training_years") != [2020, 2021, 2022, 2023, 2024]:
        raise ValueError("最终基线训练期必须固定为2020—2024")
    if config.get("markets") != ["CAISO", "ERCOT", "NYISO", "PJM"]:
        raise ValueError("最终基线市场顺序不匹配")
    parent = ROOT / PARENT
    if not parent.is_file():
        raise FileNotFoundError(f"缺少主模型预注册: {parent}")
    return {
        "protocol_version": "allmarket_final_2025_baselines_v1",
        "frozen_at_local_date": "2026-10-17",
        "parent_protocol": PARENT,
        "parent_protocol_sha256": sha256(parent),
        "training_years": [2020, 2021, 2022, 2023, 2024],
        "locked_test_year": 2025,
        "selection_uses_2025": False,
        "baselines": {
            "timesfm3": "official_frozen_no_project_finetuning",
            "timesfm25": "all_market_lora_seed0_fixed_epochs",
            "pooled_xgboost": "all_market_no_id_fair_training_data",
            "per_node_xgboost": "target_market_node_specific_supervised_upper_baseline",
            "lear": "target_market_cutoff_safe_daily_refit_da_and_rt_at_da",
            "dlinear": "all_market_price_history_only_seed0_fixed_epochs",
            "seasonal": "weekly_lag_168",
            "oracle": "same_battery_contract_perfect_price_upper_bound",
        },
        "evaluation": {
            "same_information_cutoff": True,
            "same_prediction_horizons": True,
            "same_ercot_lp_mpc_contract": True,
            "report_without_reselection": True,
        },
        "file_sha256": {name: sha256(ROOT / name) for name in FILES},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    current = build()
    if args.verify:
        frozen = json.loads(output.read_text(encoding="utf-8"))
        if frozen != current:
            raise ValueError("最终基线/2025统一入口已偏离冻结协议")
        print(json.dumps({"event": "allmarket_2025_baseline_protocol_verified"}))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".sha256").write_text(
        f"{sha256(output)}  {output.relative_to(ROOT).as_posix()}\n", encoding="utf-8"
    )
    print(json.dumps({"event": "allmarket_2025_baseline_protocol_frozen", "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
