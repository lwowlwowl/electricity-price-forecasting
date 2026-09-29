#!/usr/bin/env python
"""打印RT-at-DA样本时间合同并执行基本防泄漏断言。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_rt_da import build_rt_at_da_datasets  # noqa: E402


def _serialise(contract: dict) -> dict:
    return {key: str(value) for key, value in contract.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/rt_at_da_ercot_v1.yaml"
    )
    args = parser.parse_args()
    cfg = PilotConfig.from_yaml(args.config)
    train_ds, val_ds, test_ds, _ = build_rt_at_da_datasets(cfg)
    first = test_ds[0]
    contract = test_ds.sample_time_contract(0)

    assert contract["context_end"] < contract["issue_time"]
    assert contract["issue_time"] < contract["target_start"]
    assert first["price_rt_at_da_tgt"].shape == (24,)
    assert "price_da_tgt_known" not in first
    assert "price_da_tgt" not in first and "price_rt_tgt" not in first
    assert "load_tgt" not in first and "system_tgt" not in first
    print(json.dumps({
        "samples": {
            "train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)
        },
        "first_test_contract": _serialise(contract),
        "visible_future": ["calendar only"],
        "target": "delivery-day 24 hourly RT prices",
        "checks": {
            "context_ends_before_issue": True,
            "issue_precedes_delivery": True,
            "future_actual_covariates_hidden": True,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
