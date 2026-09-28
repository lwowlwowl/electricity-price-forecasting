#!/usr/bin/env python
"""只读审计一个固定截止时点、完整交付日的 DA 样本。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import build_da_datasets, _market_ts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/formal_ercot_v8.yaml"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    train_ds, val_ds, test_ds, wide_df = build_da_datasets(cfg)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = datasets[args.split]
    if not 0 <= args.index < len(ds):
        raise IndexError(f"index={args.index} 超出 {args.split} 样本范围 0..{len(ds)-1}")

    sample = ds[args.index]
    contract = ds.sample_time_contract(args.index)
    window = ds.windows[args.index]
    raw_context_last = float(ds.price_da[window.context_end - 1])
    reconstructed_last = float(
        sample["price_da_ctx"][-1] * sample["price_da_std"]
        + sample["price_da_mean"]
    )

    train_lo, train_hi = cfg.split_bounds("train")
    train_mask = (
        (wide_df.index >= _market_ts(train_lo, train_ds.timezone))
        & (wide_df.index <= _market_ts(train_hi, train_ds.timezone))
    )
    manual_da = wide_df.loc[train_mask, "price_da"].to_numpy(dtype=np.float32)
    issue = contract["issue_time"]
    target_start = contract["target_start"]
    target_end = contract["target_end"]

    checks = {
        "issue_is_previous_day_10am_local": (
            issue.hour == cfg.da_issue_hour_local
            and issue.date() == target_start.date() - pd.Timedelta(days=1)
        ),
        "context_ends_one_hour_before_issue": (
            issue - contract["context_end"] == pd.Timedelta(hours=1)
        ),
        "target_is_one_local_calendar_day": (
            target_start.date() == target_end.date()
            and target_start.hour == 0
            and target_end.hour == 23
        ),
        "context_length_is_configured": sample["price_da_ctx"].shape[0]
        == cfg.context_len,
        "target_length_is_24": sample["price_da_tgt"].shape[0] == 24,
        "normalization_reconstructs_raw_value": np.isclose(
            reconstructed_last, raw_context_last, rtol=1e-5, atol=1e-4
        ),
        "normalization_mean_uses_train_period": np.isclose(
            train_ds.norm_stats["price_da"]["mean"], manual_da.mean(), rtol=1e-7
        ),
        "validation_reuses_train_stats": val_ds.norm_stats is train_ds.norm_stats,
        "test_reuses_train_stats": test_ds.norm_stats is train_ds.norm_stats,
    }

    result = {
        "config": args.config,
        "market": cfg.market,
        "node": cfg.node,
        "split": args.split,
        "sample_index": args.index,
        "dataset_samples": len(ds),
        "context_hours": cfg.context_len,
        "target_hours": 24,
        "time_contract": {key: str(value) for key, value in contract.items()},
        "shapes": {key: list(value.shape) for key, value in sample.items()},
        "excluded_non_24h_delivery_days": len(ds.excluded_non_24h_days),
        "checks": {key: bool(value) for key, value in checks.items()},
        "warnings": [
            "Parquet没有记录timestamp代表区间开始还是区间结束；当前按区间开始解释，仍需向数据提供方确认。",
            "load/wind/solar上下文目前都是截止时点前的历史实际量；未来可用预测协变量尚未接入。",
        ],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("审计失败：至少一个检查未通过")


if __name__ == "__main__":
    main()
