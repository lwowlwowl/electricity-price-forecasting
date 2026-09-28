#!/usr/bin/env python
"""打印独立RT滚动样本的实际时间合同和张量形状。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402


def main():
    cfg = PilotConfig.from_yaml("configs/decision_aware/rt_ercot_v1.yaml")
    train, val, test, _ = build_rt_datasets(cfg)
    contract = {
        key: str(value) for key, value in test.sample_time_contract(0).items()
    }
    shapes = {key: list(value.shape) for key, value in test[0].items()}
    print(json.dumps({
        "samples": {"train": len(train), "validation": len(val), "test": len(test)},
        "first_test_contract": contract,
        "first_test_shapes": shapes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
