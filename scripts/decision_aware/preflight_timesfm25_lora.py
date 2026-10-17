#!/usr/bin/env python
"""Validate the pre-registered TimesFM-2.5 LoRA LOMO protocol.

The command is intentionally read-only.  It validates the four-fold contract
and reports whether the official Hugging Face Transformers + PEFT runtime and
checkpoint are locally available; it never downloads packages or weights.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs" / "decision_aware" / "timesfm25_lora_source_lomo_v1.yaml"
)
EXPECTED_MARKETS = {"CAISO", "ERCOT", "NYISO", "PJM"}
EXPECTED_HORIZONS = {"da": 24, "rt_at_da": 24, "rt": 4}
MODEL_ID = "google/timesfm-2.5-200m-transformers"


def validate_protocol(raw: dict[str, Any]) -> dict[str, Any]:
    model = raw.get("model", {})
    data = raw.get("data", {})
    tasks = raw.get("tasks", {})
    adapter = raw.get("adapter", {})
    training = raw.get("training", {})
    if model.get("model_id") != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID}")
    if model.get("implementation") != "huggingface_transformers_peft":
        raise ValueError("TimesFM-2.5 LoRA must use the official HF/PEFT path")
    if bool(model.get("frozen_zero_shot")):
        raise ValueError("the LoRA baseline is supervised adaptation, not frozen")
    if int(data.get("context_hours", 0)) != 168:
        raise ValueError("context_hours must be 168")
    if data.get("target_mode") != "per_node_univariate":
        raise ValueError("official TimesFM-2.5 LoRA comparison must be univariate")
    if data.get("input_features") != ["target_price_history"]:
        raise ValueError("input capability disclosure must remain target-history-only")
    forbidden_true = (
        "uses_node_id",
        "uses_market_id",
        "uses_holdout_labels_for_training",
        "uses_future_true_price_load_wind_solar",
    )
    if any(bool(data.get(key)) for key in forbidden_true):
        raise ValueError(f"identity/leakage flag enabled: {forbidden_true}")

    folds = data.get("folds", [])
    if len(folds) != 4:
        raise ValueError("exactly four LOMO folds are required")
    seen_holdouts: set[str] = set()
    for fold in folds:
        holdout = str(fold.get("holdout", ""))
        sources = {str(item) for item in fold.get("sources", [])}
        if holdout not in EXPECTED_MARKETS:
            raise ValueError(f"unknown holdout market: {holdout}")
        if holdout in sources:
            raise ValueError(f"holdout {holdout} appears in its source markets")
        if sources != EXPECTED_MARKETS - {holdout}:
            raise ValueError(f"fold {holdout} does not contain the other three markets")
        seen_holdouts.add(holdout)
    if seen_holdouts != EXPECTED_MARKETS:
        raise ValueError("four folds do not cover every market exactly once")

    if set(tasks) != set(EXPECTED_HORIZONS):
        raise ValueError("DA, RT-at-DA, and rolling RT must all be configured")
    for task, horizon in EXPECTED_HORIZONS.items():
        if int(tasks[task].get("horizon", 0)) != horizon:
            raise ValueError(f"{task} horizon must be {horizon}")
    if adapter.get("method") != "lora" or adapter.get("target_modules") != "all-linear":
        raise ValueError("official comparison must use all-linear LoRA")
    if int(adapter.get("rank", 0)) < 1:
        raise ValueError("LoRA rank must be positive")
    if not bool(training.get("market_balanced_sampling")):
        raise ValueError("source-market training must use market-balanced sampling")
    if bool(training.get("decision_profit_backpropagation")):
        raise ValueError("LP/MPC profit must not backpropagate into TimesFM-2.5")
    return {
        "model_id": MODEL_ID,
        "folds": len(folds),
        "tasks": dict(EXPECTED_HORIZONS),
        "context_hours": 168,
        "input_features": ["target_price_history"],
        "lora_rank": int(adapter["rank"]),
        "lora_target_modules": adapter["target_modules"],
        "status": raw.get("status"),
    }


def runtime_status() -> dict[str, Any]:
    available = {
        package: importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "peft", "accelerate")
    }
    hub_root = Path(
        os.environ.get("HF_HUB_CACHE", ROOT / "hf_cache" / "hub")
    )
    checkpoint_dir = hub_root / "models--google--timesfm-2.5-200m-transformers"
    return {
        "packages": available,
        "checkpoint_cache": str(checkpoint_dir),
        "checkpoint_present": checkpoint_dir.exists(),
        "ready": all(available.values()) and checkpoint_dir.exists(),
        "downloads_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument(
        "--output",
        help="可选JSON审计产物路径；只写本地状态，不下载包或权重",
    )
    args = parser.parse_args()
    path = Path(args.config)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    report = {
        "protocol": validate_protocol(raw),
        "runtime": runtime_status(),
        "config": path.resolve().as_posix(),
    }
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.require_runtime and not report["runtime"]["ready"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
