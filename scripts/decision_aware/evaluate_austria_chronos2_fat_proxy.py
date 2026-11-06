#!/usr/bin/env python
"""Approximate the teacher's Chronos2-ZS-fat8192 using no extra feature families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external/chronos-forecasting/src"))
# Reuse the vendored Chronos environment's lightweight einops dependency while
# keeping the main environment's CUDA-enabled PyTorch earlier on sys.path.
sys.path.append(str(ROOT / "external/chronos-forecasting/.venv/Lib/site-packages"))

from decision_aware.austria_chronos2_fat_proxy import (  # noqa: E402
    KNOWN_FUTURE_COLUMNS,
    PAST_ONLY_COLUMNS,
    UNAVAILABLE_TEACHER_FEATURES,
    build_chronos2_fat_proxy,
    make_chronos2_tasks,
)
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
)


TEACHER_MAE = {
    "all": 18.139,
    "neg": 22.793,
    "spike": 22.947,
    "ordinary": 13.786,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/Austria/奥地利_20250801_20260930.parquet")
    parser.add_argument("--max-context-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output-dir", default="data/results/austria_chronos2_fat_proxy_v1"
    )
    args = parser.parse_args()

    frame = build_chronos2_fat_proxy(pd.read_parquet(ROOT / args.raw))
    origins = pd.date_range(
        pd.Timestamp("2025-09-30 22:00", tz="UTC"),
        pd.Timestamp("2026-09-30 22:00", tz="UTC"),
        freq="1D",
        inclusive="left",
    )
    tasks = make_chronos2_tasks(
        frame, origins, max_context_length=args.max_context_length
    )
    if len(tasks) != 365:
        raise ValueError(f"应有365个预测块，实际{len(tasks)}")

    from chronos import Chronos2Pipeline

    snapshots = sorted(
        (ROOT / "hf_cache/hub/models--amazon--chronos-2/snapshots").iterdir()
    )
    if len(snapshots) != 1:
        raise ValueError(f"Chronos2本地快照数量异常: {len(snapshots)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(
        str(snapshots[0]),
        device_map=device,
        dtype=torch.float32,
        local_files_only=True,
    )
    model_inputs = [
        {
            "target": item["target"],
            "past_covariates": item["past_covariates"],
            "future_covariates": item["future_covariates"],
        }
        for item in tasks
    ]
    _quantiles, medians = pipeline.predict_quantiles(
        model_inputs,
        prediction_length=96,
        quantile_levels=[0.5],
        batch_size=args.batch_size,
    )
    prediction = np.concatenate(
        [value.detach().cpu().numpy().reshape(96) for value in medians]
    ).astype(np.float64)
    truth = np.concatenate([np.asarray(item["truth"]) for item in tasks]).astype(np.float64)
    timestamps = tasks[0]["index"].append([item["index"] for item in tasks[1:]])
    local = timestamps.tz_convert("Europe/Vienna")
    groups = classify_days(local, truth, spike_floor=SPIKE_FLOOR_EUR_PER_MWH)
    metrics = forecast_metrics(local, truth, prediction, groups)
    compact = {
        name: float(metrics[name]["point_weighted_mae_eur_per_mwh"])
        for name in TEACHER_MAE
    }
    comparison = {
        name: {
            "ours": compact[name],
            "teacher": TEACHER_MAE[name],
            "absolute_gap_ours_minus_teacher": compact[name] - TEACHER_MAE[name],
            "relative_gap_percent": 100.0 * (compact[name] / TEACHER_MAE[name] - 1.0),
        }
        for name in TEACHER_MAE
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        timestamp_unix_seconds=np.asarray(
            [int(item.timestamp()) for item in timestamps], dtype=np.int64
        ),
        truth_eur_per_mwh=truth.astype(np.float32),
        prediction_eur_per_mwh=prediction.astype(np.float32),
    )
    context_lengths = [len(np.asarray(item["target"])) for item in tasks]
    summary = {
        "model": "Chronos2-ZS-fat-proxy14plus3",
        "status": "approximate_reproduction_not_exact_teacher_run",
        "protocol": "365 local days; native 15-minute truth and prediction",
        "teacher_protocol_difference": "teacher board has 366 days; local file has 365 days",
        "context_length": {
            "maximum": args.max_context_length,
            "minimum_available": int(min(context_lengths)),
        },
        "used_known_future_teacher_family_proxies": list(KNOWN_FUTURE_COLUMNS),
        "used_past_only_teacher_family_proxies": list(PAST_ONLY_COLUMNS),
        "unavailable_teacher_features": list(UNAVAILABLE_TEACHER_FEATURES),
        "added_non_teacher_features": [],
        "austria_training_or_tuning_used": False,
        "metrics": compact,
        "teacher_comparison": comparison,
        "device": device,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

