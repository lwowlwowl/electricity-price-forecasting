#!/usr/bin/env python
"""Evaluate frozen TimesFM3 with the teacher-specified causal EXAA input."""
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
sys.path.insert(0, str(ROOT / "external/timesfm/src"))

from decision_aware.austria_mainu11_proxy import (  # noqa: E402
    PROXY_COLUMNS_WITH_EXAA,
    UNAVAILABLE_MAINU11_COLUMNS,
    build_mainu11_proxy_with_exaa,
    make_tasks,
)
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
)


TEACHER_MAE = {
    "all": 12.603,
    "neg": 16.898,
    "spike": 15.159,
    "ordinary": 9.687,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/Austria/奥地利_20250801_20260930.parquet")
    parser.add_argument(
        "--exaa",
        default="data/Austria/external_public/EXAA_AT_2025_2026_hourly.csv",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--batch-days", type=int, default=64)
    parser.add_argument(
        "--output-dir", default="data/results/austria_timesfm3_mainu11_exaa_v1"
    )
    args = parser.parse_args()

    raw = pd.read_parquet(ROOT / args.raw)
    exaa = pd.read_csv(ROOT / args.exaa, parse_dates=["timestamp_utc"]).set_index(
        "timestamp_utc"
    )
    frame = build_mainu11_proxy_with_exaa(raw, exaa)
    origins = pd.date_range(
        pd.Timestamp("2025-09-30 22:00:00", tz="UTC"),
        pd.Timestamp("2026-09-30 22:00:00", tz="UTC"),
        freq="1D",
        inclusive="left",
    )
    tasks = make_tasks(
        frame,
        origins,
        context_length=args.context_length,
        covariate_columns=PROXY_COLUMNS_WITH_EXAA,
    )
    if len(tasks) != 365:
        raise ValueError(f"expected 365 forecast blocks, got {len(tasks)}")

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimesFM3Evaluator(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=16 if device == "cuda" else 4,
        device=device,
        cache_dir=str(ROOT / "hf_cache/hub"),
        local_files_only=True,
    ))
    predictions: list[np.ndarray] = []
    for start in range(0, len(tasks), args.batch_days):
        batch = tasks[start:start + args.batch_days]
        outputs = model.predict_batch(
            contexts=[np.asarray(item["context"]) for item in batch],
            horizon=96,
            past_future_covariates=[
                np.concatenate(
                    [np.asarray(item["hist_cov"]), np.asarray(item["future_cov"])],
                    axis=0,
                ).T
                for item in batch
            ],
            return_quantiles=False,
            use_symmetric_averaging=False,
            make_positive=False,
        )
        predictions.extend(
            np.asarray(item.forecast, dtype=np.float32).reshape(96)
            for item in outputs
        )

    prediction = np.stack(predictions).reshape(-1).astype(np.float64)
    truth = np.stack([np.asarray(item["truth"]) for item in tasks]).reshape(-1)
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
            "gap_ours_minus_teacher": compact[name] - TEACHER_MAE[name],
        }
        for name in TEACHER_MAE
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        timestamp_unix_seconds=np.asarray([int(item.timestamp()) for item in timestamps]),
        truth_eur_per_mwh=truth.astype(np.float32),
        prediction_eur_per_mwh=prediction.astype(np.float32),
    )
    summary = {
        "model": "TimesFM3-ZS-mainU11-proxy6+EXAA",
        "status": "frozen_zero_shot_approximation_of_teacher_feature_protocol",
        "austria_training_tuning_or_selection_used": False,
        "teacher_feature_families_used": list(PROXY_COLUMNS_WITH_EXAA),
        "unavailable_teacher_features": list(UNAVAILABLE_MAINU11_COLUMNS),
        "exaa_availability": "earlier auction price known before forecast gate",
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
