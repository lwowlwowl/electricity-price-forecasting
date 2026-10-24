#!/usr/bin/env python
"""对冻结DA分位数头的正尖峰概率做留一市场Beta校准。

拟合只使用2023的其他两个源市场，评价对应留出市场2024。
ERCOT和2025都不参与。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.tail_quantile_adapter import DEFAULT_QUANTILES  # noqa: E402
from scripts.decision_aware.run_da_tail_distribution_posttraining import (  # noqa: E402
    _config_from_checkpoint,
    _json_ready,
    _node_thresholds,
)


MARKETS = ("CAISO", "NYISO", "PJM")


def _raw_probability(
    quantiles: np.ndarray,
    node_ids: np.ndarray,
    thresholds: dict[int, dict[str, float]],
) -> np.ndarray:
    levels = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
    boundary = np.asarray([
        thresholds[int(node)]["q99"] for node in node_ids
    ], dtype=np.float64)
    cdf = np.asarray([
        np.interp(value, row, levels, left=0.0, right=1.0)
        for value, row in zip(boundary, quantiles)
    ])
    return np.clip(1.0 - cdf, 1.0e-6, 1.0 - 1.0e-6)


def _features(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return np.column_stack([
        np.log(probability),
        np.log1p(-probability),
    ])


def _metrics(
    event: np.ndarray,
    probability: np.ndarray,
    node_ids: np.ndarray,
) -> dict:
    event = event.astype(bool)
    probability = np.asarray(probability, dtype=np.float64)
    rate = float(event.mean())
    brier = float(np.mean((probability - event.astype(float)) ** 2))
    constant = float(np.mean((event.astype(float) - rate) ** 2))
    pr_auc = float(average_precision_score(event, probability))
    per_node = []
    for node in sorted(set(node_ids.tolist())):
        mask = node_ids == node
        node_rate = float(event[mask].mean())
        node_brier = float(np.mean(
            (probability[mask] - event[mask].astype(float)) ** 2
        ))
        node_constant = float(np.mean(
            (event[mask].astype(float) - node_rate) ** 2
        ))
        per_node.append({
            "node_id": int(node),
            "samples": int(mask.sum()),
            "event_rate": node_rate,
            "mean_probability": float(probability[mask].mean()),
            "brier": node_brier,
            "constant_brier": node_constant,
        })
    return {
        "samples": int(len(event)),
        "events": int(event.sum()),
        "event_rate": rate,
        "mean_probability": float(probability.mean()),
        "pr_auc": pr_auc,
        "pr_auc_lift": float(pr_auc / rate),
        "brier": brier,
        "constant_brier": constant,
        "brier_improvement_vs_constant": constant - brier,
        "node_equal_brier": float(np.mean([item["brier"] for item in per_node])),
        "node_equal_constant_brier": float(np.mean([
            item["constant_brier"] for item in per_node
        ])),
        "node_metrics": per_node,
    }


def main() -> None:
    checkpoint_path = ROOT / (
        "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/"
        "da_inner_seed0/best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: PilotConfig = _config_from_checkpoint(checkpoint)
    if tuple(cfg.global_training_markets) != MARKETS:
        raise ValueError("概率校准只允许三个源市场")
    mappings = load_formal_entity_mappings(
        ROOT / "configs/nodes.yaml"
    ).subset(MARKETS)
    entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    train_dataset, _ = build_global_task_datasets(
        cfg,
        "da",
        entities,
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    thresholds = _node_thresholds(train_dataset)
    names = [
        market for market, _ in sorted(
            mappings.market_to_id.items(), key=lambda item: item[1]
        )
    ]
    inputs = ROOT / "data/results/da_tail_distribution_calibration_inputs_v1"
    arrays = {
        year: dict(np.load(inputs / f"hourly_{year}.npz"))
        for year in ("2023", "2024")
    }
    prepared = {}
    for year, values in arrays.items():
        probability = _raw_probability(
            values["quantiles"], values["node_id"], thresholds
        )
        boundary = np.asarray([
            thresholds[int(node)]["q99"] for node in values["node_id"]
        ])
        prepared[year] = {
            **values,
            "probability": probability,
            "event": values["truth"] > boundary,
        }

    folds = {}
    for holdout_name in MARKETS:
        holdout_id = names.index(holdout_name)
        train_mask = prepared["2023"]["market_id"] != holdout_id
        test_mask = prepared["2024"]["market_id"] == holdout_id
        train_market_ids = prepared["2023"]["market_id"][train_mask]
        sample_weight = np.zeros(train_mask.sum(), dtype=np.float64)
        for market_id in sorted(set(train_market_ids.tolist())):
            mask = train_market_ids == market_id
            sample_weight[mask] = 1.0 / mask.sum()
        sample_weight *= len(sample_weight) / sample_weight.sum()
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=20261020,
        )
        model.fit(
            _features(prepared["2023"]["probability"][train_mask]),
            prepared["2023"]["event"][train_mask].astype(int),
            sample_weight=sample_weight,
        )
        raw = prepared["2024"]["probability"][test_mask]
        calibrated = model.predict_proba(_features(raw))[:, 1]
        event = prepared["2024"]["event"][test_mask]
        node_ids = prepared["2024"]["node_id"][test_mask]
        raw_metrics = _metrics(event, raw, node_ids)
        calibrated_metrics = _metrics(event, calibrated, node_ids)
        passes = bool(
            calibrated_metrics["brier"] < raw_metrics["brier"]
            and calibrated_metrics["brier"]
            < calibrated_metrics["constant_brier"]
            and calibrated_metrics["node_equal_brier"]
            < calibrated_metrics["node_equal_constant_brier"]
            and calibrated_metrics["pr_auc_lift"] >= 1.25
        )
        folds[holdout_name] = {
            "fit_markets": [name for name in MARKETS if name != holdout_name],
            "fit_year": 2023,
            "evaluation_year": 2024,
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "passes": passes,
        }

    passed = sum(item["passes"] for item in folds.values())
    summary = _json_ready({
        "experiment": "da_positive_spike_beta_calibration_lomo_v1",
        "status": "complete",
        "uses_ercot": False,
        "test_2025_read": False,
        "probability_source": (
            "piecewise-linear CDF of frozen non-crossing q01..q99 adapter"
        ),
        "calibrator": "Beta calibration: logistic(log(p), log(1-p)), C=1",
        "folds": folds,
        "gate": {
            "passed_markets": int(passed),
            "required_markets": 2,
            "passes": bool(passed >= 2),
        },
    })
    output_dir = ROOT / "data/results/da_tail_probability_calibration_lomo_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

