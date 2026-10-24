#!/usr/bin/env python
"""在冻结DA Transformer表示上对比条件均值与非交叉分位数后训练。

这是源市场方法筛选入口：默认只读CAISO/NYISO/PJM的2020—2024，
不读ERCOT，不读2025。基础模型全部冻结，两个后训练头在相同batch上同时学习：

* mean：MSE条件均值修正，用作“中位数/均值目标错配”对照；
* quantile：等权pinball的非交叉多分位数头，用固定数值积分得到条件均值。

方法不搜索尾部权重、事件阈值、网络深度或学习率。
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import (  # noqa: E402
    AnnualStage,
    FORMAL_EXPERIMENT_CONTRACT,
)
from decision_aware.global_data import (  # noqa: E402
    MarketNodeBalancedSampler,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.tail_quantile_adapter import (  # noqa: E402
    DEFAULT_QUANTILES,
    MeanResidualAdapter,
    NonCrossingQuantileAdapter,
    RankPreservingPeakScaleAdapter,
    log_peak_excursion_loss,
    multi_quantile_pinball_loss,
)


ALLOWED_MARKETS = ("CAISO", "NYISO", "PJM")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    """将numpy标量与非有限浮点数转为严格JSON。"""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _config_from_checkpoint(checkpoint: dict) -> PilotConfig:
    allowed = {item.name for item in fields(PilotConfig)}
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in allowed
    })


def _prepare_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _node_thresholds(train_dataset) -> dict[int, dict[str, float]]:
    """由每个节点的训练期DA目标精确拟合q01/q99，不重复抽样。"""
    result: dict[int, dict[str, float]] = {}
    for (market, node), dataset in zip(
        train_dataset.entities, train_dataset.datasets
    ):
        _, node_id = train_dataset.mappings.identity(market, node)
        values = np.concatenate([
            dataset.price_da[window.target_positions]
            for window in dataset.windows
        ]).astype(np.float64)
        result[node_id] = {
            "q01": float(np.quantile(values, 0.01)),
            "q99": float(np.quantile(values, 0.99)),
        }
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1.0e-12 else float("nan")


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _safe_ratio(fp, fp + tn),
    }


def _point_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    node_ids: np.ndarray,
    thresholds: dict[int, dict[str, float]],
) -> dict[str, float | dict]:
    lower = np.asarray([thresholds[int(node)]["q01"] for node in node_ids])
    upper = np.asarray([thresholds[int(node)]["q99"] for node in node_ids])
    positive_truth = truth > upper
    positive_prediction = prediction > upper
    negative_truth = truth < 0.0
    negative_prediction = prediction < 0.0
    ordinary = (~positive_truth) & (~negative_truth)
    positive_amplitude_truth = np.maximum(truth[positive_truth] - upper[positive_truth], 0.0)
    positive_amplitude_prediction = np.maximum(
        prediction[positive_truth] - upper[positive_truth], 0.0
    )
    negative_amplitude_truth = np.maximum(-truth[negative_truth], 0.0)
    negative_amplitude_prediction = np.maximum(-prediction[negative_truth], 0.0)
    return {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "ordinary_mae": float(np.mean(np.abs(prediction[ordinary] - truth[ordinary]))),
        "positive_spike": {
            **_binary_metrics(positive_truth, positive_prediction),
            "event_hours": int(positive_truth.sum()),
            "conditional_mae": float(np.mean(np.abs(
                prediction[positive_truth] - truth[positive_truth]
            ))) if positive_truth.any() else float("nan"),
            "exceedance_amplitude_ratio": _safe_ratio(
                float(positive_amplitude_prediction.mean()) if positive_truth.any() else 0.0,
                float(positive_amplitude_truth.mean()) if positive_truth.any() else 0.0,
            ),
        },
        "negative_price": {
            **_binary_metrics(negative_truth, negative_prediction),
            "event_hours": int(negative_truth.sum()),
            "conditional_mae": float(np.mean(np.abs(
                prediction[negative_truth] - truth[negative_truth]
            ))) if negative_truth.any() else float("nan"),
            "amplitude_ratio": _safe_ratio(
                float(negative_amplitude_prediction.mean()) if negative_truth.any() else 0.0,
                float(negative_amplitude_truth.mean()) if negative_truth.any() else 0.0,
            ),
        },
    }


def _distribution_metrics(
    truth: np.ndarray,
    quantile_prediction: np.ndarray,
) -> dict:
    errors = truth[..., None] - quantile_prediction
    levels = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
    pinball = np.maximum(levels * errors, (levels - 1.0) * errors)
    coverage = np.mean(truth[..., None] <= quantile_prediction, axis=0)
    return {
        "mean_pinball": float(pinball.mean()),
        "coverage": {
            f"q{int(round(level * 100)):02d}": float(value)
            for level, value in zip(levels, coverage)
        },
        "coverage_absolute_error": {
            f"q{int(round(level * 100)):02d}": float(abs(value - level))
            for level, value in zip(levels, coverage)
        },
        "central_90_interval_coverage": float(np.mean(
            (truth >= quantile_prediction[:, 1])
            & (truth <= quantile_prediction[:, -2])
        )),
        "central_98_interval_coverage": float(np.mean(
            (truth >= quantile_prediction[:, 0])
            & (truth <= quantile_prediction[:, -1])
        )),
    }


def _event_probability_metrics(
    truth: np.ndarray,
    quantile_prediction: np.ndarray,
    node_ids: np.ndarray,
    thresholds: dict[int, dict[str, float]],
    *,
    side: str,
) -> dict:
    """用单调分位数函数的分段线性CDF得到事件概率。"""
    levels = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
    if side == "positive":
        boundary = np.asarray([
            thresholds[int(node)]["q99"] for node in node_ids
        ], dtype=np.float64)
        event = truth > boundary
    elif side == "negative":
        boundary = np.zeros_like(truth, dtype=np.float64)
        event = truth < 0.0
    else:
        raise ValueError("side必须是positive或negative")
    cdf = np.asarray([
        np.interp(value, row, levels, left=0.0, right=1.0)
        for value, row in zip(boundary, quantile_prediction)
    ])
    probability = 1.0 - cdf if side == "positive" else cdf
    probability = np.clip(probability, 0.0, 1.0)
    event_float = event.astype(np.float64)
    event_rate = float(event_float.mean())
    brier = float(np.mean((probability - event_float) ** 2))
    constant_brier = float(np.mean((event_float - event_rate) ** 2))
    pr_auc = (
        float(average_precision_score(event, probability))
        if event.any() and (~event).any()
        else float("nan")
    )
    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        if index == 9:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            bins.append({
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "samples": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()),
                "event_rate": float(event_float[mask].mean()),
            })
    return {
        "samples": int(len(truth)),
        "events": int(event.sum()),
        "event_rate": event_rate,
        "mean_probability": float(probability.mean()),
        "pr_auc": pr_auc,
        "pr_auc_lift": _safe_ratio(pr_auc, event_rate),
        "brier": brier,
        "constant_event_rate_brier": constant_brier,
        "brier_improvement_vs_constant": constant_brier - brier,
        "calibration_bins": bins,
    }


def _market_report(
    arrays: dict[str, np.ndarray],
    thresholds: dict[int, dict[str, float]],
) -> dict:
    market_ids = arrays["market_id"]
    report = {}
    for market_id in sorted(set(market_ids.tolist())):
        mask = market_ids == market_id
        market_name = arrays["market_names"][int(market_id)]
        truth = arrays["truth"][mask]
        node_ids = arrays["node_id"][mask]
        report[market_name] = {
            "samples": int(mask.sum()),
            "original": _point_metrics(
                truth, arrays["original"][mask], node_ids, thresholds
            ),
            "mean_posttraining": _point_metrics(
                truth, arrays["mean"][mask], node_ids, thresholds
            ),
            "quantile_mean": _point_metrics(
                truth, arrays["quantile_mean"][mask], node_ids, thresholds
            ),
            "quantile_median": _point_metrics(
                truth, arrays["quantiles"][mask, 4], node_ids, thresholds
            ),
            "quantile_q01": _point_metrics(
                truth, arrays["quantiles"][mask, 0], node_ids, thresholds
            ),
            "quantile_q99": _point_metrics(
                truth, arrays["quantiles"][mask, -1], node_ids, thresholds
            ),
            "peak_scale": _point_metrics(
                truth, arrays["peak_scale"][mask], node_ids, thresholds
            ),
            "distribution": _distribution_metrics(
                truth, arrays["quantiles"][mask]
            ),
            "positive_spike_probability": _event_probability_metrics(
                truth,
                arrays["quantiles"][mask],
                node_ids,
                thresholds,
                side="positive",
            ),
            "negative_price_probability": _event_probability_metrics(
                truth,
                arrays["quantiles"][mask],
                node_ids,
                thresholds,
                side="negative",
            ),
        }
    return report


@torch.no_grad()
def _evaluate(
    base_model,
    mean_adapter,
    quantile_adapter,
    peak_adapter,
    loader,
    device,
    amp_enabled,
    thresholds,
    market_names,
) -> tuple[dict, dict[str, np.ndarray]]:
    base_model.eval()
    mean_adapter.eval()
    quantile_adapter.eval()
    peak_adapter.eval()
    collected: dict[str, list[np.ndarray]] = {
        "truth": [], "original": [], "mean": [], "quantile_mean": [],
        "peak_scale": [],
        "quantiles": [], "node_id": [], "market_id": [],
    }
    for batch in loader:
        batch = _prepare_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = base_model(batch)
        representation = output["rep"].float()
        base_normalized = output["p_da_normalized"].float()
        mean_normalized = mean_adapter(representation, base_normalized)
        quantiles_normalized = quantile_adapter(representation, base_normalized)
        quantile_mean_normalized = quantile_adapter.conditional_mean(
            quantiles_normalized
        )
        peak_scale_normalized, _ = peak_adapter(
            representation, base_normalized
        )
        mean = batch["price_da_mean"].float().reshape(-1, 1)
        std = batch["price_da_std"].float().reshape(-1, 1)
        lo, hi = base_model.cfg.pred_clamp
        point_mean = (mean_normalized * std + mean).clamp(float(lo), float(hi))
        quantile_mean = (
            quantile_mean_normalized * std + mean
        ).clamp(float(lo), float(hi))
        peak_scale = (
            peak_scale_normalized * std + mean
        ).clamp(float(lo), float(hi))
        quantiles = (
            quantiles_normalized * std.unsqueeze(-1) + mean.unsqueeze(-1)
        ).clamp(float(lo), float(hi))
        batch_size, horizon = batch["price_da_tgt"].shape
        collected["truth"].append(batch["price_da_tgt"].cpu().numpy().reshape(-1))
        collected["original"].append(output["p_da"].cpu().numpy().reshape(-1))
        collected["mean"].append(point_mean.cpu().numpy().reshape(-1))
        collected["quantile_mean"].append(quantile_mean.cpu().numpy().reshape(-1))
        collected["peak_scale"].append(peak_scale.cpu().numpy().reshape(-1))
        collected["quantiles"].append(
            quantiles.cpu().numpy().reshape(-1, len(DEFAULT_QUANTILES))
        )
        collected["node_id"].append(
            batch["node_id"].reshape(-1, 1).repeat(1, horizon).cpu().numpy().reshape(-1)
        )
        collected["market_id"].append(
            batch["market_id"].reshape(-1, 1).repeat(1, horizon).cpu().numpy().reshape(-1)
        )
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    arrays["market_names"] = market_names
    report = {
        "markets": _market_report(arrays, thresholds),
        "pooled": {
            "original": _point_metrics(
                arrays["truth"], arrays["original"], arrays["node_id"], thresholds
            ),
            "mean_posttraining": _point_metrics(
                arrays["truth"], arrays["mean"], arrays["node_id"], thresholds
            ),
            "quantile_mean": _point_metrics(
                arrays["truth"], arrays["quantile_mean"], arrays["node_id"], thresholds
            ),
            "quantile_median": _point_metrics(
                arrays["truth"], arrays["quantiles"][:, 4], arrays["node_id"], thresholds
            ),
            "quantile_q01": _point_metrics(
                arrays["truth"], arrays["quantiles"][:, 0], arrays["node_id"], thresholds
            ),
            "quantile_q99": _point_metrics(
                arrays["truth"], arrays["quantiles"][:, -1], arrays["node_id"], thresholds
            ),
            "peak_scale": _point_metrics(
                arrays["truth"], arrays["peak_scale"], arrays["node_id"], thresholds
            ),
            "distribution": _distribution_metrics(
                arrays["truth"], arrays["quantiles"]
            ),
            "positive_spike_probability": _event_probability_metrics(
                arrays["truth"],
                arrays["quantiles"],
                arrays["node_id"],
                thresholds,
                side="positive",
            ),
            "negative_price_probability": _event_probability_metrics(
                arrays["truth"],
                arrays["quantiles"],
                arrays["node_id"],
                thresholds,
                side="negative",
            ),
        },
    }
    return report, arrays


def _compact_gate(report: dict) -> dict:
    """每个市场的机制筛选摘要；只作源市场证据，不含盈利。"""
    gates = {}
    for market, values in report["markets"].items():
        original = values["original"]
        candidate = values["quantile_mean"]
        original_amp = original["positive_spike"]["exceedance_amplitude_ratio"]
        candidate_amp = candidate["positive_spike"]["exceedance_amplitude_ratio"]
        gates[market] = {
            "ordinary_mae_ratio": _safe_ratio(
                candidate["ordinary_mae"], original["ordinary_mae"]
            ),
            "positive_spike_mae_ratio": _safe_ratio(
                candidate["positive_spike"]["conditional_mae"],
                original["positive_spike"]["conditional_mae"],
            ),
            "positive_amplitude_distance_improves": (
                abs(candidate_amp - 1.0) < abs(original_amp - 1.0)
                if np.isfinite(candidate_amp) and np.isfinite(original_amp)
                else False
            ),
            "negative_recall_change": (
                candidate["negative_price"]["recall"]
                - original["negative_price"]["recall"]
            ),
            "passes_screen": bool(
                candidate["ordinary_mae"] <= 1.02 * original["ordinary_mae"]
                and candidate["positive_spike"]["conditional_mae"]
                < original["positive_spike"]["conditional_mae"]
                and np.isfinite(candidate_amp)
                and np.isfinite(original_amp)
                and abs(candidate_amp - 1.0) < abs(original_amp - 1.0)
            ),
        }
    return {
        "per_market": gates,
        "passed_markets": int(sum(item["passes_screen"] for item in gates.values())),
        "required_markets": 2,
    }


def _peak_scale_gate(report: dict) -> dict:
    gates = {}
    for market, values in report["markets"].items():
        original = values["original"]
        candidate = values["peak_scale"]
        original_amp = original["positive_spike"]["exceedance_amplitude_ratio"]
        candidate_amp = candidate["positive_spike"]["exceedance_amplitude_ratio"]
        gates[market] = {
            "ordinary_mae_ratio": _safe_ratio(
                candidate["ordinary_mae"], original["ordinary_mae"]
            ),
            "positive_spike_mae_ratio": _safe_ratio(
                candidate["positive_spike"]["conditional_mae"],
                original["positive_spike"]["conditional_mae"],
            ),
            "positive_amplitude_distance_improves": (
                abs(candidate_amp - 1.0) < abs(original_amp - 1.0)
                if np.isfinite(candidate_amp) and np.isfinite(original_amp)
                else False
            ),
            "passes_screen": bool(
                candidate["ordinary_mae"] <= 1.02 * original["ordinary_mae"]
                and candidate["positive_spike"]["conditional_mae"]
                < original["positive_spike"]["conditional_mae"]
                and np.isfinite(candidate_amp)
                and np.isfinite(original_amp)
                and abs(candidate_amp - 1.0) < abs(original_amp - 1.0)
            ),
        }
    return {
        "per_market": gates,
        "passed_markets": int(sum(item["passes_screen"] for item in gates.values())),
        "required_markets": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/"
            "da_inner_seed0/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/da_tail_distribution_posttraining_source_v1",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-market", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--load-adapters",
        default="",
        help="跳过训练并读取已冻结adapters.pt，仅补做评价",
    )
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)
    if checkpoint.get("task") != "da" or checkpoint.get("phase") != "inner":
        raise ValueError("必须使用DA inner checkpoint")
    cfg = _config_from_checkpoint(checkpoint)
    training_markets = tuple(cfg.global_training_markets)
    if training_markets != ALLOWED_MARKETS:
        raise ValueError(
            f"首轮只允许{ALLOWED_MARKETS}，实际为{training_markets}"
        )
    if args.epochs != 6 or args.learning_rate != 1.0e-3:
        raise ValueError("预注册v1固定为6轮且学习率1e-3，不允许搜索")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    mappings = load_formal_entity_mappings(
        ROOT / "configs" / "nodes.yaml"
    ).subset(training_markets)
    mappings.validate_checkpoint(checkpoint)
    entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    train_dataset, validation_2023 = build_global_task_datasets(
        cfg,
        "da",
        entities,
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    _, validation_2024 = build_global_task_datasets(
        cfg,
        "da",
        entities,
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.OUTER_VALIDATION,
    )
    sampler = MarketNodeBalancedSampler(
        train_dataset,
        samples_per_market=args.samples_per_market,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loaders = {
        "2023": DataLoader(
            validation_2023,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "2024": DataLoader(
            validation_2024,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
    }
    thresholds = _node_thresholds(train_dataset)
    market_names = [
        market
        for market, _ in sorted(
            mappings.market_to_id.items(), key=lambda item: item[1]
        )
    ]

    base_model = DecisionAwareDAForecaster(cfg).to(device)
    base_model.load_state_dict(checkpoint["model_state"], strict=True)
    base_model.eval().requires_grad_(False)
    mean_adapter = MeanResidualAdapter(cfg.d_model).to(device)
    quantile_adapter = NonCrossingQuantileAdapter(cfg.d_model).to(device)
    peak_adapter = RankPreservingPeakScaleAdapter(cfg.d_model).to(device)
    optimizer = torch.optim.AdamW(
        [
            *mean_adapter.parameters(),
            *quantile_adapter.parameters(),
            *peak_adapter.parameters(),
        ],
        lr=args.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    if args.load_adapters:
        frozen_adapters = torch.load(
            ROOT / args.load_adapters, map_location="cpu", weights_only=False
        )
        mean_adapter.load_state_dict(frozen_adapters["mean_adapter_state"])
        quantile_adapter.load_state_dict(frozen_adapters["quantile_adapter_state"])
        peak_adapter.load_state_dict(frozen_adapters["peak_adapter_state"])
        history = list(frozen_adapters.get("training_history", []))
    for epoch in ([] if args.load_adapters else range(1, args.epochs + 1)):
        sampler.set_epoch(epoch)
        mean_adapter.train()
        quantile_adapter.train()
        peak_adapter.train()
        mse_total = 0.0
        pinball_total = 0.0
        peak_total = 0.0
        elements = 0
        started = perf_counter()
        for batch in train_loader:
            batch = _prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = base_model(batch)
            representation = output["rep"].float().detach()
            base_normalized = output["p_da_normalized"].float().detach()
            target = (
                batch["price_da_tgt"].float()
                - batch["price_da_mean"].float().reshape(-1, 1)
            ) / batch["price_da_std"].float().reshape(-1, 1)
            mean_prediction = mean_adapter(representation, base_normalized)
            quantile_prediction = quantile_adapter(
                representation, base_normalized
            )
            peak_prediction, _ = peak_adapter(
                representation, base_normalized
            )
            mse = F.mse_loss(mean_prediction, target)
            pinball = multi_quantile_pinball_loss(
                quantile_prediction, target
            )
            peak_loss = log_peak_excursion_loss(peak_prediction, target)
            loss = mse + pinball + peak_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("后训练出现NaN或Inf")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    *mean_adapter.parameters(),
                    *quantile_adapter.parameters(),
                    *peak_adapter.parameters(),
                ],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            count = int(target.numel())
            mse_total += float(mse.detach()) * count
            pinball_total += float(pinball.detach()) * count
            peak_total += float(peak_loss.detach()) * count
            elements += count
        record = {
            "epoch": epoch,
            "mse": mse_total / elements,
            "pinball": pinball_total / elements,
            "log_peak_excursion": peak_total / elements,
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean_adapter_state": mean_adapter.state_dict(),
            "quantile_adapter_state": quantile_adapter.state_dict(),
            "peak_adapter_state": peak_adapter.state_dict(),
            "quantiles": list(DEFAULT_QUANTILES),
            "base_checkpoint": args.checkpoint,
            "base_checkpoint_sha256": _sha256(checkpoint_path),
            "training_history": history,
            "uses_ercot": False,
            "test_2025_read": False,
        },
        output_dir / "adapters.pt",
    )
    evaluations = {}
    for year, loader in validation_loaders.items():
        evaluation, arrays = _evaluate(
            base_model,
            mean_adapter,
            quantile_adapter,
            peak_adapter,
            loader,
            device,
            amp_enabled,
            thresholds,
            market_names,
        )
        evaluations[year] = evaluation
        np.savez_compressed(
            output_dir / f"hourly_{year}.npz",
            **{
                key: value
                for key, value in arrays.items()
                if key != "market_names"
            },
        )
    summary = {
        "experiment": "da_tail_distribution_posttraining_source_v1",
        "status": "complete",
        "mechanisms": {
            "original": "frozen MAE Transformer point forecast",
            "mean_posttraining": "frozen backbone + linear conditional-mean MSE head",
            "quantile_posttraining": (
                "frozen backbone + non-crossing q01/q05/q10/q25/q50/"
                "q75/q90/q95/q99 pinball head; point forecast is fixed quantile integral"
            ),
            "rank_preserving_peak_scale": (
                "frozen backbone + positive daily excursion scale learned with "
                "log1p peak-excursion Smooth-L1; hourly ranking is unchanged"
            ),
        },
        "protocol": {
            "training_markets": list(training_markets),
            "train_years": [2020, 2021, 2022],
            "evaluation_years": [2023, 2024],
            "uses_ercot": False,
            "test_2025_read": False,
            "epochs": args.epochs,
            "samples_per_market_per_epoch": args.samples_per_market,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "backbone_frozen": True,
            "thresholds": "per-node 2020-2022 target q01/q99",
        },
        "training_history": history,
        "evaluations": evaluations,
        "screening_gate_2024": _compact_gate(evaluations["2024"]),
        "peak_scale_gate_2024": _peak_scale_gate(evaluations["2024"]),
    }
    summary = _json_ready(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "da_tail_distribution_posttraining_complete",
        "output": str((output_dir / "summary.json").relative_to(ROOT)),
        "quantile_gate": summary["screening_gate_2024"],
        "peak_scale_gate": summary["peak_scale_gate_2024"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

