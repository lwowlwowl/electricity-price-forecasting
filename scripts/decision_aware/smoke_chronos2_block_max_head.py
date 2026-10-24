#!/usr/bin/env python
"""Screen a distributional daily block-maximum head on aligned Chronos-2.

The backbone is frozen in this first information-value screen.  The head reads
Chronos forecast-token embeddings and predicts the full daily-maximum quantile
function.  All training days contribute a proper pinball loss; no outcome-based
sampling or validation tuning is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.block_max_head import (  # noqa: E402
    DailyBlockMaxQuantileHead,
    MonotoneLogitCalibrator,
    conditional_tail_mean_from_quantiles,
    event_probability_from_scalar_quantiles,
    expected_tail_localized_adjustment,
    extend_upper_quantiles,
    peak_localized_adjustment,
    scalar_quantile_score,
    select_oof_tail_shrinkage,
    select_delivery_patch_embeddings,
)
from scripts.decision_aware.smoke_chronos2_tail_posttraining import (  # noqa: E402
    DEFAULT_CONFIG,
    MODEL_SNAPSHOT,
    _configure_chronos_imports,
    _required_output_patches,
    _slice_delivery_predictions,
    evaluate,
    load_source_node,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_values(
    values: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    *,
    use_arcsinh: bool,
) -> torch.Tensor:
    normalized = (values - loc) / scale
    return torch.arcsinh(normalized) if use_arcsinh else normalized


def _extract_features(
    model: torch.nn.Module,
    contexts: np.ndarray,
    targets: np.ndarray,
    offsets: np.ndarray,
    quantiles: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    use_all_forecast_embeddings: bool = False,
) -> dict[str, np.ndarray]:
    num_output_patches = _required_output_patches(
        offsets, model.chronos_config.output_patch_size
    )
    predictions: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    baseline_normalized: list[np.ndarray] = []
    target_normalized: list[np.ndarray] = []
    locs: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    captured: dict[str, torch.Tensor] = {}

    def capture_hidden(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        captured["hidden"] = output[0]

    hook = model.encoder.register_forward_hook(capture_hidden)
    try:
        for start in range(0, len(contexts), batch_size):
            stop = min(start + batch_size, len(contexts))
            context = torch.from_numpy(contexts[start:stop]).to(device)
            batch_offsets = torch.from_numpy(offsets[start:stop]).to(device)
            with torch.no_grad():
                result = model(context=context, num_output_patches=num_output_patches)
                direct = result.quantile_preds
                aligned = _slice_delivery_predictions(direct, batch_offsets)
                forecast_embeddings = captured.pop("hidden")[:, -num_output_patches:]
                selected = (
                    forecast_embeddings
                    if use_all_forecast_embeddings
                    else select_delivery_patch_embeddings(
                        forecast_embeddings,
                        batch_offsets,
                        patch_size=model.chronos_config.output_patch_size,
                    )
                )
                _, (loc, scale) = model.instance_norm(context)
                baseline = extend_upper_quantiles(aligned, quantiles, quantiles)
                baseline_norm = _normalized_values(
                    baseline,
                    loc,
                    scale,
                    use_arcsinh=model.chronos_config.use_arcsinh,
                )
                target_max = torch.from_numpy(targets[start:stop]).to(device).amax(dim=1)
                target_norm = _normalized_values(
                    target_max[:, None],
                    loc,
                    scale,
                    use_arcsinh=model.chronos_config.use_arcsinh,
                ).squeeze(1)
            predictions.append(aligned.cpu().numpy())
            embeddings.append(selected.float().cpu().numpy())
            baseline_normalized.append(baseline_norm.float().cpu().numpy())
            target_normalized.append(target_norm.float().cpu().numpy())
            locs.append(loc.float().cpu().numpy())
            scales.append(scale.float().cpu().numpy())
    finally:
        hook.remove()
    return {
        "hourly_quantiles": np.concatenate(predictions),
        "embeddings": np.concatenate(embeddings),
        "baseline_normalized": np.concatenate(baseline_normalized),
        "target_normalized": np.concatenate(target_normalized),
        "loc": np.concatenate(locs),
        "scale": np.concatenate(scales),
    }


def _inverse_normalization(
    normalized: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    *,
    use_arcsinh: bool,
) -> torch.Tensor:
    values = torch.sinh(normalized) if use_arcsinh else normalized
    return values * scale + loc


def _predict_head(
    head: DailyBlockMaxQuantileHead,
    features: dict[str, np.ndarray],
    *,
    batch_size: int,
    use_arcsinh: bool,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features["embeddings"]), batch_size):
            stop = min(start + batch_size, len(features["embeddings"]))
            embeddings = torch.from_numpy(features["embeddings"][start:stop]).to(device)
            baseline = torch.from_numpy(
                features["baseline_normalized"][start:stop]
            ).to(device)
            normalized = head(embeddings, baseline)
            loc = torch.from_numpy(features["loc"][start:stop]).to(device)
            scale = torch.from_numpy(features["scale"][start:stop]).to(device)
            outputs.append(
                _inverse_normalization(
                    normalized, loc, scale, use_arcsinh=use_arcsinh
                ).cpu().numpy()
            )
    return np.concatenate(outputs)


def _fit_block_head(
    features: dict[str, np.ndarray],
    quantiles: torch.Tensor,
    indices: np.ndarray,
    *,
    model_dim: int,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[DailyBlockMaxQuantileHead, list[float]]:
    torch.manual_seed(seed)
    head = DailyBlockMaxQuantileHead(
        model_dim=model_dim,
        num_output_patches=features["embeddings"].shape[1],
        num_quantiles=len(quantiles),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    losses: list[float] = []
    head.train()
    for _ in range(steps):
        batch_indices = generator.choice(indices, size=batch_size, replace=True)
        embeddings = torch.from_numpy(features["embeddings"][batch_indices]).to(device)
        baseline = torch.from_numpy(
            features["baseline_normalized"][batch_indices]
        ).to(device)
        target = torch.from_numpy(features["target_normalized"][batch_indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = head(embeddings, baseline)
        loss = scalar_quantile_score(prediction, target, quantiles)
        if not torch.isfinite(loss):
            raise FloatingPointError("block-max head loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return head, losses


def _cross_fitted_train_predictions(
    features: dict[str, np.ndarray],
    train_target: np.ndarray,
    quantiles: torch.Tensor,
    *,
    positive_threshold: float,
    event_temperature: float,
    num_entities: int,
    model_dim: int,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    seed: int,
    use_arcsinh: bool,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    per_entity = len(train_target) // num_entities
    positions = np.arange(per_entity)
    folds = np.array_split(positions, 5)
    predictions = np.full(
        (len(train_target), len(quantiles)), np.nan, dtype=np.float32
    )
    audit: list[dict[str, Any]] = []
    for fold_index, fold_positions in enumerate(folds):
        validation_indices = np.concatenate(
            [fold_positions + entity * per_entity for entity in range(num_entities)]
        )
        fit_mask = np.ones(len(train_target), dtype=bool)
        fit_mask[validation_indices] = False
        fit_indices = np.flatnonzero(fit_mask)
        fold_head, fold_losses = _fit_block_head(
            features,
            quantiles,
            fit_indices,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 1000 + fold_index,
            device=device,
        )
        fold_features = {
            key: value[validation_indices]
            for key, value in features.items()
        }
        fold_predictions = _predict_head(
            fold_head,
            fold_features,
            batch_size=eval_batch_size,
            use_arcsinh=use_arcsinh,
            device=device,
        )
        predictions[validation_indices] = fold_predictions
        labels = np.max(train_target[validation_indices], axis=1) > positive_threshold
        audit.append(
            {
                "fold": fold_index,
                "fit_days": int(len(fit_indices)),
                "held_out_days": int(len(validation_indices)),
                "held_out_event_days": int(labels.sum()),
                "loss_first": fold_losses[0],
                "loss_last": fold_losses[-1],
            }
        )
        del fold_head
    if not np.all(np.isfinite(predictions)):
        raise AssertionError("cross-fitted predictions are incomplete")
    return predictions, audit


def _calibrate_probabilities(
    raw_train: np.ndarray,
    train_labels: np.ndarray,
    raw_validation: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, MonotoneLogitCalibrator, float]:
    calibrator = MonotoneLogitCalibrator().to(device)
    probabilities = torch.from_numpy(raw_train.astype(np.float32)).to(device)
    labels = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    optimizer = torch.optim.LBFGS(
        calibrator.parameters(), lr=0.2, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        calibrated = calibrator(probabilities)
        loss = torch.nn.functional.binary_cross_entropy(calibrated, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_loss = float(
            torch.nn.functional.binary_cross_entropy(
                calibrator(probabilities), labels
            ).detach()
        )
        validation = calibrator(
            torch.from_numpy(raw_validation.astype(np.float32)).to(device)
        ).cpu().numpy()
    return validation, calibrator, final_loss


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _block_metrics(
    quantile_predictions: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
    threshold: float,
    temperature: float,
    event_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    target_max = np.max(target, axis=1)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    point = quantile_predictions[:, median_index]
    labels = target_max > threshold
    probabilities = (
        np.asarray(event_probabilities, dtype=np.float32)
        if event_probabilities is not None
        else event_probability_from_scalar_quantiles(
            torch.from_numpy(quantile_predictions),
            torch.from_numpy(quantiles),
            threshold,
            temperature=temperature,
        ).numpy()
    )
    if probabilities.shape != (len(target),):
        raise ValueError("event probabilities must have one value per day")
    calibration = []
    for lower, upper in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if np.any(mask):
            calibration.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "observed_rate": float(labels[mask].mean()),
                }
            )
    return {
        "daily_max_mae": float(np.mean(np.abs(point - target_max))),
        "event_days": int(labels.sum()),
        "event_conditional_mae": (
            float(np.mean(np.abs(point[labels] - target_max[labels])))
            if np.any(labels)
            else None
        ),
        "event_brier": float(np.mean((probabilities - labels.astype(np.float32)) ** 2)),
        "event_average_precision": _average_precision(labels, probabilities),
        "predicted_probability_mean": float(probabilities.mean()),
        "observed_event_rate": float(labels.mean()),
        "coverage": {
            str(float(level)): float(np.mean(target_max <= quantile_predictions[:, index]))
            for index, level in enumerate(quantiles)
            if level in {0.5, 0.9, 0.95, 0.99}
        },
        "calibration_bins": calibration,
        "probabilities": probabilities,
    }


def _adjust_hourly_quantiles(
    hourly: np.ndarray,
    block_quantiles: np.ndarray,
    probabilities: np.ndarray,
    quantiles: np.ndarray,
    threshold: float,
    temperature: float,
) -> np.ndarray:
    adjusted = np.empty_like(hourly)
    tail_mean = conditional_tail_mean_from_quantiles(
        torch.from_numpy(block_quantiles),
        torch.from_numpy(quantiles),
        threshold,
        temperature=temperature,
    ).numpy()
    for index in range(hourly.shape[1]):
        adjusted[:, index, :] = expected_tail_localized_adjustment(
            hourly[:, index, :],
            tail_mean,
            probabilities,
        )
    return np.maximum.accumulate(adjusted, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--market", default="CAISO")
    parser.add_argument("--node")
    parser.add_argument("--all-market-nodes", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--event-temperature", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="data/results/chronos2_blockmax_head_caiso3_v1_quick"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this Chronos-2 screen requires the project CUDA environment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = perf_counter()

    (
        train_context,
        train_target,
        train_offsets,
        _train_calendar_context,
        _train_calendar_future,
        val_context,
        val_target,
        val_offsets,
        _val_calendar_context,
        _val_calendar_future,
        time_contract,
        node,
    ) = load_source_node(
        ROOT / args.config,
        args.market,
        args.node,
        all_market_nodes=args.all_market_nodes,
    )
    positive_threshold = float(np.quantile(train_target, 0.99))
    train_event_rate = float(np.mean(np.max(train_target, axis=1) > positive_threshold))

    _configure_chronos_imports()
    from chronos import Chronos2Pipeline

    device = torch.device("cuda")
    pipeline = Chronos2Pipeline.from_pretrained(
        str(MODEL_SNAPSHOT),
        device_map="cuda",
        dtype=torch.float32,
        local_files_only=True,
    )
    model = pipeline.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    quantiles = torch.tensor(model.chronos_config.quantiles, device=device)
    quantiles_np = quantiles.cpu().numpy()

    train_features = _extract_features(
        model,
        train_context,
        train_target,
        train_offsets,
        quantiles,
        batch_size=args.eval_batch_size,
        device=device,
    )
    val_features = _extract_features(
        model,
        val_context,
        val_target,
        val_offsets,
        quantiles,
        batch_size=args.eval_batch_size,
        device=device,
    )

    num_entities = len(node.split(","))
    if len(train_target) % num_entities != 0:
        raise ValueError("pooled entity samples cannot be split into aligned daily groups")
    oof_predictions, cross_fit_audit = _cross_fitted_train_predictions(
        train_features,
        train_target,
        quantiles,
        positive_threshold=positive_threshold,
        event_temperature=args.event_temperature,
        num_entities=num_entities,
        model_dim=model.model_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_arcsinh=model.chronos_config.use_arcsinh,
        device=device,
    )
    all_indices = np.arange(len(train_target))
    head, losses = _fit_block_head(
        train_features,
        quantiles,
        all_indices,
        model_dim=model.model_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    parameters = list(head.parameters())

    block_predictions = _predict_head(
        head,
        val_features,
        batch_size=args.eval_batch_size,
        use_arcsinh=model.chronos_config.use_arcsinh,
        device=device,
    )
    train_baseline_block = np.max(train_features["hourly_quantiles"], axis=2)
    train_target_max = np.max(train_target, axis=1)
    shrinkage_alpha, shrinkage_audit = select_oof_tail_shrinkage(
        train_baseline_block,
        oof_predictions,
        train_target_max,
        quantiles_np,
        minimum_quantile=0.9,
    )
    oof_predictions = train_baseline_block + shrinkage_alpha * (
        oof_predictions - train_baseline_block
    )
    baseline_block_predictions = np.max(val_features["hourly_quantiles"], axis=2)
    block_predictions = baseline_block_predictions + shrinkage_alpha * (
        block_predictions - baseline_block_predictions
    )
    oof_probabilities = event_probability_from_scalar_quantiles(
        torch.from_numpy(oof_predictions),
        torch.from_numpy(quantiles_np),
        positive_threshold,
        temperature=args.event_temperature,
    ).numpy()
    raw_event_probabilities = event_probability_from_scalar_quantiles(
        torch.from_numpy(block_predictions),
        torch.from_numpy(quantiles_np),
        positive_threshold,
        temperature=args.event_temperature,
    ).numpy()
    calibration_labels = np.max(train_target, axis=1) > positive_threshold
    event_probabilities, calibrator, calibration_loss = _calibrate_probabilities(
        oof_probabilities,
        calibration_labels,
        raw_event_probabilities,
        device=device,
    )
    frozen_predictions = val_features["hourly_quantiles"]
    baseline_block = _block_metrics(
        baseline_block_predictions,
        val_target,
        quantiles_np,
        positive_threshold,
        args.event_temperature,
    )
    proposed_block = _block_metrics(
        block_predictions,
        val_target,
        quantiles_np,
        positive_threshold,
        args.event_temperature,
        event_probabilities=event_probabilities,
    )
    adjusted_predictions = _adjust_hourly_quantiles(
        frozen_predictions,
        block_predictions,
        event_probabilities,
        quantiles_np,
        positive_threshold,
        args.event_temperature,
    )
    frozen_metrics = evaluate(
        frozen_predictions,
        val_target,
        quantiles_np,
        positive_threshold=positive_threshold,
        event_temperature=args.event_temperature,
    )
    adjusted_metrics = evaluate(
        adjusted_predictions,
        val_target,
        quantiles_np,
        positive_threshold=positive_threshold,
        event_temperature=args.event_temperature,
    )
    for metrics in (baseline_block, proposed_block):
        metrics.pop("probabilities")

    block_brier_improvement = (
        baseline_block["event_brier"] - proposed_block["event_brier"]
    ) / baseline_block["event_brier"]
    block_conditional_improvement = (
        baseline_block["event_conditional_mae"]
        - proposed_block["event_conditional_mae"]
    ) / baseline_block["event_conditional_mae"]
    hourly_conditional_improvement = (
        frozen_metrics["positive"]["conditional_mae"]
        - adjusted_metrics["positive"]["conditional_mae"]
    ) / frozen_metrics["positive"]["conditional_mae"]
    gate = {
        "block_event_brier_relative_improvement": block_brier_improvement,
        "block_event_brier_improves_1pct": block_brier_improvement >= 0.01,
        "block_event_conditional_mae_relative_improvement": block_conditional_improvement,
        "block_event_conditional_mae_improves_0_5pct": block_conditional_improvement >= 0.005,
        "hourly_event_conditional_mae_relative_improvement": hourly_conditional_improvement,
        "hourly_event_conditional_mae_improves_0_5pct": hourly_conditional_improvement >= 0.005,
        "ordinary_mae_within_1pct": adjusted_metrics["ordinary_mae"]
        <= frozen_metrics["ordinary_mae"] * 1.01,
        "ramp_mae_within_2pct": adjusted_metrics["ramp_mae"]
        <= frozen_metrics["ramp_mae"] * 1.02,
    }
    gate["passed"] = bool(
        gate["block_event_brier_improves_1pct"]
        and gate["block_event_conditional_mae_improves_0_5pct"]
        and gate["hourly_event_conditional_mae_improves_0_5pct"]
        and gate["ordinary_mae_within_1pct"]
        and gate["ramp_mae_within_2pct"]
    )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=val_target,
        quantiles=quantiles_np,
        frozen=frozen_predictions,
        block_max=block_predictions,
        raw_event_probability=raw_event_probabilities,
        event_probability=event_probabilities,
        adjusted=adjusted_predictions,
    )
    checkpoint_path = output_dir / "block_max_head.pt"
    torch.save(
        {
            "format": "chronos2_daily_block_max_quantile_head_v1",
            "block_max_state_dict": head.state_dict(),
            "calibrator_state_dict": calibrator.state_dict(),
            "quantiles": quantiles_np,
            "positive_threshold": positive_threshold,
            "shrinkage_alpha": shrinkage_alpha,
        },
        checkpoint_path,
    )
    summary = {
        "status": "passed_screen" if gate["passed"] else "failed_screen",
        "scope": "pooled_source_market_information_value_screen_not_paper_result",
        "data_contract": {
            "market": args.market,
            "node": node,
            "all_market_nodes": args.all_market_nodes,
            "train_years": "2020-2022",
            "screen_year": 2023,
            "ercot_read": False,
            "year_2025_read": False,
            "train_days": int(len(train_target)),
            "screen_days": int(len(val_target)),
            "train_positive_event_days": int(
                np.sum(np.max(train_target, axis=1) > positive_threshold)
            ),
            "train_positive_event_rate": train_event_rate,
            "chronos_direct_forecast_alignment": time_contract,
        },
        "model": "amazon/chronos-2",
        "model_snapshot": MODEL_SNAPSHOT.relative_to(ROOT).as_posix(),
        "training_path": "frozen_chronos2_forecast_token_distribution_head",
        "objective": (
            "proper_daily_block_max_quantile_score_with_disjoint_monotone_"
            "bernoulli_probability_calibration"
        ),
        "sample_reweighting": False,
        "oof_tail_shrinkage_alpha": shrinkage_alpha,
        "oof_tail_shrinkage_audit": shrinkage_audit,
        "positive_threshold": positive_threshold,
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "event_temperature": args.event_temperature,
            "seed": args.seed,
        },
        "fit_days": int(len(train_target)),
        "probability_calibration_days": int(len(train_target)),
        "probability_calibration_event_days": int(calibration_labels.sum()),
        "probability_cross_fit": cross_fit_audit,
        "trainable_parameters": sum(p.numel() for p in parameters) + 2,
        "loss_trace": {
            "quantile_first": losses[0],
            "quantile_last": losses[-1],
            "calibration_bce": calibration_loss,
        },
        "calibrator": {
            "slope": float(torch.exp(calibrator.log_slope).detach().cpu()),
            "intercept": float(calibrator.intercept.detach().cpu()),
        },
        "frozen_hourly": frozen_metrics,
        "adjusted_hourly": adjusted_metrics,
        "baseline_block_max": baseline_block,
        "proposed_block_max": proposed_block,
        "screen_gate": gate,
        "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "torch_version": torch.__version__,
        "device": str(device),
        "wall_seconds": perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
