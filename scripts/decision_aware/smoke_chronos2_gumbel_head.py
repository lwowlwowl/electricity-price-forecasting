#!/usr/bin/env python
"""Screen a stable EVT (Gumbel) daily-maximum head over Chronos-2 tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.block_max_head import (  # noqa: E402
    DailyBlockMaxGumbelHead,
    event_probability_from_scalar_quantiles,
    gumbel_negative_log_likelihood,
    gumbel_quantiles,
)
from scripts.decision_aware.smoke_chronos2_block_max_head import (  # noqa: E402
    _adjust_hourly_quantiles,
    _block_metrics,
    _calibrate_probabilities,
    _extract_features,
    _inverse_normalization,
    _sha256,
)
from scripts.decision_aware.smoke_chronos2_tail_posttraining import (  # noqa: E402
    DEFAULT_CONFIG,
    MODEL_SNAPSHOT,
    _configure_chronos_imports,
    evaluate,
    load_source_node,
)


def _fit_head(
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
) -> tuple[DailyBlockMaxGumbelHead, list[float]]:
    torch.manual_seed(seed)
    head = DailyBlockMaxGumbelHead(
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
        batch = generator.choice(indices, size=batch_size, replace=True)
        embeddings = torch.from_numpy(features["embeddings"][batch]).to(device)
        baseline = torch.from_numpy(features["baseline_normalized"][batch]).to(device)
        target = torch.from_numpy(features["target_normalized"][batch]).to(device)
        optimizer.zero_grad(set_to_none=True)
        location, scale = head(embeddings, baseline, quantiles)
        loss = gumbel_negative_log_likelihood(location, scale, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Gumbel NLL became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return head, losses


def _predict(
    head: DailyBlockMaxGumbelHead,
    features: dict[str, np.ndarray],
    quantiles: torch.Tensor,
    *,
    use_arcsinh: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features["embeddings"]), batch_size):
            stop = min(start + batch_size, len(features["embeddings"]))
            embeddings = torch.from_numpy(features["embeddings"][start:stop]).to(device)
            baseline = torch.from_numpy(features["baseline_normalized"][start:stop]).to(device)
            location, scale = head(embeddings, baseline, quantiles)
            normalized = gumbel_quantiles(location, scale, quantiles)
            loc = torch.from_numpy(features["loc"][start:stop]).to(device)
            value_scale = torch.from_numpy(features["scale"][start:stop]).to(device)
            outputs.append(
                _inverse_normalization(
                    normalized, loc, value_scale, use_arcsinh=use_arcsinh
                ).cpu().numpy()
            )
    return np.concatenate(outputs)


def _cross_fitted_predictions(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    quantiles: torch.Tensor,
    *,
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
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    days = len(target) // num_entities
    folds = np.array_split(np.arange(days), 5)
    predictions = np.full((len(target), len(quantiles)), np.nan, dtype=np.float32)
    audit: list[dict[str, float | int]] = []
    for fold_index, positions in enumerate(folds):
        held_out = np.concatenate(
            [positions + entity * days for entity in range(num_entities)]
        )
        mask = np.ones(len(target), dtype=bool)
        mask[held_out] = False
        fit = np.flatnonzero(mask)
        head, losses = _fit_head(
            features,
            quantiles,
            fit,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 1000 + fold_index,
            device=device,
        )
        fold_features = {key: value[held_out] for key, value in features.items()}
        predictions[held_out] = _predict(
            head,
            fold_features,
            quantiles,
            use_arcsinh=use_arcsinh,
            batch_size=eval_batch_size,
            device=device,
        )
        audit.append(
            {
                "fold": fold_index,
                "fit_days": int(len(fit)),
                "held_out_days": int(len(held_out)),
                "loss_first": losses[0],
                "loss_last": losses[-1],
            }
        )
    if not np.all(np.isfinite(predictions)):
        raise AssertionError("Gumbel OOF predictions are incomplete")
    return predictions, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--market", default="CAISO")
    parser.add_argument("--node")
    parser.add_argument("--all-market-nodes", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--event-temperature", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="data/results/chronos2_gumbel_caiso3_seed0_v1"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this Chronos-2 screen requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = perf_counter()
    loaded = load_source_node(
        ROOT / args.config,
        args.market,
        args.node,
        all_market_nodes=args.all_market_nodes,
    )
    train_context, train_target, train_offsets = loaded[0], loaded[1], loaded[2]
    val_context, val_target, val_offsets = loaded[5], loaded[6], loaded[7]
    time_contract, nodes = loaded[10], loaded[11]
    threshold = float(np.quantile(train_target, 0.99))

    _configure_chronos_imports()
    from chronos import Chronos2Pipeline

    device = torch.device("cuda")
    pipeline = Chronos2Pipeline.from_pretrained(
        str(MODEL_SNAPSHOT), device_map="cuda", dtype=torch.float32, local_files_only=True
    )
    model = pipeline.model
    model.eval()
    if model.chronos_config.use_arcsinh:
        raise RuntimeError("Gumbel screen currently requires linear Chronos normalization")
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
    num_entities = len(nodes.split(","))
    oof, cross_fit_audit = _cross_fitted_predictions(
        train_features,
        train_target,
        quantiles,
        num_entities=num_entities,
        model_dim=model.model_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_arcsinh=False,
        device=device,
    )
    head, losses = _fit_head(
        train_features,
        quantiles,
        np.arange(len(train_target)),
        model_dim=model.model_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    block = _predict(
        head,
        val_features,
        quantiles,
        use_arcsinh=False,
        batch_size=args.eval_batch_size,
        device=device,
    )
    oof_probability = event_probability_from_scalar_quantiles(
        torch.from_numpy(oof), torch.from_numpy(quantiles_np), threshold,
        temperature=args.event_temperature,
    ).numpy()
    raw_probability = event_probability_from_scalar_quantiles(
        torch.from_numpy(block), torch.from_numpy(quantiles_np), threshold,
        temperature=args.event_temperature,
    ).numpy()
    labels = np.max(train_target, axis=1) > threshold
    probability, calibrator, calibration_loss = _calibrate_probabilities(
        oof_probability, labels, raw_probability, device=device
    )
    frozen = val_features["hourly_quantiles"]
    baseline_block = np.max(frozen, axis=2)
    baseline_metrics = _block_metrics(
        baseline_block, val_target, quantiles_np, threshold, args.event_temperature
    )
    block_metrics = _block_metrics(
        block, val_target, quantiles_np, threshold, args.event_temperature,
        event_probabilities=probability,
    )
    adjusted = _adjust_hourly_quantiles(
        frozen, block, probability, quantiles_np, threshold, args.event_temperature
    )
    frozen_metrics = evaluate(
        frozen, val_target, quantiles_np,
        positive_threshold=threshold, event_temperature=args.event_temperature,
    )
    adjusted_metrics = evaluate(
        adjusted, val_target, quantiles_np,
        positive_threshold=threshold, event_temperature=args.event_temperature,
    )
    baseline_metrics.pop("probabilities")
    block_metrics.pop("probabilities")
    brier_improvement = (baseline_metrics["event_brier"] - block_metrics["event_brier"]) / baseline_metrics["event_brier"]
    block_conditional_improvement = (baseline_metrics["event_conditional_mae"] - block_metrics["event_conditional_mae"]) / baseline_metrics["event_conditional_mae"]
    hourly_conditional_improvement = (frozen_metrics["positive"]["conditional_mae"] - adjusted_metrics["positive"]["conditional_mae"]) / frozen_metrics["positive"]["conditional_mae"]
    gate = {
        "block_event_brier_relative_improvement": brier_improvement,
        "block_event_brier_improves_1pct": brier_improvement >= 0.01,
        "block_event_conditional_mae_relative_improvement": block_conditional_improvement,
        "block_event_conditional_mae_improves_0_5pct": block_conditional_improvement >= 0.005,
        "hourly_event_conditional_mae_relative_improvement": hourly_conditional_improvement,
        "hourly_event_conditional_mae_improves_0_5pct": hourly_conditional_improvement >= 0.005,
        "ordinary_mae_within_1pct": adjusted_metrics["ordinary_mae"] <= frozen_metrics["ordinary_mae"] * 1.01,
        "ramp_mae_within_2pct": adjusted_metrics["ramp_mae"] <= frozen_metrics["ramp_mae"] * 1.02,
    }
    gate["passed"] = bool(all(value for key, value in gate.items() if key.endswith(("1pct", "0_5pct", "2pct"))))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz", target=val_target, quantiles=quantiles_np,
        frozen=frozen, block_max=block, event_probability=probability, adjusted=adjusted,
    )
    checkpoint = output_dir / "gumbel_head.pt"
    torch.save(
        {
            "format": "chronos2_daily_block_max_gumbel_head_v1",
            "head_state_dict": head.state_dict(),
            "calibrator_state_dict": calibrator.state_dict(),
            "quantiles": quantiles_np,
            "positive_threshold": threshold,
        }, checkpoint,
    )
    summary = {
        "status": "passed_screen" if gate["passed"] else "failed_screen",
        "scope": "stable_evt_source_market_screen_not_paper_result",
        "model": "amazon/chronos-2",
        "tail_distribution": "conditional_Gumbel_GEV_shape_fixed_zero",
        "nodes": nodes,
        "market": args.market,
        "train_years": "2020-2022",
        "screen_year": 2023,
        "year_2025_read": False,
        "train_days": int(len(train_target)),
        "screen_days": int(len(val_target)),
        "positive_threshold": threshold,
        "time_contract": time_contract,
        "hyperparameters": vars(args),
        "trainable_parameters": sum(p.numel() for p in head.parameters()) + 2,
        "loss_trace": {"first": losses[0], "last": losses[-1]},
        "probability_cross_fit": cross_fit_audit,
        "calibration_bce": calibration_loss,
        "calibrator": {
            "slope": float(torch.exp(calibrator.log_slope).detach().cpu()),
            "intercept": float(calibrator.intercept.detach().cpu()),
        },
        "frozen_hourly": frozen_metrics,
        "adjusted_hourly": adjusted_metrics,
        "baseline_block_max": baseline_metrics,
        "proposed_block_max": block_metrics,
        "screen_gate": gate,
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "wall_seconds": perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
