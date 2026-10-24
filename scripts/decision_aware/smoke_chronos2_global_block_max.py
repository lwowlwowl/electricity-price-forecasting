#!/usr/bin/env python
"""Screen one normalized Chronos-2 block-max head shared across four markets."""
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
    event_probability_from_scalar_quantiles,
    select_oof_tail_shrinkage,
)
from scripts.decision_aware.smoke_chronos2_block_max_head import (  # noqa: E402
    _adjust_hourly_quantiles,
    _block_metrics,
    _calibrate_probabilities,
    _cross_fitted_train_predictions,
    _extract_features,
    _fit_block_head,
    _predict_head,
    _sha256,
)
from scripts.decision_aware.smoke_chronos2_tail_posttraining import (  # noqa: E402
    DEFAULT_CONFIG,
    MODEL_SNAPSHOT,
    _configure_chronos_imports,
    evaluate,
    load_source_node,
)


MARKET_NODES = {
    "CAISO": "TH_SP15_GEN-APND,TH_NP15_GEN-APND,TH_ZP26_GEN-APND",
    "ERCOT": "LZ_WEST,HB_HUBAVG,HB_PAN",
    "NYISO": "N.Y.C.,WEST,LONGIL",
    "PJM": "ZONE:DOM,ZONE:PJM-RTO,HUB:CHICAGO HUB",
}


def _concat_feature_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--event-temperature", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="data/results/chronos2_global_blockmax_12node_seed0_v1"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this Chronos-2 screen requires the project CUDA environment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = perf_counter()

    datasets: dict[str, dict[str, object]] = {}
    for market, nodes in MARKET_NODES.items():
        loaded = load_source_node(ROOT / args.config, market, nodes)
        datasets[market] = {
            "train_context": loaded[0],
            "train_target": loaded[1],
            "train_offsets": loaded[2],
            "val_context": loaded[5],
            "val_target": loaded[6],
            "val_offsets": loaded[7],
            "time_contract": loaded[10],
            "nodes": loaded[11],
        }

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

    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: dict[str, dict[str, np.ndarray]] = {}
    slices: dict[str, slice] = {}
    cursor = 0
    for market, data in datasets.items():
        train_feature = _extract_features(
            model,
            data["train_context"],
            data["train_target"],
            data["train_offsets"],
            quantiles,
            batch_size=args.eval_batch_size,
            device=device,
            use_all_forecast_embeddings=True,
        )
        train_parts.append(train_feature)
        count = len(data["train_target"])
        slices[market] = slice(cursor, cursor + count)
        cursor += count
        val_parts[market] = _extract_features(
            model,
            data["val_context"],
            data["val_target"],
            data["val_offsets"],
            quantiles,
            batch_size=args.eval_batch_size,
            device=device,
            use_all_forecast_embeddings=True,
        )
    train_features = _concat_feature_parts(train_parts)
    train_target = np.concatenate(
        [datasets[market]["train_target"] for market in MARKET_NODES]
    )
    num_entities = 3 * len(MARKET_NODES)
    oof_predictions, cross_fit_audit = _cross_fitted_train_predictions(
        train_features,
        train_target,
        quantiles,
        positive_threshold=0.0,
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
    head, losses = _fit_block_head(
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

    market_results: dict[str, object] = {}
    calibrators: dict[str, dict[str, torch.Tensor]] = {}
    passed_markets = 0
    for market, data in datasets.items():
        train_slice = slices[market]
        market_target = data["train_target"]
        val_target = data["val_target"]
        threshold = float(np.quantile(market_target, 0.99))
        baseline_train = np.max(
            train_features["hourly_quantiles"][train_slice], axis=2
        )
        alpha, alpha_audit = select_oof_tail_shrinkage(
            baseline_train,
            oof_predictions[train_slice],
            np.max(market_target, axis=1),
            quantiles_np,
            minimum_quantile=0.9,
        )
        blended_oof = baseline_train + alpha * (
            oof_predictions[train_slice] - baseline_train
        )
        raw_oof_probability = event_probability_from_scalar_quantiles(
            torch.from_numpy(blended_oof),
            torch.from_numpy(quantiles_np),
            threshold,
            temperature=args.event_temperature,
        ).numpy()
        val_block = _predict_head(
            head,
            val_parts[market],
            batch_size=args.eval_batch_size,
            use_arcsinh=model.chronos_config.use_arcsinh,
            device=device,
        )
        frozen_hourly = val_parts[market]["hourly_quantiles"]
        baseline_val = np.max(frozen_hourly, axis=2)
        val_block = baseline_val + alpha * (val_block - baseline_val)
        raw_val_probability = event_probability_from_scalar_quantiles(
            torch.from_numpy(val_block),
            torch.from_numpy(quantiles_np),
            threshold,
            temperature=args.event_temperature,
        ).numpy()
        labels = np.max(market_target, axis=1) > threshold
        probabilities, calibrator, calibration_loss = _calibrate_probabilities(
            raw_oof_probability,
            labels,
            raw_val_probability,
            device=device,
        )
        calibrators[market] = {
            key: value.detach().cpu() for key, value in calibrator.state_dict().items()
        }
        baseline_block = _block_metrics(
            baseline_val,
            val_target,
            quantiles_np,
            threshold,
            args.event_temperature,
        )
        proposed_block = _block_metrics(
            val_block,
            val_target,
            quantiles_np,
            threshold,
            args.event_temperature,
            event_probabilities=probabilities,
        )
        adjusted = _adjust_hourly_quantiles(
            frozen_hourly,
            val_block,
            probabilities,
            quantiles_np,
            threshold,
            args.event_temperature,
        )
        frozen_metrics = evaluate(
            frozen_hourly,
            val_target,
            quantiles_np,
            positive_threshold=threshold,
            event_temperature=args.event_temperature,
        )
        adjusted_metrics = evaluate(
            adjusted,
            val_target,
            quantiles_np,
            positive_threshold=threshold,
            event_temperature=args.event_temperature,
        )
        baseline_block.pop("probabilities")
        proposed_block.pop("probabilities")
        brier_improvement = (
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
            "block_event_brier_relative_improvement": brier_improvement,
            "block_event_brier_improves_1pct": brier_improvement >= 0.01,
            "block_event_conditional_mae_relative_improvement": block_conditional_improvement,
            "block_event_conditional_mae_improves_0_5pct": block_conditional_improvement >= 0.005,
            "hourly_event_conditional_mae_relative_improvement": hourly_conditional_improvement,
            "hourly_event_conditional_mae_improves_0_5pct": hourly_conditional_improvement >= 0.005,
            "ordinary_mae_within_1pct": adjusted_metrics["ordinary_mae"]
            <= frozen_metrics["ordinary_mae"] * 1.01,
            "ramp_mae_within_2pct": adjusted_metrics["ramp_mae"]
            <= frozen_metrics["ramp_mae"] * 1.02,
        }
        gate["passed"] = bool(all(value for key, value in gate.items() if key.endswith(("1pct", "0_5pct", "2pct"))))
        passed_markets += int(gate["passed"])
        market_results[market] = {
            "nodes": data["nodes"],
            "positive_threshold": threshold,
            "train_days": int(len(market_target)),
            "screen_days": int(len(val_target)),
            "train_event_days": int(labels.sum()),
            "screen_event_days": int(np.sum(np.max(val_target, axis=1) > threshold)),
            "oof_tail_shrinkage_alpha": alpha,
            "oof_tail_shrinkage_audit": alpha_audit,
            "calibration_bce": calibration_loss,
            "calibrator": {
                "slope": float(torch.exp(calibrator.log_slope).detach().cpu()),
                "intercept": float(calibrator.intercept.detach().cpu()),
            },
            "frozen_hourly": frozen_metrics,
            "adjusted_hourly": adjusted_metrics,
            "baseline_block_max": baseline_block,
            "proposed_block_max": proposed_block,
            "screen_gate": gate,
            "time_contract": data["time_contract"],
        }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "global_block_max_head.pt"
    torch.save(
        {
            "format": "chronos2_global_daily_block_max_quantile_head_v1",
            "head_state_dict": head.state_dict(),
            "calibrator_state_dict": calibrators,
            "quantiles": quantiles_np,
        },
        checkpoint,
    )
    summary = {
        "status": "passed_screen" if passed_markets == len(MARKET_NODES) else "failed_screen",
        "scope": "four_market_three_representative_node_screen_not_paper_result",
        "model": "amazon/chronos-2",
        "model_snapshot": MODEL_SNAPSHOT.relative_to(ROOT).as_posix(),
        "data_contract": {
            "train_years": "2020-2022",
            "screen_year": 2023,
            "year_2025_read": False,
            "representative_nodes_per_market": 3,
            "train_days": int(len(train_target)),
        },
        "training_path": "one_shared_normalized_block_max_head_market_local_threshold_calibration",
        "hyperparameters": vars(args),
        "trainable_parameters": sum(p.numel() for p in head.parameters()) + 2 * len(MARKET_NODES),
        "loss_trace": {"first": losses[0], "last": losses[-1]},
        "probability_cross_fit": cross_fit_audit,
        "markets_passed": passed_markets,
        "markets_total": len(MARKET_NODES),
        "markets": market_results,
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
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
