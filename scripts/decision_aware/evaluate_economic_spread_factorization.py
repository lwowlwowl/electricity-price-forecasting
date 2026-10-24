#!/usr/bin/env python
"""评价在ERCOT 2023拟合并冻结的经济价差分解Transformer。"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.economic_spread_factorization import (  # noqa: E402
    EconomicSpreadFactorizedTransformer,
    economic_spread_regime,
)
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from scripts.decision_aware.export_global_f0_f1_predictions import (  # noqa: E402
    _dates_and_timestamps,
)


def _config(raw: dict) -> PilotConfig:
    allowed = {field.name for field in fields(PilotConfig)}
    return PilotConfig(**{key: value for key, value in raw.items() if key in allowed})


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _state_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    per_state = {}
    f1_values = []
    for state, name in enumerate(("neutral", "da_high", "rt_high")):
        true_mask = truth == state
        predicted_mask = prediction == state
        tp = int(np.sum(true_mask & predicted_mask))
        precision = tp / max(int(predicted_mask.sum()), 1)
        recall = tp / max(int(true_mask.sum()), 1)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0 else 0.0
        )
        f1_values.append(f1)
        per_state[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(true_mask.sum()),
            "predicted": int(predicted_mask.sum()),
        }
    return {
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(np.mean(f1_values)),
        "per_state": per_state,
    }


def _report(
    da: np.ndarray,
    rt: np.ndarray,
    base_da: np.ndarray,
    base_rt: np.ndarray,
    target_da: np.ndarray,
    target_rt: np.ndarray,
    state: np.ndarray,
    threshold: float,
) -> dict:
    true_spread = target_da - target_rt
    spread = da - rt
    base_spread = base_da - base_rt
    truth_state = np.zeros_like(state, dtype=np.int64)
    truth_state[true_spread > threshold] = 1
    truth_state[true_spread < -threshold] = 2
    neutral = truth_state == 0
    tradeable = ~neutral
    return {
        "state": _state_metrics(truth_state.reshape(-1), state.reshape(-1)),
        "da_mae": float(np.mean(np.abs(da - target_da))),
        "rt_at_da_mae": float(np.mean(np.abs(rt - target_rt))),
        "ordinary_da_mae": float(np.mean(np.abs(da[neutral] - target_da[neutral]))),
        "ordinary_rt_at_da_mae": float(np.mean(np.abs(rt[neutral] - target_rt[neutral]))),
        "base_da_mae": float(np.mean(np.abs(base_da - target_da))),
        "base_rt_at_da_mae": float(np.mean(np.abs(base_rt - target_rt))),
        "spread_mae": float(np.mean(np.abs(spread - true_spread))),
        "base_spread_mae": float(np.mean(np.abs(base_spread - true_spread))),
        "mean_absolute_spread": float(np.mean(np.abs(spread))),
        "base_mean_absolute_spread": float(np.mean(np.abs(base_spread))),
        "true_mean_absolute_spread": float(np.mean(np.abs(true_spread))),
        "economic_crossing_rate": float(np.mean(np.abs(spread) > threshold)),
        "base_economic_crossing_rate": float(
            np.mean(np.abs(base_spread) > threshold)
        ),
        "true_economic_crossing_rate": float(
            np.mean(np.abs(true_spread) > threshold)
        ),
        "tradeable_spread_direction_accuracy": float(
            np.mean(np.sign(spread[tradeable]) == np.sign(true_spread[tradeable]))
        ),
        "base_tradeable_spread_direction_accuracy": float(
            np.mean(
                np.sign(base_spread[tradeable]) == np.sign(true_spread[tradeable])
            )
        ),
        "conditional_excess_mae": float(
            np.mean(
                np.abs(
                    (np.abs(spread[tradeable]) - threshold)
                    - (np.abs(true_spread[tradeable]) - threshold)
                )
            )
        ),
    }


def _save_archive(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout-market", default="ERCOT")
    parser.add_argument("--stage", choices=("inner", "outer_frozen"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rolling-rt-archive", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "economic_spread_factorization_transformer_v1":
        raise ValueError("checkpoint不是经济价差分解Transformer v1")
    if checkpoint.get("phase") != "inner":
        raise ValueError("2023/2024冻结评价只能使用inner checkpoint")
    if checkpoint.get("data_boundary", {}).get("year_2025_used") is not False:
        raise ValueError("checkpoint没有声明2025未使用")
    training_markets = tuple(checkpoint.get("training_markets", ()))
    adaptation_market = checkpoint.get("adaptation_market")
    if adaptation_market != args.holdout_market:
        raise ValueError("评价市场必须与2023价差头拟合市场一致")
    if checkpoint.get("trainable_scope") != "head":
        raise ValueError("公平v1只允许训练联合价差头")
    if checkpoint.get("data_boundary", {}).get(
        "year_2024_used_for_selection"
    ) is not False:
        raise ValueError("checkpoint没有声明2024未参与选择")
    cfg = _config(checkpoint["config"])
    if cfg.global_use_entity_embeddings:
        raise ValueError("公平v1要求无身份embedding基础Transformer")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = EconomicSpreadFactorizedTransformer(
        DecisionAwareDAForecaster(cfg),
        DecisionAwareRTAtDAForecaster(cfg),
        d_model=cfg.d_model,
        threshold=float(checkpoint["threshold_usd_per_mwh"]),
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    validation_stage = (
        AnnualStage.INNER_VALIDATION
        if args.stage == "inner" else AnnualStage.OUTER_VALIDATION
    )
    evaluation_end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(
        validation_stage
    ).end
    read_end_utc = (
        pd.Timestamp(evaluation_end, tz=market_timezone(args.holdout_market))
        + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    mappings = load_formal_entity_mappings(
        ROOT / "configs" / "nodes.yaml"
    ).subset([args.holdout_market])
    entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    if args.smoke:
        entities = entities[:1]
    _, dataset = build_global_task_datasets(
        replace(cfg, global_training_markets=[args.holdout_market]),
        "da",
        entities,
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=validation_stage,
        read_end_utc=read_end_utc,
    )

    collected = {
        key: []
        for key in (
            "da", "rt", "base_da", "base_rt", "target_da", "target_rt",
            "state", "probability", "excess", "da_adjustment", "rt_adjustment",
        )
    }
    common_dates = None
    per_node = {}
    threshold = float(checkpoint["threshold_usd_per_mwh"])
    for entity_index, ((_, node), node_dataset) in enumerate(
        zip(dataset.entities, dataset.datasets)
    ):
        start = dataset.offsets[entity_index]
        end = dataset.offsets[entity_index + 1]
        selected_end = min(end, start + 2) if args.smoke else end
        loader = DataLoader(
            Subset(dataset, range(start, selected_end)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        node_values = {key: [] for key in collected}
        for batch in loader:
            node_values["target_da"].append(batch["price_da_tgt"].numpy())
            node_values["target_rt"].append(batch["price_rt_tgt"].numpy())
            output = model(_move(dict(batch), device))
            key_map = {
                "da": "hard_da_price",
                "rt": "hard_rt_at_da_price",
                "base_da": "base_da_price",
                "base_rt": "base_rt_at_da_price",
                "state": "regime_class",
                "probability": "regime_probability",
                "excess": "conditional_excess",
                "da_adjustment": "da_price_adjustment",
                "rt_adjustment": "rt_at_da_price_adjustment",
            }
            for local_key, output_key in key_map.items():
                node_values[local_key].append(
                    output[output_key].detach().float().cpu().numpy()
                )
        node_values = {
            key: np.concatenate(values).astype(
                np.int8 if key == "state" else np.float32
            )
            for key, values in node_values.items()
        }
        dates, _ = _dates_and_timestamps(node_dataset, "da")
        if common_dates is None:
            common_dates = dates
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"节点日期不一致: {node}")
        per_node[node] = _report(
            node_values["da"], node_values["rt"],
            node_values["base_da"], node_values["base_rt"],
            node_values["target_da"], node_values["target_rt"],
            node_values["state"], threshold,
        )
        for key in collected:
            collected[key].append(node_values[key])
        print(json.dumps({
            "event": "node_complete", "node": node,
            "state_macro_f1": per_node[node]["state"]["macro_f1"],
            "spread_mae": per_node[node]["spread_mae"],
        }, ensure_ascii=False), flush=True)

    arrays = {key: np.stack(values) for key, values in collected.items()}
    aggregate = _report(
        arrays["da"], arrays["rt"], arrays["base_da"], arrays["base_rt"],
        arrays["target_da"], arrays["target_rt"], arrays["state"], threshold,
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_phase = "inner" if args.stage == "inner" else "outer"
    metadata = {
        "method": "economic_spread_factorization_transformer_v1",
        "evaluation_stage": args.stage,
        "phase": archive_phase,
        "seed": int(cfg.seed),
        "holdout_market": args.holdout_market,
        "training_markets": list(training_markets),
        "backbone_training_markets": list(
            checkpoint.get("backbone_training_markets", ())
        ),
        "adaptation_market": adaptation_market,
        "selection_allowed": args.stage == "inner",
        "year_2025_used": False,
        "threshold_usd_per_mwh": threshold,
        "checkpoint": args.checkpoint,
    }
    nodes = np.asarray([node for _, node in dataset.entities])
    common = {
        "nodes": nodes,
        "dates": common_dates,
        "target_da": arrays["target_da"],
        "target_rt": arrays["target_rt"],
    }
    prefix = "economic_spread_factorization"
    suffix = f"{archive_phase}_seed{cfg.seed}.npz"
    _save_archive(
        output_dir / f"{prefix}_da_{suffix}",
        {
            **common,
            "prediction": arrays["da"],
            "metadata": np.asarray(json.dumps({
                **metadata, "family": prefix, "task": "da",
            }, ensure_ascii=False)),
        },
    )
    _save_archive(
        output_dir / f"{prefix}_rt_at_da_{suffix}",
        {
            **common,
            "prediction": arrays["rt"],
            "metadata": np.asarray(json.dumps({
                **metadata, "family": prefix, "task": "rt_at_da",
            }, ensure_ascii=False)),
        },
    )
    _save_archive(
        output_dir / f"{prefix}_diagnostics_{suffix}",
        {
            **common,
            "metadata": np.asarray(json.dumps({
                **metadata, "family": prefix, "task": "diagnostics",
            }, ensure_ascii=False)),
            **{
                key: arrays[key]
                for key in (
                    "base_da", "base_rt", "state", "probability", "excess",
                    "da_adjustment", "rt_adjustment",
                )
            },
        },
    )
    if args.rolling_rt_archive:
        source = ROOT / args.rolling_rt_archive
        with np.load(source, allow_pickle=False) as archive:
            if not np.array_equal(nodes, archive["nodes"]):
                raise ValueError("rolling RT archive节点与分解模型不一致")
            rolling = {key: archive[key] for key in archive.files}
            source_metadata = json.loads(str(rolling["metadata"].item()))
        rolling["metadata"] = np.asarray(json.dumps({
            **source_metadata,
            "family": prefix,
            "task": "rt",
            "phase": archive_phase,
            "method": "unchanged_original_transformer_rolling_rt",
            "economic_spread_factorization_applied": False,
        }, ensure_ascii=False))
        _save_archive(output_dir / f"{prefix}_rt_{suffix}", rolling)
    report = {
        "protocol": "economic_spread_factorization_lomo_holdout_v1",
        "metadata": metadata,
        "aggregate": aggregate,
        "per_node": per_node,
    }
    report_path = output_dir / f"report_{args.stage}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete", "report": str(report_path),
        "aggregate": aggregate,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

