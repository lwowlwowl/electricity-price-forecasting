#!/usr/bin/env python
"""Export leakage-safe source-market caches for the pure-v4 multiscale head.

The cache contains only inputs already consumed by v4, the three frozen v4 DA
members, and a deterministic 168-hour seasonal anchor.  It deliberately does
not contain EXAA, foundation-model forecasts, or target-market fitted values.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    EntityMappings,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import load_market_unified_nodes  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.portable_normalization import (  # noqa: E402
    normalize,
    robust_context_stats,
)


CHECKPOINT_TEMPLATE = (
    "data/checkpoints/global_joint_implicit_qp_final_v4/"
    "implicitqp_final_decoder_balanced_seed{seed}_v4/da.pt"
)
CONTEXT_KEYS = (
    "price_da_ctx",
    "price_rt_ctx",
    "load_ctx",
    "system_ctx",
    "cal_ctx",
)
FORECAST_WEATHER = (
    "forecast_temperature_2m_c",
    "forecast_relative_humidity_2m_pct",
    "forecast_dew_point_2m_c",
    "forecast_cloud_cover_pct",
    "forecast_precipitation_mm",
    "forecast_wind_speed_10m_ms",
    "forecast_wind_gust_10m_ms",
    "forecast_shortwave_radiation_wm2",
)


def _config(checkpoint: dict) -> PilotConfig:
    allowed = {item.name for item in fields(PilotConfig)}
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in allowed
    })


def _context(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    streams = []
    for key in CONTEXT_KEYS:
        value = batch[key].float()
        if value.ndim == 2:
            value = value.unsqueeze(-1)
        streams.append(value)
    return torch.cat(streams, dim=-1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markets", nargs="+", default=["CAISO", "NYISO", "PJM", "ERCOT"]
    )
    parser.add_argument(
        "--output-dir", default="data/results/pure_v4_multiscale_cache_v1"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--stage", choices=("inner_validation", "final_train"),
        default="inner_validation",
    )
    parser.add_argument(
        "--one-node-per-market", action="store_true",
        help="Use one stable representative node per market for long multi-year coverage.",
    )
    args = parser.parse_args()

    checkpoint_paths = [ROOT / CHECKPOINT_TEMPLATE.format(seed=seed) for seed in range(3)]
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    config = _config(checkpoints[0])
    compatibility_fields = (
        "context_len", "horizon_da", "d_model", "n_heads_enc",
        "n_heads_fusion", "n_layers_enc", "n_layers_fusion", "dim_ff",
        "da_fusion_mode", "global_normalization_mode",
        "global_use_entity_embeddings",
    )
    for checkpoint in checkpoints[1:]:
        member_config = _config(checkpoint)
        mismatches = {
            name: (getattr(config, name), getattr(member_config, name))
            for name in compatibility_fields
            if getattr(config, name) != getattr(member_config, name)
        }
        if mismatches:
            raise ValueError(f"三个v4成员的结构配置不一致: {mismatches}")
    if config.global_use_entity_embeddings:
        raise ValueError("纯v4跨市场缓存禁止身份embedding")
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = []
    for checkpoint in checkpoints:
        model = DecisionAwareDAForecaster(config).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        models.append(model)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    for market in args.markets:
        mappings = full_mappings.subset([market])
        entities = [
            (mappings.node_to_market[node], node) for node in mappings.node_to_id
        ]
        if args.one_node_per_market:
            entities = entities[:1]
            node = entities[0][1]
            mappings = EntityMappings(
                market_to_id={market: 0},
                node_to_id={node: 0},
                node_to_market={node: market},
            )
        evaluation_config = replace(config, global_use_critical_event_weighting=False)
        train_dataset, validation_dataset = build_global_task_datasets(
            evaluation_config, "da", entities, mappings,
            train_stage=(
                AnnualStage.FINAL_TRAIN
                if args.stage == "final_train" else AnnualStage.INNER_TRAIN
            ),
            validation_stage=(
                AnnualStage.LOCKED_RETROSPECTIVE_TEST
                if args.stage == "final_train" else AnnualStage.INNER_VALIDATION
            ),
        )
        dataset = (
            train_dataset if args.stage == "final_train" else validation_dataset
        )
        weather_by_node = load_market_unified_nodes(
            market,
            [node for _, node in dataset.entities],
            start="2020-01-01",
            end=("2025-01-02" if args.stage == "final_train" else "2024-01-02"),
            include_weather=True,
            dropna=False,
        )

        node_rows: dict[str, list[np.ndarray]] = {
            "context": [], "calendar": [], "experts": [], "target": [],
            "mean": [], "std": [], "dates": [], "future_weather": [],
            "fundamental_target": [], "aligned_da_history": [],
        }
        for entity_index, ((_, node), node_dataset) in enumerate(
            zip(dataset.entities, dataset.datasets)
        ):
            start, stop = dataset.offsets[entity_index:entity_index + 2]
            loader = DataLoader(
                Subset(dataset, range(start, stop)),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            rows = {key: [] for key in ("context", "calendar", "experts", "target", "mean", "std")}
            for raw in loader:
                model_batch = {
                    key: value.to(device, non_blocking=True)
                    if torch.is_tensor(value) else value
                    for key, value in raw.items()
                }
                predictions = torch.stack(
                    [model(model_batch)["p_da"].float().cpu() for model in models],
                    dim=-1,
                )
                rows["context"].append(_context(raw).half().numpy())
                rows["calendar"].append(raw["cal_tgt"].half().numpy())
                rows["experts"].append(predictions.numpy())
                rows["target"].append(raw["price_da_tgt"].float().numpy())
                rows["mean"].append(raw["price_da_mean"].float().numpy())
                rows["std"].append(raw["price_da_std"].float().numpy())

            seasonal = np.stack([
                node_dataset.price_da[window.target_positions - 168]
                for window in node_dataset.windows
            ]).astype(np.float32)
            experts = np.concatenate(rows["experts"], axis=0)
            if seasonal.shape != experts.shape[:2]:
                raise ValueError(f"{market}/{node}的周周期锚点形状不一致")
            experts = np.concatenate([experts, seasonal[..., None]], axis=-1)
            price_mean = np.concatenate(rows["mean"], axis=0)
            price_std = np.concatenate(rows["std"], axis=0)
            # The seven most recent complete DA curves are already published
            # before the next delivery day.  Unlike the noon-to-noon model
            # context, these profiles are aligned to delivery hour 0..23.
            aligned_da_history = np.stack([
                np.stack([
                    node_dataset.price_da[window.target_positions - 24 * lag]
                    for lag in range(7, 0, -1)
                ])
                for window in node_dataset.windows
            ]).astype(np.float32)
            aligned_da_history = (
                aligned_da_history - price_mean[:, None, None]
            ) / np.maximum(price_std[:, None, None], 1.0e-4)
            weather_frame = weather_by_node[node].reindex(node_dataset.index)
            future_weather = np.stack([
                weather_frame.iloc[window.target_positions][
                    list(FORECAST_WEATHER)
                ].to_numpy(np.float32)
                for window in node_dataset.windows
            ])
            if not np.isfinite(future_weather).all():
                raise ValueError(f"{market}/{node}的2023未来天气预测不完整")
            fundamental_target = []
            for window in node_dataset.windows:
                context_slice = slice(window.context_start, window.context_end)
                target_positions = window.target_positions
                columns = []
                for stream, column in (("load", 0), ("system", 0), ("system", 1)):
                    values = node_dataset.streams[stream][:, column]
                    context_values = values[context_slice]
                    stats = robust_context_stats(context_values)
                    # Solar has many physically valid near-zero night values.
                    # Its median absolute deviation can therefore be tiny but
                    # non-zero, which bypasses the generic zero-MAD fallback.
                    # Keep the causal robust centre, but floor the scale by the
                    # same 168-hour context's standard deviation and central
                    # 90% range.  No target-window value is consulted.
                    q05, q95 = np.quantile(context_values, [0.05, 0.95])
                    stable_stats = dict(stats)
                    stable_stats["std"] = max(
                        float(stats["std"]),
                        float(np.std(context_values)),
                        0.20 * float(q95 - q05),
                        0.05 * abs(float(stats["mean"])),
                        1.0,
                    )
                    columns.append(normalize(values[target_positions], stable_stats))
                fundamental_target.append(np.stack(columns, axis=-1))
            node_rows["context"].append(np.concatenate(rows["context"], axis=0))
            node_rows["calendar"].append(np.concatenate(rows["calendar"], axis=0))
            node_rows["experts"].append(experts)
            node_rows["aligned_da_history"].append(
                aligned_da_history.astype(np.float16)
            )
            node_rows["target"].append(np.concatenate(rows["target"], axis=0))
            node_rows["mean"].append(np.concatenate(rows["mean"], axis=0))
            node_rows["std"].append(np.concatenate(rows["std"], axis=0))
            node_rows["future_weather"].append(future_weather.astype(np.float16))
            node_rows["fundamental_target"].append(
                np.stack(fundamental_target).astype(np.float16)
            )
            node_rows["dates"].append(np.asarray([
                str(window.delivery_date) for window in node_dataset.windows
            ]))
            print(f"{market}/{node}: {len(node_dataset)} days", flush=True)

        dates = node_rows["dates"][0]
        if any(not np.array_equal(dates, item) for item in node_rows["dates"][1:]):
            raise ValueError(f"{market}各节点日期轴不一致")
        np.savez_compressed(
            output_dir / f"{market.lower()}_{'2020_2024' if args.stage == 'final_train' else '2023'}.npz",
            nodes=np.asarray([node for _, node in dataset.entities]),
            dates=dates,
            context=np.stack(node_rows["context"]),
            calendar=np.stack(node_rows["calendar"]),
            experts=np.stack(node_rows["experts"]),
            target=np.stack(node_rows["target"]),
            price_mean=np.stack(node_rows["mean"]),
            price_std=np.stack(node_rows["std"]),
            future_weather=np.stack(node_rows["future_weather"]),
            fundamental_target=np.stack(node_rows["fundamental_target"]),
            aligned_da_history=np.stack(node_rows["aligned_da_history"]),
            weather_names=np.asarray(FORECAST_WEATHER),
            expert_names=np.asarray([
                "v4_seed0", "v4_seed1", "v4_seed2", "seasonal_lag168"
            ]),
            protocol=np.asarray("pure_v4_multiscale_cache_v1"),
        )


if __name__ == "__main__":
    main()
