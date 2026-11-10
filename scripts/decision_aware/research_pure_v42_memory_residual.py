#!/usr/bin/env python
"""Causal experience-conditioned memory correction for pure v4.

Inspired by MemCast (ICML 2026), this module retrieves previously revealed
delivery days with similar frozen-v4 curves and operating states, then carries
their forecast residual forward.  It never fits target-market parameters: the
memory rule and all hyperparameters are selected on U.S. source markets, while
Austria only supplies already-published past experiences at inference time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import AustriaExternalData, calendar_features  # noqa: E402
from scripts.decision_aware.evaluate_austria_da_mae_365 import (  # noqa: E402
    extend_rt_history,
    load_apcs_imbalance,
)
from scripts.decision_aware.run_austria_external_15min import _forecast_axes  # noqa: E402


SOURCE_MARKETS = ("caiso", "nyiso", "pjm")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _source_arrays(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = payload["price_mean"][0].astype(np.float64)[:, None]
    std = np.maximum(payload["price_std"][0].astype(np.float64)[:, None], 1.0)
    experts = (payload["experts"][0].astype(np.float64) - mean[:, :, None]) / std[:, :, None]
    anchor = (2.0 / 3.0) * experts[:, :, 0] + (1.0 / 3.0) * np.median(
        experts[:, :, :3], axis=2
    )
    truth = (payload["target"][0].astype(np.float64) - mean) / std
    return {
        "anchor": anchor,
        "truth": truth,
        "fundamentals": payload["fundamental_target"][0].astype(np.float64),
        "calendar": payload["calendar"][0, :, 0, 2:].astype(np.float64),
    }


def _memory_predict(
    anchor: np.ndarray,
    truth: np.ndarray,
    fundamentals: np.ndarray,
    calendar: np.ndarray,
    *,
    memory_days: int,
    neighbors: int,
    base_weight: float,
    physical_weight: float,
    calendar_weight: float,
    gate: float,
    aggregation: str,
    granularity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = anchor.copy()
    retrieved = np.full((len(anchor), 24, neighbors), -1, dtype=np.int16)
    confidence = np.zeros(len(anchor), dtype=np.float64)
    residual = truth - anchor
    for day in range(1, len(anchor)):
        first = max(0, day - memory_days)
        candidates = np.arange(first, day)
        calendar_distance = np.mean(
            (calendar[candidates] - calendar[day]) ** 2, axis=1
        )
        if granularity == "day":
            distances = [
                base_weight * np.mean((anchor[candidates] - anchor[day]) ** 2, axis=1)
                + physical_weight * np.mean(
                    (fundamentals[candidates] - fundamentals[day]) ** 2, axis=(1, 2)
                )
                + calendar_weight * calendar_distance
            ] * 24
        else:
            distances = [
                base_weight * (anchor[candidates, hour] - anchor[day, hour]) ** 2
                + physical_weight * np.mean(
                    (fundamentals[candidates, hour] - fundamentals[day, hour]) ** 2,
                    axis=1,
                )
                + calendar_weight * calendar_distance
                for hour in range(24)
            ]
        hour_confidence = []
        for hour, distance in enumerate(distances):
            count = min(neighbors, len(candidates))
            selected_pos = np.argpartition(distance, count - 1)[:count]
            selected = candidates[selected_pos]
            selected_distance = distance[selected_pos]
            retrieved[day, hour, :count] = selected
            scale = max(float(np.median(distance)), 1.0e-6)
            local_confidence = float(np.exp(-float(selected_distance.min()) / scale))
            hour_confidence.append(local_confidence)
            values = residual[selected, hour]
            if aggregation == "median":
                correction = float(np.nanmedian(values))
            else:
                logits = -selected_distance / scale
                logits -= logits.max()
                weights = np.exp(logits)
                available = np.isfinite(values)
                correction = float(
                    np.nansum(weights * values) / max(float(weights[available].sum()), 1.0e-12)
                )
            if not np.isfinite(correction):
                correction = 0.0
            prediction[day, hour] = (
                anchor[day, hour] + gate * local_confidence * correction
            )
        confidence[day] = float(np.mean(hour_confidence))
    return prediction, retrieved, confidence


def _evaluate_source(arrays: dict, config: dict, start: int) -> float:
    prediction, _, _ = _memory_predict(**arrays, **config)
    return float(np.mean(np.abs(prediction[start:] - arrays["truth"][start:])))


def _select(source: dict[str, dict]) -> tuple[dict, list[dict]]:
    rows = []
    # The day-level source audit already selected the memory geometry below.
    # The hour-level ablation changes only aggregation granularity and gate,
    # avoiding a second broad search after the external-test protocol began.
    for memory_days in (112,):
        for neighbors in (7,):
            for base_weight, physical_weight, calendar_weight in (
                (1.0, 2.0, 0.5),
            ):
                for gate in (0.25, 0.5, 0.75, 1.0):
                    for aggregation in ("mean", "median"):
                        for granularity in ("hour",):
                            config = {
                                "memory_days": memory_days,
                                "neighbors": neighbors,
                                "base_weight": base_weight,
                                "physical_weight": physical_weight,
                                "calendar_weight": calendar_weight,
                                "gate": gate,
                                "aggregation": aggregation,
                                "granularity": granularity,
                            }
                            scores = {
                                market: _evaluate_source(
                                    source[market], config,
                                    len(source[market]["anchor"]) - 365,
                                )
                                for market in SOURCE_MARKETS
                            }
                            rows.append({
                                **config,
                                "source_scores": scores,
                                "selection_score": float(
                                    np.mean(list(scores.values()))
                                    + 0.25 * max(scores.values())
                                ),
                            })
    rows.sort(key=lambda item: item["selection_score"])
    keys = (
        "memory_days", "neighbors", "base_weight", "physical_weight",
        "calendar_weight", "gate", "aggregation", "granularity",
    )
    return {key: rows[0][key] for key in keys}, rows


def _austria_arrays(
    cache_path: Path,
    hourly_anchor_path: Path,
    protocol: Path,
    september_apcs: Path,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    cache = _load(cache_path)
    hourly_anchor = _load(hourly_anchor_path)["da_usd_per_mwh"].astype(np.float64)
    mean = cache["price_context_mean_usd_per_mwh"].astype(np.float64)[:, None]
    std = np.maximum(cache["price_context_std_usd_per_mwh"].astype(np.float64)[:, None], 1.0)
    anchor = (hourly_anchor - mean) / std
    fundamentals = cache["known_fundamentals"].astype(np.float64)
    dates = pd.to_datetime(cache["delivery_dates"].astype(str))
    calendar = np.stack([
        calendar_features(pd.date_range(day, periods=24, freq="h"))[0, 2:]
        for day in dates
    ]).astype(np.float64)

    base, _ = AustriaExternalData.load(ROOT, protocol)
    extended = extend_rt_history(base, load_apcs_imbalance(september_apcs))
    quarter = extended.evaluation_quarters(complete_settlement=False)
    forecast_dates, _, local = _forecast_axes(extended, quarter)
    date_pos = {date: pos for pos, date in enumerate(forecast_dates)}
    truth_eur = quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    table = pd.DataFrame({
        "day": [date_pos[item] for item in local.date],
        "hour": local.hour,
        "truth": truth_eur * extended.eur_to_usd,
    }).groupby(["day", "hour"])["truth"].mean().unstack()
    truth_usd = table.reindex(index=range(len(dates)), columns=range(24)).to_numpy(np.float64)
    truth = (truth_usd - mean) / std
    return {
        "anchor": anchor,
        "truth": truth,
        "fundamentals": fundamentals,
        "calendar": calendar,
    }, cache, mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", default="data/results/pure_v4_multiscale_cache_5y_1node_v1")
    parser.add_argument("--austria-cache", default="data/results/austria_pure_v4_multiscale_v1/predictions.npz")
    parser.add_argument("--hourly-anchor", default="data/results/austria_pure_v4_consensus_v1/predictions.npz")
    parser.add_argument("--protocol", default="configs/decision_aware/austria_external_15min_v1.json")
    parser.add_argument("--september-apcs", default="data/Austria/Statistik-RZ_202609_NEU.xlsx")
    parser.add_argument("--output-dir", default="data/results/austria_pure_v42_memory_residual_v1")
    parser.add_argument("--reuse-selected", action="store_true")
    args = parser.parse_args()

    source_root = ROOT / args.source_cache
    source = {
        market: _source_arrays(_load(source_root / f"{market}_2020_2024.npz"))
        for market in (*SOURCE_MARKETS, "ercot")
    }
    output = ROOT / args.output_dir
    if args.reuse_selected and (output / "summary.json").exists():
        prior_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        selected = prior_summary["source_selected_config"]
        search = json.loads((output / "source_search.json").read_text(encoding="utf-8"))
    else:
        selected, search = _select(source)
    ercot_prediction, _, _ = _memory_predict(**source["ercot"], **selected)
    ercot_start = len(ercot_prediction) - 365
    ercot = {
        "base_normalized_mae": float(np.mean(np.abs(source["ercot"]["anchor"][ercot_start:] - source["ercot"]["truth"][ercot_start:]))),
        "candidate_normalized_mae": float(np.mean(np.abs(ercot_prediction[ercot_start:] - source["ercot"]["truth"][ercot_start:]))),
    }
    austria, raw, mean, std = _austria_arrays(
        ROOT / args.austria_cache, ROOT / args.hourly_anchor,
        ROOT / args.protocol, ROOT / args.september_apcs,
    )
    prediction_z, retrieved, confidence = _memory_predict(**austria, **selected)
    prediction_usd = mean + std * prediction_z

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        delivery_dates=raw["delivery_dates"],
        da_usd_per_mwh=prediction_usd.astype(np.float32),
        da_eur_per_mwh=(prediction_usd / 1.0824).astype(np.float32),
        confidence=confidence.astype(np.float32),
        retrieved_day_indices=retrieved,
    )
    summary = {
        "method": "pure-v4.2-causal-experience-conditioned-memory-candidate",
        "literature_basis": "MemCast (ICML 2026)",
        "source_selected_config": selected,
        "source_selection_scores": search[0]["source_scores"],
        "ercot_heldout": ercot,
        "uses_exaa": False,
        "uses_external_model_predictions": False,
        "uses_other_european_training_data": False,
        "uses_austria_labels_for_training_or_selection": False,
        "uses_only_past_published_austria_prices_as_causal_memory": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_search.json").write_text(
        json.dumps(search, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
