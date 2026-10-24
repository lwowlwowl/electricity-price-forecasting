#!/usr/bin/env python
"""Build and freeze an interpretable profit-first Transformer/DLinear router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.tail_expert_router import (  # noqa: E402
    blocked_oof_probability,
    build_feature_cube,
    fit_router,
    model_explanation,
    router_probability,
)


TASKS = ("da", "rt_at_da", "rt")
KINDS = ("logistic", "tree")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80)
DEFAULT_LABEL = "da_universal__rt_at_da_universal__rt_universal"
ALT_LABEL = "da_dlinear__rt_at_da_dlinear__rt_universal"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _sources(root: Path, phase: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    result = {
        (family, task): _load(root / f"{family}_{task}_{phase}_seed0.npz")
        for family in ("universal", "dlinear") for task in TASKS
    }
    reference = result[("universal", "da")]
    for (family, task), item in result.items():
        if not np.array_equal(item["nodes"], reference["nodes"]):
            raise ValueError(f"节点不一致: {family}/{task}")
    for task in TASKS:
        left = result[("universal", task)]
        right = result[("dlinear", task)]
        if not np.array_equal(left["dates"], right["dates"]):
            raise ValueError(f"两专家日期不一致: {task}")
        for key in ("target_da", "target_rt"):
            if key in left and (
                key not in right
                or left[key].shape != right[key].shape
                or not np.allclose(left[key], right[key], rtol=0.0, atol=1e-5)
            ):
                raise ValueError(f"真值不一致: {task}/{key}")
    for family in ("universal", "dlinear"):
        if not np.array_equal(
            result[(family, "da")]["dates"], result[(family, "rt_at_da")]["dates"]
        ):
            raise ValueError(f"{family}的DA与RT-at-DA日期不一致")
    return result


def _final_sources(root: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load the already-frozen 2025 Transformer ensemble and DLinear baseline."""
    result = {
        (family, task): _load(
            root / f"{'ensemble' if family == 'universal' else 'dlinear'}_{task}_final_seed0.npz"
        )
        for family in ("universal", "dlinear") for task in TASKS
    }
    reference = result[("universal", "da")]
    for (family, task), item in result.items():
        if not np.array_equal(item["nodes"], reference["nodes"]):
            raise ValueError(f"2025节点不一致: {family}/{task}")
    # The frozen Transformer DA archives contain 31 December whereas both the
    # rolling-RT contract and DLinear end on 30 December.  Route only dates
    # present for both experts; on an unmatched date the alternate prediction
    # is set equal to the Transformer so the decision safely defaults to it.
    for task in TASKS:
        left = result[("universal", task)]
        right = result[("dlinear", task)]
        if not np.array_equal(left["dates"], right["dates"]):
            right_index = {
                str(value): index for index, value in enumerate(right["dates"].astype(str))
            }
            aligned_prediction = np.array(left["prediction"], copy=True)
            for left_index, value in enumerate(left["dates"].astype(str)):
                if value in right_index:
                    aligned_prediction[:, left_index] = right["prediction"][:, right_index[value]]
            aligned = dict(left)
            aligned["prediction"] = aligned_prediction
            result[("dlinear", task)] = aligned
    for family in ("universal", "dlinear"):
        if not np.array_equal(
            result[(family, "da")]["dates"], result[(family, "rt_at_da")]["dates"]
        ):
            raise ValueError(f"2025 {family}的DA与RT-at-DA日期不一致")
    return result


def _dataset(
    sources: dict[tuple[str, str], dict[str, np.ndarray]],
    settlement_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    universal_da = sources[("universal", "da")]
    cube, feature_names = build_feature_cube(
        universal_da["prediction"],
        sources[("universal", "rt_at_da")]["prediction"],
        sources[("dlinear", "da")]["prediction"],
        sources[("dlinear", "rt_at_da")]["prediction"],
        universal_da["dates"],
    )
    report = json.loads(settlement_path.read_text(encoding="utf-8"))
    if ALT_LABEL not in report["systems"] or DEFAULT_LABEL not in report["systems"]:
        raise ValueError("历史结算缺少纯Transformer或默认混合系统")
    nodes = universal_da["nodes"].astype(str)
    dates = universal_da["dates"].astype(str)
    alt = report["systems"][ALT_LABEL]["nodes"]
    default = report["systems"][DEFAULT_LABEL]["nodes"]
    common_dates = sorted(
        set(dates)
        & set(alt[nodes[0]]["daily_net_revenue"])
        & set(default[nodes[0]]["daily_net_revenue"])
    )
    if not common_dates:
        raise ValueError("预测与结算没有共同日期")
    date_index = {date: index for index, date in enumerate(dates)}
    day_indices = np.asarray([date_index[date] for date in common_dates], dtype=np.int64)
    advantage = np.empty((len(nodes), len(common_dates)), dtype=np.float64)
    for node_index, node in enumerate(nodes):
        if set(alt[node]["daily_net_revenue"]) != set(alt[nodes[0]]["daily_net_revenue"]):
            raise ValueError(f"节点结算日期不一致: {node}")
        for selected_index, date in enumerate(common_dates):
            advantage[node_index, selected_index] = (
                float(alt[node]["daily_net_revenue"][date])
                - float(default[node]["daily_net_revenue"][date])
            )
    selected_cube = cube[:, day_indices]
    row_dates = np.tile(np.asarray(common_dates), len(nodes))
    return (
        selected_cube.reshape(-1, selected_cube.shape[-1]),
        advantage.reshape(-1),
        row_dates,
        cube,
        feature_names,
        day_indices,
    )


def _write_family(
    sources: dict[tuple[str, str], dict[str, np.ndarray]],
    output: Path,
    family: str,
    phase: str,
    select_universal: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    reference = sources[("universal", "da")]
    expected = reference["prediction"].shape[:2]
    if select_universal.shape != expected:
        raise ValueError(f"路由决策形状错误: {select_universal.shape} != {expected}")
    output.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        universal = sources[("universal", task)]
        dlinear = sources[("dlinear", task)]
        if task == "rt":
            prediction = universal["prediction"]
        else:
            prediction = np.where(
                select_universal[..., None], universal["prediction"], dlinear["prediction"]
            )
        arrays = dict(universal)
        arrays["prediction"] = prediction.astype(np.float32)
        arrays["metadata"] = np.asarray(json.dumps({
            **metadata,
            "family": family,
            "task": task,
            "phase": phase,
            "selection_rate_universal": float(np.mean(select_universal)),
            "uses_future_rt_forecast_for_routing": False,
            "locked_2025_read": bool(metadata.get("locked_2025_read", False)),
        }, ensure_ascii=False, sort_keys=True))
        np.savez_compressed(output / f"{family}_{task}_{phase}_seed0.npz", **arrays)


def _classification_record(
    advantage: np.ndarray,
    probability: np.ndarray,
    decision: np.ndarray,
) -> dict[str, Any]:
    target = advantage > 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, decision, average="binary", zero_division=0
    )
    return {
        "event_rate": float(target.mean()),
        "selection_rate_alternate_hybrid": float(decision.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            float(roc_auc_score(target, probability))
            if np.unique(target).size == 2 else None
        ),
        "mean_realized_advantage_when_selected": (
            float(advantage[decision].mean()) if decision.any() else None
        ),
    }


def build_inner(args: argparse.Namespace) -> None:
    source_root = ROOT / args.source_root
    output = ROOT / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    sources = _sources(source_root, "inner")
    features, advantage, row_dates, cube, feature_names, day_indices = _dataset(
        sources, ROOT / args.settlement
    )
    node_count, day_count = cube.shape[:2]
    _write_family(
        sources, output, "universal", "inner", np.ones((node_count, day_count), dtype=bool),
        {"protocol": "tail_expert_router_v1", "role": "pure_transformer"},
    )
    _write_family(
        sources, output, "hybrid", "inner", np.zeros((node_count, day_count), dtype=bool),
        {"protocol": "tail_expert_router_v1", "role": "dlinear_da_rt_at_da_transformer_rt"},
    )
    candidates: list[dict[str, Any]] = []
    for kind in KINDS:
        probability, folds = blocked_oof_probability(
            kind, features, advantage, row_dates, blocks=5
        )
        fold_explanations = [
            {
                "fold": item["fold"],
                "held_start": item["held_dates"][0],
                "held_end": item["held_dates"][-1],
                "explanation": model_explanation(item["model"], feature_names),
            }
            for item in folds
        ]
        complexity = max(
            item["explanation"]["feature_count"] for item in fold_explanations
        )
        for threshold in THRESHOLDS:
            selected_hybrid = (probability >= threshold).reshape(node_count, len(day_indices))
            # Default to the 2023 stronger pure Transformer.  Only fold-out
            # evidence may route a node-day to the DLinear DA/RT-at-DA expert.
            select_universal = np.ones((node_count, day_count), dtype=bool)
            select_universal[:, day_indices] = ~selected_hybrid
            family = f"router_{kind}_{int(round(threshold * 100))}"
            _write_family(
                sources,
                output,
                family,
                "inner",
                select_universal,
                {
                    "protocol": "tail_expert_router_v1",
                    "role": "blocked_oof_candidate",
                    "kind": kind,
                    "threshold": threshold,
                },
            )
            candidates.append({
                "family": family,
                "kind": kind,
                "threshold": threshold,
                "feature_count": complexity,
                **_classification_record(advantage, probability, selected_hybrid.reshape(-1)),
            })
    manifest = {
        "protocol": "tail_expert_router_v1",
        "phase": "inner_oof",
        "objective": "complete_net_revenue_primary",
        "default_system": DEFAULT_LABEL,
        "alternate_system": ALT_LABEL,
        "features": list(feature_names),
        "thresholds": list(THRESHOLDS),
        "candidates": candidates,
        "locked_2025_read": False,
    }
    (output / "inner_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "inner_router_candidates_ready", "candidates": len(candidates)}))


def _system_label(family: str) -> str:
    return f"da_{family}__rt_at_da_{family}__rt_{family}"


def freeze_outer(args: argparse.Namespace) -> None:
    source_root = ROOT / args.source_root
    output = ROOT / args.output_root
    manifest = json.loads((output / "inner_manifest.json").read_text(encoding="utf-8"))
    settlement = json.loads((ROOT / args.inner_router_settlement).read_text(encoding="utf-8"))
    ranked: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        label = _system_label(candidate["family"])
        ranked.append({
            **candidate,
            "inner_profit": float(settlement["systems"][label]["macro_mean_daily_revenue"]),
        })
    best_profit = max(item["inner_profit"] for item in ranked)
    contenders = [item for item in ranked if best_profit - item["inner_profit"] < 1.0]
    selected = sorted(
        contenders,
        key=lambda item: (
            item["feature_count"],
            0 if item["kind"] == "logistic" else 1,
            -item["inner_profit"],
            item["threshold"],
        ),
    )[0]

    inner_sources = _sources(source_root, "inner")
    features, advantage, _, _, feature_names, _ = _dataset(
        inner_sources, ROOT / args.inner_cross_settlement
    )
    model = fit_router(selected["kind"], features, advantage)
    model_path = output / "router_frozen_2023.joblib"
    joblib.dump(model, model_path)

    outer_sources = _sources(source_root, "outer")
    outer_reference = outer_sources[("universal", "da")]
    outer_cube, outer_names = build_feature_cube(
        outer_reference["prediction"],
        outer_sources[("universal", "rt_at_da")]["prediction"],
        outer_sources[("dlinear", "da")]["prediction"],
        outer_sources[("dlinear", "rt_at_da")]["prediction"],
        outer_reference["dates"],
    )
    if outer_names != feature_names:
        raise ValueError("内外层特征顺序不一致")
    probability = router_probability(model, outer_cube.reshape(-1, outer_cube.shape[-1]))
    select_hybrid = probability >= float(selected["threshold"])
    shape = outer_cube.shape[:2]
    _write_family(
        outer_sources, output, "universal", "outer", np.ones(shape, dtype=bool),
        {"protocol": "tail_expert_router_v1", "role": "pure_transformer"},
    )
    _write_family(
        outer_sources, output, "hybrid", "outer", np.zeros(shape, dtype=bool),
        {"protocol": "tail_expert_router_v1", "role": "dlinear_da_rt_at_da_transformer_rt"},
    )
    _write_family(
        outer_sources,
        output,
        "router_final",
        "outer",
        (~select_hybrid).reshape(shape),
        {
            "protocol": "tail_expert_router_v1",
            "role": "frozen_2023_router_applied_to_2024",
            "kind": selected["kind"],
            "threshold": selected["threshold"],
        },
    )
    frozen = {
        "protocol": "tail_expert_router_v1",
        "selected_on": "2023 blocked out-of-fold complete LP/MPC profit",
        "selected": selected,
        "candidate_ranking": sorted(ranked, key=lambda item: item["inner_profit"], reverse=True),
        "model_explanation": model_explanation(model, feature_names),
        "outer_selection_rate_hybrid": float(select_hybrid.mean()),
        "locked_2025_read": False,
        "outer_model_changes_allowed": False,
    }
    (output / "frozen_router.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "outer_router_ready", "selected": selected}, ensure_ascii=False))


def build_followup(args: argparse.Namespace) -> None:
    """Refit the frozen 2023 mechanism on 2023+2024 OOS records and apply once."""
    source_root = ROOT / args.source_root
    output = ROOT / args.output_root
    frozen = json.loads((output / "frozen_router.json").read_text(encoding="utf-8"))
    selected = frozen["selected"]

    feature_parts: list[np.ndarray] = []
    advantage_parts: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for phase, settlement_path in (
        ("inner", ROOT / args.inner_cross_settlement),
        ("outer", ROOT / args.outer_cross_settlement),
    ):
        sources = _sources(source_root, phase)
        features, advantage, _, _, names, _ = _dataset(sources, settlement_path)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("2023与2024特征顺序不一致")
        feature_parts.append(features)
        advantage_parts.append(advantage)
    if feature_names is None:
        raise AssertionError("未载入历史路由训练数据")

    features = np.concatenate(feature_parts, axis=0)
    advantage = np.concatenate(advantage_parts, axis=0)
    model = fit_router(str(selected["kind"]), features, advantage)
    joblib.dump(model, output / "router_refit_2023_2024.joblib")

    final_sources = _final_sources(ROOT / args.final_source_root)
    reference = final_sources[("universal", "da")]
    cube, final_names = build_feature_cube(
        reference["prediction"],
        final_sources[("universal", "rt_at_da")]["prediction"],
        final_sources[("dlinear", "da")]["prediction"],
        final_sources[("dlinear", "rt_at_da")]["prediction"],
        reference["dates"],
    )
    if final_names != feature_names:
        raise ValueError("历史与2025特征顺序不一致")
    probability = router_probability(model, cube.reshape(-1, cube.shape[-1]))
    select_hybrid = (probability >= float(selected["threshold"])).reshape(cube.shape[:2])
    _write_family(
        final_sources,
        output,
        "router_followup",
        "final",
        ~select_hybrid,
        {
            "protocol": "tail_expert_router_v1",
            "role": "fixed_post_original_test_followup",
            "kind": selected["kind"],
            "threshold": selected["threshold"],
            "locked_2025_read": True,
        },
    )
    followup = {
        "protocol": "tail_expert_router_v1",
        "status": "post_original_2025_test_fixed_followup",
        "mechanism_selected_on": "2023 blocked OOF",
        "outer_confirmation": "2024 evaluated once without reselection",
        "refit_records": "ERCOT 2023+2024 out-of-sample node-days",
        "kind": selected["kind"],
        "threshold": selected["threshold"],
        "model_explanation": model_explanation(model, feature_names),
        "selection_rate_hybrid_2025": float(select_hybrid.mean()),
        "further_2025_tuning_allowed": False,
    }
    (output / "followup_router.json").write_text(
        json.dumps(followup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "followup_router_ready", **followup}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True,
        choices=("build_inner", "freeze_outer", "build_followup"),
    )
    parser.add_argument(
        "--source-root", default="data/results/historical_transformer_dlinear_profit_v1"
    )
    parser.add_argument(
        "--output-root", default="data/results/tail_expert_router_v1"
    )
    parser.add_argument(
        "--settlement",
        default="data/results/historical_dlinear_router_inputs_v1/cross_task_inner_mpc_v1.json",
    )
    parser.add_argument(
        "--inner-router-settlement",
        default="data/results/tail_expert_router_v1/inner_router_mpc_v1.json",
    )
    parser.add_argument(
        "--inner-cross-settlement",
        default="data/results/historical_dlinear_router_inputs_v1/cross_task_inner_mpc_v1.json",
    )
    parser.add_argument(
        "--outer-cross-settlement",
        default="data/results/historical_dlinear_router_inputs_v1/cross_task_outer_mpc_v1.json",
    )
    parser.add_argument(
        "--final-source-root",
        default="data/results/allmarket_locked_2025_all_profit_v2",
    )
    args = parser.parse_args()
    if args.mode == "build_inner":
        build_inner(args)
    elif args.mode == "freeze_outer":
        freeze_outer(args)
    else:
        build_followup(args)


if __name__ == "__main__":
    main()
