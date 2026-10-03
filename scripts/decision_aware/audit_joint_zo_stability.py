#!/usr/bin/env python
"""对三模型联合训练的零阶信号做不更新参数的稳定性审计。

这个脚本不使用验证收益选超参数。它只回答三个工程问题：
1. 相同episode重复估计时，零阶梯度方向是否稳定；
2. 扰动是否小到几乎不换动作，或大到几乎每日都换动作；
3. 修正后代理梯度与真值Huber梯度的参数量级相差多大。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "decision_aware"))
os.chdir(ROOT)

import train_joint_decision_aware as trainer  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from decision_aware.joint_system import (  # noqa: E402
    estimate_system_zo_gradient,
    joint_proxy_loss,
    settle_joint_episode,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from decision_aware.zero_order import compute_epsilon  # noqa: E402


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/joint_decision_aware_v1.yaml"
    )
    parser.add_argument("--episode-days", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--setting-filter", default="",
        help="只运行名称包含该文本的设置；空字符串表示全部",
    )
    parser.add_argument(
        "--output", default="data/results/joint_zo_stability_audit.json"
    )
    return parser.parse_args()


def _load(cfg: PilotConfig, device: torch.device):
    paths = {
        "da": cfg.joint_da_checkpoint,
        "rt_at_da": cfg.joint_rt_at_da_checkpoint,
        "rt": cfg.joint_rt_checkpoint,
    }
    checkpoints = {
        name: trainer._load_checkpoint(path) for name, path in paths.items()
    }
    model_cfgs = {
        name: trainer._config_from_checkpoint(checkpoint)
        for name, checkpoint in checkpoints.items()
    }
    models = {
        "da": DecisionAwareDAForecaster(model_cfgs["da"]),
        "rt_at_da": DecisionAwareRTAtDAForecaster(model_cfgs["rt_at_da"]),
        "rt": DecisionAwareRTForecaster(model_cfgs["rt"]),
    }
    for name, model in models.items():
        model.load_state_dict(checkpoints[name]["model_state"])
        model.to(device).eval()
    trainer._assert_independent(models)
    train_ds, _, _, _ = build_joint_datasets(
        cfg, norm_stats=checkpoints["da"]["norm_stats"]
    )
    return models, train_ds, checkpoints


def _spike_block(dataset, episode_days: int) -> tuple[list[int], dict]:
    maxima = []
    for index, window in enumerate(dataset.windows):
        target = dataset.rt_at_da_dataset[
            window.rt_at_da_index
        ]["price_rt_at_da_tgt"]
        maxima.append(float(torch.max(torch.abs(target))))
    spike_index = int(np.argmax(maxima))
    start = max(0, min(
        spike_index - episode_days // 2,
        len(dataset) - episode_days,
    ))
    indices = list(range(start, start + episode_days))
    return indices, {
        "spike_index": spike_index,
        "spike_abs_price": maxima[spike_index],
        "start_index": start,
        "end_index": start + episode_days - 1,
        "delivery_dates": [
            str(dataset.windows[index].delivery_date) for index in indices
        ],
    }


def _batch(dataset, indices: list[int], device: torch.device):
    raw = next(iter(DataLoader(
        Subset(dataset, indices), batch_size=len(indices), shuffle=False,
        num_workers=0,
    )))
    return trainer._move(raw, device)


def _cosine_summary(gradients: list[torch.Tensor]) -> dict:
    values = []
    for left, right in itertools.combinations(gradients, 2):
        left_flat = left.reshape(-1).double()
        right_flat = right.reshape(-1).double()
        denominator = (
            torch.linalg.vector_norm(left_flat)
            * torch.linalg.vector_norm(right_flat)
        )
        values.append(
            float(torch.dot(left_flat, right_flat) / denominator)
            if float(denominator) > 0 else float("nan")
        )
    finite = [value for value in values if math.isfinite(value)]
    return {
        "pair_count": len(values),
        "finite_pair_count": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "minimum": float(np.min(finite)) if finite else None,
        "maximum": float(np.max(finite)) if finite else None,
    }


def _gradient_norms(loss: torch.Tensor, models: dict) -> dict:
    parameters = [
        parameter
        for model in models.values()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    result = {}
    offset = 0
    for name, model in models.items():
        model_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        squares = 0.0
        for gradient in gradients[offset:offset + len(model_parameters)]:
            if gradient is not None:
                squares += float(gradient.double().square().sum().cpu())
        result[name] = math.sqrt(squares)
        offset += len(model_parameters)
    return result


def main():
    args = _args()
    if args.repeats < 2:
        raise ValueError("repeats至少为2，否则无法检查方向稳定性")
    cfg = PilotConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, dataset, checkpoints = _load(cfg, device)
    indices, episode = _spike_block(dataset, args.episode_days)
    batch = _batch(dataset, indices, device)
    with torch.no_grad():
        outputs = trainer._forward(
            models, batch, device.type == "cuda", device
        )
    p_da, p_rt_at_da, p_rt = [value.detach() for value in outputs]
    spread = p_da - p_rt_at_da
    true_da = batch["da"]["price_da_tgt"].float()
    true_rt = batch["rt_at_da"]["price_rt_at_da_tgt"].float()
    reset = batch["starts_new_segment"]

    def settle(da_value, rt_da_value, rt_value):
        return settle_joint_episode(
            da_value, rt_da_value, rt_value,
            true_da, true_rt, reset, cfg,
        )

    epsilon_da_raw = compute_epsilon(
        checkpoints["da"]["norm_stats"]["price_da"]["std"], cfg.zo_rho
    )
    epsilon_rt_raw = compute_epsilon(
        checkpoints["da"]["norm_stats"]["price_rt"]["std"], cfg.zo_rho
    )
    settings = [
        {
            "name": "v1_scalar_gaussian_k2",
            "feedback_mode": "episode_scalar",
            "direction_mode": "gaussian",
            "directions": 2,
            "epsilon_spread": epsilon_da_raw,
            "epsilon_rt": epsilon_rt_raw,
        },
        *[
            {
                "name": f"per_day_orthogonal_k{k}_eps{epsilon:g}",
                "feedback_mode": "per_day",
                "direction_mode": "orthogonal",
                "directions": k,
                "epsilon_spread": epsilon,
                "epsilon_rt": epsilon,
            }
            for epsilon in (2.5, 5.0, 10.0)
            for k in (2, 4, 8)
        ],
        {
            "name": "per_day_gaussian_k8_eps5",
            "feedback_mode": "per_day",
            "direction_mode": "gaussian",
            "directions": 8,
            "epsilon_spread": 5.0,
            "epsilon_rt": 5.0,
        },
        *[
            {
                "name": f"per_group_orthogonal_k4_eps{epsilon:g}",
                "feedback_mode": "per_group",
                "direction_mode": "orthogonal",
                "directions": 4,
                "epsilon_spread": epsilon,
                "epsilon_rt": epsilon,
            }
            for epsilon in (2.5, 5.0, 10.0)
        ],
    ]
    if args.setting_filter:
        settings = [
            setting for setting in settings
            if args.setting_filter in setting["name"]
        ]
        if not settings:
            raise ValueError("没有零阶设置匹配setting-filter")
    audit_rows = []
    saved_candidate = None
    for setting_index, setting in enumerate(settings):
        branch_rows = {}
        candidate_gradients = {}
        for branch_name, prediction, objective, epsilon_key in (
            (
                "spread", spread,
                lambda value: settle(
                    value, torch.zeros_like(value), p_rt
                )["daily_revenue"],
                "epsilon_spread",
            ),
            (
                "rt", p_rt,
                lambda value: settle(p_da, p_rt_at_da, value)[
                    "hourly_revenue"
                    if setting["feedback_mode"] == "per_group"
                    else "daily_revenue"
                ],
                "epsilon_rt",
            ),
        ):
            gradients = []
            diagnostics = []
            for repeat in range(args.repeats):
                torch.manual_seed(10000 * setting_index + 100 * repeat + 7)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(
                        10000 * setting_index + 100 * repeat + 7
                    )
                gradient, diagnostic = estimate_system_zo_gradient(
                    prediction,
                    objective,
                    epsilon=float(setting[epsilon_key]),
                    directions=int(setting["directions"]),
                    pred_clamp=tuple(cfg.pred_clamp),
                    direction_mode=setting["direction_mode"],
                    feedback_mode=setting["feedback_mode"],
                    tail_weight=0.10,
                    tail_fraction=0.10,
                )
                gradients.append(gradient.cpu())
                diagnostics.append(diagnostic)
            norms = np.asarray([
                diagnostic["gradient_norm"] for diagnostic in diagnostics
            ])
            branch_rows[branch_name] = {
                "repeat_cosine": _cosine_summary(gradients),
                "gradient_norm_mean": float(norms.mean()),
                "gradient_norm_cv": float(
                    norms.std() / norms.mean() if norms.mean() else np.nan
                ),
                "changed_direction_fraction_mean": float(np.mean([
                    row["changed_direction_fraction"] for row in diagnostics
                ])),
                "changed_day_fraction_mean": float(np.mean([
                    row["changed_day_fraction"] for row in diagnostics
                ])),
            }
            candidate_gradients[branch_name] = gradients[0].to(device)
        audit_rows.append({**setting, "branches": branch_rows})
        if setting["name"] in {
            "per_day_orthogonal_k8_eps5",
            "per_group_orthogonal_k4_eps5",
        }:
            saved_candidate = candidate_gradients

    if saved_candidate is None:
        raise RuntimeError("缺少预设的V2梯度候选")

    # 只打开三个模型的price head，检查修正后代理梯度量级。
    trainable_counts = trainer._configure_trainable_scope(models, "price_head")
    outputs_graph = trainer._forward(
        models, batch, device.type == "cuda", device
    )
    audit_cfg = PilotConfig(**cfg.to_dict())
    audit_cfg.joint_prediction_loss_mode = "truth_huber"
    audit_cfg.joint_pred_weight_spread = 1.0
    truth_loss, _ = trainer._prediction_loss(
        outputs_graph, batch, audit_cfg
    )
    graph_da, graph_rt_da, graph_rt = outputs_graph
    proxy_spread = joint_proxy_loss(
        graph_da - graph_rt_da,
        saved_candidate["spread"],
        scale=10.0,
        reduction="sample_sum_mean",
    )
    proxy_rt = joint_proxy_loss(
        graph_rt,
        saved_candidate["rt"],
        scale=10.0,
        reduction="sample_sum_mean",
    )
    proxy_loss = 0.5 * (proxy_spread + proxy_rt)
    gradient_scale = {
        "trainable_scope": "price_head",
        "trainable_parameter_counts": trainable_counts,
        "truth_huber_parameter_gradient_norm": _gradient_norms(
            truth_loss, models
        ),
        "v2_proxy_beta1_parameter_gradient_norm": _gradient_norms(
            proxy_loss, models
        ),
        "note": (
            "source_anchor在初始checkpoint处梯度为0；训练后只在偏离"
            "原输出时产生回拉信号。"
        ),
    }

    result = {
        "device": str(device),
        "config": args.config,
        "episode": episode,
        "episode_days": args.episode_days,
        "repeats": args.repeats,
        "raw_epsilon": {
            "da": epsilon_da_raw,
            "rt": epsilon_rt_raw,
        },
        "settings": audit_rows,
        "gradient_scale": gradient_scale,
        "interpretation_contract": {
            "purpose": "gradient stability and scale audit only",
            "not_model_selection": True,
            "not_evidence_of_revenue_improvement": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "complete",
        "output": str(output),
        "device": str(device),
        "episode": episode,
        "gradient_scale": gradient_scale,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
