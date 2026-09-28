#!/usr/bin/env python
"""在固定DA时间合同上运行季节性、树模型和RNN基线。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.baselines_da import (  # noqa: E402
    DAGRUBaseline,
    dataset_to_numpy,
    fit_extra_trees,
    fit_xgboost,
    weekly_seasonal_predictions,
)
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.policy import (  # noqa: E402
    BESSSimulator,
    HardTopKPolicy,
    lp_oracle_revenue,
)


def _indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit, dtype=int).tolist()


def _evaluate(predictions: np.ndarray, targets: np.ndarray, simulator, policy):
    pred = torch.as_tensor(predictions, dtype=torch.float32)
    truth = torch.as_tensor(targets, dtype=torch.float32)
    with torch.no_grad():
        action = policy(pred)
        revenue = simulator(action, truth)
        oracle = lp_oracle_revenue(truth, simulator)
    return {
        "mae": float(torch.mean(torch.abs(pred - truth))),
        "rmse": float(torch.mean((pred - truth) ** 2).sqrt()),
        "mean_revenue": float(revenue.mean()),
        "mean_oracle": float(oracle.mean()),
        "mean_regret": float((oracle - revenue).mean()),
        "positive_day_rate": float((revenue > 0).float().mean()),
        "days": int(len(targets)),
    }


def _train_rnn(train_ds, val_ds, cfg, epochs: int, max_train: int | None):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAGRUBaseline(hidden_size=64).to(device)
    train_indices = _indices(len(train_ds), max_train)
    loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=min(cfg.batch_size, 64),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state, best_mae = None, float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            target_n = (
                batch["price_da_tgt"] - batch["price_da_mean"].unsqueeze(-1)
            ) / batch["price_da_std"].unsqueeze(-1)
            prediction_n = model(batch)
            loss = torch.nn.functional.smooth_l1_loss(prediction_n, target_n)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                price = model.to_price(model(batch), batch)
                errors.append(torch.abs(price - batch["price_da_tgt"]).cpu())
        val_mae = float(torch.cat(errors).mean())
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_mae": val_mae}
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, device, history


def _predict_rnn(model, device, dataset, indices: list[int]):
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False)
    outputs, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs.append(model.to_price(model(batch), batch).cpu().numpy())
            targets.append(batch["price_da_tgt"].cpu().numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/da_ercot_v1.yaml")
    parser.add_argument("--models", default="seasonal,extra_trees,rnn,xgboost")
    parser.add_argument("--rnn-epochs", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--output", default="data/results/da_baselines_v1.json")
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    train_ds, val_ds, test_ds, _ = build_da_datasets(cfg)
    test_indices = _indices(len(test_ds), args.max_test)
    targets = np.stack([test_ds[index]["price_da_tgt"].numpy() for index in test_indices])
    simulator = BESSSimulator(
        cfg.bess_power_mw,
        cfg.bess_energy_mwh,
        cfg.bess_eta,
        cfg.bess_init_soc_frac,
        kappa=cfg.bess_kappa,
        soc_min=cfg.bess_soc_min,
        soc_max=cfg.bess_soc_max,
        e_cyc=cfg.bess_e_cyc,
    )
    policy = HardTopKPolicy(
        cfg.topk_k_charge,
        cfg.topk_k_discharge,
        spread_threshold=cfg.resolved_spread_threshold,
    )
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    report = {
        "contract": "ERCOT D-1 10:00 local -> D 00:00-23:00",
        "revenue_scope": "DA standalone physical-execution proxy; not full dual settlement",
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_indices),
        "models": {},
        "errors": {},
    }

    for name in requested:
        started = time.time()
        try:
            if name == "seasonal":
                predictions = weekly_seasonal_predictions(test_ds, test_indices)
            elif name in ("extra_trees", "xgboost"):
                train_indices = _indices(len(train_ds), args.max_train)
                X_train, y_train = dataset_to_numpy(train_ds, train_indices)
                X_test, _ = dataset_to_numpy(test_ds, test_indices)
                estimator = (
                    fit_extra_trees(X_train, y_train, cfg.seed)
                    if name == "extra_trees"
                    else fit_xgboost(X_train, y_train, cfg.seed)
                )
                predictions = estimator.predict(X_test).astype(np.float32)
            elif name == "rnn":
                model, device, history = _train_rnn(
                    train_ds, val_ds, cfg, args.rnn_epochs, args.max_train
                )
                predictions, _ = _predict_rnn(model, device, test_ds, test_indices)
                report["models"][name] = {"training_history": history}
            else:
                raise ValueError(f"未知基线: {name}")
            metrics = _evaluate(predictions, targets, simulator, policy)
            metrics["seconds"] = round(time.time() - started, 2)
            report["models"].setdefault(name, {}).update(metrics)
            print(name, json.dumps(metrics, ensure_ascii=False))
        except Exception as exc:
            report["errors"][name] = f"{type(exc).__name__}: {exc}"
            print(name, report["errors"][name])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
