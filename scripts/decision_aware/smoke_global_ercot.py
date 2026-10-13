#!/usr/bin/env python
"""ERCOT两节点、三任务、一个epoch的本地global smoke。

只读 ``data/markets``；输出写入 ``data/checkpoints/global_ercot_smoke`` 和
``data/results/global_ercot_two_node_smoke.json``。这不是正式训练结果。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    NodeBalancedSampler,
    build_global_task_datasets,
    flatten_rt_delivery_day_batch,
    global_checkpoint_metadata,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


TASKS = {
    "da": {
        "model": DecisionAwareDAForecaster,
        "prediction": "p_da_normalized",
        "target": "price_da_tgt",
        "mean": "price_da_mean",
        "std": "price_da_std",
    },
    "rt_at_da": {
        "model": DecisionAwareRTAtDAForecaster,
        "prediction": "p_rt_at_da_normalized",
        "target": "price_rt_at_da_tgt",
        "mean": "price_rt_mean",
        "std": "price_rt_std",
    },
    "rt": {
        "model": DecisionAwareRTForecaster,
        "prediction": "p_rt_normalized",
        "target": "price_rt_tgt",
        "mean": "price_rt_mean",
        "std": "price_rt_std",
    },
}


def _config(num_nodes: int, num_markets: int, seed: int) -> PilotConfig:
    return PilotConfig(
        use_dual_settlement=True,
        use_deviation_penalty=True,
        bess_kappa=5.7,
        bess_power_mw=1.0,
        bess_energy_mwh=4.0,
        bess_eta=0.95,
        bess_init_soc_frac=0.5,
        bess_soc_min=0.4,
        bess_soc_max=3.6,
        bess_e_cyc=4.0,
        context_len=168,
        horizon_da=24,
        horizon_rt=4,
        train_stride=1,
        val_stride=24,
        eval_stride=24,
        d_model=32,
        n_heads_enc=4,
        n_heads_fusion=4,
        n_layers_enc=1,
        n_layers_fusion=1,
        dim_ff=64,
        dropout=0.0,
        use_amp=False,
        batch_size=2,
        epochs=1,
        global_use_entity_embeddings=True,
        global_num_nodes=num_nodes,
        global_num_markets=num_markets,
        seed=seed,
    )


def _normalized_target(batch: dict, task: dict) -> torch.Tensor:
    target = batch[task["target"]]
    mean = batch[task["mean"]].reshape(-1, 1)
    std = batch[task["std"]].reshape(-1, 1)
    return (target - mean) / std


def _parameter_ids(model: torch.nn.Module) -> set[int]:
    return {id(parameter) for parameter in model.parameters()}


def run_smoke(samples_per_node: int, seed: int) -> dict:
    torch.manual_seed(seed)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(
        ["ERCOT"]
    )
    # 按冻结清单字母序取前两个节点，只用于连通性smoke，不用于研究结论。
    nodes = list(mappings.node_to_id)[:2]
    entities = [("ERCOT", node) for node in nodes]
    cfg = _config(len(mappings.node_to_id), len(mappings.market_to_id), seed)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)

    checkpoint_dir = ROOT / "data" / "checkpoints" / "global_ercot_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    parameter_ids: dict[str, set[int]] = {}
    optimizer_parameter_ids: dict[str, set[int]] = {}

    for task_name, task in TASKS.items():
        started = perf_counter()
        train_dataset, validation_dataset = build_global_task_datasets(
            cfg,
            task_name,
            entities,
            mappings,
        )
        sampler = NodeBalancedSampler(
            train_dataset,
            samples_per_node=samples_per_node,
            seed=seed,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=0,
        )
        model = task["model"](cfg).cpu().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        parameter_ids[task_name] = _parameter_ids(model)
        optimizer_parameter_ids[task_name] = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        losses: list[float] = []
        for batch in loader:
            if task_name == "rt":
                batch = flatten_rt_delivery_day_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            target = _normalized_target(batch, task)
            loss = F.smooth_l1_loss(output[task["prediction"]], target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{task_name} smoke出现非有限loss")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            ):
                raise RuntimeError(f"{task_name} smoke出现缺失或非有限梯度")
            optimizer.step()
            losses.append(float(loss.detach()))

        metadata = global_checkpoint_metadata(
            cfg,
            train_dataset,
            require_all_nodes=False,
        )
        checkpoint = {
            "smoke_only": True,
            "task": task_name,
            "epoch": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.to_dict(),
            **metadata,
        }
        checkpoint_path = checkpoint_dir / f"{task_name}.pt"
        torch.save(checkpoint, checkpoint_path)

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(loaded)
        mappings.validate_checkpoint(loaded)
        restored = task["model"](cfg).cpu()
        restored.load_state_dict(loaded["model_state"], strict=True)
        validation_sample = validation_dataset[0]
        validation_batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in validation_sample.items()
        }
        if task_name == "rt":
            validation_batch = flatten_rt_delivery_day_batch(validation_batch)
        with torch.no_grad():
            restored.eval()(validation_batch)

        reports[task_name] = {
            "train_samples_available": len(train_dataset),
            "validation_samples_available": len(validation_dataset),
            "smoke_samples": len(sampler),
            "optimizer_steps": len(losses),
            "losses": losses,
            "all_losses_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_reload_ok": True,
            "elapsed_seconds": perf_counter() - started,
        }
        del restored, loaded, checkpoint, optimizer, model
        del loader, sampler, validation_dataset, train_dataset
        gc.collect()

    task_names = tuple(TASKS)
    for index, left in enumerate(task_names):
        for right in task_names[index + 1:]:
            if not parameter_ids[left].isdisjoint(parameter_ids[right]):
                raise RuntimeError(f"{left}和{right}共享Parameter对象")
            if not optimizer_parameter_ids[left].isdisjoint(
                optimizer_parameter_ids[right]
            ):
                raise RuntimeError(f"{left}和{right}的optimizer参数重叠")

    return {
        "status": "passed",
        "smoke_only": True,
        "market": "ERCOT",
        "nodes": nodes,
        "epochs": 1,
        "samples_per_node": samples_per_node,
        "three_models_share_no_parameter_objects": True,
        "three_optimizers_are_independent": True,
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "entity_mapping_hash": mappings.mapping_hash,
        "tasks": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-node", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_smoke(args.samples_per_node, args.seed)
    output = ROOT / "data" / "results" / "global_ercot_two_node_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
