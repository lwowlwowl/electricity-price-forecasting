"""
run_fusion.py — ElecFM 一键入口
=================================
从 YAML 配置文件读取参数，执行：
  Step 1. 验证 15 层剪枝后的基准（零样本，不训练）
  Step 2. 两阶段训练
  Step 3. τ* 搜索 + 三窗口评估

用法：
    external/timesfm/.venv/bin/python src/fusion_model/run_fusion.py \\
        --config configs/fusion/electfm.yaml \\
        [--skip-train]        # 只评估（使用已有 checkpoint）
        [--step1-only]        # 只跑剪枝基准验证（不训练）

运行环境：external/timesfm/.venv
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
sys.path.insert(0, os.path.join(_ROOT, "src", "evaluation"))

from model    import ElecFM
from train    import TrainConfig, train
from evaluate import EvalConfig, run_evaluation
from dataset  import build_datasets


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def step1_verify_pruning(cfg: dict, device: torch.device):
    """
    Step 1：零样本验证——剪枝后 15 层模型 vs 原始 20 层 TimesFM。
    目标：SMAPE 退化 < 5%（5 层保守方案实测 +4.4%）。
    """
    print("=" * 60)
    print("Step 1：零样本剪枝基准验证（不训练）")
    print("=" * 60)
    import numpy as np
    import loader
    from evaluate import EvalConfig, TEST_WINDOWS, _build_origins, _inference_batch, _compute_spike_f1

    eval_cfg = EvalConfig(
        market=cfg["market"], nodes=cfg["nodes"],
        context_len=cfg["context_len"], horizon=cfg["horizon"],
        stride_hours=cfg.get("stride_hours", 24),
        max_origins=cfg.get("max_origins", 30),
    )

    # 加载 ElecFM（已剪枝，随机初始化 spike head，但 backbone 是预训练权重）
    print("  加载 ElecFM（剪枝后，backbone 使用 TimesFM 预训练权重）...")
    model = ElecFM(
        pretrained_id=cfg.get("pretrained_id", "google/timesfm-2.5-200m-pytorch"),
        horizon=cfg["horizon"],
    ).to(device)
    model.eval()

    df = loader.load_slice(
        market=eval_cfg.market, nodes=eval_cfg.nodes, freq=eval_cfg.freq,
        start="2025-07-01", end="2025-08-31",  # 用 W1 窗口附近数据快速验
    )
    target_cols = [f"price__{n}" for n in eval_cfg.nodes]

    win_start, win_end = "2025-08-01", "2025-08-31"
    origins = _build_origins(df.index, win_start, win_end,
                             eval_cfg.context_len, eval_cfg.horizon,
                             eval_cfg.stride_hours, eval_cfg.max_origins)

    preds = _inference_batch(model, df, origins, target_cols,
                             eval_cfg.context_len, eval_cfg.horizon, device)

    smape_vals = []
    for oi in origins:
        actual = df[target_cols].iloc[oi: oi + eval_cfg.horizon].to_numpy(float).ravel()
        mean_p = preds[oi]["mean"].ravel()
        from metrics import smape
        smape_vals.append(smape(actual, mean_p))

    avg_smape = float(np.mean(smape_vals))
    timesfm_baseline = 27.67   # 来自 v2.0 消融基准（W1）

    delta = (avg_smape - timesfm_baseline) / timesfm_baseline * 100
    print(f"\n  15 层 ElecFM（零样本）SMAPE = {avg_smape:.2f}")
    print(f"  原始 20 层 TimesFM baseline SMAPE = {timesfm_baseline:.2f}")
    print(f"  SMAPE 退化 = {delta:+.1f}%")

    if delta < 5:
        print("  ✅ 退化 < 5%，剪枝基准通过，可继续 Step 2 训练")
    elif delta < 10:
        print("  ⚠️  退化 5-10%，接受但请在 Step 5 额外记录基准差距")
    else:
        print("  ❌ 退化 ≥ 10%，建议切换至 5 层保守方案")
        print("     仅移除 {L6, L8, L9, L13, L15}，重新运行 Step 1 验证")

    return avg_smape, delta


def main():
    parser = argparse.ArgumentParser(description="ElecFM 融合模型训练与评估")
    parser.add_argument("--config",      required=True,      help="YAML 配置文件路径")
    parser.add_argument("--skip-train",  action="store_true", help="跳过训练，直接评估")
    parser.add_argument("--step1-only",  action="store_true", help="只跑剪枝基准验证")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # 设备优先级：CUDA > MPS（Apple Silicon）> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")

    # ── Step 1：剪枝基准验证 ──────────────────────────────────────────────────
    step1_verify_pruning(cfg, device)
    if args.step1_only:
        return

    # ── 构建数据集 ────────────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n构建训练/验证数据集...")
        train_ds, val_ds = build_datasets(
            market=cfg["market"],
            nodes=cfg["nodes"],
            context_len=cfg["context_len"],
            horizon=cfg["horizon"],
            stride=cfg.get("train_stride", 1),
        )
        print(f"  训练样本：{len(train_ds)}  验证样本：{len(val_ds)}")

        from torch.utils.data import DataLoader
        _pin = torch.cuda.is_available()
        # num_workers=0：macOS 下多进程 DataLoader 有 spawn 兼容性问题；
        # 训练数据集较小（~25K 样本），单进程加载开销可接受
        train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32),
                                  shuffle=True, num_workers=0, pin_memory=_pin)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.get("batch_size", 32),
                                  shuffle=False, num_workers=0, pin_memory=_pin)

        # 统计实际 pos_weight
        if hasattr(train_ds, "spike_pos_weight"):
            pos_weight = train_ds.spike_pos_weight
        elif hasattr(train_ds, "datasets"):   # ConcatDataset
            pos_weight = train_ds.datasets[0].spike_pos_weight
        else:
            pos_weight = 19.0
        print(f"  实际 spike pos_weight = {pos_weight:.1f}")

    # ── 初始化模型 ────────────────────────────────────────────────────────────
    print("\n初始化 ElecFM...")
    model = ElecFM(
        pretrained_id=cfg.get("pretrained_id", "google/timesfm-2.5-200m-pytorch"),
        horizon=cfg["horizon"],
    ).to(device)

    # ── Step 2：两阶段训练 ────────────────────────────────────────────────────
    checkpoint_dir = os.path.join(_ROOT, cfg.get("checkpoint_dir", "data/checkpoints/electfm"))
    if not args.skip_train:
        train_cfg = TrainConfig(
            stage1_epochs=cfg.get("stage1_epochs", 10),
            stage1_lr=cfg.get("stage1_lr", 1e-4),
            stage2_epochs=cfg.get("stage2_epochs", 40),
            stage2_lr=cfg.get("stage2_lr", 5e-6),
            stage2_lr_min=cfg.get("stage2_lr_min", 5e-7),
            batch_size=cfg.get("batch_size", 32),
            gradient_clip=cfg.get("gradient_clip", 1.0),
            weight_decay=cfg.get("weight_decay", 0.01),
            early_stop_patience=cfg.get("early_stop_patience", 10),
            lambda_pinball=cfg.get("lambda_pinball", 0.8),
            lambda_spike=cfg.get("lambda_spike", 0.2),
            spike_pos_weight=pos_weight,
            use_amp=cfg.get("use_amp", True),
            checkpoint_dir=checkpoint_dir,
        )
        print("\n开始两阶段训练...")
        best_ckpt = train(model, train_loader, val_loader, train_cfg, device)
    else:
        best_ckpt = os.path.join(checkpoint_dir, "electfm_best.pt")
        print(f"\n跳过训练，使用已有 checkpoint：{best_ckpt}")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Checkpoint 不存在：{best_ckpt}")

    # ── Step 3：评估 ──────────────────────────────────────────────────────────
    print("\n开始三窗口评估...")
    eval_cfg = EvalConfig(
        market=cfg["market"],
        nodes=cfg["nodes"],
        freq=cfg.get("freq", "1h"),
        context_len=cfg["context_len"],
        horizon=cfg["horizon"],
        stride_hours=cfg.get("stride_hours", 24),
        max_origins=cfg.get("max_origins", 30),
        spike_quantile=cfg.get("spike_quantile", 0.95),
        tau_search_range=tuple(cfg.get("tau_search_range", [0.05, 0.95])),
        tau_search_step=cfg.get("tau_search_step", 0.05),
    )
    output_root = os.path.join(_ROOT, "data", "results", "fusion")
    run_evaluation(model, eval_cfg, best_ckpt, output_root, device)

    print("\n✅ ElecFM 全流程完成")


if __name__ == "__main__":
    main()
