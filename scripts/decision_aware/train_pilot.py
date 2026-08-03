#!/usr/bin/env python
"""train_pilot.py — 先行版训练入口.

用法（用带 torch 的 venv）:
  external/chronos-forecasting/.venv/bin/python \\
    scripts/decision_aware/train_pilot.py \\
    --config configs/decision_aware/pilot_ercot.yaml [--epochs N] [--smoke]

流程: 加载配置 → 构建多流数据集 → 训练（α-β 退火）→ 存 ckpt + norm_stats
      → test 段评估（MAE/SMAPE/Regret/收益）→ 可选: 跑一个 backtest 起报点验证 predict().
"""
from __future__ import annotations

import argparse
import os
import sys

# 让 `import decision_aware` 与 `import loader` 都可用
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
os.chdir(_ROOT)

import torch  # noqa: E402

from decision_aware.config import PilotConfig            # noqa: E402
from decision_aware.dataset import build_datasets        # noqa: E402
from decision_aware.model import DecisionAwareTSFM      # noqa: E402
from decision_aware.train import train, evaluate, get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/pilot_ercot.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    ap.add_argument("--anneal-epochs", type=int, default=None, help="覆盖 α-β 退火区间")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="关 early-stop（patience=99），跑到 β=1 之后看 regret 趋势")
    ap.add_argument("--smoke", action="store_true",
                    help="烟雾测试: 3 epoch + 跑满 train/val，不早停")
    args = ap.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.anneal_epochs is not None:
        cfg.anneal_epochs = args.anneal_epochs
    if args.no_early_stop:
        cfg.early_stop_patience = 99          # 不早停，跑满看趋势
    if args.smoke:
        cfg.epochs = 3
        cfg.early_stop_patience = 99
        cfg.train_stride = 4
        cfg.eval_stride = 48
    torch.manual_seed(cfg.seed)

    print("=" * 70)
    print(f"先行版 DA-TSFM 训练 | {cfg.market} {cfg.node} | d={cfg.d_model} "
          f"| epochs={cfg.epochs}{' (SMOKE)' if args.smoke else ''}")
    print("=" * 70)

    train_ds, val_ds, test_ds, _ = build_datasets(cfg)
    if len(train_ds) == 0:
        print("❌ train 集为空，检查数据/时间划分"); return

    model = DecisionAwareTSFM(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数: {n:,} (~{n/1e6:.2f}M)")

    device = get_device()
    print(f"device: {device}\n")

    res = train(model, train_ds, val_ds, cfg, device=device, verbose=True)

    # 存 norm_stats（forecaster 推理时反归一化用，与 train 段统计一致）
    stats_path = cfg.checkpoint_path("stats")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    torch.save(train_ds.norm_stats, stats_path)
    print(f"\nnorm_stats → {stats_path}")

    # ── test 段评估 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Test 段评估")
    print("=" * 70)
    te = evaluate(model, test_ds, cfg, device=device, ckpt_path=res["best_path"])
    print(f"  样本数    : {te['n_samples']}")
    print(f"  MAE       : {te['mae']:.3f} $/MWh")
    print(f"  SMAPE     : {te['smape']*100:.2f}%")
    print(f"  平均收益R : {te['R_model']:.1f}  (Oracle R*={te['R_star']:.1f})")
    print(f"  平均Regret: {te['regret']:.1f}")

    # ── 可选: 用 Forecaster.predict() 跑一个 backtest 起报点，验证接口 ────────
    if args.smoke:
        print("\n--- Forecaster.predict() 接口冒烟 ---")
        from decision_aware.forecaster import DecisionAwareForecaster
        fc = DecisionAwareForecaster(cfg, ckpt_path=res["best_path"], stats_path=stats_path)
        # 取 test 段第一个起报点构造 context_df
        import pandas as pd
        ctx_len = cfg.context_len
        i = test_ds.valid_starts[0]
        ctx_df = test_ds_index_to_df(test_ds, i, ctx_len)
        f = fc.predict(ctx_df, future_covariates=None, horizon=cfg.horizon_da)
        print(f"  Forecast.mean shape={f.mean.shape}  前几点={f.mean[:3].round(2).tolist()}")
        actual = test_ds.price[i + ctx_len: i + ctx_len + cfg.horizon_da]
        print(f"  真值前几点      ={actual[:3].round(2).tolist()}")
        print(f"  index[0]={f.index[0]}  (expect {test_ds.index[i+ctx_len]})")
    print("\n✅ 先行版跑通")


def test_ds_index_to_df(test_ds, i: int, ctx_len: int):
    """从 test_ds 还原一个 context_df（含 price__<node> + 协变量列）。
    用 STREAM_COLS 的列序从 test_ds.streams 数组拼回 wide 格式。"""
    import pandas as pd
    from decision_aware.config import STREAM_COLS
    idx = test_ds.index[i: i + ctx_len]
    df = pd.DataFrame({test_ds.cfg.price_col: test_ds.price[i: i + ctx_len]}, index=idx)
    for s, cols in STREAM_COLS.items():
        arr = test_ds.streams[s][i: i + ctx_len]
        if arr.shape[1] == len(cols):
            for j, c in enumerate(cols):
                df[c] = arr[:, j]
    return df


if __name__ == "__main__":
    main()
