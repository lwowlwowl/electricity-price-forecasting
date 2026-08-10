#!/usr/bin/env python
"""eval_v3_da_oracle.py — 用 DA 价 oracle 重新评估 v3 checkpoints（修复 #1）.

修复 #1：oracle 从 RT 价改为 DA 价。
验证：训练梯度不受影响（R_star 是 detached，R_model 计算里 prt_t 未使用），
所以现有 ckpt 权重有效，只需重新评估报正确数字。

本脚本同时报 R*_RT（旧，错）和 R*_DA（新，对），以验证 R_model 不变、只有 R* 变。

用法:
  external/chronos-forecasting/.venv/bin/python scripts/decision_aware/eval_v3_da_oracle.py \
      --config configs/decision_aware/pilot_ercot_v3.yaml \
      --policy topk --ckpt-dir data/checkpoints/da_tsfm_pilot_v3 --tags best last

  # v3-STE（只有 best，训练崩溃无 last）:
  external/chronos-forecasting/.venv/bin/python scripts/decision_aware/eval_v3_da_oracle.py \
      --config configs/decision_aware/pilot_ercot_v3.yaml \
      --policy ste --ckpt-dir data/checkpoints/da_tsfm_pilot_v3_ste --tags best
"""
from __future__ import annotations
import argparse, os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))
os.chdir(_ROOT)

import numpy as np, torch
from torch.utils.data import DataLoader
from decision_aware.config import PilotConfig
from decision_aware.dataset_v3 import build_datasets_v3, collate_v3
from decision_aware.model import DecisionAwareTSFM
from decision_aware.policy import BESSSimulator, TopKPolicy, STEPolicy, lp_oracle_revenue


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def evaluate(cfg, ckpt_tag, dev, test_ds, sim, pol):
    """加载 ckpt，在 test 上评估，返回指标 dict。"""
    model = DecisionAwareTSFM(cfg)
    ckpt_path = os.path.join(cfg.checkpoint_dir.replace(cfg.checkpoint_dir, _ckpt_dir_override),
                             f"pilot_{cfg.node}_{ckpt_tag}.pt")
    # 用 --ckpt-dir 覆盖
    ckpt_path = os.path.join(_ckpt_dir_override, f"pilot_{cfg.node}_{ckpt_tag}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [{ckpt_tag}] ckpt 不存在: {ckpt_path}，跳过")
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    model = model.to(dev).eval()

    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_v3, num_workers=0)

    R_mods, R_stars_da, R_stars_rt, maes = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(batch)
            p_da = out["p_da"]
            price_da = batch["price_da_tgt"]
            price_rt = batch["price_rt_tgt"]

            u = pol(p_da)
            # R_model：双结算（DA 腿用 DA 价；prt_t 未使用，修复前后一致）
            R_model = sim(u, price_da, price_da=price_da if cfg.use_dual_settlement else None)
            # R*_DA（修复后，正确）：LP oracle 用 DA 价
            R_star_da = lp_oracle_revenue(price_da, sim)
            # R*_RT（修复前，错误）：LP oracle 用 RT 价 — 仅对比用
            R_star_rt = lp_oracle_revenue(price_rt, sim)

            mae = (p_da - price_da).abs().mean()
            R_mods.append(R_model.cpu().numpy())
            R_stars_da.append(R_star_da.cpu().numpy())
            R_stars_rt.append(R_star_rt.cpu().numpy())
            maes.append(float(mae.item()))

    R_m = np.concatenate(R_mods)
    R_sd = np.concatenate(R_stars_da)
    R_sr = np.concatenate(R_stars_rt)
    mae = float(np.mean(maes))
    return {
        "R_model": float(R_m.mean()),
        "R_star_DA": float(R_sd.mean()),       # 修复后（正确）
        "R_star_RT": float(R_sr.mean()),       # 修复前（错误，对比用）
        "regret_DA": float((R_sd - R_m).mean()),
        "regret_RT": float((R_sr - R_m).mean()),
        "PCR_DA": float(R_m.mean() / max(1e-8, R_sd.mean()) * 100),
        "MAE": mae,
        "n": len(R_m),
    }


_ckpt_dir_override = ""


def main():
    global _ckpt_dir_override
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_aware/pilot_ercot_v3.yaml")
    ap.add_argument("--policy", choices=["topk", "ste"], required=True,
                    help="v3-TopK 用 topk，v3-STE 用 ste")
    ap.add_argument("--ckpt-dir", required=True, help="checkpoint 目录")
    ap.add_argument("--tags", default="best,last", help="评估哪些 ckpt，逗号分隔")
    args = ap.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    cfg.policy_type = args.policy
    _ckpt_dir_override = args.ckpt_dir

    dev = get_device()
    print("=" * 78)
    print(f"v3 重新评估（DA 价 oracle） | policy={args.policy} | ckpt_dir={args.ckpt_dir}")
    print("=" * 78)

    train_ds, val_ds, test_ds, _ = build_datasets_v3(cfg)
    print(f"  test 样本数: {len(test_ds)}")

    sim = BESSSimulator(cfg.bess_power_mw, cfg.bess_energy_mwh, cfg.bess_eta,
                        cfg.bess_init_soc_frac, kappa=cfg.bess_kappa,
                        soc_min=cfg.bess_soc_min, soc_max=cfg.bess_soc_max,
                        e_cyc=cfg.bess_e_cyc)
    pol = (TopKPolicy(cfg.topk_k_charge, cfg.topk_k_discharge) if args.policy == "topk"
           else STEPolicy(cfg.ste_k))

    print(f"  BESS: P={cfg.bess_power_mw} E={cfg.bess_energy_mwh} η={cfg.bess_eta} "
          f"κ={cfg.bess_kappa} SOC[{cfg.bess_soc_min},{cfg.bess_soc_max}]")
    print()

    for tag in args.tags.split(","):
        tag = tag.strip()
        print(f"--- {tag} ---")
        m = evaluate(cfg, tag, dev, test_ds, sim, pol)
        if m is None:
            continue
        print(f"  R_model      = {m['R_model']:8.1f}   ← 模型收益（修复前后不变，验证训练未受影响）")
        print(f"  R*_RT (旧)   = {m['R_star_RT']:8.1f}   ← 修复前 oracle（RT 价，错）")
        print(f"  R*_DA (新)   = {m['R_star_DA']:8.1f}   ← 修复后 oracle（DA 价，对）")
        print(f"  regret_DA    = {m['regret_DA']:8.1f}   ← 正确 regret（R*_DA − R_model）")
        print(f"  regret_RT    = {m['regret_RT']:8.1f}   ← 旧 regret（文档现值，对比）")
        print(f"  PCR_DA       = {m['PCR_DA']:7.1f}%   ← 占 DA oracle（≤100% 才干净）")
        print(f"  MAE          = {m['MAE']:8.1f}")
        print()


if __name__ == "__main__":
    main()
