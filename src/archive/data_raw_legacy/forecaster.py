"""forecaster.py — DecisionAwareForecaster(Forecaster).

接 src/evaluation/backtest.py：predict(context_df, future_covariates, horizon) → Forecast。
context_df 携带 price__<node> + 协变量列（backtest 在 supports_covariates=True 时合并进来）。
Query Decoder 是非自回归的，不需要 future_covariates（忽略）。

归一化 stats（price + 各流 train mean/std）由训练时存到 cfg.checkpoint_path('stats')，
此处加载；缺失则用 context_df 自身统计兜底（退化但能跑）。
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))           # for `from models.base import ...`
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))

from models.base import Forecaster, Forecast              # noqa: E402

from .config import PilotConfig, STREAM_COLS_V12 as STREAM_COLS  # noqa: E402
from .model import DecisionAwareTSFM                      # noqa: E402
from .dataset import _calendar_features                    # noqa: E402


class DecisionAwareForecaster(Forecaster):
    name = "DA-TSFM-Pilot"
    needs_training = True
    supports_covariates = True
    supports_multivariate = False                          # 先行版单节点

    def __init__(self, cfg: PilotConfig, ckpt_path: Optional[str] = None,
                 stats_path: Optional[str] = None, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = DecisionAwareTSFM(cfg).to(self.device)
        self.norm_stats = None
        if stats_path and os.path.exists(stats_path):
            # stats 是本仓库自己生成的 dict（含 numpy 数组），非外部权重，可信：
            self.norm_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(sd)
        self.model.eval()

    # ── 从 context_df 构 6 流 batch（单样本，B=1）─────────────────────────────
    def _build_batch(self, context_df: pd.DataFrame) -> dict:
        cfg = self.cfg
        ctx = context_df.sort_index()
        if ctx.index.tz is None:
            ctx.index = ctx.index.tz_localize("UTC")
        else:
            ctx.index = ctx.index.tz_convert("UTC")
        # 取最后 context_len 行（backtest 传入长度可能恰为 context_len）
        if len(ctx) > cfg.context_len:
            ctx = ctx.iloc[-cfg.context_len:]
        assert len(ctx) == cfg.context_len, \
            f"context_df 长度 {len(ctx)} ≠ context_len {cfg.context_len}"

        price_col = cfg.price_col
        price = ctx[price_col].to_numpy(dtype=np.float32) if price_col in ctx else \
                np.zeros(cfg.context_len, dtype=np.float32)

        # 归一化 stats：优先加载的 train stats；否则用 context 自身（退化兜底）
        st = self.norm_stats
        local = st is None
        if local:
            st = {"price": {"mean": float(price.mean()), "std": float(price.std() + 1e-8)}}
            for s, cols in STREAM_COLS.items():
                have = [c for c in cols if c in ctx.columns]
                arr = ctx[have].to_numpy(dtype=np.float32) if have else np.zeros((len(ctx), len(cols)), np.float32)
                st[s] = {"mean": arr.mean(0), "std": arr.std(0) + 1e-8}

        pst = st["price"]
        price_n = (price - pst["mean"]) / pst["std"]
        batch = {
            "price_ctx": torch.from_numpy(price_n).unsqueeze(0),
            "price_mean": torch.tensor([pst["mean"]]),
            "price_std":  torch.tensor([pst["std"]]),
            "cal_ctx": torch.from_numpy(_calendar_features(ctx.index)).unsqueeze(0),
        }
        for s, cols in STREAM_COLS.items():
            have = [c for c in cols if c in ctx.columns]
            if have:
                arr = ctx[have].to_numpy(dtype=np.float32)
            else:
                arr = np.zeros((len(ctx), len(cols)), dtype=np.float32)
                have = cols                       # 对齐列序
            m = st[s]["mean"]; sd = st[s]["std"]
            arr = (arr - m) / sd
            batch[f"{s}_ctx"] = torch.from_numpy(arr).unsqueeze(0)
        return batch

    def predict(self, context_df: pd.DataFrame,
                future_covariates: Optional[pd.DataFrame] = None,
                horizon: int = 24) -> Forecast:
        cfg = self.cfg
        batch = {k: v.to(self.device) for k, v in self._build_batch(context_df).items()}
        with torch.no_grad():
            out = self.model(batch)
        mean = out["p_da"][0, :horizon].detach().cpu().numpy().astype(float)
        index = self._future_index(context_df, horizon)
        return Forecast(mean=mean, q10=None, q50=None, q90=None,
                        index=index, series_names=[self.cfg.node])
