"""config.py — 先行版配置（dataclass + yaml 加载）.

v1/v2: d=128/1层~1.1M, 17月ERCOT单RT, STE/greedy
v3:    d=256/2层~5M, 6.5年ERCOT DA+RT双结算, TopK/LP, w10规范BESS参数
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Tuple


# ── v1/v2 旧协变量列分组（17月 model_ready）─────────────────────────────────
STREAM_COLS_V12 = {
    "load":    ["load"],
    "weather": ["temperature"],
    "system":  ["wind", "solar", "gas_share", "renewable_share",
                "gas_gen_mwh", "wind_gen_mwh", "solar_gen_mwh"],
    "econ":    ["henry_hub_usd_per_mmbtu", "wti_usd_per_barrel",
                "natgas_storage_bcf", "storm_event_count"],
}
ALL_COVARIATES_V12 = sum(STREAM_COLS_V12.values(), [])

# ── v3 新协变量列分组（6.5年统一表，含 DA+RT+load+wind+solar+holiday）─────
STREAM_COLS_V3 = {
    "load":    ["load"],
    "system":  ["wind", "solar"],       # 风光出力
    # weather/econ 在 v3 统一表里没有，后续可从 model_ready 合并
}
STREAM_COLS_V3_ALL = ["price_da", "price_rt", "load", "wind", "solar",
                      "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                      "is_weekend", "is_holiday"]


# ── 时间划分 ────────────────────────────────────────────────────────────────
# v1/v2（旧 ERCOT 17 个月）
TRAIN_START_V12 = "2025-01-01"
TRAIN_END_V12   = "2025-05-31 23:00"
VAL_START_V12   = "2025-06-01 00:00"
VAL_END_V12     = "2025-06-30 23:00"
TEST_START_V12  = "2025-07-01 00:00"
TEST_END_V12    = "2026-06-01 23:00"

# v3（新 ERCOT 6.5 年，2020-2026）
TRAIN_START_V3 = "2020-01-01"
TRAIN_END_V3   = "2024-12-31 23:00"   # 5 年训练
VAL_START_V3   = "2025-01-01 00:00"
VAL_END_V3     = "2025-06-30 23:00"   # 6 个月验证
TEST_START_V3  = "2025-07-01 00:00"
TEST_END_V3    = "2026-06-01 23:00"   # 11 个月测试


@dataclass
class PilotConfig:
    # ── 数据 ─────────────────────────────────────────────────────────────────
    market: str = "ERCOT"
    node: str = "LZ_LCRA"
    freq: str = "1h"
    context_len: int = 168
    horizon_da: int = 24
    horizon_rt: int = 4            # w10 §2: 实时滚动预测 H=4
    train_stride: int = 1
    eval_stride: int = 24

    # ── 数据版本（v12=旧17月单RT，v3=新6.5年DA+RT）─────────────────────────
    data_version: str = "v3"
    use_dual_settlement: bool = True  # v3: 真双结算（DA+RT）
    use_dual_split: bool = False      # DA/RT 分离决策（w10 §4.3），False=简化版(uDA=uRT)
    use_deviation_penalty: bool = False  # 偏差罚金（w10 §5.1），3%容忍+双倍RT价

    # ── 模型（v3: d256/2层~5M；v1/v2: d128/1层~1.1M）───────────────────────
    d_model: int = 256
    n_heads_enc: int = 4
    n_heads_fusion: int = 4          # C1: 融合层多头（原 1，跨模态融合最该用多头）
    n_layers_enc: int = 1            # B1: 编码器层数（v7 设 2；原硬编码 1）
    dim_ff: int = 1024               # v3 放大（v1/v2=512）
    dropout: float = 0.1
    use_rope: bool = True

    # ── BESS 模拟器（w10 第7节规范值）──────────────────────────────────────
    bess_power_mw: float = 1.0
    bess_energy_mwh: float = 4.0
    bess_eta: float = 0.95           # v3 对齐 w10（v1/v2=0.9）
    bess_init_soc_frac: float = 0.5
    bess_kappa: float = 27.0         # 退化+交易成本 USD/MWh（w10，v1/v2=0）
    bess_soc_min: float = 0.4        # SOC 下限 MWh（w10: 0.4）
    bess_soc_max: float = 3.6        # SOC 上限 MWh（w10: 3.6）
    bess_e_cyc: float = 4.0          # 每日放电上限 MWh（w10 §4.2 E_cyc）

    # ── 策略与 Oracle ────────────────────────────────────────────────────────
    policy_type: str = "topk"        # "ste" / "topk"(soft) / "hard_topk"(正式版)
    oracle_type: str = "lp"          # "greedy" 或 "lp"
    topk_k_charge: int = 4
    topk_k_discharge: int = 4
    topk_spread_threshold: float = -1.0   # <0 = 自动 κ/η（w10 §4.1 价差门控）；0=关闭
    ste_k: float = 5.0

    # ── 零阶梯度（w10 §6，正式版 hard_topk 专用）────────────────────────────
    zo_rho: float = 0.05             # 扰动比例 ε = ρ·σ_train（w10 §6.1，搜索范围 0.02~0.2）
    zo_K: int = 2                    # 成对扰动方向数（w10: 2 或 4）
    zo_sigma: float = 0.0            # 训练集价格 std（从 norm_stats 自动填充，0=自动）
    proxy_scale: float = 1.0         # L_proxy 归一化（1.0=不归一；若梯度爆炸调大）

    # ── 损失 / 退火 ──────────────────────────────────────────────────────────
    alpha0: float = 1.0
    beta0: float = 0.0
    alpha_end: float = 0.0
    beta_end: float = 1.0
    anneal_epochs: int = 10
    pretrain_epochs: int = 0   # 前 N epoch 纯预测(β=0)，之后才开始退火
    use_mse_loss: bool = True  # w10 §3 半平方误差（False=Huber，向后兼容）
    huber_delta: float = 1.0
    pred_scale: float = 25.0
    bus_scale: float = 250.0
    pred_clamp: list = field(default_factory=lambda: [-100.0, 5000.0])

    # ── 训练 ────────────────────────────────────────────────────────────────
    lr: float = 1.0e-3
    weight_decay: float = 0.01
    batch_size: int = 64             # v3 数据量大，batch 放大（v1/v2=32）
    epochs: int = 20                 # v3 数据量大，epoch 减少（v1/v2=30）
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    use_amp: bool = True
    monitor: str = "regret"
    num_workers: int = 0

    # ── 路径 ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "data/checkpoints/da_tsfm_pilot_v3"
    seed: int = 0

    # ── 便捷 ─────────────────────────────────────────────────────────────────
    @property
    def horizon(self) -> int:
        return self.horizon_da

    @property
    def resolved_spread_threshold(self) -> float:
        """w10 §4.1 价差门控阈值：<0 → 自动 κ/η，0 → 关闭。"""
        if self.topk_spread_threshold < 0:
            return self.bess_kappa / self.bess_eta
        return self.topk_spread_threshold

    @property
    def price_col(self) -> str:
        return f"price__{self.node}"

    def split_bounds(self, split: str) -> Tuple[str, str]:
        if self.data_version == "v3":
            return {"train": (TRAIN_START_V3, TRAIN_END_V3),
                    "val":   (VAL_START_V3,   VAL_END_V3),
                    "test":  (TEST_START_V3,  TEST_END_V3)}[split]
        else:
            return {"train": (TRAIN_START_V12, TRAIN_END_V12),
                    "val":   (VAL_START_V12,   VAL_END_V12),
                    "test":  (TEST_START_V12,  TEST_END_V12)}[split]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "PilotConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def checkpoint_path(self, tag: str = "best") -> str:
        return os.path.join(self.checkpoint_dir, f"pilot_{self.node}_{tag}.pt")
