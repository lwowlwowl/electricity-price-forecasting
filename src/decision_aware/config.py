"""config.py — 先行版配置（dataclass + yaml 加载）.

v1/v2: d=128/1层~1.1M, 17月ERCOT单RT, STE/Greedy hindsight baseline
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

# ── v3 当前启用协变量列（6.5年统一表）─────────────────────────────────────
# 统一表还提供实际天气和 t+24 天气预报；当前 v8 尚未将天气接入 Dataset/Model，
# 不能据此声称 v8 已使用 Xweather。DA-only 设计时须先定义预报发布时间与目标时点映射。
STREAM_COLS_V3 = {
    "load":    ["load"],
    "system":  ["wind", "solar"],       # 市场级实际风光出力
}
STREAM_COLS_V3_ALL = ["price_da", "price_rt", "load", "wind", "solar",
                      "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                      "month_sin", "month_cos", "is_weekend", "is_holiday"]


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
    da_issue_hour_local: int = 10  # ERCOT正常DAM提交截止：D-1日10:00 CPT
    horizon_rt: int = 4            # w10 §2: 实时滚动预测 H=4
    train_stride: int = 1
    eval_stride: int = 24           # test 用（保持独立日窗口，DM/GW 检验 i.i.d. 前提）
    val_stride: int = 24            # val 用（v8 设 6 加密止住 regret 摇号；默认 24 向后兼容）

    # 可选的实验级时间切分。留空时沿用上面的版本默认值；正式消融用配置文件
    # 显式冻结，避免为了某个结果在代码里反复改全局常量。
    split_train_start: str | None = None
    split_train_end: str | None = None
    split_val_start: str | None = None
    split_val_end: str | None = None
    split_test_start: str | None = None
    split_test_end: str | None = None

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
    n_layers_fusion: int = 1         # 融合层数（v8 设 2；原硬编码 1，被指多源数据处理太薄）
    dim_ff: int = 1024               # v3 放大（v1/v2=512）
    dropout: float = 0.1
    use_rope: bool = True
    da_fusion_mode: str = "source_attention"
    rt_at_da_fusion_mode: str = "source_attention"
    rt_fusion_mode: str = "source_attention"

    # ── BESS 模拟器（w10 第7节规范值）──────────────────────────────────────
    bess_power_mw: float = 1.0
    bess_energy_mwh: float = 4.0
    bess_eta: float = 0.95           # v3 对齐 w10（v1/v2=0.9）
    bess_init_soc_frac: float = 0.5
    bess_kappa: float = 5.7          # 项目主合同的单位运行与退化成本；偏差罚金另行计算
    bess_soc_min: float = 0.4        # SOC 下限 MWh（w10: 0.4）
    bess_soc_max: float = 3.6        # SOC 上限 MWh（w10: 3.6）
    bess_e_cyc: float = 4.0          # 每日放电上限 MWh（w10 §4.2 E_cyc）

    # ── 策略与 Oracle ────────────────────────────────────────────────────────
    policy_type: str = "topk"        # "ste" / "topk"(soft) / "hard_topk"(正式版)
    oracle_type: str = "lp"          # Regret 只使用求解成功的 LP Oracle
    topk_k_charge: int = 4
    topk_k_discharge: int = 4
    # 滚动RT的单个H=4窗口候选数。历史口径固定为1/1；
    # 偏差罚金场景允许在验证集显式比较0/0（不额外偏离DA）。
    rt_topk_k_charge: int = 1
    rt_topk_k_discharge: int = 1
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
    grad_accum_steps: int = 1

    # 融合消融使用的冻结完整系统。普通训练入口不会读取这些字段。
    # RT-at-DA消融同时对三个冻结DA-F1 seed计分，避免结论依赖单个DA seed。
    frozen_da_checkpoints: list[str] = field(default_factory=list)
    frozen_rt_at_da_checkpoints: list[str] = field(default_factory=list)
    frozen_rt_at_da_checkpoint: str = ""
    frozen_rt_checkpoint: str = ""
    frozen_coordination_mode: str = "rt_only"
    # 0表示不强制天数；新正式口径设为182，防止误用旧验证集选epoch。
    frozen_expected_validation_days: int = 0
    bootstrap_samples: int = 2000

    # ── 三模型联合 decision-aware 微调 ────────────────────────────────────
    # 这三个路径是联合训练的唯一初始化清单。结构由各checkpoint自己的config
    # 恢复；这里的实验配置只冻结数据、物理结算和联合优化合同。
    joint_da_checkpoint: str = ""
    joint_rt_at_da_checkpoint: str = ""
    joint_rt_checkpoint: str = ""
    # 价格预测网络的形状不依赖退化成本。仅在显式kappa灵敏性/迁移实验中
    # 允许source checkpoint的kappa不同；其余物理合同仍必须完全一致。
    joint_allow_source_kappa_mismatch: bool = False
    # spread使用p_DA-p_RT|DA制定DA计划；da_only只使用p_DA。
    # 当罚金使RT严格跟随DA时，必须比较两者，不能默认spread。
    joint_da_signal_mode: str = "spread"
    # rt_only保留无罚金历史口径；plan_track_topk对应w10 §4.3：
    # 先执行DA计划，再用剩余功率/SOC执行RT TopK。
    joint_coordination_mode: str = "rt_only"
    joint_episode_days: int = 16
    joint_validation_batch_days: int = 8
    joint_pred_weight_da: float = 1.0
    joint_pred_weight_rt_at_da: float = 1.0
    joint_pred_weight_rt: float = 1.0
    # DA动作只读取p_DA-p_RT|DA；单独约束两条价格曲线并不能保证价差排序稳定。
    # 0保持v1复现；v2可显式加入真实DA-RT价差的Huber辅助损失。
    joint_pred_weight_spread: float = 0.0
    # truth_huber保留v1；source_anchor约束联合微调不偏离三个已按
    # 验证收益选出的强checkpoint输出。
    joint_prediction_loss_mode: str = "truth_huber"
    # 三路零阶代理都先按各自输出元素求mean，再除以同一尺度；这样滚动RT的
    # 24×4个输出不会仅因元素更多而压过两个24点日模型。
    joint_proxy_scale: float = 10.0
    joint_alpha: float = 1.0
    joint_beta_start: float = 0.05
    joint_beta_end: float = 0.30
    joint_beta_warmup_epochs: int = 4
    # v1分别扰动p_DA和p_RT|DA；v2直接扰动共同决策变量spread；
    # da_only用于只由p_DA制定计划的罚金合同，只允许DA获得收益代理。
    joint_da_proxy_mode: str = "independent_prices"
    # gaussian保持v1；orthogonal使用等范数正交高斯方向降低同一batch内方差。
    joint_zo_direction_mode: str = "gaussian"
    # episode_scalar保留v1的单标量反馈；per_day利用系统已有的逐日
    # 收益，为每个交付日单独分配零阶信号，降低高维交叉噪声。
    joint_zo_feedback_mode: str = "episode_scalar"
    # 滚动RT每小时有一个H维窗口，v2可用per_group将该小时
    # 结算反馈直接分配给对应窗口，避免24个窗口共用一个日标量。
    joint_zo_rt_feedback_mode: str = "episode_scalar"
    # >0时覆盖rho*全局价格std。ERCOT全局std受尖峰支配，v2使用显式、较小扰动。
    joint_zo_epsilon_spread: float = 0.0
    joint_zo_epsilon_rt: float = 0.0
    # 0时共用历史zo_K；v2允许DA价差和4维RT窗口用不同K。
    joint_zo_directions_spread: int = 0
    joint_zo_directions_rt: int = 0
    joint_proxy_weight_spread: float = 1.0
    joint_proxy_weight_rt: float = 1.0
    # element_mean保留v1；sample_sum_mean先对每日的决策维求和再对天数
    # 取均值；global_sum用于已经是完整episode目标梯度的episode_scalar。
    joint_proxy_reduction: str = "element_mean"
    # 风险效用=(1-w)*mean_revenue+w*lower_tail_mean；0保持只优化均值。
    joint_tail_weight: float = 0.0
    joint_tail_fraction: float = 0.10
    # checkpoint选择使用同形式的验证效用；0保持v1按日均收益选择。
    joint_selection_tail_weight: float = 0.0
    # all保持v1；decoder_head只更新各自decoder和price_head，保护已选骨干。
    joint_trainable_scope: str = "all"
    # 显式列出本次允许更新的独立模型。未列出的模型仍参与前向与结算，
    # 但参数完全冻结，也不会为它创建optimizer或无意义的零阶估计。
    joint_trainable_models: list[str] = field(
        default_factory=lambda: ["da", "rt_at_da", "rt"]
    )
    # v1单AdamW+全局clip；v2为三个独立模型分别建optimizer和clip。
    joint_separate_optimizers: bool = False
    # v2把未微调的强基线作为epoch 0候选；训练若没有真正
    # 提高验证效用，最终bundle至少不会被更差的epoch覆盖。
    joint_include_baseline_candidate: bool = False
    # source_anchor必须和确定性基线输出比较；v2在微调时保持eval
    # 以关闭dropout，但不使用no_grad，因此price head仍正常反传。
    joint_disable_dropout: bool = False
    # 极端RT批次会放大零阶收益代理的price-head梯度。正式罚金训练使用1.0，
    # 保留float16前向的同时避免GradScaler再次放大反向梯度。
    joint_amp_init_scale: float = 1.0

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
            defaults = {
                "train": (TRAIN_START_V3, TRAIN_END_V3),
                "val":   (VAL_START_V3,   VAL_END_V3),
                "test":  (TEST_START_V3,  TEST_END_V3),
            }
        else:
            defaults = {
                "train": (TRAIN_START_V12, TRAIN_END_V12),
                "val":   (VAL_START_V12,   VAL_END_V12),
                "test":  (TEST_START_V12,  TEST_END_V12),
            }
        if split not in defaults:
            raise ValueError(f"未知split: {split}")
        default_start, default_end = defaults[split]
        override_start = getattr(self, f"split_{split}_start")
        override_end = getattr(self, f"split_{split}_end")
        return override_start or default_start, override_end or default_end

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
