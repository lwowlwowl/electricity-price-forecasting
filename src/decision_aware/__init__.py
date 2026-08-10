"""Decision-aware 多模态 TSFM（先行版 / Pilot + 正式版 / Formal）.

从零训练的小型多模态 Transformer：混合 Encoder + Cross-Attn 融合 + Query Decoder +
BESS 策略。架构对应 docs/model_architecture_v2.drawio。

先行版（v1/v2/v3）: STE / soft TopK 可微策略 + regret 退火。
正式版（formal）: Hard TopK 不可微策略 + 零阶双点高斯梯度（w10 §6）。

模块
----
config.py       PilotConfig 配置（dataclass + from_yaml）
dataset.py       DecisionAwareDataset 多流滑窗（复用 data_processing.loader，无泄漏）
dataset_v3.py    DecisionAwareDatasetV3（6.5年 DA+RT 双价）
model.py         DecisionAwareTSFM：混合编码器 + 融合 + Query Decoder + Heads
policy.py        BESSSimulator + STEPolicy + TopKPolicy + HardTopKPolicy + LP Oracle
loss.py          total_loss（先行版 regret 退火）+ total_loss_zo（正式版零阶梯度）
zero_order.py    estimate_zo_gradient + compute_l_proxy（w10 §6）
train.py         先行版训练回路
forecaster.py    DecisionAwareForecaster(Forecaster) 接 src/evaluation/backtest.py
"""

from .config import PilotConfig
from .model import DecisionAwareTSFM
from .policy import (BESSSimulator, STEPolicy, TopKPolicy, HardTopKPolicy,
                     lp_oracle_revenue, lp_oracle_revenue_dual)
from .loss import total_loss, total_loss_zo, anneal_alpha_beta
from .zero_order import (estimate_zo_gradient, estimate_zo_gradient_dual,
                         compute_l_proxy, compute_epsilon)
from .train import train, evaluate, get_device
from .forecaster import DecisionAwareForecaster

__all__ = [
    "PilotConfig", "DecisionAwareTSFM",
    "BESSSimulator", "STEPolicy", "TopKPolicy", "HardTopKPolicy",
    "lp_oracle_revenue", "lp_oracle_revenue_dual",
    "total_loss", "total_loss_zo", "anneal_alpha_beta",
    "estimate_zo_gradient", "estimate_zo_gradient_dual",
    "compute_l_proxy", "compute_epsilon",
    "train", "evaluate", "get_device",
    "DecisionAwareForecaster",
]
