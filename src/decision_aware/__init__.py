"""Decision-aware 多模态 TSFM（先行版 / Pilot）.

从零训练的小型多模态 Transformer：混合 Encoder + Cross-Attn 融合 + Query Decoder +
可微 BESS 策略 (STE)。架构对应 docs/model_architecture_v2.drawio 的「先行版」两页。

模块
----
config.py     PilotConfig 配置（dataclass + from_yaml）
dataset.py     DecisionAwareDataset 多流滑窗（复用 data_processing.loader，无泄漏）
model.py       DecisionAwareTSFM：6 编码器 + 融合 + Query Decoder + Heads
policy.py      BESSSimulator + STEPolicy（可微 π）+ Regret
loss.py        Huber L_pred + L_bus + α/β 退火
train.py       训练回路（镜像 src/archive/fusion_model/train.py 脚手架）
forecaster.py  DecisionAwareForecaster(Forecaster) 接 src/evaluation/backtest.py
"""

from .config import PilotConfig
from .model import DecisionAwareTSFM
from .policy import BESSSimulator, STEPolicy
from .loss import total_loss, anneal_alpha_beta
from .train import train, evaluate, get_device
from .forecaster import DecisionAwareForecaster

__all__ = [
    "PilotConfig", "DecisionAwareTSFM",
    "BESSSimulator", "STEPolicy",
    "total_loss", "anneal_alpha_beta",
    "train", "evaluate", "get_device",
    "DecisionAwareForecaster",
]
