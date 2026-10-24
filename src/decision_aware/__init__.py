"""Decision-aware 多模态 TSFM（先行版 / Pilot + 正式版 / Formal）.

从零训练的小型多模态 Transformer：混合 Encoder + Cross-Attn 融合 + Query Decoder +
BESS 策略。当前结构图见 docs/model_architecture.drawio。

先行版（v1/v2/v3）: STE / soft TopK 可微策略 + regret 退火。
正式版（formal）: Hard TopK 不可微策略 + 零阶双点高斯梯度（w10 §6）。

模块
----
config.py       PilotConfig 配置（dataclass + from_yaml）
dataset_v3.py    DecisionAwareDatasetV3（6.5年 DA+RT 双价）
dataset_da.py    固定D-1截止时点、预测完整交付日的DA-only样本
dataset_rt.py    每小时起报、只执行第一步的RT滚动样本
loader_v2.py     统一 Parquet 数据加载器（当前唯一数据入口）
model.py         DecisionAwareTSFM：混合编码器 + 融合 + Query Decoder + Heads
model_da.py      独立DA Transformer（同小时来源交互）
model_rt.py      独立RT Transformer（4小时滚动预测）
policy.py        BESSSimulator + STEPolicy + TopKPolicy + HardTopKPolicy + LP Oracle
loss.py          total_loss（先行版 regret 退火）+ total_loss_zo（正式版零阶梯度）
zero_order.py    estimate_zo_gradient + compute_l_proxy（w10 §6）

旧 data/raw 加载器、v1/v2 Dataset 和 Forecaster 已移到
src/archive/data_raw_legacy/，不再由当前包导入。
"""

from .config import PilotConfig
from .contracts import (FORMAL_EXPERIMENT_CONTRACT, AnnualSplitContract,
                        AnnualStage,
                        ExperimentContract, SettlementContract,
                        formal_checkpoint_metadata)
from .global_data import (EntityMappings, MultiNodeDataset,
                          MarketNodeBalancedSampler, NodeBalancedSampler,
                          RTDeliveryDayDataset,
                          build_global_task_datasets,
                          flatten_rt_delivery_day_batch,
                          global_checkpoint_metadata, load_formal_entity_mappings)
from .model import DecisionAwareTSFM
from .model_da import DecisionAwareDAForecaster
from .model_rt import DecisionAwareRTForecaster
from .policy import (BESSSimulator, LookaheadMPCPolicy, STEPolicy, TopKPolicy, HardTopKPolicy,
                     greedy_hindsight_revenue, lp_oracle_revenue,
                     lp_oracle_revenue_dual, milp_oracle_revenue_dual,
                     solve_dual_penalty_milp_day,
                     lp_oracle_revenue_dual_plan_tracking_restricted,
                     LPSolveStatus, MILPSolveStatus, DualMILPDispatch,
                     DualMILPOracleResult, LPOracleSolveError,
                     MILPOracleSolveError)
from .loss import total_loss, total_loss_zo, anneal_alpha_beta
from .loss_da import da_decision_aware_loss
from .zero_order import (estimate_zo_gradient, estimate_zo_gradient_dual,
                         compute_l_proxy, compute_epsilon)
from .backtest_dual import locked_dual_backtest_by_node
from .continuous_mpc import (
    locked_continuous_mpc_backtest,
    locked_continuous_mpc_backtest_by_node,
    solve_fixed_da_penalty_lp_window,
    solve_predicted_da_penalty_lp_day,
)
from .economic_spread_factorization import (
    DA_HIGH,
    NEUTRAL,
    RT_HIGH,
    EconomicSpreadFactorizationHead,
    EconomicSpreadFactorizedTransformer,
    economic_spread_factorization_loss,
    economic_spread_regime,
)

__all__ = [
    "PilotConfig", "FORMAL_EXPERIMENT_CONTRACT", "AnnualSplitContract",
    "AnnualStage",
    "ExperimentContract", "SettlementContract", "formal_checkpoint_metadata",
    "EntityMappings", "MultiNodeDataset", "NodeBalancedSampler",
    "MarketNodeBalancedSampler", "build_global_task_datasets",
    "RTDeliveryDayDataset", "flatten_rt_delivery_day_batch",
    "global_checkpoint_metadata", "load_formal_entity_mappings",
    "DecisionAwareTSFM", "DecisionAwareDAForecaster",
    "DecisionAwareRTForecaster",
    "BESSSimulator", "LookaheadMPCPolicy", "STEPolicy", "TopKPolicy", "HardTopKPolicy",
    "greedy_hindsight_revenue", "lp_oracle_revenue", "lp_oracle_revenue_dual",
    "milp_oracle_revenue_dual", "solve_dual_penalty_milp_day",
    "lp_oracle_revenue_dual_plan_tracking_restricted",
    "LPSolveStatus", "MILPSolveStatus", "DualMILPDispatch",
    "DualMILPOracleResult", "LPOracleSolveError", "MILPOracleSolveError",
    "total_loss", "total_loss_zo", "anneal_alpha_beta", "da_decision_aware_loss",
    "estimate_zo_gradient", "estimate_zo_gradient_dual",
    "compute_l_proxy", "compute_epsilon",
    "locked_dual_backtest_by_node",
    "locked_continuous_mpc_backtest",
    "locked_continuous_mpc_backtest_by_node",
    "solve_fixed_da_penalty_lp_window",
    "solve_predicted_da_penalty_lp_day",
    "NEUTRAL", "DA_HIGH", "RT_HIGH",
    "EconomicSpreadFactorizationHead",
    "EconomicSpreadFactorizedTransformer",
    "economic_spread_factorization_loss",
    "economic_spread_regime",
]
