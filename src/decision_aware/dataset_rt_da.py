"""DA截止时点的RT日预测样本。

该任务与独立DA使用完全相同的D-1 10:00信息边界和交付日24小时，
区别只有监督目标：这里预测交付日实际RT价格。复用已审计窗口可以避免
两个模型因各自实现日期逻辑而错开一天。
"""
from __future__ import annotations

from .config import PilotConfig
from .dataset_da import DecisionAwareDADataset
from .loader_v2 import load_market_unified


class DecisionAwareRTAtDADataset(DecisionAwareDADataset):
    """在D-1 10:00预测D日00:00--23:00的24个RT价格。"""

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        # 使用明确名称，避免把这个长提前量目标与小时滚动RT目标混淆。
        sample["price_rt_at_da_tgt"] = sample.pop("price_rt_tgt")
        # 该任务不监督DA价格，专用batch不携带交付日真实DA标签。
        sample.pop("price_da_tgt")
        return sample


def build_rt_at_da_datasets(cfg: PilotConfig):
    """返回(train, val, test, wide_df)，不修改任何原始数据文件。"""
    wide_df = load_market_unified(
        market=cfg.market,
        node=cfg.node,
        start="2020-01-01",
        end="2026-06-02",
    )
    train_ds = DecisionAwareRTAtDADataset(wide_df, cfg, "train")
    val_ds = DecisionAwareRTAtDADataset(
        wide_df, cfg, "val", norm_stats=train_ds.norm_stats
    )
    test_ds = DecisionAwareRTAtDADataset(
        wide_df, cfg, "test", norm_stats=train_ds.norm_stats
    )
    return train_ds, val_ds, test_ds, wide_df
