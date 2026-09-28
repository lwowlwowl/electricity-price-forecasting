"""独立RT模型的逐小时滚动执行诊断。"""
from __future__ import annotations

from collections import defaultdict

import torch

from .policy import HardTopKPolicy


@torch.no_grad()
def rolling_rt_backtest(forecasts: torch.Tensor, realized_first_price: torch.Tensor,
                        local_dates, cfg):
    """每个预测窗口只执行第一个动作，并连续维护真实SOC。

    这是RT-only诊断，不含DA头寸，也不冒充双结算收益。每天只重置累计放电
    上限，SOC跨日连续保留。
    """
    forecasts = forecasts.detach().float().cpu()
    prices = realized_first_price.detach().float().cpu()
    if forecasts.ndim != 2 or prices.ndim != 1:
        raise ValueError("forecasts应为[N,H]，realized_first_price应为[N]")
    if len(forecasts) != len(prices) or len(local_dates) != len(prices):
        raise ValueError("预测、价格和日期长度必须一致")
    policy = HardTopKPolicy(
        k_charge=1,
        k_discharge=1,
        spread_threshold=cfg.resolved_spread_threshold,
    )
    intended = policy(forecasts)[:, 0]
    soc = cfg.bess_energy_mwh * cfg.bess_init_soc_frac
    daily_discharge = 0.0
    previous_date = None
    daily_revenue = defaultdict(float)
    actual_actions = []

    for action, price, local_date in zip(intended.tolist(), prices.tolist(), local_dates):
        if previous_date is None or local_date != previous_date:
            daily_discharge = 0.0
            previous_date = local_date
        discharge = max(action, 0.0) * cfg.bess_power_mw
        charge = max(-action, 0.0) * cfg.bess_power_mw
        discharge = min(
            discharge,
            max(0.0, (soc - cfg.bess_soc_min) * cfg.bess_eta),
            max(0.0, cfg.bess_e_cyc - daily_discharge),
        )
        charge = min(
            charge,
            max(0.0, (cfg.bess_soc_max - soc) / cfg.bess_eta),
        )
        revenue = (discharge - charge) * price
        revenue -= cfg.bess_kappa * (discharge + charge)
        soc = soc - discharge / cfg.bess_eta + charge * cfg.bess_eta
        daily_discharge += discharge
        daily_revenue[local_date] += revenue
        actual_actions.append(discharge - charge)

    revenues = list(daily_revenue.values())
    total = float(sum(revenues))
    return {
        "total_revenue": total,
        "mean_daily_revenue": total / max(1, len(revenues)),
        "positive_day_rate": sum(value > 0 for value in revenues) / max(1, len(revenues)),
        "days": len(revenues),
        "hours": len(prices),
        "final_soc": float(soc),
        "intended_nonzero_rate": float((intended != 0).float().mean()) if len(intended) else 0.0,
        "actual_actions": actual_actions,
    }
