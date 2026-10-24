"""源市场验证上的局部峰值动态候选选择规则。"""
from __future__ import annotations

from collections.abc import Sequence


def select_peak_dynamic_candidate(
    records: Sequence[dict],
    *,
    mae_guard_fraction: float = 0.02,
) -> dict:
    """在普通MAE守门后，选择日峰值幅度误差最小的候选。

    epoch 0必须是未修改的初始化模型。目标市场指标和盈利不属于输入，因而
    不能静默参与选择。
    """
    if not 0.0 <= mae_guard_fraction < 1.0:
        raise ValueError("mae_guard_fraction必须位于[0,1)")
    normalized: list[dict] = []
    for record in records:
        validation = record.get("validation")
        if not isinstance(validation, dict):
            continue
        try:
            normalized.append({
                "epoch": int(record["epoch"]),
                "market_macro_mae": float(validation["market_macro_mae"]),
                "market_macro_daily_peak_excursion_mae": float(
                    validation["market_macro_daily_peak_excursion_mae"]
                ),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("候选缺少冻结的源市场验证指标") from exc
    baselines = [item for item in normalized if item["epoch"] == 0]
    if len(baselines) != 1:
        raise ValueError("必须且只能有一个epoch 0初始化候选")
    baseline = baselines[0]
    mae_limit = baseline["market_macro_mae"] * (1.0 + mae_guard_fraction)
    eligible = [
        item for item in normalized
        if item["market_macro_mae"] <= mae_limit
    ]
    if not eligible:
        raise RuntimeError("普通MAE守门后没有合格候选")
    selected = min(
        eligible,
        key=lambda item: (
            item["market_macro_daily_peak_excursion_mae"],
            item["market_macro_mae"],
            item["epoch"],
        ),
    )
    return {
        **selected,
        "baseline_epoch": 0,
        "baseline_market_macro_mae": baseline["market_macro_mae"],
        "baseline_market_macro_daily_peak_excursion_mae": baseline[
            "market_macro_daily_peak_excursion_mae"
        ],
        "mae_guard_fraction": mae_guard_fraction,
        "mae_limit": mae_limit,
        "eligible_epochs": [item["epoch"] for item in eligible],
        "selection_uses_target_market": False,
        "selection_uses_profit": False,
    }
