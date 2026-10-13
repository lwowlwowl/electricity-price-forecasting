"""正式年度实验与双结算合同。

这里保存的是正式 global 实验必须共同遵守的不可变口径。训练脚本可以有不同
模型和超参数，但不能自行改写年度切分、储能物理参数或结算公式。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Mapping


def _canonical_json(value: Mapping[str, Any]) -> str:
    """返回跨平台稳定的 JSON；它是合同哈希的唯一输入。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class LocalDateRange:
    """闭区间，边界均按各市场本地时间解释。"""

    start: str
    end: str


class AnnualStage(str, Enum):
    INNER_TRAIN = "inner_train"
    INNER_VALIDATION = "inner_validation"
    OUTER_TRAIN = "outer_train"
    OUTER_VALIDATION = "outer_validation"
    FINAL_TRAIN = "final_train"
    LOCKED_RETROSPECTIVE_TEST = "locked_retrospective_test"
    REPORT_ONLY = "report_only"


@dataclass(frozen=True)
class AnnualSplitContract:
    """模型选择和最终测试的唯一年度协议。"""

    timezone_basis: str = "market_local"
    inner_train: LocalDateRange = LocalDateRange("2020-01-01", "2022-12-31")
    inner_validation: LocalDateRange = LocalDateRange("2023-01-01", "2023-12-31")
    outer_train: LocalDateRange = LocalDateRange("2020-01-01", "2023-12-31")
    outer_validation: LocalDateRange = LocalDateRange("2024-01-01", "2024-12-31")
    final_train: LocalDateRange = LocalDateRange("2020-01-01", "2024-12-31")
    locked_retrospective_test: LocalDateRange = LocalDateRange(
        "2025-01-01", "2025-12-31"
    )
    report_only: LocalDateRange = LocalDateRange("2026-01-01", "2026-06-01")
    report_only_can_select_model: bool = False
    locked_test_can_select_model: bool = False

    def range_for(self, stage: AnnualStage | str) -> LocalDateRange:
        try:
            resolved = AnnualStage(stage)
        except ValueError as exc:
            raise ValueError(f"未知年度阶段: {stage!r}") from exc
        return getattr(self, resolved.value)


@dataclass(frozen=True)
class SettlementContract:
    """唯一正式双结算与储能物理合同。"""

    dual_settlement: bool = True
    deviation_penalty_enabled: bool = True
    degradation_cost_usd_per_mwh: float = 5.7
    degradation_throughput_basis: str = "actual_charge_plus_actual_discharge"
    deviation_tolerance_fraction_of_abs_da: float = 0.03
    deviation_penalty_rt_abs_price_multiplier: float = 2.0
    deviation_basis: str = "abs(actual_rt_dispatch_minus_locked_da_dispatch)"
    bess_power_mw: float = 1.0
    bess_energy_mwh: float = 4.0
    round_trip_component_efficiency: float = 0.95
    soc_min_mwh: float = 0.4
    soc_max_mwh: float = 3.6
    initial_soc_mwh: float = 2.0
    daily_discharge_cap_mwh: float = 4.0
    da_plan_locked: bool = True
    settle_each_delivery_hour_once: bool = True
    soc_continuity: str = "within_node_contiguous_timestamps_only"
    reset_soc_on: str = "node_change_or_true_time_gap"
    terminal_soc_value_usd_per_mwh: float = 0.0

    def validate_pilot_config(self, config: Any) -> None:
        """硬校验旧 ``PilotConfig`` 是否满足正式物理与结算口径。"""
        mismatches: dict[str, tuple[Any, Any]] = {}

        expected_booleans = {
            "use_dual_settlement": self.dual_settlement,
            "use_deviation_penalty": self.deviation_penalty_enabled,
        }
        for field, expected in expected_booleans.items():
            actual = getattr(config, field, None)
            if actual is not expected:
                mismatches[field] = (actual, expected)

        expected_numbers = {
            "bess_kappa": self.degradation_cost_usd_per_mwh,
            "bess_power_mw": self.bess_power_mw,
            "bess_energy_mwh": self.bess_energy_mwh,
            "bess_eta": self.round_trip_component_efficiency,
            "bess_soc_min": self.soc_min_mwh,
            "bess_soc_max": self.soc_max_mwh,
            "bess_e_cyc": self.daily_discharge_cap_mwh,
        }
        for field, expected in expected_numbers.items():
            actual = getattr(config, field, None)
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                mismatches[field] = (actual, expected)

        energy = getattr(config, "bess_energy_mwh", None)
        initial_fraction = getattr(config, "bess_init_soc_frac", None)
        if isinstance(energy, (int, float)) and isinstance(
            initial_fraction, (int, float)
        ):
            initial_soc = float(energy) * float(initial_fraction)
        else:
            initial_soc = None
        if initial_soc is None or not math.isclose(
            initial_soc, self.initial_soc_mwh, rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches["initial_soc_mwh"] = (initial_soc, self.initial_soc_mwh)

        if mismatches:
            details = "; ".join(
                f"{field}: actual={actual!r}, expected={expected!r}"
                for field, (actual, expected) in sorted(mismatches.items())
            )
            raise ValueError(f"配置不符合正式结算合同: {details}")


@dataclass(frozen=True)
class ExperimentContract:
    """可序列化、可哈希、可写入 checkpoint 的正式实验合同。"""

    contract_name: str
    contract_version: int
    data_audit_version: int
    annual_split: AnnualSplitContract
    settlement: SettlementContract

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def contract_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "experiment_contract": self.to_dict(),
            "contract_hash": self.contract_hash,
        }

    def validate_pilot_config(self, config: Any) -> None:
        self.settlement.validate_pilot_config(config)

    def validate_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """拒绝无合同、被改写或属于另一合同的 checkpoint。"""
        saved_contract = checkpoint.get("experiment_contract")
        saved_hash = checkpoint.get("contract_hash")
        if not isinstance(saved_contract, Mapping) or not isinstance(saved_hash, str):
            raise ValueError("checkpoint缺少experiment_contract或contract_hash")

        actual_saved_hash = sha256(
            _canonical_json(dict(saved_contract)).encode("utf-8")
        ).hexdigest()
        if actual_saved_hash != saved_hash:
            raise ValueError("checkpoint合同内容与其contract_hash不一致")
        if saved_hash != self.contract_hash:
            raise ValueError(
                "checkpoint合同与当前正式合同不一致: "
                f"saved={saved_hash}, expected={self.contract_hash}"
            )


FORMAL_EXPERIMENT_CONTRACT = ExperimentContract(
    contract_name="multi_node_global_annual",
    contract_version=1,
    data_audit_version=1,
    annual_split=AnnualSplitContract(),
    settlement=SettlementContract(),
)


def formal_checkpoint_metadata(config: Any) -> dict[str, Any]:
    """校验配置后，返回应原样写入 checkpoint 的正式合同字段。"""
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(config)
    return FORMAL_EXPERIMENT_CONTRACT.checkpoint_metadata()
