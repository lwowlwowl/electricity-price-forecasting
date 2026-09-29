"""policy.py — 可微 BESS 策略 + 模拟器 + LP Oracle（先行版）.

三层策略，按 w10 规范逐步升级：
  1. STEPolicy      — 先行版 v1：sign 阈值 + STE，最轻量（保留向后兼容）
  2. TopKPolicy     — 先行版 v2：TopK/BotK 候选 + SOC 约束，对应 w10 第 4 节
  3. LP Oracle      — 真上界：scipy.linprog 解线性规划，对应 w10 第 5.2 节

BESS 模拟器保持不变（SOC 守恒 + 效率 + 可行性 clip）。
LP Oracle 用 scipy（无需 cvxpy/Gurobi），逐样本求解。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class LPSolveStatus:
    """One HiGHS solve record exposed for audit/debugging."""

    sample_index: int
    leg: str
    success: bool
    status_code: int
    message: str
    objective: float | None


@dataclass(frozen=True)
class BESSActionProjection:
    """一段动作在电池约束下的可执行结果。

    ``action`` 仍使用 [-1, 1] 的归一化动作；``net_energy_mwh`` 使用市场
    结算符号（放电为正、充电为负）。``soc_path_mwh`` 包含起点，因此长度
    比动作多 1。这个结构既可用于 DA 计划可行性检查，也可用于小数字审计。
    """

    action: torch.Tensor
    net_energy_mwh: torch.Tensor
    discharge_mwh: torch.Tensor
    charge_mwh: torch.Tensor
    soc_path_mwh: torch.Tensor
    clipped_energy_mwh: torch.Tensor


class LPOracleSolveError(RuntimeError):
    """Raised when an LP Oracle solve is not certified optimal."""

    def __init__(self, solve_status: LPSolveStatus):
        self.solve_status = solve_status
        super().__init__(
            "LP Oracle solve failed "
            f"(sample={solve_status.sample_index}, leg={solve_status.leg}, "
            f"status={solve_status.status_code}): {solve_status.message}"
        )


# ════════════════════════════════════════════════════════════════════════════
# BESS 模拟器（不变）
# ════════════════════════════════════════════════════════════════════════════
class BESSSimulator(nn.Module):
    """电池收益模拟器。u_t ∈ [-1,1]：+1=放电(P MW)、-1=充电(P MW)。

    v3 支持：
    - 真双结算：日前腿按 DA 价、实时偏差腿按 RT 价（w10 第5节）
    - 退化成本 κ（w10 第7节：27 USD/MWh）
    - SOC 上下限 [s_min, s_max]（w10：0.4-3.6 MWh）
    - 效率 η=0.95（w10，v1/v2=0.9）

    单结算模式（v1/v2 向后兼容）：只传 price（RT），DA 价=RT 价。
    """

    def __init__(self, power_mw: float, energy_mwh: float, eta: float,
                 init_soc_frac: float = 0.5, dt: float = 1.0,
                 kappa: float = 0.0, soc_min: float = 0.0, soc_max: float = None,
                 e_cyc: float = None):
        super().__init__()
        self.P = float(power_mw)
        self.E = float(energy_mwh)
        self.eta = float(eta)
        self.init_soc_frac = float(init_soc_frac)
        self.dt = float(dt)
        self.kappa = float(kappa)           # 退化+交易成本 USD/MWh
        self.s_min = float(soc_min)         # SOC 下限
        self.s_max = float(soc_max if soc_max is not None else energy_mwh)
        # w10 §4.2: 每日累计放电上限 E_cyc（默认=E，即满容量一次循环）
        self.e_cyc = float(e_cyc if e_cyc is not None else energy_mwh)

    @torch.no_grad()
    def project_actions(
        self,
        u: torch.Tensor,
        initial_soc_mwh: float | torch.Tensor | None = None,
    ) -> BESSActionProjection:
        """把意图动作投影为满足 SOC 和单段放电量上限的动作。

        该函数不修改模拟器状态。对 DA 使用时得到的是一条独立的“计划
        SOC”，不会更新真实 SOC；真实 SOC 仍只能由 RT 实际动作更新。
        ``u`` 的每一行被视为一个独立计划段，累计放电上限从 0 开始。
        """
        if u.ndim != 2:
            raise ValueError(f"u应为[B,H]，实际为{tuple(u.shape)}")
        batch_size, horizon = u.shape
        if initial_soc_mwh is None:
            soc = torch.full(
                (batch_size,), self.E * self.init_soc_frac,
                dtype=u.dtype, device=u.device,
            )
        else:
            soc = torch.as_tensor(
                initial_soc_mwh, dtype=u.dtype, device=u.device
            ).reshape(-1)
            if soc.numel() == 1:
                soc = soc.expand(batch_size).clone()
            elif soc.numel() != batch_size:
                raise ValueError(
                    "initial_soc_mwh必须是标量或长度与batch一致的向量"
                )
        tolerance = 1e-6
        if torch.any(soc < self.s_min - tolerance) or torch.any(
            soc > self.s_max + tolerance
        ):
            raise ValueError("initial_soc_mwh超出SOC上下限")
        # 长序列反复乘除效率后可能出现3.6000001这类纯浮点误差。
        soc = soc.clamp(min=self.s_min, max=self.s_max)

        feasible = torch.zeros_like(u)
        discharge_all = torch.zeros_like(u)
        charge_all = torch.zeros_like(u)
        soc_path = torch.empty(
            (batch_size, horizon + 1), dtype=u.dtype, device=u.device
        )
        soc_path[:, 0] = soc
        clipped = torch.zeros(batch_size, dtype=u.dtype, device=u.device)
        discharged = torch.zeros(batch_size, dtype=u.dtype, device=u.device)
        scale = self.P * self.dt

        for step in range(horizon):
            intended = u[:, step]
            requested_discharge = torch.clamp(intended, min=0.0) * scale
            requested_charge = torch.clamp(-intended, min=0.0) * scale
            discharge = torch.minimum(
                requested_discharge,
                torch.clamp((soc - self.s_min) * self.eta, min=0.0),
            )
            discharge = torch.minimum(
                discharge,
                torch.clamp(self.e_cyc - discharged, min=0.0),
            )
            charge = torch.minimum(
                requested_charge,
                torch.clamp((self.s_max - soc) / self.eta, min=0.0),
            )
            discharge_all[:, step] = discharge
            charge_all[:, step] = charge
            feasible[:, step] = (discharge - charge) / scale
            clipped = clipped + (
                requested_discharge + requested_charge - discharge - charge
            )
            soc = soc - discharge / self.eta + charge * self.eta
            discharged = discharged + discharge
            soc_path[:, step + 1] = soc

        return BESSActionProjection(
            action=feasible,
            net_energy_mwh=discharge_all - charge_all,
            discharge_mwh=discharge_all,
            charge_mwh=charge_all,
            soc_path_mwh=soc_path,
            clipped_energy_mwh=clipped,
        )

    def forward(self, u: torch.Tensor, price: torch.Tensor,
                price_da: torch.Tensor = None) -> torch.Tensor:
        """u: [B,H] 动作；price: [B,H] RT 价；price_da: [B,H] DA 价（可选）→ R: [B]。

        真双结算（price_da 传入时）：
            R = Σ (pDA·uDA + pRT·Δu − κ|u|)    （w10 第5节）
        单结算（price_da=None，v1/v2 向后兼容）：
            R = Σ (pRT·u − κ|u|)
        含 w10 §4.2 SOC + 循环约束（s_min/s_max + E_cyc/q_t）。
        """
        B, H = u.shape
        P, eta, dt = self.P, self.eta, self.dt
        s0 = self.E * self.init_soc_frac
        kappa = self.kappa
        e_cyc = self.e_cyc

        # DA 价：没传就用 RT 价（单结算退化）
        if price_da is None:
            price_da = price

        soc = torch.full((B,), s0, dtype=u.dtype, device=u.device)
        q_t = torch.zeros(B, dtype=u.dtype, device=u.device)   # 当日累计放电量（w10 §4.2）
        revenue = torch.zeros(B, dtype=u.dtype, device=u.device)
        for t in range(H):
            ut = u[:, t]
            pda_t = price_da[:, t]
            prt_t = price[:, t]
            dis = torch.clamp(ut, min=0.0) * P * dt
            chg = torch.clamp(-ut, min=0.0) * P * dt
            # SOC 可行性 clip（w10: s_min ≤ soc ≤ s_max）
            d_act = torch.clamp(dis, max=(soc - self.s_min) * eta)
            c_act = torch.clamp(chg, max=(self.s_max - soc) / eta)
            # 循环约束 clip（w10 §4.2: q_t ≤ E_cyc）
            d_act = torch.clamp(d_act, max=(e_cyc - q_t))
            revenue = revenue + (d_act - c_act) * pda_t - kappa * (d_act + c_act)
            soc = soc - d_act / eta + c_act * eta
            q_t = q_t + d_act
        return revenue

    def forward_dual(self, u_da: torch.Tensor, u_rt: torch.Tensor,
                     price_da: torch.Tensor, price_rt: torch.Tensor,
                     use_deviation_penalty: bool = False,
                     plan_track: bool = False,
                     enforce_da_feasibility: bool = True) -> torch.Tensor:
        """DA/RT 分离决策的真双结算（w10 §4.3 + §5）。

        u_da: [B, 24] 在截止时点锁定的交付日计划动作
        u_rt: [B, 24] 交付日内逐小时得到的实时实际动作
        price_da: [B, 24] 真实 DA 价
        price_rt: [B, 24] 真实 RT 价
        use_deviation_penalty: 是否启用偏差罚金（w10 §5.1）
        plan_track: 预留的“RT跟踪DA再套利”策略；当前未实现，传True会报错
        enforce_da_feasibility: 是否先用独立计划SOC把DA动作投影为可行计划

        两条 SOC 轨迹分离（w10 §4.3）：
          - 计划 SOC：只用于检查/修正DA提交可行性，不改变真实SOC
          - 实际 SOC：u_rt 在 24h 独立从 s0 起跑，更新退化成本
        E_cyc 循环约束在 24h 交易日内生效（第 25h 重置，w10 §4.2）。

        R = Σ_{t<24} (pDA·uDA + pRT·Δu − κ|uRT|)   Δu = uRT − uDA
        若 use_deviation_penalty: R -= P_dev（3% 容忍 + 双倍 RT 价，w10 §5.1）
        """
        B, H_da = u_da.shape
        H_rt = u_rt.shape[1]
        if plan_track:
            raise NotImplementedError(
                "plan_track尚未实现，不能静默当作False使用"
            )
        if any(value.ndim != 2 for value in (u_da, u_rt, price_da, price_rt)):
            raise ValueError("u_da/u_rt/price_da/price_rt都必须是[B,H]")
        if any(value.shape[0] != B for value in (u_rt, price_da, price_rt)):
            raise ValueError("DA/RT动作和价格的batch维必须一致")
        P, eta, dt = self.P, self.eta, self.dt
        s0 = self.E * self.init_soc_frac
        kappa = self.kappa
        e_cyc = self.e_cyc
        H_settle = min(H_da, H_rt, price_da.shape[1], price_rt.shape[1], 24)
        if H_settle < 1:
            raise ValueError("结算窗口不能为空")
        u_da_settle = u_da[:, :H_settle]
        if enforce_da_feasibility:
            u_da_settle = self.project_actions(u_da_settle).action

        # ── 实际 SOC 轨迹 + 结算（24h，w10 §4.3）────────────────────────────
        soc_act = torch.full((B,), s0, dtype=u_da.dtype, device=u_da.device)
        q_act = torch.zeros(B, dtype=u_da.dtype, device=u_da.device)
        revenue = torch.zeros(B, dtype=u_da.dtype, device=u_da.device)
        penalty = torch.zeros(B, dtype=u_da.dtype, device=u_da.device)
        for t in range(H_settle):
            uda_t = u_da_settle[:, t]
            urt_t = u_rt[:, t]
            pda_t = price_da[:, t]
            prt_t = price_rt[:, t]

            # 日前腿：按 DA 价结算日前计划量
            da_dis = torch.clamp(uda_t, min=0.0) * P * dt
            da_chg = torch.clamp(-uda_t, min=0.0) * P * dt
            da_leg = (da_dis - da_chg) * pda_t

            # 实际动作的 SOC 裁剪（w10 §4.2）——先算 clipped actual，
            # E3 修复：RT 偏差腿和罚金用 actual（计量值），不用 intent。
            act_dis = torch.clamp(urt_t, min=0.0) * P * dt
            act_chg = torch.clamp(-urt_t, min=0.0) * P * dt
            d_act = torch.clamp(act_dis, max=(soc_act - self.s_min) * eta)
            d_act = torch.clamp(d_act, max=(e_cyc - q_act))
            c_act = torch.clamp(act_chg, max=(self.s_max - soc_act) / eta)
            deg_cost = kappa * (d_act + c_act)

            # RT 偏差腿：按 RT 价结算「实际偏差量」（w10 §5: Δu = uRT_actual − uDA）
            # uRT_actual = (d_act − c_act)/Δt（SOC 裁剪后的实际净放电），
            # uDA = da_dis − da_chg（DA 计划量，DA 侧不裁剪——日前申报按计划结算）。
            # 市场按实际计量值结算偏差，不是意图值。
            actual_rt_net = (d_act - c_act)   # 实际净放电量（已裁剪）
            da_net = (da_dis - da_chg)        # DA 计划净量
            delta_actual = actual_rt_net - da_net   # 实际偏差量
            rt_leg = delta_actual * prt_t

            revenue = revenue + da_leg + rt_leg - deg_cost

            # 偏差罚金（w10 §5.1）：偏差超 3%|uDA| 的部分按双倍 RT 价罚
            # E3: 用 actual 偏差（计量值），非 intent
            if use_deviation_penalty:
                threshold = 0.03 * torch.abs(da_net)   # 3% of DA planned volume
                excess = torch.clamp(torch.abs(delta_actual) - threshold, min=0.0)
                penalty = penalty + 2.0 * torch.abs(prt_t) * excess

            soc_act = soc_act - d_act / eta + c_act * eta
            q_act = q_act + d_act

        if use_deviation_penalty:
            revenue = revenue - penalty
        return revenue


# ════════════════════════════════════════════════════════════════════════════
# 策略 1：STE（v1，保留向后兼容）
# ════════════════════════════════════════════════════════════════════════════
class LookaheadMPCPolicy(nn.Module):
    """在短时域内联合选择充/停/放路径，而不做单小时贪心判断。

    动作集默认为 ``{-1, 0, +1}``，对 H=4 只需比较 3^4=81 条路径。
    每条路径都按相同的功率、SOC、效率、每日放电上限和吞吐成本
    执行。回测时每小时重新规划并只执行第一步。

    在无偏差罚金的价格接受者双结算中，已锁定的 DA 头寸对
    所有 RT 候选路径是同一个常数项，不会改变最优 RT 动作。
    只有启用偏差罚金时，``da_plan`` 才会影响路径选择。
    """

    def __init__(
        self,
        power_mw: float,
        energy_mwh: float,
        eta: float,
        init_soc_frac: float = 0.5,
        kappa: float = 0.0,
        soc_min: float = 0.0,
        soc_max: float | None = None,
        e_cyc: float | None = None,
        action_levels: tuple[float, ...] = (-1.0, 0.0, 1.0),
        terminal_value_weight: float = 1.0,
    ):
        super().__init__()
        if not action_levels or any(abs(value) > 1.0 for value in action_levels):
            raise ValueError("action_levels必须是[-1,1]内的非空动作集")
        self.P = float(power_mw)
        self.E = float(energy_mwh)
        self.eta = float(eta)
        self.init_soc_frac = float(init_soc_frac)
        self.kappa = float(kappa)
        self.s_min = float(soc_min)
        self.s_max = float(soc_max if soc_max is not None else energy_mwh)
        self.e_cyc = float(e_cyc if e_cyc is not None else energy_mwh)
        self.action_levels = tuple(float(value) for value in action_levels)
        self.terminal_value_weight = float(terminal_value_weight)

    def _sequences(self, horizon: int, reference: torch.Tensor) -> torch.Tensor:
        levels = torch.tensor(
            self.action_levels, dtype=reference.dtype, device=reference.device
        )
        if horizon == 1:
            return levels.reshape(-1, 1)
        return torch.cartesian_prod(*([levels] * horizon)).reshape(-1, horizon)

    @torch.no_grad()
    def forward(
        self,
        predicted_price: torch.Tensor,
        initial_soc_mwh: float | torch.Tensor | None = None,
        initial_discharged_mwh: float | torch.Tensor = 0.0,
        da_plan: torch.Tensor | None = None,
        use_deviation_penalty: bool = False,
        cycle_reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回每个样本在预测价格下的最优离散动作路径。"""
        if predicted_price.ndim != 2:
            raise ValueError("predicted_price必须是[B,H]")
        batch_size, horizon = predicted_price.shape
        if horizon < 1:
            raise ValueError("前瞻时域不能为空")
        sequences = self._sequences(horizon, predicted_price)
        sequence_count = sequences.shape[0]
        actions = sequences.unsqueeze(0).expand(batch_size, -1, -1)

        def _state(value, default: float, name: str) -> torch.Tensor:
            if value is None:
                value = default
            tensor = torch.as_tensor(
                value, dtype=predicted_price.dtype, device=predicted_price.device
            ).reshape(-1)
            if tensor.numel() == 1:
                tensor = tensor.expand(batch_size)
            if tensor.numel() != batch_size:
                raise ValueError(f"{name}必须是标量或长度为B的向量")
            return tensor

        soc0 = _state(
            initial_soc_mwh, self.E * self.init_soc_frac, "initial_soc_mwh"
        )
        q0 = _state(initial_discharged_mwh, 0.0, "initial_discharged_mwh")
        tolerance = 1e-6
        if torch.any(soc0 < self.s_min - tolerance) or torch.any(
            soc0 > self.s_max + tolerance
        ):
            raise ValueError("initial_soc_mwh超出SOC上下限")
        soc = soc0.clamp(self.s_min, self.s_max).unsqueeze(1).expand(
            -1, sequence_count
        ).clone()
        discharged = q0.clamp(min=0.0, max=self.e_cyc).unsqueeze(1).expand(
            -1, sequence_count
        ).clone()
        value = torch.zeros_like(soc)
        throughput = torch.zeros_like(soc)

        if da_plan is None:
            da_plan = torch.zeros_like(predicted_price)
        else:
            da_plan = torch.as_tensor(
                da_plan,
                dtype=predicted_price.dtype,
                device=predicted_price.device,
            )
            if da_plan.shape != predicted_price.shape:
                raise ValueError("da_plan必须与predicted_price同形状")
        if cycle_reset_mask is None:
            cycle_reset_mask = torch.zeros_like(predicted_price, dtype=torch.bool)
        else:
            cycle_reset_mask = torch.as_tensor(
                cycle_reset_mask, dtype=torch.bool, device=predicted_price.device
            )
            if cycle_reset_mask.shape != predicted_price.shape:
                raise ValueError("cycle_reset_mask必须与predicted_price同形状")

        for step in range(horizon):
            reset = cycle_reset_mask[:, step].unsqueeze(1)
            discharged = torch.where(reset, torch.zeros_like(discharged), discharged)
            intended = actions[:, :, step]
            requested_discharge = torch.clamp(intended, min=0.0) * self.P
            requested_charge = torch.clamp(-intended, min=0.0) * self.P
            actual_discharge = torch.minimum(
                requested_discharge,
                torch.clamp((soc - self.s_min) * self.eta, min=0.0),
            )
            actual_discharge = torch.minimum(
                actual_discharge,
                torch.clamp(self.e_cyc - discharged, min=0.0),
            )
            actual_charge = torch.minimum(
                requested_charge,
                torch.clamp((self.s_max - soc) / self.eta, min=0.0),
            )
            actual_net = actual_discharge - actual_charge
            da_net = da_plan[:, step].unsqueeze(1) * self.P
            price = predicted_price[:, step].unsqueeze(1)
            step_throughput = actual_discharge + actual_charge
            value = value + (actual_net - da_net) * price
            value = value - self.kappa * step_throughput
            if use_deviation_penalty:
                tolerance_band = 0.03 * torch.abs(da_net)
                excess = torch.clamp(
                    torch.abs(actual_net - da_net) - tolerance_band, min=0.0
                )
                value = value - 2.0 * torch.abs(price) * excess
            throughput = throughput + step_throughput
            soc = soc - actual_discharge / self.eta + actual_charge * self.eta
            discharged = discharged + actual_discharge

        # 有限时域末端的存电不能被当成“没有价值”。用最后一个
        # 可见预测价格估计剩余能量的机会价值，避免四小时窗口在末端
        # 无条件把电池放空。负价时持有库存不会被赋予负价值。
        terminal_price = torch.clamp(predicted_price[:, -1], min=0.0).unsqueeze(1)
        terminal_value = (
            self.terminal_value_weight * soc * self.eta * terminal_price
        )
        # 同收益时轻微偏好更少吞吐，避免选到无效充放。
        intended_effort = torch.abs(actions).sum(dim=2)
        score = value + terminal_value - throughput * 1e-6 - intended_effort * 1e-7
        best = torch.argmax(score, dim=1)
        return sequences[best]


class STEPolicy(nn.Module):
    """可微 greedy：前向 sign(p̂-mean)，反向 tanh 软阈值。最轻量。"""
    def __init__(self, k: float = 5.0):
        super().__init__()
        self.k = float(k)
    def forward(self, p_hat: torch.Tensor) -> torch.Tensor:
        thr = p_hat.mean(dim=-1, keepdim=True)
        soft = torch.tanh(self.k * (p_hat - thr))
        hard = torch.sign(soft)
        return hard + (soft - soft.detach())


# ════════════════════════════════════════════════════════════════════════════
# 策略 2：TopK/BotK（v2，对应 w10 第 4 节）
# ════════════════════════════════════════════════════════════════════════════
class TopKPolicy(nn.Module):
    """TopK/BotK 候选 + SOC 约束（w10 第 4 节）。

    给定价格信号 c（预测价或预测价差），选预测价最高的 K_d 个时段放电（TopK）、
    最低的 K_c 个时段充电（BotK），按时间顺序执行并做 SOC 可行性修正。
    可微性：用 soft TopK（sigmoid 加权）近似 hard TopK，前向接近 hard、反向有梯度。

    K_c / K_d 由储能物理决定：4MWh/1MW → 最多充/放 4 小时 → K=4。
    """
    def __init__(self, k_charge: int = 4, k_discharge: int = 4, tau: float = 0.1):
        super().__init__()
        self.k_c = k_charge
        self.k_d = k_discharge
        self.tau = tau  # soft TopK 温度，越小越接近 hard

    def _soft_topk_mask(self, x: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
        """soft TopK：返回 [B,H] mask，top-k 时段≈1其余≈0，可微。

        用排序 scatter 构造排名：原始位置 i 在排序后的名次 rank_i，
        rank_i < k → sigmoid 给≈1，否则≈0。
        """
        sign = 1.0 if largest else -1.0
        sorted_idx = torch.argsort(sign * x, dim=-1, descending=True)  # [B,H]
        # 构造排名张量：rank[i] = 原始位置 i 在排序后的名次
        B, H = x.shape
        ranks = torch.empty(B, H, dtype=torch.long, device=x.device)
        arange_b = torch.arange(B, device=x.device).unsqueeze(1).expand(B, H)
        ranks.scatter_(-1, sorted_idx, torch.arange(H, device=x.device).unsqueeze(0).expand(B, H))
        # rank < k → 1, else → 0，sigmoid 软化
        logits = (k - 0.5 - ranks.float()) / self.tau
        return torch.sigmoid(logits)

    def forward(self, p_hat: torch.Tensor) -> torch.Tensor:
        """p_hat: [B,H] → u: [B,H] ∈ [-1,1]。放电=+1(TopK高价)，充电=-1(BotK低价)。"""
        dis_mask = self._soft_topk_mask(p_hat, self.k_d, largest=True)   # 高价时段放电
        chg_mask = self._soft_topk_mask(p_hat, self.k_c, largest=False)  # 低价时段充电
        u = dis_mask - chg_mask  # +1 放电 / -1 充电 / 0 静置
        return u


# ════════════════════════════════════════════════════════════════════════════
# Greedy hindsight baseline（v1 历史参考，非 Oracle）
# ════════════════════════════════════════════════════════════════════════════
def greedy_hindsight_revenue(price: torch.Tensor,
                             simulator: BESSSimulator) -> torch.Tensor:
    """Greedy hindsight baseline: sign(price-mean), not an optimal bound."""
    with torch.no_grad():
        thr = price.mean(dim=-1, keepdim=True)
        u_greedy = torch.sign(price - thr)
        R_greedy = simulator(u_greedy, price)
    return R_greedy.detach()


# ════════════════════════════════════════════════════════════════════════════
# Oracle 2：LP（v2，真上界，对应 w10 第 5.2 节）
# ════════════════════════════════════════════════════════════════════════════
def lp_oracle_revenue(price: torch.Tensor, simulator: BESSSimulator,
                      *, return_status: bool = False):
    """LP Oracle：用真实电价解线性规划求最优充放电 → R*（真上界）。

    对每个样本解：
        max  Σ_t (discharge_t - charge_t) · price_t - κ·(dis_t + chg_t)
        s.t. SOC 守恒 + 容量/功率约束 + 效率 + E_cyc 循环约束（w10 §4.2）

    用 scipy.linprog（无需 cvxpy）。默认返回 [B] 张量，无梯度。
    return_status=True 时额外返回每个样本的 LPSolveStatus，用于审计求解状态。
    任一样本未获得最优证明时抛出 LPOracleSolveError，不回退到 Greedy。
    模型不可能超过 LP oracle（它是真最优），故 Regret ≥ 0 恒成立。

    修复：原版收益没减 κ、缺 E_cyc 约束 → R* 被高估、regret 被放大。
    现直接复用 _lp_revenue_one（已正确含两者），与双结算 Oracle 同口径。
    """
    P, E, eta = simulator.P, simulator.E, simulator.eta
    s0 = E * simulator.init_soc_frac
    s_min, s_max = simulator.s_min, simulator.s_max  # w10: 0.4-3.6
    kappa = simulator.kappa
    e_cyc = simulator.e_cyc        # w10 §4.2: E_cyc 循环约束
    dt = simulator.dt
    device, dtype = price.device, price.dtype

    price_np = price.detach().cpu().numpy().astype(np.float64)  # [B,H]
    B = price_np.shape[0]
    results = np.empty(B, dtype=np.float64)
    statuses = []

    for b in range(B):
        results[b], status = _lp_revenue_one(
            price_np[b], P, E, eta, s0, s_min, s_max, kappa, dt, e_cyc,
            sample_index=b, leg="single",
        )
        statuses.append(status)

    values = torch.from_numpy(results).to(device=device, dtype=dtype).detach()
    return (values, tuple(statuses)) if return_status else values


# ════════════════════════════════════════════════════════════════════════════
# Oracle 3：双结算 LP（w10 §5.2，正式版）
# ════════════════════════════════════════════════════════════════════════════
def _lp_revenue_one(price_np, P, E, eta, s0, s_min, s_max, kappa, dt, e_cyc,
                    *, sample_index: int = -1, leg: str = "single"):
    """单序列 LP：max Σ(dis-chg)·p − κ(dis+chg)  s.t. SOC + 容量/功率 + E_cyc 循环约束。

    返回 (最优收益, LPSolveStatus)。kappa=0 时退化为无退化成本（用于 DA 腿）。
    w10 §4.2: 累计放电量 q_t = Σ_{τ≤t} dis_τ ≤ E_cyc。
    """
    from scipy.optimize import linprog
    H = len(price_np)
    c = np.concatenate([-price_np + kappa, price_np + kappa])
    A_ub, b_ub = [], []
    for t in range(H):
        # SOC 守恒约束（累积）
        row_dis = np.zeros(H)
        row_chg = np.zeros(H)
        row_dis[:t + 1] = -1.0 / eta
        row_chg[:t + 1] = eta
        row = np.concatenate([row_dis, row_chg])
        A_ub.append(row);  b_ub.append(s_max - s0)
        A_ub.append(-row); b_ub.append(s0 - s_min)
        # 循环约束：Σ_{τ≤t} dis_τ ≤ E_cyc（w10 §4.2）
        row_cyc = np.concatenate([np.zeros(H), np.zeros(H)])
        row_cyc[:t + 1] = 1.0   # dis 部分
        A_ub.append(row_cyc);   b_ub.append(e_cyc)
    bounds = [(0, P * dt)] * (2 * H)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method='highs')
    status = LPSolveStatus(
        sample_index=sample_index,
        leg=leg,
        success=bool(res.success),
        status_code=int(res.status),
        message=str(res.message),
        objective=float(res.fun) if res.success else None,
    )
    if not res.success:
        raise LPOracleSolveError(status)
    x = res.x
    dis, chg = x[:H], x[H:]
    revenue = float(np.sum((dis - chg) * price_np) - kappa * np.sum(dis + chg))
    return revenue, status


def lp_oracle_revenue_dual(price_da: torch.Tensor, price_rt: torch.Tensor,
                           simulator: BESSSimulator,
                           use_deviation_penalty: bool = False,
                           *, return_status: bool = False):
    """双结算 LP Oracle（w10 §5.2）。

    无偏差罚金（use_deviation_penalty=False）:
      R* = LP_DA(spread) + LP_RT(pRT)，两条 SOC 轨迹独立（§4.3 计划/实际分离）：
        DA 腿: max Σ(pDA−pRT)·uDA   s.t. 计划 SOC + 功率，无 κ（退化只对 uRT）
        RT 腿: max Σ(pRT·uRT − κ|uRT|)  s.t. 实际 SOC + 功率
      两个 LP 结构相同、独立求解相加。返回 [B] 张量，无梯度。
      return_status=True 时同时返回 DA/RT 各腿的 LPSolveStatus。

    有偏差罚金（use_deviation_penalty=True）:
      当前评估的是 plan_track 受限策略类：uDA≠0 时业务规则强制 uRT=uDA；
      uDA=0 时，2×|pRT| 罚金使任意偏差都不划算。因此该受限策略类下退化为
      单结算 LP: max Σ pDA·uDA − κ|uDA|  s.t. SOC+E_cyc+功率。
      注意：由于罚金含 3% 免罚容忍带，这不是“所有可行 DA/RT 动作”的无限制全局上界；
      R* 及 Regret 必须明确标注为 plan_track 策略类口径。
    """
    P, E, eta = simulator.P, simulator.E, simulator.eta
    s0 = E * simulator.init_soc_frac
    s_min, s_max = simulator.s_min, simulator.s_max
    kappa = simulator.kappa
    e_cyc = simulator.e_cyc
    dt = simulator.dt
    device, dtype = price_da.device, price_da.dtype

    da_np = price_da.detach().cpu().numpy().astype(np.float64)
    rt_np = price_rt.detach().cpu().numpy().astype(np.float64)
    B = da_np.shape[0]
    results = np.empty(B, dtype=np.float64)
    statuses = []

    if use_deviation_penalty:
        # plan_track 受限策略类下 uRT=uDA → 单结算 LP（DA 价）。
        # 这是受限策略类上界，非含 3% 容忍带的无限制 DA/RT 全局上界。
        for b in range(B):
            results[b], status = _lp_revenue_one(
                da_np[b], P, E, eta, s0, s_min, s_max, kappa, dt, e_cyc,
                sample_index=b, leg="penalty_plan_tracking_restricted",
            )
            statuses.append(status)
    else:
        # 无罚金：双 LP 独立求解（DA腿 spread + RT腿 pRT）
        for b in range(B):
            spread = da_np[b] - rt_np[b]                   # DA 腿价格 = 价差
            r_da, status_da = _lp_revenue_one(
                spread, P, E, eta, s0, s_min, s_max, 0.0, dt, e_cyc,
                sample_index=b, leg="day_ahead",
            )
            r_rt, status_rt = _lp_revenue_one(
                rt_np[b], P, E, eta, s0, s_min, s_max, kappa, dt, e_cyc,
                sample_index=b, leg="real_time",
            )
            results[b] = r_da + r_rt
            statuses.extend((status_da, status_rt))
    values = torch.from_numpy(results).to(device=device, dtype=dtype).detach()
    return (values, tuple(statuses)) if return_status else values


# ════════════════════════════════════════════════════════════════════════════
# 计划跟踪优先规则（w10 §4.3，启用偏差罚金时）
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def plan_track_override(u_da: torch.Tensor, u_rt_topk: torch.Tensor) -> torch.Tensor:
    """w10 §4.3：启用偏差罚金时，RT 策略先尽量执行日前动作 uDA，再套利。

    规则：在每个时段，若 uDA≠0，RT 动作优先取 uDA（消除偏差罚金）；
    若 uDA=0，保留 TopK 套利动作。若 uDA 与 TopK 同号冲突，取 uDA（计划优先）。
    这是预先确定的业务规则，不参与学习。

    u_da:     [B, 24] 日前计划动作 ∈ {-1,0,+1}
    u_rt_topk:[B, 24] RT TopK 套利动作 ∈ {-1,0,+1}
    返回:     [B, 24] 合成后的 uRT
    """
    # uDA≠0 的时段：强制 uRT=uDA（避免偏差罚金）
    mask_da = (u_da != 0).float()
    u_rt = mask_da * u_da + (1.0 - mask_da) * u_rt_topk
    return u_rt


# ════════════════════════════════════════════════════════════════════════════
# 统一入口
# ════════════════════════════════════════════════════════════════════════════
def compute_regret(p_hat: torch.Tensor, price: torch.Tensor,
                   simulator: BESSSimulator, policy,
                   oracle: str = "lp"):
    """返回 (R_model, R_star, regret)。

    Oracle/Regret 只保留给求解成功的 LP。Greedy hindsight baseline
    应单独用 greedy_hindsight_revenue 计算 R_greedy 和 gap_to_greedy。
    regret 的梯度穿过 policy 回传到 p_hat；R_star 无梯度。
    """
    u = policy(p_hat)
    R_model = simulator(u, price)
    if oracle != "lp":
        raise ValueError(
            "compute_regret only accepts oracle='lp'. For the historical Greedy "
            "hindsight baseline, compute R_greedy with greedy_hindsight_revenue "
            "and report gap_to_greedy instead of Regret."
        )
    R_star = lp_oracle_revenue(price, simulator)
    regret = R_star - R_model
    return R_model, R_star, regret


# ════════════════════════════════════════════════════════════════════════════
# 策略 3：Hard TopK（正式版，w10 §4，不可微，配合零阶梯度使用）
# ════════════════════════════════════════════════════════════════════════════
class HardTopKPolicy(nn.Module):
    """w10 §4 硬 TopK/BotK 策略（不可微，用于零阶梯度）。

    选预测价最高 K_d 个时段放电(+1)、最低 K_c 个时段充电(-1)，其余静置(0)。
    非可微：torch.topk + 赋值。SOC 可行性由 BESSSimulator 内部 clip 处理。

    K_c / K_d 由储能物理决定：4MWh/1MW → 最多充/放 4 小时 → K=4。
    与 STEPolicy/TopKPolicy 的区别：前向是真正硬选择（无 sigmoid 近似），
    因此不能直接反向传播——必须通过零阶梯度（zero_order.py）估计梯度。
    """
    def __init__(self, k_charge: int = 4, k_discharge: int = 4,
                 spread_threshold: float = 0.0):
        super().__init__()
        self.k_c = k_charge
        self.k_d = k_discharge
        # w10 §4.1 价差门控：候选充放电对价差须覆盖效率损失+运行成本才保留。
        # >0 时启用；默认 0=关闭（向后兼容）。RT 策略建议 κ/η；DA 价差口径见 loss.py 注释。
        self.spread_threshold = float(spread_threshold)

    @torch.no_grad()
    def forward(self, p_hat: torch.Tensor) -> torch.Tensor:
        """p_hat: [B,H] → u: [B,H] ∈ {-1,0,+1}。放电=+1(TopK高价)，充电=-1(BotK低价)。

        价差门控（w10 §4.1）：按候选值排序配对（最高放电 vs 最低充电），
        逐对检查 c[dis]−c[chg] > spread_threshold，不满足的剔除（置 0）。
        两序列均有序，价差随配对序号递减，故首次不满足即可 break。
        """
        B, H = p_hat.shape
        u = torch.zeros_like(p_hat)
        k_d = min(self.k_d, H)
        k_c = min(self.k_c, H)
        thr = self.spread_threshold
        for b in range(B):
            x = p_hat[b]
            top_idx = torch.topk(x, k_d).indices                  # 放电候选（高 c），降序
            bot_idx = torch.topk(x, k_c, largest=False).indices   # 充电候选（低 c），升序
            if thr > 0:
                top_vals = x[top_idx]      # [k_d] 降序
                bot_vals = x[bot_idx]      # [k_c] 升序
                n_pairs = min(k_d, k_c)
                keep_dis, keep_chg = [], []
                for i in range(n_pairs):
                    if float(top_vals[i] - bot_vals[i]) > thr:
                        keep_dis.append(top_idx[i])
                        keep_chg.append(bot_idx[i])
                    else:
                        break  # 后续配对价差更小，均不保留
                if keep_dis:
                    u[b, torch.stack(keep_dis)] = 1.0
                if keep_chg:
                    u[b, torch.stack(keep_chg)] = -1.0
            else:
                u[b, top_idx] = 1.0
                u[b, bot_idx] = -1.0
        return u
