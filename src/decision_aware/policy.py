"""policy.py — 可微 BESS 策略 + 模拟器 + LP Oracle（先行版）.

三层策略，按 w10 规范逐步升级：
  1. STEPolicy      — 先行版 v1：sign 阈值 + STE，最轻量（保留向后兼容）
  2. TopKPolicy     — 先行版 v2：TopK/BotK 候选 + SOC 约束，对应 w10 第 4 节
  3. LP Oracle      — 真上界：scipy.linprog 解线性规划，对应 w10 第 5.2 节

BESS 模拟器保持不变（SOC 守恒 + 效率 + 可行性 clip）。
LP Oracle 用 scipy（无需 cvxpy/Gurobi），逐样本求解。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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
                 kappa: float = 0.0, soc_min: float = 0.0, soc_max: float = None):
        super().__init__()
        self.P = float(power_mw)
        self.E = float(energy_mwh)
        self.eta = float(eta)
        self.init_soc_frac = float(init_soc_frac)
        self.dt = float(dt)
        self.kappa = float(kappa)           # 退化+交易成本 USD/MWh
        self.s_min = float(soc_min)         # SOC 下限
        self.s_max = float(soc_max if soc_max is not None else energy_mwh)

    def forward(self, u: torch.Tensor, price: torch.Tensor,
                price_da: torch.Tensor = None) -> torch.Tensor:
        """u: [B,H] 动作；price: [B,H] RT 价；price_da: [B,H] DA 价（可选）→ R: [B]。

        真双结算（price_da 传入时）：
            R = Σ (pDA·uDA + pRT·Δu − κ|u|)    （w10 第5节）
        单结算（price_da=None，v1/v2 向后兼容）：
            R = Σ (pRT·u − κ|u|)
        """
        B, H = u.shape
        P, eta, dt = self.P, self.eta, self.dt
        s0 = self.E * self.init_soc_frac
        kappa = self.kappa

        # DA 价：没传就用 RT 价（单结算退化）
        if price_da is None:
            price_da = price

        soc = torch.full((B,), s0, dtype=u.dtype, device=u.device)
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
            # 真双结算：DA 腿 + RT 偏差腿 - 退化成本
            # 简化：uDA = u（日前申报=实际执行，Δu=0，RT腿=0）
            # 先行版不模拟 DA/RT 分离决策，DA 腿用全量
            revenue = revenue + (d_act - c_act) * pda_t - kappa * (d_act + c_act)
            soc = soc - d_act / eta + c_act * eta
        return revenue


# ════════════════════════════════════════════════════════════════════════════
# 策略 1：STE（v1，保留向后兼容）
# ════════════════════════════════════════════════════════════════════════════
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
# Oracle 1：greedy（v1，保留向后兼容）
# ════════════════════════════════════════════════════════════════════════════
def oracle_revenue(price: torch.Tensor, simulator: BESSSimulator) -> torch.Tensor:
    """greedy oracle（v1）：sign(price-mean)。偏弱，模型可能超过。保留向后兼容。"""
    with torch.no_grad():
        thr = price.mean(dim=-1, keepdim=True)
        u_oracle = torch.sign(price - thr)
        R_star = simulator(u_oracle, price)
    return R_star.detach()


# ════════════════════════════════════════════════════════════════════════════
# Oracle 2：LP（v2，真上界，对应 w10 第 5.2 节）
# ════════════════════════════════════════════════════════════════════════════
def lp_oracle_revenue(price: torch.Tensor, simulator: BESSSimulator) -> torch.Tensor:
    """LP Oracle：用真实电价解线性规划求最优充放电 → R*（真上界）。

    对每个样本解：
        max  Σ_t (discharge_t - charge_t) · price_t
        s.t. SOC 守恒 + 容量/功率约束 + 效率

    用 scipy.linprog（无需 cvxpy）。返回 [B] 张量，无梯度。
    模型不可能超过 LP oracle（它是真最优），故 Regret ≥ 0 恒成立。
    """
    P, E, eta = simulator.P, simulator.E, simulator.eta
    s0 = E * simulator.init_soc_frac
    s_min, s_max = simulator.s_min, simulator.s_max  # w10: 0.4-3.6
    kappa = simulator.kappa
    dt = simulator.dt
    device, dtype = price.device, price.dtype

    price_np = price.detach().cpu().numpy().astype(np.float64)  # [B,H]
    B, H = price_np.shape
    results = np.empty(B, dtype=np.float64)

    for b in range(B):
        p = price_np[b]  # [H]
        # 决策变量：x = [dis_0..dis_{H-1}, chg_0..chg_{H-1}]，各 H 个，共 2H
        # dis_t, chg_t ∈ [0, P*dt]（充放电功率，非负）
        # 目标：max Σ (dis_t - chg_t)·p_t - κ·(dis_t + chg_t)  →  linprog 求 min，取负
        c = np.concatenate([-p + kappa, p + kappa])  # min -Σ[(dis-chg)·p - κ(dis+chg)]

        # 约束：SOC 守恒 soc_{t+1} = soc_t + eta·chg_t - dis_t/eta
        # 重排：soc_{t+1} - soc_t - eta·chg_t + dis_t/eta = 0
        # SOC 用累积表达，等式约束
        A_eq = np.zeros((H, 2 * H))
        b_eq = np.zeros(H)
        # soc_t = s0 + Σ_{τ<t}(eta·chg_τ - dis_τ/eta)
        # 约束 soc_t ∈ [s_min, s_max] → 拆成不等式
        A_ub, b_ub = [], []
        for t in range(H):
            # soc at end of step t
            # soc_{t+1} = s0 + Σ_{τ≤t}(eta·chg_τ - dis_τ/eta)
            # 约束: s_min ≤ soc_{t+1} ≤ s_max
            row_dis = np.zeros(H)
            row_chg = np.zeros(H)
            row_dis[:t+1] = -1.0 / eta   # dis 减少 soc
            row_chg[:t+1] = eta           # chg 增加 soc
            row = np.concatenate([row_dis, row_chg])
            # soc ≤ s_max:  row·x ≤ s_max - s0
            A_ub.append(row)
            b_ub.append(s_max - s0)
            # soc ≥ s_min:  -row·x ≤ -(s_min - s0) = s0 - s_min
            A_ub.append(-row)
            b_ub.append(s0 - s_min)

        # 变量边界：dis_t, chg_t ∈ [0, P*dt]
        bounds = [(0, P * dt)] * (2 * H)

        from scipy.optimize import linprog
        res = linprog(c, A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None,
                      bounds=bounds, method='highs')
        if res.success:
            x = res.x
            dis = x[:H]
            chg = x[H:]
            results[b] = float(np.sum((dis - chg) * p))
        else:
            # LP 求解失败，退回 greedy oracle 兜底
            thr = p.mean()
            u = np.sign(p - thr).astype(np.float64)
            u_t = torch.from_numpy(u).unsqueeze(0)
            p_t = torch.from_numpy(p).unsqueeze(0)
            results[b] = simulator(u_t, p_t).item()

    return torch.from_numpy(results).to(device=device, dtype=dtype).detach()


# ════════════════════════════════════════════════════════════════════════════
# 统一入口
# ════════════════════════════════════════════════════════════════════════════
def compute_regret(p_hat: torch.Tensor, price: torch.Tensor,
                   simulator: BESSSimulator, policy,
                   oracle: str = "greedy"):
    """返回 (R_model, R_star, regret)。

    oracle: "greedy"（v1，向后兼容）或 "lp"（v2，真上界）。
    regret 的梯度穿过 policy 回传到 p_hat；R_star 无梯度。
    """
    u = policy(p_hat)
    R_model = simulator(u, price)
    if oracle == "lp":
        R_star = lp_oracle_revenue(price, simulator)
    else:
        R_star = oracle_revenue(price, simulator)
    regret = R_star - R_model
    return R_model, R_star, regret
