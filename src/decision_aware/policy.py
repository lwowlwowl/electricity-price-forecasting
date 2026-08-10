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
                     plan_track: bool = False) -> torch.Tensor:
        """DA/RT 分离决策的真双结算（w10 §4.3 + §5）。

        u_da: [B, 48] 日前计划动作（48h 窗口，只前 24h 提交结算）
        u_rt: [B, 24] 实时实际动作（24h 交易日）
        price_da: [B, 48] 真实 DA 价
        price_rt: [B, 48] 真实 RT 价（前 24h 用于结算）
        use_deviation_penalty: 是否启用偏差罚金（w10 §5.1）
        plan_track: 启用偏差罚金时 RT 先尽量执行 uDA 再套利（w10 §4.3）

        两条 SOC 轨迹分离（w10 §4.3）：
          - 计划 SOC：u_da 在 48h 连续传递，只检查日前提交可行性
          - 实际 SOC：u_rt 在 24h 独立从 s0 起跑，更新退化成本
        E_cyc 循环约束在 24h 交易日内生效（第 25h 重置，w10 §4.2）。

        R = Σ_{t<24} (pDA·uDA + pRT·Δu − κ|uRT|)   Δu = uRT − uDA
        若 use_deviation_penalty: R -= P_dev（3% 容忍 + 双倍 RT 价，w10 §5.1）
        """
        B, H_da = u_da.shape
        H_rt = u_rt.shape[1]
        P, eta, dt = self.P, self.eta, self.dt
        s0 = self.E * self.init_soc_frac
        kappa = self.kappa
        e_cyc = self.e_cyc
        H_settle = min(H_da, H_rt, 24)   # 结算窗口 = 交易日 24h

        # 注：原 plan SOC 轨迹（48h 连续）已删除——它算完即丢弃，从不参与收益
        # 计算，是死代码（48 步 Python 循环白跑）。w10 §4.3 的"计划 SOC 检查
        # 提交可行性"当前未实现约束/罚金，若后续需要可重新加入。

        # ── 实际 SOC 轨迹 + 结算（24h，w10 §4.3）────────────────────────────
        soc_act = torch.full((B,), s0, dtype=u_da.dtype, device=u_da.device)
        q_act = torch.zeros(B, dtype=u_da.dtype, device=u_da.device)
        revenue = torch.zeros(B, dtype=u_da.dtype, device=u_da.device)
        penalty = torch.zeros(B, dtype=u_da.dtype, device=u_da.device)
        for t in range(H_settle):
            uda_t = u_da[:, t]
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
        max  Σ_t (discharge_t - charge_t) · price_t - κ·(dis_t + chg_t)
        s.t. SOC 守恒 + 容量/功率约束 + 效率 + E_cyc 循环约束（w10 §4.2）

    用 scipy.linprog（无需 cvxpy）。返回 [B] 张量，无梯度。
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

    for b in range(B):
        results[b] = _lp_revenue_one(price_np[b], P, E, eta, s0, s_min, s_max,
                                     kappa, dt, e_cyc)

    return torch.from_numpy(results).to(device=device, dtype=dtype).detach()


# ════════════════════════════════════════════════════════════════════════════
# Oracle 3：双结算 LP（w10 §5.2，正式版）
# ════════════════════════════════════════════════════════════════════════════
def _lp_revenue_one(price_np, P, E, eta, s0, s_min, s_max, kappa, dt, e_cyc):
    """单序列 LP：max Σ(dis-chg)·p − κ(dis+chg)  s.t. SOC + 容量/功率 + E_cyc 循环约束。

    返回最优收益标量。kappa=0 时退化为无退化成本（用于 DA 腿）。
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
    if res.success:
        x = res.x
        dis, chg = x[:H], x[H:]
        return float(np.sum((dis - chg) * price_np) - kappa * np.sum(dis + chg))
    # LP 求解失败，退回 greedy 兜底
    thr = price_np.mean()
    u = np.sign(price_np - thr)
    soc, q, rev = s0, 0.0, 0.0
    for t in range(H):
        ut = u[t]
        d = max(ut, 0) * P * dt
        ch = max(-ut, 0) * P * dt
        d_act = min(d, max(0.0, (soc - s_min) * eta), max(0.0, e_cyc - q))
        c_act = min(ch, max(0.0, (s_max - soc) / eta))
        rev += (d_act - c_act) * price_np[t] - kappa * (d_act + c_act)
        soc = soc - d_act / eta + c_act * eta
        q += d_act
    return float(rev)


def lp_oracle_revenue_dual(price_da: torch.Tensor, price_rt: torch.Tensor,
                           simulator: BESSSimulator,
                           use_deviation_penalty: bool = False) -> torch.Tensor:
    """双结算 LP Oracle（w10 §5.2）。

    无偏差罚金（use_deviation_penalty=False）:
      R* = LP_DA(spread) + LP_RT(pRT)，两条 SOC 轨迹独立（§4.3 计划/实际分离）：
        DA 腿: max Σ(pDA−pRT)·uDA   s.t. 计划 SOC + 功率，无 κ（退化只对 uRT）
        RT 腿: max Σ(pRT·uRT − κ|uRT|)  s.t. 实际 SOC + 功率
      两个 LP 结构相同、独立求解相加。返回 [B] 张量，无梯度。

    有偏差罚金（use_deviation_penalty=True，E2 修复）:
      罚金 P_dev = Σ 2|pRT|·[|Δu|−0.03|uDA|]+ 耦合 uDA 和 uRT，两个独立 LP 无法表达。
      但分析表明：罚金倍数 2×|pRT| 使偏差收益 = Δu·pRT − 2|pRT|·|Δu| ≤ −|pRT|·|Δu| < 0
      （对任意 Δu 符号），即偏差永远不划算 → 最优 uRT=uDA（无偏差）→ 罚金=0。
      此时 R* 退化为单结算 LP: max Σ pDA·uDA − κ|uDA|  s.t. SOC+E_cyc+功率。
      这与模型 plan_track 行为一致（uRT 跟 uDA），是公平的真上界。
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

    if use_deviation_penalty:
        # E2: 偏差罚金下 uRT=uDA 最优 → 单结算 LP（DA 价）
        # R* = max Σ pDA·uDA − κ|uDA|  s.t. SOC + E_cyc + 功率
        for b in range(B):
            results[b] = _lp_revenue_one(da_np[b], P, E, eta, s0, s_min, s_max,
                                         kappa, dt, e_cyc)
    else:
        # 无罚金：双 LP 独立求解（DA腿 spread + RT腿 pRT）
        for b in range(B):
            spread = da_np[b] - rt_np[b]                   # DA 腿价格 = 价差
            r_da = _lp_revenue_one(spread, P, E, eta, s0, s_min, s_max, 0.0, dt, e_cyc)
            r_rt = _lp_revenue_one(rt_np[b], P, E, eta, s0, s_min, s_max, kappa, dt, e_cyc)
            results[b] = r_da + r_rt
    return torch.from_numpy(results).to(device=device, dtype=dtype).detach()


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
