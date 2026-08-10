"""loss.py — 先行版联合损失 + α/β 退火（todo3 #2 决策 B）。

L = α·L_pred(Huber) + β·L_bus(Regret)
退火：α: alpha0→alpha_end，β: beta0→beta_end，线性跨 anneal_epochs，之后恒定。
当 π 可微时 β→1 即纯业务损失驱动（与正式版同口径）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import PilotConfig
from .policy import BESSSimulator, STEPolicy, compute_regret


def huber_loss(pred: torch.Tensor, true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """逐元素 Huber，对电价尖峰更鲁棒（todo3 #3 先行版）。"""
    return F.huber_loss(pred, true, reduction="mean", delta=delta)


def half_se_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """w10 §3 半平方误差：0.5·‖pred−true‖²（mean over all elements）。

    与 XGBoost 接口 §6.3 的半平方误差一致（G_j = p̂_j − p_j 的曲率 1）。
    强制 fp32 计算：电价尖峰~1000+，err² 可超 fp16 上限(65504)→inf→NaN。
    """
    pred_f = pred.float()
    true_f = true.float()
    return 0.5 * F.mse_loss(pred_f, true_f, reduction="mean")


def _pred_loss(pred, true, cfg):
    """按 cfg.use_mse_loss 选 MSE（w10 §3）或 Huber（向后兼容）。"""
    if getattr(cfg, "use_mse_loss", True):
        return half_se_loss(pred, true)
    return huber_loss(pred, true, delta=cfg.huber_delta)


def anneal_alpha_beta(epoch: int, cfg: PilotConfig):
    """线性退火 (alpha0,beta0) → (alpha_end,beta_end)，跨 anneal_epochs 后恒定。

    若 cfg.pretrain_epochs > 0，前 pretrain_epochs 个 epoch 保持 α=1/β=0（纯预测），
    之后再开始退火。
    """
    pt = getattr(cfg, "pretrain_epochs", 0)
    if pt > 0 and epoch < pt:
        return 1.0, 0.0
    eff_epoch = epoch - pt
    frac = min(1.0, max(0.0, eff_epoch / max(1, cfg.anneal_epochs)))
    alpha = cfg.alpha0 + frac * (cfg.alpha_end - cfg.alpha0)
    beta = cfg.beta0 + frac * (cfg.beta_end - cfg.beta0)
    return float(alpha), float(beta)


def total_loss(
    p_da: torch.Tensor,          # [B, H] 真实尺度预测（模型已反归一化 + clamp）
    price_tgt: torch.Tensor,     # [B, H] 真实电价
    simulator: BESSSimulator,
    policy,
    alpha: float,
    beta: float,
    delta: float = 1.0,
    pred_scale: float = 1.0,      # 把 L_pred 归一到 O(1)，与 L_bus 同尺度，防 β→1 梯度爆炸
    bus_scale: float = 1.0,       # 把 L_bus  归一到 O(1)
    oracle: str = "greedy",       # "greedy"(v1) 或 "lp"(v2 真上界)
):
    """返回 (total_loss, metrics_dict)。metrics_dict 全为 .detach().item() 标量，供日志。

    total = α·(L_pred/pred_scale) + β·(L_bus/bus_scale)，两分量均 O(1)，
    α/β 退火才是真正的权衡（否则 regret 尺度 ~10× 于 pred，β=1 直接 NaN）。
    oracle="lp" 时 R* 用 LP 真上界（模型不可能超过，regret≥0）。
    """
    l_pred_raw = huber_loss(p_da, price_tgt, delta=delta)
    R_model, R_star, regret = compute_regret(p_da, price_tgt, simulator, policy, oracle=oracle)
    l_bus_raw = regret.mean()
    l_pred = l_pred_raw / pred_scale
    l_bus = l_bus_raw / bus_scale
    total = alpha * l_pred + beta * l_bus
    metrics = {
        "loss_total": float(total.detach().item()),       # 归一化后的总损失（O(1)）
        "loss_pred": float(l_pred_raw.detach().item()),  # 原始 Huber（$ 尺度，~25）
        "loss_bus": float(l_bus_raw.detach().item()),    # 原始 Regret（$ 尺度，~250）
        "R_model": float(R_model.detach().mean().item()),
        "R_star": float(R_star.detach().mean().item()),
        "regret": float(regret.detach().mean().item()),
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return total, metrics


# ════════════════════════════════════════════════════════════════════════════
# 正式版损失：零阶梯度 + L_proxy（w10 §6.2）
# ════════════════════════════════════════════════════════════════════════════
def total_loss_zo(
    model_out: dict,              # 模型输出：{"p_da", "p_rt_da"(可选), "p_rt"(可选)}
    price_da_tgt: torch.Tensor,   # [B, H_tgt] 真实 DA 价
    price_rt_tgt: torch.Tensor,   # [B, H_tgt] 真实 RT 价
    simulator,                     # BESSSimulator
    policy_hard,                   # HardTopKPolicy
    alpha: float,
    beta: float,
    cfg,                           # PilotConfig
    epsilon_da: float,             # ε_DA = ρ·σ_DA（w10 §6.1 每任务独立 ε_m）
    epsilon_rt: float,             # ε_RT = ρ·σ_RT（RT|DA 与 RT 共用 RT 的 σ）
    oracle_train: bool = True,
):
    """正式版损失：L = α·(L_pred/pred_scale) + β·(L_proxy/proxy_scale)。

    dual_split 模式（w10 §3+§6.2）:
      L_pred = LD + LR
        LD = Huber(p̂DA, pDA) + Huber(p̂RT|DA, pRT)    两条 48h 曲线
        LR = Huber(p̂RT, pRT)                           24h 滚动 RT 预测
      L_proxy = p̂DA·ĝ^DA + p̂RT|DA·ĝ^{RT|D} + p̂RT·ĝ^{RT}  三项分别注入

    非 dual_split 模式（向后兼容 v4/v5）:
      L_pred = Huber(p̂DA, pDA)
      L_proxy = p̂DA·ĝ^DA
    """
    from .policy import lp_oracle_revenue, lp_oracle_revenue_dual, plan_track_override
    from .zero_order import estimate_zo_gradient, estimate_zo_gradient_dual, compute_l_proxy_scaled

    use_dual_split = getattr(cfg, "use_dual_split", False)
    use_dev_penalty = getattr(cfg, "use_deviation_penalty", False)
    p_da = model_out["p_da"]      # [B, H] or [B, 48]
    H_da = p_da.shape[1]

    # ── L_pred（w10 §3 半平方误差 + /96 归一化）─────────────────────────────
    if use_dual_split:
        # w10 §3: LD = 两条 48h 曲线（pDA + pRT|DA），LR = 24 个滚动窗口取平均
        p_rt_da = model_out.get("p_rt_da")      # [B, 48]
        p_rt_windows = model_out.get("p_rt_windows")  # [B, 24, H_rt]（w10 §3）
        p_rt_roll = model_out.get("p_rt")       # [B, 24] 动作信号（兼容旧字段）
        da_tgt_48 = price_da_tgt[:, :H_da]
        rt_tgt_48 = price_rt_tgt[:, :H_da]
        ld = _pred_loss(p_da, da_tgt_48, cfg)
        if p_rt_da is not None:
            ld = ld + _pred_loss(p_rt_da, rt_tgt_48, cfg)
        # LR: w10 §3 — 24 个 H_rt 小时滚动窗口取平均
        lr = torch.tensor(0.0, device=p_da.device)
        if p_rt_windows is not None:
            H_rt = p_rt_windows.shape[-1]
            # 构造 [B, 24, H_rt] 真实 RT 窗口：窗口 t 从 t 开始取 H_rt 步
            rt_tgt_full = price_rt_tgt[:, :cfg.horizon_da + H_rt - 1]  # [B, 24+H-1]
            rt_windows_tgt = rt_tgt_full.unfold(-1, H_rt, 1)[:, :cfg.horizon_da, :]  # [B,24,H]
            lr = _pred_loss(p_rt_windows, rt_windows_tgt, cfg)
        elif p_rt_roll is not None:
            # fallback：无窗口输出时用动作信号退化为单窗口（不合规，兼容旧 ckpt）
            lr = _pred_loss(p_rt_roll, price_rt_tgt[:, :cfg.horizon_da], cfg)
        l_pred_raw = ld + lr
    else:
        l_pred_raw = _pred_loss(p_da, price_da_tgt[:, :H_da], cfg)
    l_pred = l_pred_raw / cfg.pred_scale

    # ── L_proxy（零阶梯度注入）──────────────────────────────────────────────
    # 只对前 24h 做充放电决策（日前提交前 24h，w10 §4.3）
    H_action = cfg.horizon_da  # 24
    p_da_24 = p_da[:, :H_action]
    da_tgt_24 = price_da_tgt[:, :H_action]
    rt_tgt_24 = price_rt_tgt[:, :H_action]

    if use_dual_split:
        # ── 双结算零阶梯度（w10 §6.1+§5+§4.1）──────────────────────────────
        # uDA 由价差 d̂=p̂DA−p̂RT|DA 决定（§4.1），uRT 由 p̂RT 决定。
        # 扰动某条曲线时其余保持不变（§6.2），收益用 forward_dual。
        # plan_track（w10 §4.3）：启用偏差罚金时 uRT 先跟 uDA 再套利。
        # 注：DA/RT|DA 扰动在 48h 完整曲线上（forward_dual DA腿跑 48h），
        #     RT 扰动在 24h 动作信号上（RT 腿只跑 24h）。L_proxy 仍按原曲线注入。
        p_da_full = p_da                                               # [B, 48] 有梯度
        p_da_full_d = p_da_full.detach()
        # 保留有梯度的原版给 L_proxy 注入；detach 版给 action_fn 闭包用（闭包在
        # no_grad 下跑，用哪个都一样，但显式 detach 防止意外梯度穿透策略仿真）。
        p_rt_da_orig = model_out.get("p_rt_da")                        # [B, 48] 有梯度，可能 None
        p_rt_da_full = (p_rt_da_orig if p_rt_da_orig is not None
                        else torch.zeros_like(p_da_full)).detach()     # [B, 48] 闭包用
        p_rt_roll = model_out.get("p_rt")
        # uRT 动作信号：有 p̂RT 用预测，否则退回真实 RT 价（RT Oracle，fallback）
        p_rt_24_orig = p_rt_roll[:, :H_action] if p_rt_roll is not None else rt_tgt_24  # 有梯度
        p_rt_24 = p_rt_24_orig.detach()                                # 闭包用
        da_tgt_48 = price_da_tgt[:, :H_da]                              # [B, 48]
        rt_tgt_48 = price_rt_tgt[:, :H_da]                              # [B, 48]

        def _make_u_rt(u_da_24_cur, p_rt_signal):
            """构造 uRT：plan_track 时先跟 uDA，否则纯 TopK。"""
            u_rt_topk = policy_hard(p_rt_signal)
            if use_dev_penalty:   # 偏差罚金启用 → 计划跟踪优先（w10 §4.3）
                return plan_track_override(u_da_24_cur, u_rt_topk)
            return u_rt_topk

        u_da_48_fixed = policy_hard(p_da_full_d - p_rt_da_full)         # [B, 48]
        u_da_24_fixed = u_da_48_fixed[:, :H_action]
        u_rt_fixed = _make_u_rt(u_da_24_fixed, p_rt_24)

        # ĝ^DA: 扰动 p̂DA(48h) → 价差变 → uDA(48h) 变；uRT 在 plan_track 时也跟变
        def _action_da(p_da_pert):
            u_da_48 = policy_hard(p_da_pert - p_rt_da_full)
            return u_da_48, _make_u_rt(u_da_48[:, :H_action], p_rt_24)
        grad_da = estimate_zo_gradient_dual(
            p_da_full, da_tgt_48, rt_tgt_48, simulator, _action_da,
            epsilon_da, cfg.zo_K, cfg.pred_clamp, use_dev_penalty)
        # E1: DA 项 w10 是点积（不除 horizon），用 per_sample_sum=True
        l_proxy = compute_l_proxy_scaled(p_da_full, grad_da, cfg.proxy_scale,
                                         per_sample_sum=True)

        # ĝ^{RT|D}: 扰动 p̂RT|DA(48h) → 价差变 → uDA(48h) 变；uRT 跟变
        if p_rt_da_orig is not None:
            def _action_rtda(p_rtda_pert):
                u_da_48 = policy_hard(p_da_full_d - p_rtda_pert)
                return u_da_48, _make_u_rt(u_da_48[:, :H_action], p_rt_24)
            grad_rt_da = estimate_zo_gradient_dual(
                p_rt_da_full, da_tgt_48, rt_tgt_48, simulator, _action_rtda,
                epsilon_rt, cfg.zo_K, cfg.pred_clamp, use_dev_penalty)
            # E1: RT|DA 项同 DA（48h 点积），用 per_sample_sum=True
            l_proxy = l_proxy + compute_l_proxy_scaled(p_rt_da_orig, grad_rt_da,
                                                       cfg.proxy_scale, per_sample_sum=True)

        # ĝ^RT: 扰动 p̂RT(24h) → uRT 的 TopK 部分变；uDA(48h) 固定
        if p_rt_roll is not None:
            def _action_rt(p_rt_pert):
                return u_da_48_fixed, _make_u_rt(u_da_24_fixed, p_rt_pert)
            grad_rt = estimate_zo_gradient_dual(
                p_rt_24, da_tgt_24, rt_tgt_24, simulator, _action_rt,
                epsilon_rt, cfg.zo_K, cfg.pred_clamp, use_dev_penalty)
            # L_proxy 用有梯度的原版
            l_proxy = l_proxy + compute_l_proxy_scaled(p_rt_24_orig, grad_rt, cfg.proxy_scale)

        g_norm = float(grad_da.norm().item())
    else:
        # ── 非 dual_split（v4 向后兼容）：单结算 ZO ─────────────────────────
        grad_da = estimate_zo_gradient(
            p_da_24, da_tgt_24, simulator, policy_hard,
            epsilon_da, cfg.zo_K, cfg.pred_clamp,
            price_da=da_tgt_24 if getattr(cfg, "use_dual_settlement", False) else None,
        )
        l_proxy = compute_l_proxy_scaled(p_da_24, grad_da, cfg.proxy_scale)
        g_norm = float(grad_da.norm().item())

    # ── 总损失 ──────────────────────────────────────────────────────────────
    total = alpha * l_pred + beta * l_proxy

    # ── 日志指标 ────────────────────────────────────────────────────────────
    with torch.no_grad():
        mae = (p_da[:, :H_action] - da_tgt_24).abs().mean()
        rmse = ((p_da[:, :H_action] - da_tgt_24) ** 2).mean().sqrt()   # w10 §7 指标
        if use_dual_split:
            # w10 §4.1: uDA 用价差 d̂=p̂DA−p̂RT|DA，48h 完整窗口（§4.3）
            p_rt_da_full = (model_out["p_rt_da"] if "p_rt_da" in model_out
                            else torch.zeros_like(p_da))
            u_da_48 = policy_hard(p_da - p_rt_da_full)             # [B, 48]
            u_da_24 = u_da_48[:, :H_action]                        # [B, 24] 用于 plan_track
            if "p_rt" in model_out:
                p_rt_sig = model_out["p_rt"][:, :H_action]
            else:
                p_rt_sig = rt_tgt_24                                # fallback: RT Oracle
            u_rt = _make_u_rt(u_da_24, p_rt_sig)                   # [B, 24]
            # forward_dual: u_da=[B,48], u_rt=[B,24]，结算前 24h
            R_model = simulator.forward_dual(u_da_48, u_rt, da_tgt_24, rt_tgt_24,
                                             use_deviation_penalty=use_dev_penalty)
        else:
            u_da = policy_hard(p_da_24)
            R_model = simulator(u_da, da_tgt_24,
                                price_da=da_tgt_24 if getattr(cfg, "use_dual_settlement", False) else None)
        R_model_val = float(R_model.detach().mean().item())

        if oracle_train:
            if use_dual_split:
                # w10 §5.2: 双结算 LP Oracle（DA腿+RT腿），与 R_model 同口径
                R_star = lp_oracle_revenue_dual(da_tgt_24, rt_tgt_24, simulator,
                                                use_deviation_penalty=use_dev_penalty)
            else:
                R_star = lp_oracle_revenue(da_tgt_24, simulator)
            R_star_val = float(R_star.detach().mean().item())
            regret_val = R_star_val - R_model_val
        else:
            R_star_val = 0.0
            regret_val = 0.0

    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_pred": float(l_pred_raw.detach().item()),
        "loss_proxy": float(l_proxy.detach().item()),
        "R_model": R_model_val,
        "R_star": R_star_val,
        "regret": regret_val,
        "mae": float(mae.detach().item()),
        "rmse": float(rmse.detach().item()),
        "g_norm": g_norm,
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return total, metrics
