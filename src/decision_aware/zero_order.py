"""zero_order.py — w10 §6 双点零阶高斯梯度 + L_proxy 注入.

正式版核心机制：因为硬 TopK 策略（argsort+选择）不可微，无法直接反向传播，
所以用 Nesterov-Spokoiny 双点高斯平滑梯度估计绕过不可微操作。

原理（w10 §6.1）:
    对业务损失 F(p̂) = -R(π(p̂); p)，定义高斯平滑目标:
        F_ε(p̂) = E_z[F(p̂ + εz)],  z ~ N(0, I)
    用 K 个成对扰动方向得到 Monte Carlo 估计:
        ĝ = (1/2Kε) · Σ_k [F(p̂+εz_k) - F(p̂-εz_k)] · z_k
    这是 smoothed 梯度的无偏估计（[7] Nesterov-Spokoiny 2017, [8] Duchi et al 2015）。

L_proxy 注入（w10 §6.2）:
    L_proxy = p̂ · stopgrad(ĝ)
    ∇_{p̂} L_proxy = ĝ  （标准链式法则，stopgrad 使 ĝ 为常数）
    再由 autodiff 更新 TSFM 参数。

关键: ĝ 必须在 no_grad 下计算（F 评估用 detached 扰动预测），
      L_proxy 的梯度只穿过 p̂（模型输出），不穿过 ĝ。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_epsilon(sigma: float, rho: float) -> float:
    """w10 §6.1: ε = ρ·σ_train。

    sigma: 训练集价格 std（从 norm_stats["price_da"]["std"] 读，约 370）
    rho: 扰动比例（搜索范围 0.02~0.2，默认 0.05）
    → ε ≈ 18.5（MAE 级别扰动，足够触发 TopK 排名变化但不偏离预测太远）
    """
    return float(rho * sigma)


def estimate_zo_gradient(
    p_hat: torch.Tensor,
    price: torch.Tensor,
    simulator: nn.Module,
    policy: nn.Module,
    epsilon: float,
    K: int,
    pred_clamp: tuple = (-500.0, 10000.0),
    price_da: torch.Tensor = None,
) -> torch.Tensor:
    """w10 §6.1 双点零阶高斯梯度估计。

    参数:
        p_hat: [B, H] 模型预测（真实尺度，有梯度——但内部会 detach）
        price: [B, H] 真实电价（RT 价，结算用）
        simulator: BESSSimulator，forward(u, price, price_da) → R [B]
        policy: HardTopKPolicy，forward(p_hat) → u [B,H]（不可微）
        epsilon: 扰动幅度 ε = ρ·σ_train
        K: 成对扰动方向数（2 或 4）
        pred_clamp: 预测截断范围（防扰动后溢出）
        price_da: [B, H] DA 价（双结算模式，None=单结算退化）

    返回:
        grad_zo: [B, H] 梯度估计（detached，无梯度）
        调用方用它构造 L_proxy = p̂ · stopgrad(grad_zo)
    """
    p_hat_d = p_hat.detach()
    B, H = p_hat_d.shape
    lo, hi = pred_clamp

    # 在 fp32 下累积梯度（AMP autocast 下 randn 可能是 fp16，精度不够）
    g = torch.zeros(B, H, dtype=torch.float32, device=p_hat_d.device)

    for _ in range(K):
        z = torch.randn(B, H, dtype=torch.float32, device=p_hat_d.device)
        p_plus = (p_hat_d + epsilon * z).clamp(lo, hi)
        p_minus = (p_hat_d - epsilon * z).clamp(lo, hi)

        with torch.no_grad():
            # 硬策略前向（不可微）
            u_plus = policy(p_plus)
            u_minus = policy(p_minus)
            # 收益评估：F = -R（业务损失 = 负收益）
            R_plus = simulator(u_plus, price, price_da=price_da)
            R_minus = simulator(u_minus, price, price_da=price_da)
            F_plus = -R_plus    # [B]
            F_minus = -R_minus  # [B]
            # 双点估计: (F+ - F-) / (2ε) * z
            diff = (F_plus - F_minus) / (2.0 * epsilon)  # [B]
            g = g + diff.unsqueeze(-1) * z                 # [B, H]

    g = g / K
    return g.detach()


def compute_l_proxy(p_hat: torch.Tensor, grad_zo: torch.Tensor,
                    per_sample_sum: bool = False) -> torch.Tensor:
    """w10 §6.2 L_proxy = p̂ · stopgrad(ĝ)。

    ∇_{p̂} L_proxy = ĝ（标准链式法则注入）。
    grad_zo 必须 detached（在 estimate_zo_gradient 里已 detach）。

    聚合方式（E1 修复）:
      per_sample_sum=False（默认，RT 项用）: .mean() = Σ/(B·H)
        w10 §6.2 RT 项要求 (1/24)Σ_{t=1..24}，mean over batch → Σ/(B·24)，匹配。
      per_sample_sum=True（DA/RT|DA 项用）: per-sample 点积 Σ_h，再 mean over batch = Σ/B
        w10 §6.2 DA/RT|DA 项是点积 (p̂DA)^T ĝ^DA，不除 horizon。
        原 .mean() 对 48h 项除以 48 → 梯度比 w10 小 48×。

    返回标量 loss。
    """
    g = grad_zo.detach()
    if per_sample_sum:
        # DA/RT|DA: w10 点积，per-sample Σ_h 再 mean over batch
        return (p_hat * g).sum(dim=-1).mean()
    # RT: w10 (1/24)Σ，mean over batch×horizon 等效
    return (p_hat * g).mean()


def compute_l_proxy_scaled(p_hat: torch.Tensor, grad_zo: torch.Tensor,
                           proxy_scale: float = 1.0,
                           per_sample_sum: bool = False) -> torch.Tensor:
    """带归一化的 L_proxy（proxy_scale 把 L_proxy 归到 O(1) 与 L_pred 同尺度）。"""
    return compute_l_proxy(p_hat, grad_zo, per_sample_sum=per_sample_sum) / proxy_scale


def estimate_zo_gradient_dual(
    p_hat: torch.Tensor,
    price_da: torch.Tensor,
    price_rt: torch.Tensor,
    simulator: nn.Module,
    action_fn,
    epsilon: float,
    K: int,
    pred_clamp: tuple = (-500.0, 10000.0),
    use_deviation_penalty: bool = False,
) -> torch.Tensor:
    """双结算零阶梯度（w10 §6.1 + §5）。

    与 estimate_zo_gradient 的区别：收益用 forward_dual（DA腿+RT偏差腿），
    动作由 action_fn 构建——调用方决定如何从「被扰动的曲线」+「其他固定曲线」
    生成 (u_da, u_rt)。这样日前扰动 p̂DA 时 uRT 保持固定、扰动 p̂RT|DA 时
    uDA 用 (p̂DA − p̂RT|DA_perturbed) 重算，符合 §6.2「分别扰动、其他不变」。

    参数:
        p_hat: [B, H] 被扰动的那条曲线（真实尺度，内部 detach）
        price_da/price_rt: [B, H] 真实价（结算用，不扰动）
        action_fn: callable(perturbed_p_hat) -> (u_da, u_rt)，闭包捕获固定曲线与策略
        epsilon: 该任务的扰动幅度 ε_m = ρ·σ_m
        K: 成对扰动方向数
    返回:
        grad_zo: [B, H] detached 梯度估计
    """
    p_hat_d = p_hat.detach()
    B, H = p_hat_d.shape
    lo, hi = pred_clamp
    g = torch.zeros(B, H, dtype=torch.float32, device=p_hat_d.device)

    for _ in range(K):
        z = torch.randn(B, H, dtype=torch.float32, device=p_hat_d.device)
        p_plus = (p_hat_d + epsilon * z).clamp(lo, hi)
        p_minus = (p_hat_d - epsilon * z).clamp(lo, hi)
        with torch.no_grad():
            u_da_p, u_rt_p = action_fn(p_plus)
            u_da_m, u_rt_m = action_fn(p_minus)
            R_plus = simulator.forward_dual(u_da_p, u_rt_p, price_da, price_rt,
                                            use_deviation_penalty=use_deviation_penalty)
            R_minus = simulator.forward_dual(u_da_m, u_rt_m, price_da, price_rt,
                                             use_deviation_penalty=use_deviation_penalty)
            diff = ((-R_plus) - (-R_minus)) / (2.0 * epsilon)   # [B]
            g = g + diff.unsqueeze(-1) * z                       # [B, H]
    g = g / K
    return g.detach()
