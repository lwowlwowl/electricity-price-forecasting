"""
Chronos-2 worker（运行在 external/chronos-forecasting/.venv）
=============================================================
读取 request.npz，加载一次 Chronos-2，批量预测所有任务，写回 response.npz。
协议见同目录 README.md。

Chronos-2 是三个模型里协变量能力最原生的：支持 list-of-dicts 格式同时传
target / past_covariates / future_covariates。本 worker 自动判断：
  - 请求里带协变量 → 用 dict 格式喂协变量（发挥 Chronos-2 强项）
  - 不带协变量      → 退化为单变量 tensor 输入

多节点：逐节点（逐 series）独立预测后堆叠。

用法（由 foundation.py 自动调用）：
    python worker_chronos2.py  request.npz  response.npz
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch


def _predict_one_series(pipe, target, hist_cov, fut_cov, horizon):
    """对单条序列预测，返回 (mean, q10, q90)，各形状 (horizon,)。"""
    if hist_cov is not None and fut_cov is not None:
        n_cov = hist_cov.shape[1]
        inputs = [{
            "target": target,
            "past_covariates": {f"c{k}": hist_cov[:, k] for k in range(n_cov)},
            "future_covariates": {f"c{k}": fut_cov[:, k] for k in range(n_cov)},
        }]
    else:
        # 单变量：(n_series=1, n_variates=1, history)
        inputs = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)

    quantiles, mean = pipe.predict_quantiles(
        inputs=inputs, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0].detach().cpu().numpy()   # (n_variates, horizon, 3)
    m = mean[0].detach().cpu().numpy()        # (n_variates, horizon)
    return m[0], q[0, :, 0], q[0, :, 2]


def _predict_multivariate(pipe, ctx, horizon):
    """多变量联合预测：把所有节点作为一条多变量记录的 target 一次喂入。
    ctx 形状 (T, n_series)；返回三个 (horizon, n_series) 数组。

    Chronos-2 的多变量约定（见 chronos2/dataset.py）：一个 dict，键 'target'
    取 2-D 张量 (n_variates, history_length)，模型把这些 variate 作为协同序列
    联合预测 → 这正是“多节点联合建模”要的效果。
    """
    # (T, n_series) → (n_variates=n_series, history_length=T)
    target = ctx.T.astype(np.float32)
    inputs = [{"target": target}]
    quantiles, mean = pipe.predict_quantiles(
        inputs=inputs, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0].detach().cpu().numpy()   # (n_variates, horizon, 3)
    m = mean[0].detach().cpu().numpy()        # (n_variates, horizon)
    # variate 顺序与 target 行顺序一致 → 直接转成 (horizon, n_series)
    return m.T.astype(np.float32), q[:, :, 0].T.astype(np.float32), q[:, :, 2].T.astype(np.float32)


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])
    keys = set(req.files)
    # 多变量旋钮（手册 §6 消融 C）：1=多节点联合建模，0=逐列单变量。
    multivariate = bool(int(req["multivariate"])) if "multivariate" in keys else False

    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map=device, dtype=torch.float32)
    pipe.model.eval()

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)        # (T, n_series)
        n_series = ctx.shape[1]
        has_cov = f"future_cov__{i}" in keys and f"hist_cov__{i}" in keys

        # 多变量且无协变量时，走联合建模一次预测全部节点。
        # （协变量 + 多变量的组合较复杂，本期协变量路径仍逐列，保证稳定。）
        if multivariate and n_series > 1 and not has_cov:
            mean2d, q10_2d, q90_2d = _predict_multivariate(pipe, ctx, horizon)
            out[f"mean__{i}"] = mean2d
            out[f"q10__{i}"] = q10_2d
            out[f"q90__{i}"] = q90_2d
            continue

        hist_cov = req[f"hist_cov__{i}"].astype(np.float32) if has_cov else None
        fut_cov = req[f"future_cov__{i}"].astype(np.float32) if has_cov else None
        means, q10s, q90s = [], [], []
        for j in range(n_series):
            m, q10, q90 = _predict_one_series(
                pipe, ctx[:, j], hist_cov, fut_cov, horizon)
            means.append(m)
            q10s.append(q10)
            q90s.append(q90)

        out[f"mean__{i}"] = np.stack(means, axis=1).astype(np.float32)  # (horizon, n_series)
        out[f"q10__{i}"] = np.stack(q10s, axis=1).astype(np.float32)
        out[f"q90__{i}"] = np.stack(q90s, axis=1).astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
