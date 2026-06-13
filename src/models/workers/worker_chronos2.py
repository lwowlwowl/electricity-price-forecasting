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


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])
    keys = set(req.files)

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

        means, q10s, q90s = [], [], []
        for j in range(n_series):
            target = ctx[:, j]
            if has_cov:
                hist_cov = req[f"hist_cov__{i}"].astype(np.float32)   # (T, n_cov)
                fut_cov = req[f"future_cov__{i}"].astype(np.float32)  # (horizon, n_cov)
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
                inputs=inputs,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            q = quantiles[0]            # (n_variates, horizon, 3) tensor
            m = mean[0]                 # (n_variates, horizon)
            q = q.detach().cpu().numpy()
            m = m.detach().cpu().numpy()
            means.append(m[0])          # 第 0 个 variate 是 target
            q10s.append(q[0, :, 0])
            q90s.append(q[0, :, 2])

        out[f"mean__{i}"] = np.stack(means, axis=1).astype(np.float32)  # (horizon, n_series)
        out[f"q10__{i}"] = np.stack(q10s, axis=1).astype(np.float32)
        out[f"q90__{i}"] = np.stack(q90s, axis=1).astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
