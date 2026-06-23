"""
Chronos-2 结构消融 worker
==========================
在 worker_chronos2.py 基础上增加消融钩子。

用法（由 foundation_ablation.py 调用）：
    python worker_chronos2_ablation.py  request.npz  response.npz
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SA_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(SA_DIR)
ROOT = os.path.dirname(SRC_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, SA_DIR)  # for ablations.py

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
        inputs = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)

    quantiles, mean = pipe.predict_quantiles(
        inputs=inputs, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0].detach().cpu().numpy()
    m = mean[0].detach().cpu().numpy()
    return m[0], q[0, :, 0], q[0, :, 2]


def _predict_multivariate(pipe, ctx, horizon):
    """多变量联合预测。"""
    target = ctx.T.astype(np.float32)
    inputs = [{"target": target}]
    quantiles, mean = pipe.predict_quantiles(
        inputs=inputs, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0].detach().cpu().numpy()
    m = mean[0].detach().cpu().numpy()
    return m.T.astype(np.float32), q[:, :, 0].T.astype(np.float32), q[:, :, 2].T.astype(np.float32)


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])
    keys = set(req.files)
    multivariate = bool(int(req["multivariate"])) if "multivariate" in keys else False

    # 读取消融配置
    ablation_type = str(req["ablation_type"]) if "ablation_type" in keys else None

    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map=device, dtype=torch.float32)
    pipe.model.eval()

    # ★ 应用结构消融
    if ablation_type:
        from ablations import apply_ablation
        pipe.model = apply_ablation(pipe.model, "chronos2", ablation_type)
        pipe.model.eval()

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)
        n_series = ctx.shape[1]
        has_cov = f"future_cov__{i}" in keys and f"hist_cov__{i}" in keys

        try:
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

            out[f"mean__{i}"] = np.stack(means, axis=1).astype(np.float32)
            out[f"q10__{i}"] = np.stack(q10s, axis=1).astype(np.float32)
            out[f"q90__{i}"] = np.stack(q90s, axis=1).astype(np.float32)

        except Exception as e:
            print(f"⚠️  Chronos-2 消融后推理失败 (task {i}): {e}", file=sys.stderr)
            out[f"mean__{i}"] = np.full((horizon, n_series), np.nan, dtype=np.float32)
            out[f"q10__{i}"] = np.full((horizon, n_series), np.nan, dtype=np.float32)
            out[f"q90__{i}"] = np.full((horizon, n_series), np.nan, dtype=np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
