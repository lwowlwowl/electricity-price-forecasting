"""
TimesFM-3 worker（运行在 external/timesfm/.venv）
================================================
读取主框架打包的 request.npz，加载一次官方 frozen TimesFM-3 权重，批量预测
所有任务，再把结果写回 response.npz。协议见同目录 README.md。

TimesFM-3 原生支持多变量目标和外生协变量。本 worker 服从 request 中的
``multivariate`` 开关：关闭时每个节点独立预测；开启时同一任务的全部节点
联合预测。两种模式均只使用起报时刻已知的信息，并允许负电价输出。

用法（由 foundation.py 自动调用）：
    python worker_timesfm.py request.npz response.npz
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

# 指向项目内权重缓存；正式回测禁止 worker 临时联网。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
from runtime_env import configure_runtime_environment

configure_runtime_environment(ROOT, "timesfm")


MODEL_ID = "google/timesfm-3.0-pytorch"


def _past_future_covariates(req: Any, task_idx: int) -> np.ndarray | None:
    """Return TimesFM-3 past+future covariates as (n_cov, context+horizon)."""
    hist_key = f"hist_cov__{task_idx}"
    future_key = f"future_cov__{task_idx}"
    if hist_key not in req.files or future_key not in req.files:
        return None
    hist = np.asarray(req[hist_key], dtype=np.float32)
    future = np.asarray(req[future_key], dtype=np.float32)
    if hist.ndim != 2 or future.ndim != 2 or hist.shape[1] != future.shape[1]:
        raise ValueError(
            f"task {task_idx} covariate shapes incompatible: "
            f"hist={hist.shape}, future={future.shape}"
        )
    return np.concatenate([hist, future], axis=0).T


def _as_worker_arrays(output: Any, n_series: int, horizon: int) -> tuple[np.ndarray, ...]:
    """Convert one TimesFM-3 output to worker protocol shape (horizon, n_series)."""
    forecast = np.asarray(output.forecast, dtype=np.float32)
    quantiles = np.asarray(output.quantiles, dtype=np.float32)

    if n_series == 1:
        forecast = forecast.reshape(horizon, 1)
        quantiles = quantiles.reshape(horizon, -1)[:, None, :]
    else:
        forecast = forecast.reshape(n_series, horizon).T
        quantiles = quantiles.reshape(n_series, horizon, -1).transpose(1, 0, 2)

    # 官方 TimesFM-3 输出 0.1, ..., 0.9 九个分位数。
    if quantiles.shape[-1] < 9:
        raise ValueError(f"TimesFM-3 quantile dimension should be 9, got {quantiles.shape}")
    return forecast, quantiles[..., 0], quantiles[..., 8]


def main(req_path: str, resp_path: str) -> None:
    with np.load(req_path, allow_pickle=True) as req:
        n_tasks = int(req["n_tasks"])
        horizon = int(req["horizon"])
        multivariate = bool(int(req.get("multivariate", 0)))

        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_batch_size = "16" if device == "cuda" else "4"
        batch_size = int(os.environ.get("TIMESFM3_BATCH_SIZE", default_batch_size))
        model = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=MODEL_ID,
                per_core_batch_size=batch_size,
                device=device,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
                local_files_only=True,
            )
        )

        out: dict[str, np.ndarray] = {}
        if multivariate:
            contexts = []
            covariates = []
            series_counts = []
            for i in range(n_tasks):
                ctx = np.asarray(req[f"context__{i}"], dtype=np.float32)
                contexts.append(ctx.T)  # TimesFM-3: (n_series, context)
                covariates.append(_past_future_covariates(req, i))
                series_counts.append(ctx.shape[1])

            outputs = list(
                model.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    past_future_covariates=covariates,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                    make_positive=False,  # 电价允许为负
                    sort_quantiles=True,
                )
            )
            for i, (prediction, n_series) in enumerate(zip(outputs, series_counts)):
                mean, q10, q90 = _as_worker_arrays(prediction, n_series, horizon)
                out[f"mean__{i}"] = mean
                out[f"q10__{i}"] = q10
                out[f"q90__{i}"] = q90
        else:
            # 节点独立模式：把每个 task 的每列拆为一个 1D 样本，一次批量推理，
            # 再按 task 还原。相同的合法协变量复制给该 task 的各节点。
            flat_contexts: list[np.ndarray] = []
            flat_covariates: list[np.ndarray | None] = []
            task_slices: list[tuple[int, int]] = []
            for i in range(n_tasks):
                ctx = np.asarray(req[f"context__{i}"], dtype=np.float32)
                cov = _past_future_covariates(req, i)
                start = len(flat_contexts)
                for j in range(ctx.shape[1]):
                    flat_contexts.append(ctx[:, j])
                    flat_covariates.append(cov)
                task_slices.append((start, len(flat_contexts)))

            flat_outputs = list(
                model.predict_batch(
                    contexts=flat_contexts,
                    horizon=horizon,
                    past_future_covariates=flat_covariates,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                    make_positive=False,
                    sort_quantiles=True,
                )
            )
            for i, (start, stop) in enumerate(task_slices):
                means, q10s, q90s = [], [], []
                for prediction in flat_outputs[start:stop]:
                    mean, q10, q90 = _as_worker_arrays(prediction, 1, horizon)
                    means.append(mean[:, 0])
                    q10s.append(q10[:, 0])
                    q90s.append(q90[:, 0])
                out[f"mean__{i}"] = np.stack(means, axis=1)
                out[f"q10__{i}"] = np.stack(q10s, axis=1)
                out[f"q90__{i}"] = np.stack(q90s, axis=1)

    out["ok"] = np.int64(1)
    out["model_id"] = np.asarray(MODEL_ID)
    out["model_version"] = np.asarray("3.0")
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
