"""
Toto worker（运行在 external/toto/.venv）
==========================================
读取 request.npz，加载一次 Toto-1.0，批量预测所有任务，写回 response.npz。
协议见同目录 README.md。

Toto 的强项是【多节点联合建模】：一次把多列序列一起喂进去，模型能利用
节点间的空间相关性。本 worker 把每个任务的 n_series 列作为一个 batch
（沿变量维度）一起预测。

设备说明：Apple Silicon 的 MPS 对 Toto 部分算子不兼容，统一用 CPU。

用法（由 foundation.py 自动调用）：
    python worker_toto.py  request.npz  response.npz
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# toto 包以源码树形式存在于 external/toto/toto，未 pip 安装；把仓库根加入路径
TOTO_REPO = os.path.join(ROOT, "external", "toto")
if TOTO_REPO not in sys.path:
    sys.path.insert(0, TOTO_REPO)

import numpy as np
import torch


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])
    interval = int(req["interval_seconds"]) if "interval_seconds" in req.files else 3600
    num_samples = int(req["num_samples"]) if "num_samples" in req.files else 200
    # 多变量旋钮（手册 §6 消融 C）：
    #   1 = 多节点联合建模，id_mask 全 0（同组），让模型利用空间相关性（Toto 强项）；
    #   0 = 单变量，每个节点 id_mask 各不相同（独立组），模型不跨节点共享信息。
    multivariate = bool(int(req["multivariate"])) if "multivariate" in req.files else False

    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

    device = "cuda" if torch.cuda.is_available() else "cpu"
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(device)
    toto.eval()
    forecaster = TotoForecaster(toto.model)

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)        # (T, n_series)
        T, n_series = ctx.shape
        series = torch.from_numpy(ctx.T).to(device)          # (n_series, T)

        start_ts = int(req[f"start_ts__{i}"]) if f"start_ts__{i}" in req.files else 0
        timestamps = np.array(
            [start_ts + k * interval for k in range(T)], dtype=np.int64)
        ts = torch.from_numpy(timestamps).unsqueeze(0).expand(n_series, -1).to(device)
        ivl = torch.full((n_series,), interval, dtype=torch.int64).to(device)

        # id_mask 决定节点是否“同组”：同组才会跨节点共享注意力（联合建模）。
        #   多变量：全 0 → 所有节点同组，利用空间相关性（Toto 强项）。
        #   单变量：每行不同 id → 节点彼此独立，等价于逐序列预测。
        if multivariate and n_series > 1:
            id_mask = torch.zeros_like(series, dtype=torch.int64)
        else:
            ids = torch.arange(n_series, dtype=torch.int64).unsqueeze(1)
            id_mask = ids.expand(n_series, T).to(device)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),
            id_mask=id_mask,
            timestamp_seconds=ts,
            time_interval_seconds=ivl,
        )

        forecast = forecaster.forecast(
            inputs=inputs,
            prediction_length=horizon,
            num_samples=num_samples,
            use_kv_cache=True,
        )
        # 本版本 Toto 输出布局：
        #   forecast.samples : (batch=1, n_series, horizon, num_samples)  采样在最后一维
        #   forecast.mean    : (batch=1, n_series, horizon)               已给点预测
        samples = forecast.samples.detach().cpu().numpy()[0]   # (n_series, horizon, num_samples)
        if getattr(forecast, "mean", None) is not None:
            mean = forecast.mean.detach().cpu().numpy()[0]     # (n_series, horizon)
        else:
            mean = samples.mean(axis=-1)
        q10 = np.percentile(samples, 10, axis=-1)              # (n_series, horizon)
        q90 = np.percentile(samples, 90, axis=-1)

        out[f"mean__{i}"] = mean.T.astype(np.float32)        # (horizon, n_series)
        out[f"q10__{i}"] = q10.T.astype(np.float32)
        out[f"q90__{i}"] = q90.T.astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
