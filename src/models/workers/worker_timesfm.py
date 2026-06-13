"""
TimesFM worker（运行在 external/timesfm/.venv）
================================================
读取主框架打包的 request.npz，加载一次 TimesFM-2.5，批量预测所有任务，
把结果写回 response.npz。协议见同目录 README.md。

TimesFM 能力：单变量为主，逐序列预测；协变量作为消融项（本 worker 暂按
单变量处理，协变量在 Chronos/Toto 上更原生，TimesFM 的协变量接口后续可扩展）。

用法（由 foundation.py 自动调用）：
    python worker_timesfm.py  request.npz  response.npz
"""

import os
import sys

# 指向本地权重缓存，禁止联网
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])

    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=2048,
        max_horizon=max(horizon, 64),
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        infer_is_positive=False,     # 电价可能为负
    ))

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)   # (T, n_series)
        n_series = ctx.shape[1]
        # TimesFM 输入：list of 1D array，每列一条序列
        inputs = [ctx[:, j] for j in range(n_series)]
        point, quant = model.forecast(horizon=horizon, inputs=inputs)
        # point: (n_series, horizon)  quant: (n_series, horizon, 10)
        point = np.asarray(point)
        quant = np.asarray(quant)
        mean = point.T                                  # (horizon, n_series)
        q10 = quant[:, :, 1].T
        q90 = quant[:, :, 9].T
        out[f"mean__{i}"] = mean.astype(np.float32)
        out[f"q10__{i}"] = q10.astype(np.float32)
        out[f"q90__{i}"] = q90.astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
