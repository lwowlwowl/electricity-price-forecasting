"""
TimesFM 结构消融 worker
=========================
在 worker_timesfm.py 基础上增加消融钩子。

用法（由 foundation_ablation.py 调用）：
    python worker_timesfm_ablation.py  request.npz  response.npz
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

import numpy as np


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])

    # 读取消融配置
    ablation_type = str(req["ablation_type"]) if "ablation_type" in req.files else None

    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")

    # ★ 应用结构消融必须在 compile() 之前执行。
    # compile() 会根据当前模型结构（特别是 num_heads）预分配 decode_cache；
    # 如果先 compile 再 halve_heads，cache 的 head 维度与缩小后的模型不一致，
    # 会在 forward 的 cache 写入处报 RuntimeError。
    if ablation_type:
        from ablations import apply_ablation
        # TimesFM 的实际 nn.Module 在 model.model 属性中
        internal_model = model.model
        apply_ablation(internal_model, "timesfm", ablation_type)

    model.compile(timesfm.ForecastConfig(
        max_context=2048,
        max_horizon=max(horizon, 64),
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        infer_is_positive=False,
    ))

    # 检查是否为 point_only 模式
    point_only = getattr(model.model, "_ablation_point_only", False) if hasattr(model, "model") else False

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)
        n_series = ctx.shape[1]
        inputs = [ctx[:, j] for j in range(n_series)]

        try:
            point, quant = model.forecast(horizon=horizon, inputs=inputs)
            point = np.asarray(point)
            quant = np.asarray(quant)
            mean = point.T  # (horizon, n_series)

            if point_only:
                # 只用点预测，分位数设为与点预测相同（无不确定性信息）
                q10 = mean.copy()
                q90 = mean.copy()
            else:
                q10 = quant[:, :, 1].T
                q90 = quant[:, :, 9].T

        except Exception as e:
            print(f"⚠️  TimesFM 消融后推理失败 (task {i}): {e}", file=sys.stderr)
            mean = np.full((horizon, n_series), np.nan, dtype=np.float32)
            q10 = mean.copy()
            q90 = mean.copy()

        out[f"mean__{i}"] = mean.astype(np.float32)
        out[f"q10__{i}"] = q10.astype(np.float32)
        out[f"q90__{i}"] = q90.astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
