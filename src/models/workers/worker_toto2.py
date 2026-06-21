"""
Toto 2.0 worker（运行在 external/toto/.venv）
==============================================
读取 request.npz，加载一次 Toto-2.0（默认 313M），批量预测所有任务，
写回 response.npz。协议见同目录 README.md。

与 Toto 1.0 的核心差异
----------------------
- 输出头：Student-T 混合采样 → 固定 9 分位数（0.1~0.9），无 num_samples
- 解码：逐 patch 自回归 → Contiguous Patch Masking（CPM），单次前向
- 输入：dict{"target","target_mask","series_ids"}，不再用 MaskedTimeseries
- 多变量：series_ids 全 0 = 同组（联合建模），各不同 = 独立

设备说明：Apple Silicon 的 MPS 对部分算子不兼容，统一用 CPU。

用法（由 foundation.py 自动调用）：
    python worker_toto2.py  request.npz  response.npz
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── 解决 toto-ts editable install 的 finder hook 劫持问题 ───────────────────
# toto-ts（Toto 1.0）以 editable 模式安装，其 finder hook 会把 toto2 和
# dd_unit_scaling 当作自己的 namespace 子包劫持到不存在的路径。
# 方案：把源码树中正确的包路径插到 sys.path 最前面，并在导入前清理
# sys.modules 中已被劫持的条目。
TOTO_REPO = os.path.join(ROOT, "external", "toto")
TOTO2_PKG = os.path.join(TOTO_REPO, "toto2")
DD_US_PKG = os.path.join(TOTO_REPO, "dd_unit_scaling")

for p in (TOTO2_PKG, DD_US_PKG, TOTO_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

# 清除被 toto-ts finder hook 污染的 namespace 模块缓存
for mod_name in list(sys.modules):
    if mod_name == "toto2" or mod_name.startswith("toto2.") \
       or mod_name == "dd_unit_scaling" or mod_name.startswith("dd_unit_scaling."):
        del sys.modules[mod_name]

import numpy as np
import torch


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])
    # 多变量旋钮（手册 §6 消融 C）：
    #   1 = 多节点联合建模，series_ids 全 0（同组）；
    #   0 = 单变量，每个节点 series_id 各不相同（独立组）。
    multivariate = bool(int(req["multivariate"])) if "multivariate" in req.files else False

    # 模型名称 / 尺寸可通过 request 传入，默认 22M（轻量，CPU 可跑）
    # 可改为 Datadog/Toto-2.0-313m / 1B / 2.5B（需 GPU + 更多显存）
    model_id = str(req["model_id"]) if "model_id" in req.files else "Datadog/Toto-2.0-22m"

    import importlib
    importlib.invalidate_caches()
    from toto2 import Toto2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Toto2Model.from_pretrained(model_id, map_location=device)
    model = model.to(device).eval()

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)        # (T, n_series)
        T, n_series = ctx.shape

        # Toto 2.0 的 patch_size=32，context 长度必须是 32 的倍数。
        # 不足则左侧 zero-pad，并在 mask 中标记 pad 位置为 False。
        patch_size = model.config.patch_size                       # 通常 32
        remainder = T % patch_size
        if remainder != 0:
            pad_len = patch_size - remainder
            ctx_padded = np.concatenate(
                [np.zeros((pad_len, n_series), dtype=np.float32), ctx], axis=0)
        else:
            pad_len = 0
            ctx_padded = ctx

        # Toto 2.0 输入格式：(batch, n_var, time)
        target = torch.from_numpy(ctx_padded.T).unsqueeze(0).to(device)   # (1, n_series, T_padded)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        if pad_len > 0:
            target_mask[:, :, :pad_len] = False                    # pad 位置不参与计算

        # series_ids 控制联合建模：同 id = 同组（共享注意力）
        #   多变量：全 0 → 联合建模
        #   单变量：[0,1,2,...] → 独立
        if multivariate and n_series > 1:
            series_ids = torch.zeros(1, n_series, dtype=torch.long, device=device)
        else:
            series_ids = torch.arange(n_series, dtype=torch.long, device=device).unsqueeze(0)

        inputs = {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

        # Toto 2.0 forecast 返回 (9, batch, n_var, horizon)
        # 分位顺序：[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # 短 horizon（≤512）用 None（单次前向，更快更准），
        # 长 horizon 用 block decoding。
        decode_bs = None if horizon <= 512 else 768
        # 如果有 padding，context 含 missing values，需要启用 missing value mask
        has_mv = pad_len > 0
        quantiles = model.forecast(
            inputs,
            horizon=horizon,
            decode_block_size=decode_bs,
            has_missing_values=has_mv,
        )
        # quantiles: (9, 1, n_series, horizon)
        q_np = quantiles.detach().cpu().numpy()[:, 0, :, :]   # (9, n_series, horizon)

        # 取 q10(idx=0), median(idx=4), q90(idx=8)
        mean = q_np[4]           # (n_series, horizon)  中位数作点预测
        q10  = q_np[0]           # (n_series, horizon)
        q90  = q_np[8]           # (n_series, horizon)

        out[f"mean__{i}"] = mean.T.astype(np.float32)        # (horizon, n_series)
        out[f"q10__{i}"]  = q10.T.astype(np.float32)
        out[f"q90__{i}"]  = q90.T.astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
