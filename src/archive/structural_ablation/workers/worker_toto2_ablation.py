"""
Toto 2.0 结构消融 worker
=========================
在 worker_toto2.py 基础上增加消融钩子：从 request.npz 读取 ablation_type，
加载模型后应用消融操作，然后执行正常预测流程。

不修改原 worker_toto2.py，完全独立文件。

用法（由 foundation_ablation.py 调用）：
    python worker_toto2_ablation.py  request.npz  response.npz
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# structural_ablation/workers/ → structural_ablation/ → src/ → ROOT
SA_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(SA_DIR)
ROOT = os.path.dirname(SRC_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── 解决 toto-ts editable install 的 finder hook 劫持问题 ───────────────────
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

# 把 ablations.py 所在目录加入 path
sys.path.insert(0, SA_DIR)

import numpy as np
import torch


def main(req_path: str, resp_path: str):
    req = np.load(req_path, allow_pickle=True)
    n_tasks = int(req["n_tasks"])
    horizon = int(req["horizon"])
    multivariate = bool(int(req["multivariate"])) if "multivariate" in req.files else False

    # 读取消融配置
    ablation_type = str(req["ablation_type"]) if "ablation_type" in req.files else None

    model_id = str(req["model_id"]) if "model_id" in req.files else "Datadog/Toto-2.0-22m"

    import importlib
    importlib.invalidate_caches()
    from toto2 import Toto2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Toto2Model.from_pretrained(model_id, map_location=device)
    model = model.to(device).eval()

    # ★ 应用结构消融
    if ablation_type:
        from ablations import apply_ablation
        model = apply_ablation(model, "toto2", ablation_type)
        model.eval()

    out = {}
    for i in range(n_tasks):
        ctx = req[f"context__{i}"].astype(np.float32)
        T, n_series = ctx.shape

        patch_size = model.config.patch_size
        remainder = T % patch_size
        if remainder != 0:
            pad_len = patch_size - remainder
            ctx_padded = np.concatenate(
                [np.zeros((pad_len, n_series), dtype=np.float32), ctx], axis=0)
        else:
            pad_len = 0
            ctx_padded = ctx

        target = torch.from_numpy(ctx_padded.T).unsqueeze(0).to(device)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        if pad_len > 0:
            target_mask[:, :, :pad_len] = False

        if multivariate and n_series > 1:
            series_ids = torch.zeros(1, n_series, dtype=torch.long, device=device)
        else:
            series_ids = torch.arange(n_series, dtype=torch.long, device=device).unsqueeze(0)

        inputs = {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

        decode_bs = None if horizon <= 512 else 768
        has_mv = pad_len > 0

        try:
            quantiles = model.forecast(
                inputs,
                horizon=horizon,
                decode_block_size=decode_bs,
                has_missing_values=has_mv,
            )
            q_np = quantiles.detach().cpu().numpy()[:, 0, :, :]
            mean = q_np[4]
            q10 = q_np[0]
            q90 = q_np[8]
        except Exception as e:
            # 消融可能导致数值爆炸，记录 NaN 作为结果
            print(f"⚠️  Toto-2.0 消融后推理失败 (task {i}): {e}", file=sys.stderr)
            mean = np.full((n_series, horizon), np.nan, dtype=np.float32)
            q10 = mean.copy()
            q90 = mean.copy()

        out[f"mean__{i}"] = mean.T.astype(np.float32)
        out[f"q10__{i}"] = q10.T.astype(np.float32)
        out[f"q90__{i}"] = q90.T.astype(np.float32)

    out["ok"] = np.int64(1)
    np.savez(resp_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
