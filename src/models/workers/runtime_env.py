"""Project-local runtime environment for foundation-model workers."""

from __future__ import annotations

import os


def configure_runtime_environment(project_root: str, worker_name: str) -> None:
    """Keep model caches and temporary files inside the project directory."""
    project_root = os.path.abspath(project_root)
    hf_home = os.path.join(project_root, "hf_cache")
    runtime_dir = os.path.join(
        project_root, "runtime_cache", "foundation_workers", worker_name
    )
    cache_dir = os.path.join(runtime_dir, "cache")

    for path in (hf_home, runtime_dir, cache_dir):
        os.makedirs(path, exist_ok=True)

    os.environ.update({
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": os.path.join(hf_home, "hub"),
        "HF_XET_CACHE": os.path.join(hf_home, "xet"),
        "HF_ASSETS_CACHE": os.path.join(hf_home, "assets"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": cache_dir,
        "TORCH_HOME": os.path.join(cache_dir, "torch"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_dir, "torchinductor"),
        "TRITON_CACHE_DIR": os.path.join(cache_dir, "triton"),
        "MPLCONFIGDIR": os.path.join(cache_dir, "matplotlib"),
        "TEMP": runtime_dir,
        "TMP": runtime_dir,
    })
