"""
结构消融实验模块 (Structural Ablation)
======================================
独立于现有参数消融代码的新模块，实现 v2.1 手册中的结构层消融实验。

模块结构：
  ablations.py             - 14 种消融操作的核心实现（需 PyTorch，仅 worker 内使用）
  foundation_ablation.py   - 适配器（AblationForecaster），调用消融 worker
  run_structural_ablation.py - 实验入口（读 YAML → 回测 → 出结果）
  workers/
    worker_toto2_ablation.py    - Toto-2.0 消融 worker
    worker_chronos2_ablation.py - Chronos-2 消融 worker
    worker_timesfm_ablation.py  - TimesFM-2.5 消融 worker

使用方式：
    python src/structural_ablation/run_structural_ablation.py \\
        configs/structural_ablation/smoke_toto2.yaml

注意：ablations.py 依赖 PyTorch，仅在 worker 子进程的 venv 中可用。
主进程只需使用 foundation_ablation.py 中的适配器类。
"""

# foundation_ablation.py 不依赖 torch，可安全导入
from .foundation_ablation import (
    AblationForecaster,
    Toto2AblationForecaster,
    Chronos2AblationForecaster,
    TimesFMAblationForecaster,
    build_ablation_forecaster,
)

# ablations.py 需要 torch，容错导入
try:
    from .ablations import apply_ablation, ABLATION_REGISTRY, ABLATION_APPLICABILITY
except ImportError:
    # torch 未安装时（主进程环境），仅提供 foundation 层接口
    apply_ablation = None
    ABLATION_REGISTRY = None
    ABLATION_APPLICABILITY = None

__all__ = [
    "apply_ablation",
    "ABLATION_REGISTRY",
    "ABLATION_APPLICABILITY",
    "AblationForecaster",
    "Toto2AblationForecaster",
    "Chronos2AblationForecaster",
    "TimesFMAblationForecaster",
    "build_ablation_forecaster",
]
