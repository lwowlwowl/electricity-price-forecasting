"""
结构消融基础模型适配器 foundation_ablation.py
=============================================
扩展 foundation.py 的子进程 worker 模式，新增 ablation_type 字段传给
结构消融 worker。不修改原 foundation.py 的任何代码。

架构：
- 复用原 foundation.py 的 Task / Forecast / 缓存 / 子进程协议
- 唯一差异：request.npz 多写一个 ablation_type 字段，worker 脚本指向消融版

使用方式：
    from foundation_ablation import (
        Toto2AblationForecaster,
        Chronos2AblationForecaster,
        TimesFMAblationForecaster,
    )
    fc = Toto2AblationForecaster(ablation_type="skip_attention")
    forecasts = fc.predict_batch(context_dfs, future_covs, horizon)
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# ── 路径常量 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(SRC_DIR)

# 添加原 models 目录以导入 base
sys.path.insert(0, os.path.join(SRC_DIR, "models"))
from base import Forecaster, Forecast  # noqa: E402

# 消融 worker 脚本位置
ABLATION_WORKERS_DIR = os.path.join(SCRIPT_DIR, "workers")

# 各模型 venv python
VENV_PYTHON = {
    "timesfm":  os.path.join(ROOT, "external", "timesfm", ".venv", "bin", "python"),
    "chronos2": os.path.join(ROOT, "external", "chronos-forecasting", ".venv", "bin", "python"),
    "toto2":    os.path.join(ROOT, "external", "toto", ".venv", "bin", "python"),
}

# 消融 worker 脚本
ABLATION_WORKER_SCRIPT = {
    "timesfm":  os.path.join(ABLATION_WORKERS_DIR, "worker_timesfm_ablation.py"),
    "chronos2": os.path.join(ABLATION_WORKERS_DIR, "worker_chronos2_ablation.py"),
    "toto2":    os.path.join(ABLATION_WORKERS_DIR, "worker_toto2_ablation.py"),
}


# ── Task 数据类（与 foundation.py 相同结构）──────────────────────────────────
class Task:
    def __init__(self, context, index, series_names, hist_cov=None,
                 future_cov=None, start_ts=0):
        self.context = context
        self.index = index
        self.series_names = series_names
        self.hist_cov = hist_cov
        self.future_cov = future_cov
        self.start_ts = start_ts


# ── 子进程调用 ──────────────────────────────────────────────────────────────
def _infer_interval_seconds(index: pd.DatetimeIndex) -> int:
    if len(index) >= 2:
        return int((index[-1] - index[-2]).total_seconds())
    return 3600


def _run_ablation_worker(kind: str, tasks: List[Task], horizon: int,
                         ablation_type: str,
                         num_samples: int = 200,
                         multivariate: bool = False) -> List[Dict[str, np.ndarray]]:
    """
    调用消融 worker 子进程。与 foundation.py 的 _run_worker 逻辑相同，
    唯一区别是 request.npz 多了 ablation_type 字段，且 worker 脚本指向消融版。
    """
    py = VENV_PYTHON[kind]
    script = ABLATION_WORKER_SCRIPT[kind]
    if not os.path.exists(py):
        raise FileNotFoundError(f"{kind} 的 venv python 不存在：{py}")
    if not os.path.exists(script):
        raise FileNotFoundError(f"{kind} 的消融 worker 不存在：{script}")

    interval = _infer_interval_seconds(tasks[0].index) if tasks else 3600

    payload: Dict[str, object] = {
        "n_tasks": np.int64(len(tasks)),
        "horizon": np.int64(horizon),
        "interval_seconds": np.int64(interval),
        "num_samples": np.int64(num_samples),
        "multivariate": np.int64(1 if multivariate else 0),
        "ablation_type": np.array(ablation_type, dtype=object),
    }
    for i, t in enumerate(tasks):
        payload[f"context__{i}"] = t.context.astype(np.float32)
        payload[f"start_ts__{i}"] = np.int64(t.start_ts)
        if t.hist_cov is not None and t.future_cov is not None:
            payload[f"hist_cov__{i}"] = t.hist_cov.astype(np.float32)
            payload[f"future_cov__{i}"] = t.future_cov.astype(np.float32)

    workdir = tempfile.mkdtemp(prefix=f"sa_{kind}_")
    req_path = os.path.join(workdir, "request.npz")
    resp_path = os.path.join(workdir, "response.npz")
    np.savez(req_path, **payload)

    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    proc = subprocess.run(
        [py, script, req_path, resp_path],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0 or not os.path.exists(resp_path):
        raise RuntimeError(
            f"[{kind} ablation worker 失败] returncode={proc.returncode}\n"
            f"ablation_type={ablation_type}\n"
            f"--- STDOUT ---\n{proc.stdout[-2000:]}\n"
            f"--- STDERR ---\n{proc.stderr[-3000:]}"
        )

    resp = np.load(resp_path, allow_pickle=True)
    if int(resp.get("ok", 0)) != 1:
        raise RuntimeError(f"[{kind} ablation worker] response.ok != 1")

    results = []
    for i in range(len(tasks)):
        results.append({
            "mean": np.asarray(resp[f"mean__{i}"], dtype=float),
            "q10":  np.asarray(resp[f"q10__{i}"], dtype=float),
            "q90":  np.asarray(resp[f"q90__{i}"], dtype=float),
        })
    return results


# ── 结构消融适配器基类 ───────────────────────────────────────────────────────
class AblationForecaster(Forecaster):
    """
    结构消融版基础模型适配器。与 FoundationForecaster 接口一致，
    但调用消融 worker 而非原 worker。
    """

    kind: str = ""
    ablation_type: str = ""
    num_samples: int = 200

    def __init__(self, ablation_type: str):
        self.ablation_type = ablation_type
        self._cache: Dict[str, List[Forecast]] = {}
        self.multivariate_used = False

    # 目标列前缀
    TARGET_PREFIX = "price__"

    def _target_columns(self, df: pd.DataFrame) -> List[str]:
        cols = [c for c in df.columns if c.startswith(self.TARGET_PREFIX)]
        if not cols:
            raise ValueError(f"找不到目标列（应以 {self.TARGET_PREFIX!r} 开头）")
        return cols

    def _future_index(self, context_df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
        idx = context_df.index
        step = idx[-1] - idx[-2] if len(idx) >= 2 else pd.Timedelta(hours=1)
        return pd.date_range(start=idx[-1] + step, periods=horizon, freq=step)

    def _make_task(self, context_df: pd.DataFrame,
                   future_covariates: Optional[pd.DataFrame],
                   horizon: int) -> Task:
        cols = self._target_columns(context_df)
        context = context_df[cols].to_numpy(dtype=float)
        fut_index = self._future_index(context_df, horizon)
        hist_cov = future_cov = None
        if self.supports_covariates and future_covariates is not None \
                and future_covariates.shape[1] > 0:
            cov_cols = list(future_covariates.columns)
            if all(c in context_df.columns for c in cov_cols):
                hist_cov = context_df[cov_cols].to_numpy(dtype=float)
                future_cov = future_covariates[cov_cols].to_numpy(dtype=float)
        start_ts = int(context_df.index[0].timestamp())
        return Task(context=context, index=fut_index, series_names=cols,
                    hist_cov=hist_cov, future_cov=future_cov, start_ts=start_ts)

    @staticmethod
    def _pack(raw: Dict[str, np.ndarray], task: Task) -> Forecast:
        mean, q10, q90 = raw["mean"], raw["q10"], raw["q90"]
        q50 = mean.copy()
        if mean.shape[1] == 1:
            return Forecast(mean[:, 0], q10[:, 0], q50[:, 0], q90[:, 0],
                            index=task.index, series_names=task.series_names)
        return Forecast(mean, q10, q50, q90,
                        index=task.index, series_names=task.series_names)

    def _cache_key(self, tasks: List[Task], horizon: int) -> str:
        h = hashlib.sha1()
        h.update(str(horizon).encode())
        h.update(self.ablation_type.encode())
        for t in tasks:
            h.update(t.context.tobytes())
            if t.hist_cov is not None:
                h.update(t.hist_cov.tobytes())
            if t.future_cov is not None:
                h.update(t.future_cov.tobytes())
        return h.hexdigest()

    def predict_tasks(self, tasks: List[Task], horizon: int,
                      multivariate: bool = False) -> List[Forecast]:
        if not tasks:
            return []
        mv = bool(multivariate and self.supports_multivariate)
        self.multivariate_used = bool(mv and tasks[0].context.shape[1] > 1)
        key = self._cache_key(tasks, horizon) + ("_mv" if mv else "_uv")
        if key in self._cache:
            return self._cache[key]
        raw_list = _run_ablation_worker(
            self.kind, tasks, horizon,
            ablation_type=self.ablation_type,
            num_samples=self.num_samples,
            multivariate=mv)
        forecasts = [self._pack(raw, t) for raw, t in zip(raw_list, tasks)]
        self._cache[key] = forecasts
        return forecasts

    def predict_batch(self, context_dfs: List[pd.DataFrame],
                      future_covs: Optional[List[Optional[pd.DataFrame]]],
                      horizon: int,
                      multivariate: bool = False) -> List[Forecast]:
        if future_covs is None:
            future_covs = [None] * len(context_dfs)
        tasks = [self._make_task(c, fc, horizon)
                 for c, fc in zip(context_dfs, future_covs)]
        return self.predict_tasks(tasks, horizon, multivariate=multivariate)

    def predict(self, context_df, future_covariates=None, horizon=24,
                multivariate: bool = False) -> Forecast:
        task = self._make_task(context_df, future_covariates, horizon)
        return self.predict_tasks([task], horizon, multivariate=multivariate)[0]


# ── 三个具体消融适配器 ───────────────────────────────────────────────────────
class Toto2AblationForecaster(AblationForecaster):
    """Toto-2.0 结构消融版。"""
    kind = "toto2"
    needs_training = False
    supports_covariates = False
    supports_multivariate = True

    def __init__(self, ablation_type: str):
        super().__init__(ablation_type)
        self.name = f"Toto2[{ablation_type}]"


class Chronos2AblationForecaster(AblationForecaster):
    """Chronos-2 结构消融版。"""
    kind = "chronos2"
    needs_training = False
    supports_covariates = True
    supports_multivariate = True

    def __init__(self, ablation_type: str):
        super().__init__(ablation_type)
        self.name = f"Chronos2[{ablation_type}]"


class TimesFMAblationForecaster(AblationForecaster):
    """TimesFM 结构消融版。"""
    kind = "timesfm"
    needs_training = False
    supports_covariates = False
    supports_multivariate = False

    def __init__(self, ablation_type: str):
        super().__init__(ablation_type)
        self.name = f"TimesFM[{ablation_type}]"


# ── 工厂函数 ────────────────────────────────────────────────────────────────
ABLATION_MODEL_REGISTRY = {
    "toto2":    Toto2AblationForecaster,
    "chronos2": Chronos2AblationForecaster,
    "timesfm":  TimesFMAblationForecaster,
}


def build_ablation_forecaster(model_type: str, ablation_type: str) -> AblationForecaster:
    """按模型类型和消融类型构造一个消融适配器。"""
    if model_type not in ABLATION_MODEL_REGISTRY:
        raise ValueError(f"未知模型 {model_type!r}。可用：{list(ABLATION_MODEL_REGISTRY)}")
    return ABLATION_MODEL_REGISTRY[model_type](ablation_type)
