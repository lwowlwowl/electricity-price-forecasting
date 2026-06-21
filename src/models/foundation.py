"""
基础模型适配器 foundation.py
=============================
把四个时序基础模型（TimesFM-2.5 / Chronos-2 / Toto-1.0 / Toto-2.0）统一封装成 Forecaster，
使其能与基线在同一回测里公平对比。

★ 核心难点：三个模型装在各自独立 venv 里，依赖互相冲突，**不能在同一个
  Python 进程里 import**。因此本文件不直接 import 任何模型，而是通过
  「子进程 worker」模式调用——把预测任务打包成 request.npz，启动模型自己
  venv 里的 worker 脚本，读回 response.npz。协议见 workers/README.md。

★ 批量调度：逐个起报点起子进程会重复加载模型（每次几秒~几十秒），不可接受。
  所以本适配器提供 `predict_batch(tasks)`：一次把全部任务打包给 worker，
  worker 只加载一次模型跑完所有任务。回测引擎优先走这条批量路径。

  为兼容旧的「逐 origin 调用 predict」，本适配器还做了【结果缓存】：
  同一组 tasks 第一次 predict_batch 后缓存，run_backtest 与 _spike_f1
  阶段重复请求时直接命中缓存，不重复起子进程。单次 predict() 退化为 1 任务批。
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from base import Forecaster, Forecast


# ── 路径常量：worker 脚本 与 各模型 venv 的 python ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                 # src/models
WORKERS_DIR = os.path.join(SCRIPT_DIR, "workers")
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))            # 项目根

# 各基础模型 venv 的解释器（Stage 0 已确认存在）
VENV_PYTHON = {
    "timesfm":  os.path.join(ROOT, "external", "timesfm", ".venv", "bin", "python"),
    "chronos2": os.path.join(ROOT, "external", "chronos-forecasting", ".venv", "bin", "python"),
    "toto":     os.path.join(ROOT, "external", "toto", ".venv", "bin", "python"),
    "toto2":    os.path.join(ROOT, "external", "toto", ".venv", "bin", "python"),  # 共用 toto venv
}
WORKER_SCRIPT = {
    "timesfm":  os.path.join(WORKERS_DIR, "worker_timesfm.py"),
    "chronos2": os.path.join(WORKERS_DIR, "worker_chronos2.py"),
    "toto":     os.path.join(WORKERS_DIR, "worker_toto.py"),
    "toto2":    os.path.join(WORKERS_DIR, "worker_toto2.py"),
}


# ── 一个预测任务（一个起报点）────────────────────────────────────────────────
@dataclass
class Task:
    """
    回测里的一个起报点对应一个 Task。
      context     : (T, n_series) 目标历史（列=节点）
      hist_cov    : (T, n_cov)    历史协变量，无则 None
      future_cov  : (horizon, n_cov) 未来协变量，无则 None
      index       : 该任务预测窗口的时间戳（用于回填 Forecast.index）
      series_names: 各列节点名
      start_ts    : context 首点 unix 秒（Toto 需要）
    """
    context: np.ndarray
    index: pd.DatetimeIndex
    series_names: List[str]
    hist_cov: Optional[np.ndarray] = None
    future_cov: Optional[np.ndarray] = None
    start_ts: int = 0


# ── 子进程调用：写 request → 跑 worker → 读 response ──────────────────────────
def _infer_interval_seconds(index: pd.DatetimeIndex) -> int:
    if len(index) >= 2:
        return int((index[-1] - index[-2]).total_seconds())
    return 3600


def _run_worker(kind: str, tasks: List[Task], horizon: int,
                num_samples: int = 200,
                multivariate: bool = False) -> List[Dict[str, np.ndarray]]:
    """
    把 tasks 打包成 request.npz，调用 kind 对应 venv 的 worker，读回结果。
    返回 list（与 tasks 等长），每项 {"mean","q10","q90"} 形状 (horizon, n_series)。

    multivariate : 是否要求多节点联合建模。作为 request 的全局标志传给 worker，
                   worker 据此在“逐列独立”与“联合”之间切换。
    """
    py = VENV_PYTHON[kind]
    script = WORKER_SCRIPT[kind]
    if not os.path.exists(py):
        raise FileNotFoundError(f"{kind} 的 venv python 不存在：{py}")
    if not os.path.exists(script):
        raise FileNotFoundError(f"{kind} 的 worker 脚本不存在：{script}")

    interval = _infer_interval_seconds(tasks[0].index) if tasks else 3600

    payload: Dict[str, np.ndarray] = {
        "n_tasks": np.int64(len(tasks)),
        "horizon": np.int64(horizon),
        "interval_seconds": np.int64(interval),
        "num_samples": np.int64(num_samples),
        "multivariate": np.int64(1 if multivariate else 0),
    }
    for i, t in enumerate(tasks):
        payload[f"context__{i}"] = t.context.astype(np.float32)
        payload[f"start_ts__{i}"] = np.int64(t.start_ts)
        if t.hist_cov is not None and t.future_cov is not None:
            payload[f"hist_cov__{i}"] = t.hist_cov.astype(np.float32)
            payload[f"future_cov__{i}"] = t.future_cov.astype(np.float32)

    workdir = tempfile.mkdtemp(prefix=f"fm_{kind}_")
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
            f"[{kind} worker 失败] returncode={proc.returncode}\n"
            f"--- STDOUT ---\n{proc.stdout[-2000:]}\n"
            f"--- STDERR ---\n{proc.stderr[-3000:]}"
        )

    resp = np.load(resp_path, allow_pickle=True)
    if int(resp.get("ok", 0)) != 1:
        raise RuntimeError(f"[{kind} worker] response.ok != 1")

    results = []
    for i in range(len(tasks)):
        results.append({
            "mean": np.asarray(resp[f"mean__{i}"], dtype=float),
            "q10":  np.asarray(resp[f"q10__{i}"], dtype=float),
            "q90":  np.asarray(resp[f"q90__{i}"], dtype=float),
        })
    return results


# ── 基础模型适配器基类 ───────────────────────────────────────────────────────
class FoundationForecaster(Forecaster):
    """
    所有基础模型适配器的父类。子类只需指定 `kind`（worker 类型）与能力开关。

    对外提供两条预测路径：
      - predict(context_df, future_cov, horizon)  : 单任务（兼容旧接口），内部退化为 1 任务批
      - predict_batch(context_dfs, future_covs, horizon) : 真·批量，回测引擎优先用

    结果缓存：以「全部任务的内容哈希」为 key 缓存上一次批量结果。回测里
    run_backtest 和 _spike_f1_for_model 会用相同 tasks 各请求一次，缓存可避免
    重复起子进程（模型重复加载）。
    """

    kind: str = ""              # "timesfm" / "chronos2" / "toto"
    num_samples: int = 200      # 仅 Toto 用

    def __init__(self):
        self._cache: Dict[str, List[Forecast]] = {}
        # 最近一次预测是否真正走了多变量联合建模（仅供调试/参考；
        # 结果表里的 multivariate_used 以 backtest 的记录为准）。
        self.multivariate_used: bool = False

    # ── 把 (context_df, future_cov) 转成内部 Task ─────────────────────────────
    def _make_task(self, context_df: pd.DataFrame,
                   future_covariates: Optional[pd.DataFrame],
                   horizon: int) -> Task:
        cols = self._target_columns(context_df)
        context = context_df[cols].to_numpy(dtype=float)        # (T, n_series)
        fut_index = self._future_index(context_df, horizon)

        hist_cov = future_cov = None
        if self.supports_covariates and future_covariates is not None \
                and future_covariates.shape[1] > 0:
            cov_cols = list(future_covariates.columns)
            # 历史协变量取 context_df 中的同名列（若有）
            if all(c in context_df.columns for c in cov_cols):
                hist_cov = context_df[cov_cols].to_numpy(dtype=float)
                future_cov = future_covariates[cov_cols].to_numpy(dtype=float)

        start_ts = int(context_df.index[0].timestamp())
        return Task(context=context, index=fut_index, series_names=cols,
                    hist_cov=hist_cov, future_cov=future_cov, start_ts=start_ts)

    # ── 把 worker 原始输出包成 Forecast（单列压一维，多列保二维）──────────────
    @staticmethod
    def _pack(raw: Dict[str, np.ndarray], task: Task) -> Forecast:
        mean, q10, q90 = raw["mean"], raw["q10"], raw["q90"]   # (horizon, n_series)
        q50 = mean.copy()
        if mean.shape[1] == 1:
            return Forecast(mean[:, 0], q10[:, 0], q50[:, 0], q90[:, 0],
                            index=task.index, series_names=task.series_names)
        return Forecast(mean, q10, q50, q90,
                        index=task.index, series_names=task.series_names)

    @staticmethod
    def _cache_key(tasks: List[Task], horizon: int) -> str:
        h = hashlib.sha1()
        h.update(str(horizon).encode())
        for t in tasks:
            h.update(t.context.tobytes())
            if t.hist_cov is not None:
                h.update(t.hist_cov.tobytes())
            if t.future_cov is not None:
                h.update(t.future_cov.tobytes())
        return h.hexdigest()

    # ── 真·批量预测：一次子进程跑完所有任务 ──────────────────────────────────
    def predict_tasks(self, tasks: List[Task], horizon: int,
                      multivariate: bool = False) -> List[Forecast]:
        if not tasks:
            return []
        # 能力诚实：只有本模型原生支持多变量、且调用方要求多变量时，才真正
        # 传 multivariate=True 给 worker（联合建模）；否则降级为逐列单变量。
        mv = bool(multivariate and self.supports_multivariate)
        self.multivariate_used = bool(mv and tasks[0].context.shape[1] > 1)
        # 缓存键带上单/多变量后缀，避免两种模式互相串味。
        key = self._cache_key(tasks, horizon) + ("_mv" if mv else "_uv")
        if key in self._cache:
            return self._cache[key]
        raw_list = _run_worker(self.kind, tasks, horizon,
                               num_samples=self.num_samples, multivariate=mv)
        forecasts = [self._pack(raw, t) for raw, t in zip(raw_list, tasks)]
        self._cache[key] = forecasts
        return forecasts

    def predict_batch(self, context_dfs: List[pd.DataFrame],
                      future_covs: Optional[List[Optional[pd.DataFrame]]],
                      horizon: int,
                      multivariate: bool = False) -> List[Forecast]:
        """回测引擎用的批量入口：传一串起报点的 context（与可选未来协变量）。

        multivariate : 是否多节点联合建模（手册 §6 消融 C）。
        """
        if future_covs is None:
            future_covs = [None] * len(context_dfs)
        tasks = [self._make_task(c, fc, horizon)
                 for c, fc in zip(context_dfs, future_covs)]
        return self.predict_tasks(tasks, horizon, multivariate=multivariate)

    # ── 单任务（兼容旧接口）：退化为 1 任务批 ────────────────────────────────
    def predict(self, context_df, future_covariates=None, horizon=24,
                multivariate: bool = False) -> Forecast:
        task = self._make_task(context_df, future_covariates, horizon)
        return self.predict_tasks([task], horizon, multivariate=multivariate)[0]


# ── 三个具体适配器 ───────────────────────────────────────────────────────────
class TimesFMForecaster(FoundationForecaster):
    """TimesFM-2.5：decoder-only 单序列模型。

    架构上是单变量预测器，**不原生支持多变量联合建模**——多节点只能逐列独立
    预测（节点间互不影响）。因此 supports_multivariate=False：当上层要求多变量
    时会如实降级为逐列，并在结果里标记 multivariate_used=False（手册 §3.3 能力诚实）。

    协变量：TimesFM-2.5 通过 XReg 原生支持外生协变量，但本期 worker 尚未接入
    该路径，故 supports_covariates=False（暂不喂协变量，也不冒充支持）。
    """
    name = "TimesFM"
    kind = "timesfm"
    needs_training = False
    supports_covariates = False        # XReg 原生支持，但本期 worker 未接入
    supports_multivariate = False      # decoder-only 单序列，不支持联合建模


class Chronos2Forecaster(FoundationForecaster):
    """Chronos-2：唯一同时原生支持多变量 + 协变量的基础模型。

    官方在单一架构内零样本支持 univariate / multivariate / covariate-informed
    三类任务：多变量时把多个协同序列联合预测，捕捉节点间依赖。
    所以两个能力标志都是 True（但本轮协变量+多变量同时开启时，worker
    仍按逐列+协变量处理以保证稳定）。
    """
    name = "Chronos2"
    kind = "chronos2"
    needs_training = False
    supports_covariates = True
    supports_multivariate = True


class TotoForecaster(FoundationForecaster):
    """Toto：生来面向多变量的时间序列基础模型（为可观测性多指标设计）。

    强项是多节点联合建模：多变量时 worker 用同组 id_mask 把多节点作为
    一个多变量序列联合预测。故 supports_multivariate=True。

    协变量：Toto 未提供变量级外生协变量接口，故 supports_covariates=False（不
    喂协变量，也不冒充支持）。
    """
    name = "Toto"
    kind = "toto"
    needs_training = False
    supports_covariates = False        # 无变量级协变量接口
    supports_multivariate = True       # 原生多变量联合建模


class Toto2Forecaster(FoundationForecaster):
    """Toto 2.0：u-μP scaling + 分位数输出头 + CPM 解码。

    相比 1.0 的核心改进：
    - 输出头从 Student-T 混合采样 → 9 分位数（Pinball Loss），消除数值爆炸；
    - 解码从逐 patch 自回归 → CPM 单次前向，更快且误差不累积；
    - 输入归一化增加 asinh 变换，对重尾分布（如电价尖峰）更鲁棒；
    - patch_size 64→32，粒度更细；patch 嵌入改为 ResidualMLP。

    多变量：通过 series_ids 控制，与 1.0 相同。
    协变量：2.0 目前版本尚未开放 fine-tuning / exogenous 接口。
    """
    name = "Toto2"
    kind = "toto2"
    needs_training = False
    supports_covariates = False        # 2.0 当前版本未开放 exogenous 接口
    supports_multivariate = True       # 原生多变量联合建模


# ── 注册表：名字 → 构造器 ─────────────────────────────────────────────────────
FOUNDATION_REGISTRY = {
    "TimesFM":  lambda **kw: TimesFMForecaster(),
    "Chronos2": lambda **kw: Chronos2Forecaster(),
    "Toto":     lambda **kw: TotoForecaster(),
    "Toto2":    lambda **kw: Toto2Forecaster(),
}


# ── 自测：仅检查 venv/worker 路径是否齐全，不实际跑模型 ──────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("基础模型适配器环境自检")
    print("=" * 60)
    for kind in ("timesfm", "chronos2", "toto", "toto2"):
        py_ok = os.path.exists(VENV_PYTHON[kind])
        sc_ok = os.path.exists(WORKER_SCRIPT[kind])
        print(f"{kind:10s}  venv={'✅' if py_ok else '❌'}  "
              f"worker={'✅' if sc_ok else '❌'}")
    print("\n注册的基础模型：", list(FOUNDATION_REGISTRY))
