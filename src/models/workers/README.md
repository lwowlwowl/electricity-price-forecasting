# 基础模型 worker 子进程协议

三个基础模型（TimesFM / Chronos-2 / Toto）各自装在独立 venv 里，**依赖互相冲突，
无法在同一个 Python 进程里 import**。因此采用「子进程 worker」模式：

```
主框架(backtest)  ──写──>  request.npz  ──启动子进程──>  worker(在模型自己的venv)
                                                              │ 加载一次模型
                                                              │ 批量推理所有任务
       主框架 <──读──  response.npz  <────────────────────────┘
```

## 为什么批量

逐个起报点起子进程会重复加载模型（每次几秒~几十秒），几十个起报点会慢到不可用。
所以一次请求里**打包某模型的全部预测任务**，worker 只加载一次模型，跑完所有任务。

## request.npz 格式

由 `foundation.py` 写出，键如下：

| 键 | 形状/类型 | 含义 |
|---|---|---|
| `n_tasks` | 标量 int | 任务数量 |
| `horizon` | 标量 int | 预测步长（所有任务相同） |
| `context__{i}` | (T, n_series) float | 第 i 个任务的历史，列=节点 |
| `future_cov__{i}` | (horizon, n_cov) float 或缺省 | 第 i 个任务的未来协变量（可选） |
| `hist_cov__{i}` | (T, n_cov) float 或缺省 | 第 i 个任务的历史协变量（可选） |
| `interval_seconds` | 标量 int | 采样间隔秒数（Toto 需要） |
| `start_ts__{i}` | 标量 int | 第 i 个任务 context 首点 unix 时间戳（Toto 需要） |
| `multivariate` | 标量 int | 1=多节点联合建模，0=逐列单变量（手册 §6 消融 C） |

> 多节点（n_series>1）行为由 `multivariate` 旋钮控制：
> - `multivariate=0`（默认）：所有 worker 逐列单变量预测，节点之间相互独立。
> - `multivariate=1`：支持联合建模的模型（Toto / Chronos-2）把多节点作为一个
>   多变量序列联合预测；TimesFM 不支持联合，自动降级为逐列，并在结果里如实
>   标记 `multivariate_used=False`。
> - 协变量与多变量同时开启时，Chronos-2 本轮仍按逐列+协变量处理以保证稳定。

## response.npz 格式

由 worker 写出：

| 键 | 形状 | 含义 |
|---|---|---|
| `mean__{i}` | (horizon, n_series) | 第 i 个任务的点预测 |
| `q10__{i}` | (horizon, n_series) | 10% 分位 |
| `q90__{i}` | (horizon, n_series) | 90% 分位 |
| `ok` | 标量 int | 1=成功 |

## 调用方式

```
{venv_python} worker_xxx.py  {request.npz}  {response.npz}
```

worker 全程不联网（HF_HOME 指向本地 hf_cache）。
