"""
LMP 数据预处理脚本
==================
从 ENLITEN 数据集的 LMP.xlsx 中提取电价时序，
输出为标准 CSV 供三个模型直接读取。

运行前准备：
  1. 激活 toto 虚拟环境：source ../toto/.venv/bin/activate
  2. 安装 openpyxl（仅首次）：pip install openpyxl

运行方式：
  python prepare.py

输出文件：
  ../../data/processed/lmp_processed.csv   — 处理好的电价时序
"""

import os
import sys
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# 输入：从 data/raw/ 读取原始数据
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "../../data/raw")
LMP_XLSX     = os.path.join(RAW_DATA_DIR, "ENLITEN-Grid-Econ-Data/WECC and NPCC Systems/Price/LMP.xlsx")

# 输出路径
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "../../data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_CSV   = os.path.join(PROCESSED_DIR, "lmp_processed.csv")

print("=" * 60)
print("LMP 数据预处理")
print("=" * 60)

# ── 1. 读取 LMP.xlsx ──────────────────────────────────────────────────────────
print(f"\n读取文件：{LMP_XLSX}")
try:
    xl = pd.ExcelFile(LMP_XLSX, engine="openpyxl")
except ModuleNotFoundError:
    print("\n❌ 缺少 openpyxl，请先运行：pip install openpyxl")
    sys.exit(1)

print(f"Sheet 列表：{xl.sheet_names}")

# ── 2. 探索数据结构 ───────────────────────────────────────────────────────────
# 分别读取 NPCC 和 WECC 两个 sheet
dfs = {}
for sheet in xl.sheet_names:
    df = xl.parse(sheet, header=None)
    print(f"\n── Sheet: {sheet} ──")
    print(f"  形状：{df.shape}  （行 × 列）")
    print(f"  前 3 行预览：")
    print(df.iloc[:3, :6].to_string())
    dfs[sheet] = df

# ── 3. 解析数据格式 ───────────────────────────────────────────────────────────
# ENLITEN LMP.xlsx 格式（两行表头）：
#   行 0 → NaN, "Time", NaN, NaN, ...（说明行，跳过）
#   行 1 → "Bus ID", 1, 2, 3, ...（列标签，跳过）
#   行 2+ → 实际数据：第 0 列=bus_id，第 1 列起=各小时的 LMP（$/MWh）
# 转置后格式为 (N_hours × N_buses)

results = {}
for sheet, df in dfs.items():
    # Excel 格式：行=节点（从第2行起），列=时间步（从第1列起）
    # 第0列=bus_id，第1列起=各小时 LMP（共 8784 列）
    data_rows = df.iloc[2:, :]                              # 去掉两行表头，剩余行=节点
    bus_ids   = data_rows.iloc[:, 0].astype(str).tolist()  # 第0列=节点编号
    data      = data_rows.iloc[:, 1:].values.astype(float) # shape: (N_buses, N_hours)

    n_buses, n_hours = data.shape
    print(f"\n── {sheet}：{n_buses} 个节点，{n_hours} 个小时")

    # 不需要转置：data 已经是 (N_buses, N_hours)，直接转为 (N_hours, N_buses)
    df_t = pd.DataFrame(data.T, columns=[f"{sheet}_bus{b}" for b in bus_ids])
    results[sheet] = df_t

# ── 4. 构造时间索引（从实际小时数推断，兼容闰年/非闰年） ──────────────────────
# 取任意 sheet 的行数作为时间步总数（ENLITEN 2020 年=闰年，共 8784 小时）
n_hours_actual = next(iter(results.values())).shape[0]
time_index = pd.date_range(start="2020-01-01 00:00", periods=n_hours_actual, freq="h")

# ── 5. 合并 NPCC 和 WECC ──────────────────────────────────────────────────────
df_combined = pd.concat(list(results.values()), axis=1)
df_combined.index = time_index
df_combined.index.name = "timestamp"

print(f"\n合并后形状：{df_combined.shape}  （{n_hours_actual} 小时 × {df_combined.shape[1]} 节点）")
print(f"时间范围：{df_combined.index[0]}  →  {df_combined.index[-1]}")
print(f"\n电价统计（$/MWh）：")
stats = df_combined.stack()
print(f"  均值：{stats.mean():.2f}  最小：{stats.min():.2f}  最大：{stats.max():.2f}  标准差：{stats.std():.2f}")

# ── 6. 检查并处理异常值 ───────────────────────────────────────────────────────
nan_count = df_combined.isna().sum().sum()
print(f"\n缺失值数量：{nan_count}")
if nan_count > 0:
    df_combined = df_combined.ffill().bfill()
    print("  已用前向/后向填充处理缺失值")

# 检查极端值（仅统计，不做处理）
extreme = (df_combined.abs() > 1000).sum().sum()
if extreme > 0:
    print(f"  ℹ️  发现 {extreme} 个极端值（|LMP| > 1000 $/MWh），已保留原始值")

# ── 7. 保存 CSV ───────────────────────────────────────────────────────────────
df_combined.to_csv(OUTPUT_CSV)
print(f"\n✅ 已保存至：{OUTPUT_CSV}")
print(f"   文件大小：{os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.1f} MB")
print(f"   列数（节点数）：{df_combined.shape[1]}")
print(f"   行数（小时数）：{df_combined.shape[0]}")

# ── 8. 打印列名供后续脚本参考 ─────────────────────────────────────────────────
print(f"\n前 10 个节点列名：")
print(df_combined.columns[:10].tolist())

print("\n" + "=" * 60)
print("✅ 预处理完成！")
print("=" * 60)
