"""
10_build_model_ready.py
=======================
按【领域逻辑】为每个市场筛选协变量，产出喂模型用的特征集。

原则（领域先验，非数据驱动猜测）：
  - 气价/库存（gas）：天然气是各市场边际燃料 → 四市场通用
  - 风暴事件（storm）：极端天气影响各市场负荷 → 四市场通用
  - 发电燃料结构（generation_mix）：供给侧，各市场都有 → 四市场通用
  - WTI 原油（oil）：德州油气经济 → 仅 ERCOT
  - 钢铁 PPI/产量（steel）：Ohio/PA 钢铁带 → 仅 PJM

【保留钢在 PJM、WTI 在 ERCOT】——不因差分线性信号为零就人工砍掉
（月度 PPI 粒度可能太粗，让模型自行判断权重，见 todo2.md 稳健性结论）。
仅移除"无因果故事"的跨市场列（如钢价→ERCOT 电价没有物理通路）。

✅ 泄漏修复（政策B：所有协变量统一滞后，模型通道保持 future_known=True）：
  经济/风暴/发电类按发布频率滞后——
  - 24h：日频现货(气价/WTI) + NOAA 风暴(事后记录) + 实际发电(T+1) + 其派生
  - 168h：周度库存（按周末对齐，下周四才发布）
  - 1440h：月度 PPI/产量（按月初对齐，次月中旬才发布）
  另 load/temperature/wind/solar（实测值）统一 shift(24) 滞后1天——预测明天电价时
  用的是"昨天已公布的真实"负荷/温度/风光，而非明天实测（无偷看）。
  → forecast 窗口全是 T-1 已知值，无泄漏、自洽。首段 NaN 落在测试期(2025-07+)前。
  注：PJM/NYISO 无 wind/solar 市场数据，按市场容错跳过。

产出:
  data/covariates/model_ready/{market}_features_hourly.csv

用法:
  python scripts/covariates/10_build_model_ready.py
"""
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MERGED_DIR = os.path.join(ROOT, 'data', 'covariates', 'merged')
OUT_DIR = os.path.join(ROOT, 'data', 'covariates', 'model_ready')
os.makedirs(OUT_DIR, exist_ok=True)

# 复用 loader 的取数逻辑（load/wind/solar 从市场目录、temperature 从天气文件）
sys.path.insert(0, os.path.join(ROOT, 'src', 'data_processing'))
from loader import _system_series, _weather_series  # noqa: E402

# 市场系统级 + 天气协变量（统一滞后 1 天，与经济协变量同档防泄漏）
# 注：PJM/NYISO 无 wind/solar 市场数据，按市场容错跳过
AUX_COVARS = ['load', 'wind', 'solar', 'temperature']

# 仅在某市场有因果故事的协变量列（钢→仅PJM，WTI→仅ERCOT）
STEEL_COLS = ['hrc_futures_usd_per_ton']
WTI_COL = ['wti_usd_per_barrel']

# 市场 → 该市场【无因果故事】需剔除的列
# （反过来即"该市场保留"的列 = 全集 - 剔除集）
MARKET_DROP = {
    'ercot': STEEL_COLS,                       # ERCOT 无钢铁产业故事 → 剔除钢
    'pjm':   WTI_COL,                          # PJM 非油气经济 → 剔除 WTI（保留钢）
    'caiso': STEEL_COLS + WTI_COL,             # CAISO 两者都无故事
    'nyiso': STEEL_COLS + WTI_COL,             # NYISO 两者都无故事
}

# 保留理由（写进每市场 sidecar 文本，便于论文/审稿追溯）
RATIONALE = {
    'ercot': '气价/库存(边际燃料) + 风暴 + 发电结构 + WTI(德州油气)；剔除钢(无钢铁产业)',
    'pjm':   '气价/库存(边际燃料) + 风暴 + 发电结构 + 钢(PJM 钢铁带 Ohio/PA)；剔除 WTI(非油气经济)',
    'caiso': '气价/库存(边际燃料) + 风暴 + 发电结构；剔除钢+WTI(均无因果故事)',
    'nyiso': '气价/库存(边际燃料) + 风暴 + 发电结构；剔除钢+WTI(均无因果故事)',
}


def main():
    print("=" * 60)
    print("市场专属协变量筛选（领域逻辑）→ model_ready 特征集（含 load/temp/wind/solar）")
    print("=" * 60)
    # 经济/风暴/发电类防泄漏滞后（按发布频率/可得性）：
    #   24h（1天）= 日频现货(气价/WTI/HRC钢价) + 事后记录(风暴) + 实际发电(T+1) + 其派生
    #   168h（7天）= 周度库存（按周末对齐，下周四才发布）
    #   （月度 PPI/产量已替换为日频 HRC，故不再有 1440h 档）
    LAG_HOURS = {
        'henry_hub_usd_per_mmbtu': 24, 'wti_usd_per_barrel': 24,
        'hrc_futures_usd_per_ton': 24,
        'storm_event_count': 24, 'storm_high_impact_count': 24, 'storm_damage_usd': 24,
        'storm_injuries': 24, 'storm_has_extreme_temp': 24, 'storm_has_wind': 24, 'storm_has_winter': 24,
        'total_gen_mwh': 24, 'gas_gen_mwh': 24, 'wind_gen_mwh': 24, 'solar_gen_mwh': 24,
        'gas_share': 24, 'renewable_share': 24, 'gas_share_diff': 24, 'renewable_shock': 24,
        'natgas_storage_bcf': 168,
    }
    for mkt in ['ercot', 'pjm', 'caiso', 'nyiso']:
        mkt_up = mkt.upper()
        src = os.path.join(MERGED_DIR, f'{mkt}_all_covariates_hourly.csv')
        df = pd.read_csv(src, parse_dates=['timestamp_utc'])
        drop = [c for c in MARKET_DROP[mkt] if c in df.columns]
        out = df.drop(columns=drop)
        # 1) 经济/风暴/发电类按 LAG_HOURS 滞后
        applied = []
        for c, lag in LAG_HOURS.items():
            if c in out.columns:
                out[c] = out[c].shift(lag)
                applied.append(f"{c}={lag}h")
        # 2) load/temperature/wind/solar（实测→必滞后1天防泄漏），按市场容错加载
        ts_idx = pd.DatetimeIndex(pd.to_datetime(out['timestamp_utc'], utc=True)).tz_convert(None)  # tz-naive UTC 主轴
        aux_applied, aux_missing = [], []
        for cov in AUX_COVARS:
            try:
                if cov in ('load', 'wind', 'solar'):
                    s = _system_series(mkt_up, cov, 'hourly')   # 传归一化后的 freq（_system_series 内部不再归一化）
                else:  # temperature
                    s = _weather_series(mkt_up, [cov], 'hourly')[cov]
                s = s.tz_convert(None) if s.index.tz is not None else s
                s = s.reindex(ts_idx).shift(24)   # 对齐到主轴 + 滞后1天
                out[cov] = s.values
                aux_applied.append(cov)
            except FileNotFoundError:
                aux_missing.append(cov)  # 该市场无此数据（如 PJM/NYISO 无 wind/solar）
        out_path = os.path.join(OUT_DIR, f'{mkt}_features_hourly.csv')
        out.to_csv(out_path, index=False)
        # sidecar
        with open(os.path.join(OUT_DIR, f'{mkt}_features.txt'), 'w') as f:
            f.write(f"{mkt_up} model_ready 特征集（无泄漏，含 load/temp/wind/solar）\n")
            f.write(f"保留理由: {RATIONALE[mkt]}\n")
            f.write(f"剔除列: {drop}\n")
            f.write(f"特征列({len(out.columns)-1}): {list(out.columns[1:])}\n")
            f.write(f"行数: {len(out)}\n")
            f.write(f"经济/风暴/发电 防泄漏滞后: {applied}\n")
            f.write(f"load/temp/wind/solar 滞后24h: {aux_applied}（缺失跳过: {aux_missing}）\n")
            f.write("\n注: 所有特征均滞后到起报时刻已发布值，forecast 窗口无泄漏（政策B）。\n")
            f.write("注: 钢/WTI 线性信号为零但保留供模型非线性学习（月度 PPI 粒度 caveat）。\n")
        print(f"\n  {mkt_up}: {len(out.columns)-1} 特征, {len(out)} 行")
        print(f"    剔除: {drop}")
        print(f"    load/temp/wind/solar 滞后24h: {aux_applied}  缺失: {aux_missing}")
        print(f"    → {out_path}")
    print("\n✓ 完成")


if __name__ == "__main__":
    main()
