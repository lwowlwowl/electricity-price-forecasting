"""
05_download_noaa_storm.py
=========================
下载 NOAA Storm Events Database 批量 CSV，处理成四个市场的日频特征，
再 forward-fill 对齐到小时级时间轴。

数据源: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
- 免费公开数据，无需 API key
- 按年归档的 gzip 压缩 CSV
- 包含事件类型、州县、时间、财产损失、伤亡等

产出:
  data/covariates/storm/raw/noaa_storm/StormEvents_details_d{year}.csv.gz  (原始下载)
  data/covariates/storm/cleaned/noaa_storm_daily.csv                       (四市场日频特征)
  data/covariates/storm/hourly/{market}_storm_hourly.csv                   (小时级特征)

用法:
  pip install pandas requests
  python scripts/covariates/05_download_noaa_storm.py
"""

import os
import gzip
import shutil
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── 路径 ──
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "covariates" / "storm" / "raw" / "noaa_storm"
CLEANED_DIR = ROOT / "data" / "covariates" / "storm" / "cleaned"
HOURLY_DIR = ROOT / "data" / "covariates" / "storm" / "hourly"
PRICE_DIR = ROOT / "data" / "raw"
for _d in (RAW_DIR, CLEANED_DIR, HOURLY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 配置 ──
# NOAA 文件命名格式: StormEvents_details-ftp_v1.0_d{YEAR}_c{YYYYMMDD}.csv.gz
# c 日期每月更新，需要从目录页匹配最新版本
BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
YEARS = [2025, 2026]

# 四个市场对应的州 (STATE 字段为全大写)
MARKET_STATES = {
    "ERCOT": ["TEXAS"],
    "PJM": [
        "PENNSYLVANIA", "OHIO", "VIRGINIA", "NEW JERSEY", "MARYLAND",
        "DELAWARE", "WEST VIRGINIA", "INDIANA", "ILLINOIS", "MICHIGAN",
        "NORTH CAROLINA", "KENTUCKY", "DISTRICT OF COLUMBIA",
    ],
    "CAISO": ["CALIFORNIA"],
    "NYISO": ["NEW YORK"],
}

# 对电价有显著影响的事件类型 (排除 Hail-only 等对电力系统影响小的事件)
HIGH_IMPACT_TYPES = {
    "Winter Storm", "Winter Weather", "Blizzard", "Ice Storm", "Heavy Snow",
    "Cold/Wind Chill", "Extreme Cold/Wind Chill", "Frost/Freeze",
    "Excessive Heat", "Heat",
    "Hurricane", "Hurricane (Typhoon)", "Tropical Storm", "Tropical Depression",
    "Tornado", "Thunderstorm Wind", "High Wind", "Strong Wind",
    "Flash Flood", "Flood", "Storm Surge/Tide",
    "Wildfire", "Drought",
    "Lake-Effect Snow",
}


def find_latest_filename(year):
    """从目录页找到指定年份的最新文件名"""
    import re
    print(f"  查找 {year} 年最新文件名...")
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c(\d{{8}})\.csv\.gz'
    matches = re.findall(pattern, resp.text)
    if not matches:
        raise FileNotFoundError(f"未找到 {year} 年的 Storm Events 文件")
    latest_date = max(matches)
    filename = f"StormEvents_details-ftp_v1.0_d{year}_c{latest_date}.csv.gz"
    print(f"  → {filename}")
    return filename


def download_year(year):
    """下载指定年份的 Storm Events 数据"""
    local_gz = RAW_DIR / f"StormEvents_details_d{year}.csv.gz"

    # 如果已存在且大小合理，跳过
    if local_gz.exists() and local_gz.stat().st_size > 100_000:
        print(f"  {year} 年数据已存在 ({local_gz.stat().st_size / 1e6:.1f} MB)，跳过下载")
        return local_gz

    remote_name = find_latest_filename(year)
    url = BASE_URL + remote_name
    print(f"  下载: {url}")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    with open(local_gz, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"  → 保存到 {local_gz} ({local_gz.stat().st_size / 1e6:.1f} MB)")
    return local_gz


def parse_damage(val):
    """将 NOAA 的损失字符串 (如 '1.00K', '2.50M') 转成数值 (美元)"""
    if pd.isna(val) or val == "":
        return 0.0
    val = str(val).strip().upper()
    if val in ("0", "0.00"):
        return 0.0
    multiplier = 1
    if val.endswith("K"):
        multiplier = 1_000
        val = val[:-1]
    elif val.endswith("M"):
        multiplier = 1_000_000
        val = val[:-1]
    elif val.endswith("B"):
        multiplier = 1_000_000_000
        val = val[:-1]
    try:
        return float(val) * multiplier
    except ValueError:
        return 0.0


def load_and_concat(years):
    """加载并合并多年数据"""
    frames = []
    for year in years:
        gz_path = RAW_DIR / f"StormEvents_details_d{year}.csv.gz"
        print(f"  读取 {gz_path.name}...")
        df = pd.read_csv(gz_path, compression="gzip", low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # 清理 STATE 列 (有些带引号)
    df["STATE"] = df["STATE"].str.strip('"').str.strip()
    df["EVENT_TYPE"] = df["EVENT_TYPE"].str.strip('"').str.strip()
    return df


def process_to_daily(df):
    """
    将原始事件数据聚合成每个市场每天的特征:
      - storm_event_count: 当日该市场区域内的总事件数
      - storm_high_impact_count: 高影响事件数 (对电力系统有直接影响)
      - storm_damage_usd: 当日总财产损失 (美元)
      - storm_injuries: 当日总伤亡人数
      - storm_has_extreme_temp: 是否有极端温度事件 (Heat/Cold)
      - storm_has_wind: 是否有大风/龙卷风事件
      - storm_has_winter: 是否有冬季风暴事件
    """
    # 解析日期
    # BEGIN_DATE_TIME 格式: "31-MAR-25 11:04:00"
    df["event_date"] = pd.to_datetime(
        df["BEGIN_DATE_TIME"].str.strip('"'),
        format="%d-%b-%y %H:%M:%S",
        errors="coerce",
    ).dt.date

    # 解析损失
    df["damage_usd"] = df["DAMAGE_PROPERTY"].apply(parse_damage) + df["DAMAGE_CROPS"].apply(parse_damage)
    df["injuries"] = df["INJURIES_DIRECT"].fillna(0) + df["INJURIES_INDIRECT"].fillna(0) if "INJURIES_INDIRECT" in df.columns else df["INJURIES_DIRECT"].fillna(0)
    df["is_high_impact"] = df["EVENT_TYPE"].isin(HIGH_IMPACT_TYPES).astype(int)
    df["is_extreme_temp"] = df["EVENT_TYPE"].isin({"Excessive Heat", "Heat", "Cold/Wind Chill", "Extreme Cold/Wind Chill"}).astype(int)
    df["is_wind"] = df["EVENT_TYPE"].isin({"Tornado", "Thunderstorm Wind", "High Wind", "Strong Wind", "Hurricane", "Hurricane (Typhoon)"}).astype(int)
    df["is_winter"] = df["EVENT_TYPE"].isin({"Winter Storm", "Winter Weather", "Blizzard", "Ice Storm", "Heavy Snow", "Lake-Effect Snow"}).astype(int)

    records = []
    for market, states in MARKET_STATES.items():
        mask = df["STATE"].isin(states)
        sub = df[mask].copy()
        if sub.empty:
            continue
        daily = sub.groupby("event_date").agg(
            storm_event_count=("EVENT_TYPE", "count"),
            storm_high_impact_count=("is_high_impact", "sum"),
            storm_damage_usd=("damage_usd", "sum"),
            storm_injuries=("injuries", "sum"),
            storm_has_extreme_temp=("is_extreme_temp", "max"),
            storm_has_wind=("is_wind", "max"),
            storm_has_winter=("is_winter", "max"),
        ).reset_index()
        daily["market"] = market
        records.append(daily)

    result = pd.concat(records, ignore_index=True)
    result["event_date"] = pd.to_datetime(result["event_date"])
    return result.sort_values(["market", "event_date"]).reset_index(drop=True)


def align_to_hourly(daily_df):
    """
    将日频特征对齐到每个市场的小时级电价时间轴 (forward-fill)。
    产出独立的 CSV: {market}_storm_hourly.csv
    """
    market_price_files = {
        "ERCOT": PRICE_DIR / "ERCOT" / "processed" / "actual_price_hourly.csv",
        "PJM": PRICE_DIR / "PJM" / "processed" / "actual_price_hourly.csv",
        "CAISO": PRICE_DIR / "CAISO" / "processed" / "actual_price_hourly.csv",
        "NYISO": PRICE_DIR / "NYISO" / "processed" / "actual_price_hourly.csv",
    }

    for market, price_path in market_price_files.items():
        print(f"  对齐 {market}...")
        if not price_path.exists():
            print(f"    ⚠ 价格文件不存在: {price_path}，跳过")
            continue

        # 读取价格文件获取时间轴
        price = pd.read_csv(price_path, usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
        hourly_index = pd.DatetimeIndex(price["timestamp_utc"].sort_values().unique())
        # 统一为 tz-naive (UTC implied) 以避免 tz mismatch
        if hourly_index.tz is not None:
            hourly_index = hourly_index.tz_localize(None)

        # 该市场的日频数据
        mkt_daily = daily_df[daily_df["market"] == market].copy()
        mkt_daily = mkt_daily.set_index("event_date").drop(columns=["market"])
        # 确保日频索引也是 tz-naive
        if mkt_daily.index.tz is not None:
            mkt_daily.index = mkt_daily.index.tz_localize(None)

        # 构建完整日期范围并填 0 (无事件日)
        full_dates = pd.date_range(
            hourly_index.min().normalize(),
            hourly_index.max().normalize(),
            freq="D",
        )
        mkt_daily = mkt_daily.reindex(full_dates, fill_value=0)

        # 将日期索引扩展到小时，forward-fill
        mkt_daily.index.name = "date"
        mkt_daily = mkt_daily.reset_index()
        mkt_daily["timestamp_utc"] = mkt_daily["date"]
        mkt_daily = mkt_daily.set_index("timestamp_utc")
        mkt_daily = mkt_daily.reindex(hourly_index, method="ffill")
        mkt_daily = mkt_daily.drop(columns=["date"], errors="ignore")

        # 保存
        out_path = HOURLY_DIR / f"{market.lower()}_storm_hourly.csv"
        mkt_daily.index.name = "timestamp_utc"
        mkt_daily.to_csv(out_path)
        print(f"    → {out_path} ({len(mkt_daily)} 行)")


def main():
    print("=" * 60)
    print("NOAA Storm Events → 市场级日频/小时频特征")
    print("=" * 60)

    # 1. 下载
    print("\n[1/4] 下载原始数据...")
    for year in YEARS:
        download_year(year)

    # 2. 加载合并
    print("\n[2/4] 加载并合并...")
    df = load_and_concat(YEARS)
    print(f"  合并后总行数: {len(df)}")

    # 3. 聚合成日频
    print("\n[3/4] 聚合为日频特征...")
    daily = process_to_daily(df)
    daily_path = CLEANED_DIR / "noaa_storm_daily.csv"
    daily.to_csv(daily_path, index=False)
    print(f"  → {daily_path} ({len(daily)} 行)")

    # 打印摘要
    for market in MARKET_STATES:
        mkt = daily[daily["market"] == market]
        print(f"  {market}: {len(mkt)} 天有事件, "
              f"日均 {mkt['storm_event_count'].mean():.1f} 事件, "
              f"总损失 ${mkt['storm_damage_usd'].sum()/1e6:.1f}M")

    # 4. 对齐到小时
    print("\n[4/4] 对齐到小时级时间轴 (forward-fill)...")
    align_to_hourly(daily)

    print("\n✓ 完成!")


if __name__ == "__main__":
    main()
