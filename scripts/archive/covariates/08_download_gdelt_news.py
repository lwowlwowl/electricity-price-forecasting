"""
08_download_gdelt_news.py
=========================
从 GDELT GKG v2 原始文件提取美国能源/天气新闻的日频特征，按市场拆分。

策略: 每天下载 1 个 GKG 文件 (12:00 UTC) → 筛选能源主题 →
      解析州代码映射到 ERCOT/PJM/CAISO/NYISO → 按市场按天聚合。

时间对齐: 采用次日规则防止未来信息泄漏。
  今天 (day T) 的新闻聚合特征，从 day T+1 00:00 UTC 开始生效。
  因为 12:00 UTC 的新闻不可能在 T 日 00:00-11:59 UTC 被观察到。

数据源: http://data.gdeltproject.org/gdeltv2/
  - 免费公开静态文件，无需 API key，不受 DOC API 限流影响

产出:
  data/covariates/news/raw/gdelt_news_raw.csv              逐条明细 (含 market 列)
  data/covariates/news/cleaned/gdelt_news_daily.csv        按市场日频聚合
  data/covariates/news/hourly/{market}_news_hourly.csv      小时频 (次日 forward-fill)

支持断点续传。

用法:
  python scripts/covariates/08_download_gdelt_news.py
"""

import io
import re
import time
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "covariates" / "news" / "raw"
NEWS_CLEANED = ROOT / "data" / "covariates" / "news" / "cleaned"
NEWS_HOURLY = ROOT / "data" / "covariates" / "news" / "hourly"
PRICE_DIR = ROOT / "data" / "raw"
for _d in (RAW_DIR, NEWS_CLEANED, NEWS_HOURLY):
    _d.mkdir(parents=True, exist_ok=True)

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 6, 2)
SAMPLE_HOUR = "120000"

ENERGY_THEMES = {
    "ECON_ENERGY", "ENV_GREEN", "ENV_CLIMATECHANGE", "NATURAL_DISASTER",
    "ECON_OILPRICE", "ECON_NATURALGAS", "EPU_ENERGY",
    "WB_2743_ENERGY", "WB_466_ENERGY_AND_EXTRACTIVES",
    "UNGP_CLEAN_ENERGY", "UNGP_AFFORDABLE_ENERGY",
    "ENV_SOLAR", "ENV_WIND", "ENV_NUCLEAR", "ENV_COAL",
    "ENV_DROUGHT", "ENV_FLOOD", "ENV_STORM", "ENV_HURRICANE",
    "ENV_HEATWAVE", "ENV_COLDWAVE", "ENV_WILDFIRE",
}

# 州 → 市场映射
STATE_TO_MARKET = {}
for s in ["TX"]:
    STATE_TO_MARKET[s] = "ERCOT"
for s in ["PA", "OH", "VA", "NJ", "MD", "DE", "WV", "IN", "IL", "MI", "NC", "KY", "DC"]:
    STATE_TO_MARKET[s] = "PJM"
for s in ["CA"]:
    STATE_TO_MARKET[s] = "CAISO"
for s in ["NY"]:
    STATE_TO_MARKET[s] = "NYISO"

# 所有四个市场的州码集合
MARKET_STATES = set(STATE_TO_MARKET.keys())

PROGRESS_FILE = RAW_DIR / "gdelt_progress.txt"
DETAIL_FILE = RAW_DIR / "gdelt_news_raw.csv"
STATE_PATTERN = re.compile(r"#US#US([A-Z]{2})#")


def get_completed():
    if PROGRESS_FILE.exists():
        return set(PROGRESS_FILE.read_text().strip().split("\n"))
    return set()


def save_completed(date_str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(date_str + "\n")


def download_day(date_str, session):
    """下载某天 12:00 UTC 的 GKG 文件，按市场拆分返回记录"""
    ts = date_str + SAMPLE_HOUR
    url = f"http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"

    try:
        resp = session.get(url, timeout=60)
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    records = []
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            fname = zf.namelist()[0]
            with zf.open(fname) as f:
                for line in f:
                    try:
                        row = line.decode("utf-8", errors="ignore").split("\t")
                    except Exception:
                        continue
                    if len(row) < 16:
                        continue

                    # 筛选能源主题
                    themes = set(row[7].split(";"))
                    if not (themes & ENERGY_THEMES):
                        continue

                    # 解析州代码
                    locations = row[9] if len(row) > 9 else ""
                    state_codes = set(STATE_PATTERN.findall(locations))

                    # 找到该记录属于哪些市场
                    matched_states = state_codes & MARKET_STATES
                    if not matched_states:
                        # 如果没有具体州码但有 #US#，归入 "US_GENERAL"
                        if "#US#" in locations:
                            markets = ["US_GENERAL"]
                        else:
                            continue
                    else:
                        markets = list(set(STATE_TO_MARKET[s] for s in matched_states))

                    # tone
                    tone_parts = row[15].split(",") if len(row) > 15 else []
                    try:
                        avg_tone = float(tone_parts[0])
                    except (ValueError, IndexError):
                        avg_tone = 0.0

                    matched_themes = themes & ENERGY_THEMES
                    for market in markets:
                        records.append({
                            "date": date_str,
                            "market": market,
                            "source": row[3] if len(row) > 3 else "",
                            "url": row[4] if len(row) > 4 else "",
                            "tone": avg_tone,
                            "themes": ";".join(sorted(matched_themes)),
                            "n_energy_themes": len(matched_themes),
                        })
    except (zipfile.BadZipFile, Exception):
        pass

    return records


def aggregate(detail_path):
    """按市场按天聚合"""
    df = pd.read_csv(detail_path, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # 对四个市场: 用该市场特定记录 + US_GENERAL 记录合并
    all_daily = []
    for market in ["ERCOT", "PJM", "CAISO", "NYISO"]:
        mkt_df = df[(df["market"] == market) | (df["market"] == "US_GENERAL")]
        if mkt_df.empty:
            continue
        daily = mkt_df.groupby("date").agg(
            news_count=("tone", "count"),
            news_tone_mean=("tone", "mean"),
            news_tone_std=("tone", "std"),
            news_tone_min=("tone", "min"),
            news_tone_max=("tone", "max"),
            news_n_sources=("source", "nunique"),
        ).reset_index()
        daily["news_tone_std"] = daily["news_tone_std"].fillna(0)
        daily["market"] = market
        all_daily.append(daily)

    result = pd.concat(all_daily, ignore_index=True)
    result = result.rename(columns={"date": "event_date"})
    return result.sort_values(["market", "event_date"]).reset_index(drop=True)


def align_to_hourly(daily_df):
    """
    次日规则 forward-fill: day T 的新闻特征从 day T+1 00:00 UTC 开始生效。
    防止未来信息泄漏。
    """
    market_price_files = {
        "ERCOT": PRICE_DIR / "ERCOT" / "processed" / "actual_price_hourly.csv",
        "PJM": PRICE_DIR / "PJM" / "processed" / "actual_price_hourly.csv",
        "CAISO": PRICE_DIR / "CAISO" / "processed" / "actual_price_hourly.csv",
        "NYISO": PRICE_DIR / "NYISO" / "processed" / "actual_price_hourly.csv",
    }

    feature_cols = ["news_count", "news_tone_mean", "news_tone_std",
                    "news_tone_min", "news_tone_max", "news_n_sources"]

    for market, price_path in market_price_files.items():
        print(f"  对齐 {market} (次日规则)...")
        if not price_path.exists():
            print(f"    ⚠ 价格文件不存在，跳过")
            continue

        price = pd.read_csv(price_path, usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
        hourly_index = pd.DatetimeIndex(price["timestamp_utc"].sort_values().unique())
        if hourly_index.tz is not None:
            hourly_index = hourly_index.tz_localize(None)

        mkt_daily = daily_df[daily_df["market"] == market].copy()
        if mkt_daily.empty:
            print(f"    ⚠ 无数据，跳过")
            continue

        mkt_daily["event_date"] = pd.to_datetime(mkt_daily["event_date"])
        # 次日规则: 将日期推后一天
        mkt_daily["effective_date"] = mkt_daily["event_date"] + timedelta(days=1)
        mkt_daily = mkt_daily.set_index("effective_date")[feature_cols]

        # 构建完整日期范围
        full_dates = pd.date_range(
            hourly_index.min().normalize(),
            hourly_index.max().normalize(),
            freq="D",
        )
        mkt_daily = mkt_daily.reindex(full_dates, fill_value=0)

        # 扩展到小时, forward-fill
        mkt_daily = mkt_daily.reindex(hourly_index, method="ffill").fillna(0)

        out_path = NEWS_HOURLY / f"{market.lower()}_news_hourly.csv"
        mkt_daily.index.name = "timestamp_utc"
        mkt_daily.to_csv(out_path)
        print(f"    → {out_path} ({len(mkt_daily)} 行)")


def main():
    print("=" * 60)
    print("GDELT GKG v2 → 按市场拆分的能源新闻日频特征")
    print("采样: 每天 12:00 UTC | 对齐: 次日规则防泄漏")
    print("=" * 60)

    completed = get_completed()
    dates = []
    current = START_DATE
    while current <= END_DATE:
        ds = current.strftime("%Y%m%d")
        if ds not in completed:
            dates.append((current, ds))
        current += timedelta(days=1)

    total = (END_DATE - START_DATE).days + 1
    done = len(completed)
    print(f"总天数: {total}, 已完成: {done}, 剩余: {len(dates)}")

    if dates:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (academic research)"})
        write_header = not DETAIL_FILE.exists()

        t_start = time.time()
        for i, (date, ds) in enumerate(dates):
            t0 = time.time()
            records = download_day(ds, session)

            if records:
                df = pd.DataFrame(records)
                df.to_csv(DETAIL_FILE, mode="a", header=write_header, index=False)
                write_header = False

            save_completed(ds)
            elapsed = time.time() - t0
            done += 1
            remaining = len(dates) - i - 1
            avg_time = (time.time() - t_start) / (i + 1)
            eta_min = avg_time * remaining / 60

            # 统计该天各市场记录数
            mkt_counts = {}
            for r in records:
                m = r["market"]
                mkt_counts[m] = mkt_counts.get(m, 0) + 1
            mkt_str = " ".join(f"{k}:{v}" for k, v in sorted(mkt_counts.items()))

            print(f"  [{done}/{total}] {ds}: {len(records)} 条 [{mkt_str}], "
                  f"{elapsed:.0f}s, ETA {eta_min:.0f}min")

    # 聚合
    print("\n按市场聚合为日频特征...")
    if DETAIL_FILE.exists():
        daily = aggregate(DETAIL_FILE)
        daily_path = NEWS_CLEANED / "gdelt_news_daily.csv"
        daily.to_csv(daily_path, index=False)
        print(f"→ {daily_path} ({len(daily)} 行)")
        for market in ["ERCOT", "PJM", "CAISO", "NYISO"]:
            mkt = daily[daily["market"] == market]
            if not mkt.empty:
                print(f"  {market}: {len(mkt)} 天, "
                      f"日均 {mkt['news_count'].mean():.0f} 条, "
                      f"tone 均值 {mkt['news_tone_mean'].mean():.2f}")

        # 对齐到小时
        print("\n对齐到小时级时间轴 (次日规则)...")
        align_to_hourly(daily)
    else:
        print("⚠ 未找到明细文件")

    print("\n✓ 完成!")


if __name__ == "__main__":
    main()
