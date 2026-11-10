#!/usr/bin/env python
"""Build a reproducible 2025--2026 raw EXAA hourly feature file.

The script reads Salzburg AG's public 2025 historical CSV and obtains 2026
records from the public endpoint used by its FlexSpot webpage.  The API key is
read from that public page at runtime and is never persisted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import html
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_exaa import (  # noqa: E402
    flexspot_records_to_hourly,
    merge_exaa_hourly,
    read_historical_exaa_csv,
)


PUBLIC_PAGE = (
    "https://www.salzburg-ag.at/strom/privat/stromprodukte/strom-flexspot.html"
)
API_BASE = "https://oftc-api.shared.salzburg-ag.tech/api/dynamic-tariff/SJESFSP/from"


def _request_bytes(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    request = Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=45) as response:  # noqa: S310 - fixed public URLs
        return response.read()


def _public_api_key() -> str:
    page = _request_bytes(PUBLIC_PAGE).decode("utf-8")
    match = re.search(r'data-api-key="([^"]+)"', page)
    if match is None:
        raise RuntimeError("public FlexSpot API key not found on the official page")
    return html.unescape(match.group(1))


def _fetch_day(day: date, api_key: str, retries: int = 4) -> list[dict[str, object]]:
    url = f"{API_BASE}/{day.isoformat()}"
    for attempt in range(retries):
        try:
            payload = json.loads(
                _request_bytes(url, headers={"X-Api-Key": api_key}).decode("utf-8")
            )
            records = payload.get("tariffs", [])
            if len(records) not in {23, 24, 25}:
                raise RuntimeError(f"{day}: expected 23/24/25 hourly rows, got {len(records)}")
            return records
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-2025",
        default="data/Austria/external_public/EXAA_AT_2025.csv",
    )
    parser.add_argument("--api-start", default="2026-01-01")
    parser.add_argument("--api-end", default="2026-09-30")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        default="data/Austria/external_public/EXAA_AT_2025_2026_hourly.csv",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be in [1, 16]")

    api_days = pd.date_range(args.api_start, args.api_end, freq="1D")
    api_key = _public_api_key()
    collected: dict[pd.Timestamp, list[dict[str, object]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch_day, timestamp.date(), api_key): timestamp
            for timestamp in api_days
        }
        for position, future in enumerate(as_completed(futures), start=1):
            timestamp = futures[future]
            collected[timestamp] = future.result()
            if position % 25 == 0 or position == len(futures):
                print(f"downloaded {position}/{len(futures)} days", flush=True)

    records = [
        record
        for timestamp in sorted(collected)
        for record in collected[timestamp]
    ]
    historical = read_historical_exaa_csv(ROOT / args.historical_2025)
    current = flexspot_records_to_hourly(records)
    combined = merge_exaa_hourly(historical, current)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.reset_index().to_csv(output, index=False)
    manifest = {
        "output": str(output.relative_to(ROOT)),
        "first_timestamp_utc": combined.index[0].isoformat(),
        "last_timestamp_utc": combined.index[-1].isoformat(),
        "hourly_rows": int(len(combined)),
        "sources": combined["source"].value_counts().to_dict(),
        "api_key_persisted": False,
        "raw_price_units": "EUR/MWh",
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
