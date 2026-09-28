"""
11_scrape_article_text.py
=========================
从 GDELT 能源新闻 URL 列表抓取文章正文，供后续 LLM 抽特征（W7 path B）。

输入: data/covariates/news/raw/gdelt_news_raw.csv  (列含 url/market/date/tone/themes/source)
输出: data/covariates/news/text/articles_text.jsonl
      每行一个唯一 URL: {url, status, text, text_len, markets, dates, tone, themes, source}

特性:
  - 按 URL 去重（同一 URL 多市场/多日只抓一次，保留全部 market/date 关联）
  - 断点续传：ok/short 及永久性失败（403/404/410/451/401/406/405/421/TooManyRedirects）
    直接计入已完成、不再重试；timeout/conn_error/429/5xx 等瞬时性失败下次重跑会继续重试
  - 线程并发（默认 8）、超时重试、浏览器 UA
  - 正文用 trafilatura 抽取（优于 newspaper3k，去导航/广告/评论）

依赖: pip install trafilatura requests
用法:
  python3 scripts/covariates/11_scrape_article_text.py              # 全量
  python3 scripts/covariates/11_scrape_article_text.py --limit 100  # 先试 100 个看成功率
  python3 scripts/covariates/11_scrape_article_text.py --workers 12 # 调并发
"""
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

try:
    import trafilatura
except ImportError:
    sys.exit("缺少 trafilatura，先装: python3 -m pip install trafilatura requests")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'covariates', 'news', 'raw', 'gdelt_news_raw.csv')
OUT_DIR = os.path.join(ROOT, 'data', 'covariates', 'news', 'text')
OUT = os.path.join(OUT_DIR, 'articles_text.jsonl')
os.makedirs(OUT_DIR, exist_ok=True)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
TIMEOUT = 15          # 缩短：超时早点失败，少占 worker
RETRIES = 1           # 缩短：超时重试多半也超时，少浪费
SLEEP = 0.5           # 每请求后节流，避免打满带宽/触发限流
BACKOFF = 1.5         # 超时/连接失败重试前退避


def fetch_text(url):
    """抓单条 URL，返回 {status, text, text_len}。每请求后 sleep 节流。"""
    global SLEEP
    last_err = ""
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            if r.status_code != 200:
                time.sleep(SLEEP)
                return {"status": f"http_{r.status_code}", "text": "", "text_len": 0}
            text = trafilatura.extract(
                r.text, include_comments=False, include_tables=False, favor_recall=True
            ) or ""
            time.sleep(SLEEP)
            if len(text) < 200:
                return {"status": "short", "text": text, "text_len": len(text)}
            return {"status": "ok", "text": text, "text_len": len(text)}
        except requests.exceptions.Timeout:
            last_err = "timeout"
            if attempt < RETRIES:
                time.sleep(BACKOFF)
                continue
        except requests.exceptions.ConnectionError:
            last_err = "conn_error"
            if attempt < RETRIES:
                time.sleep(BACKOFF)
                continue
        except Exception as e:
            last_err = type(e).__name__
            if attempt < RETRIES:
                time.sleep(BACKOFF)
                continue
    time.sleep(SLEEP)
    return {"status": last_err or "failed", "text": "", "text_len": 0}


# 永久性失败状态：重跑基本不会变好（页面不存在/无权限/法律下线等），
# 计入 done 避免反复重试浪费时间、并防止同一 URL 被反复追加冗余失败记录。
PERMANENT_FAIL = {
    "http_403", "http_404", "http_410", "http_451",
    "http_401", "http_406", "http_405", "http_421",
    "TooManyRedirects",
}
# 瞬时性失败状态（timeout/conn_error/http_429/5xx/ChunkedEncodingError 等）不计入上面的集合，
# 默认会被 load_done() 判定为未完成，下次继续重试补全。


def load_done():
    """已完成（无需再抓）的 URL 集合：
      - ok/short：抓取成功（含短文），跳过
      - 永久性失败（403/404/410/451/401/406/405/421/TooManyRedirects）：重跑也大概率不会变好，跳过
      - 其余瞬时性失败（timeout/conn_error/429/5xx等）：不计入 done，下次继续重试补全
    """
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") in ("ok", "short") or r.get("status") in PERMANENT_FAIL:
                        done.add(r["url"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 个待抓 URL（0=全量）")
    ap.add_argument("--workers", type=int, default=8, help="并发数（多遍补全策略可用 8）")
    ap.add_argument("--sleep", type=float, default=0.0, help="每请求后 sleep 秒数（0=不节流，多遍跑用 0）")
    args = ap.parse_args()
    global SLEEP
    SLEEP = args.sleep

    df = pd.read_csv(RAW, dtype=str)
    # 按 URL 去重，聚合 market/date 列表
    agg = df.groupby("url").agg(
        markets=("market", lambda s: sorted(set(s.dropna()))),
        dates=("date", lambda s: sorted(set(s.dropna().astype(str)))),
        tone=("tone", "first"),
        themes=("themes", "first"),
        source=("source", "first"),
    ).reset_index()
    url2meta = {r["url"]: r for _, r in agg.iterrows()}

    done = load_done()
    todo = [u for u in agg["url"].tolist() if u not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"唯一 URL {len(agg)} | 已抓 {len(done)} | 待抓 {len(todo)} | 并发 {args.workers}")
    if not todo:
        print("全部已抓完。"); return

    counts = {"ok": 0, "short": 0, "timeout": 0, "conn_error": 0}
    lock = threading.Lock()
    finished = [0]

    with open(OUT, "a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_text, u): u for u in todo}
        for fut in as_completed(futs):
            url = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"status": "error", "text": "", "text_len": 0}
            meta = url2meta.get(url, {})
            rec = {
                "url": url,
                "status": res["status"],
                "text": res["text"],
                "text_len": res["text_len"],
                "markets": meta.get("markets", []),
                "dates": meta.get("dates", []),
                "tone": meta.get("tone"),
                "themes": meta.get("themes"),
                "source": meta.get("source"),
            }
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                finished[0] += 1
                st = res["status"]
                if st in counts:
                    counts[st] += 1
                if finished[0] % 50 == 0:
                    print(f"  [{finished[0]}/{len(todo)}] 成功ok={counts['ok']} "
                          f"short={counts['short']} timeout={counts['timeout']} "
                          f"conn={counts['conn_error']}", flush=True)
    print(f"\n✓ 完成。已抓 {finished[0]} 条 → {OUT}")
    print(f"  成功(ok)={counts['ok']}  短文(short)={counts['short']}  "
          f"超时={counts['timeout']}  连接失败={counts['conn_error']}")
    if finished[0]:
        ok = counts['ok']
        print(f"  正文成功率 ≈ {100*ok/finished[0]:.0f}%（ok/总）")


if __name__ == "__main__":
    main()
