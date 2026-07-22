"""
12_llm_extract_news_features.py
================================
用 LLM 读取新闻正文，抽取结构化特征，作为电价预测的协变量（替代/补充 GDELT 粗粒度 tone）。

背景:
  scripts/covariates/11_scrape_article_text.py 已抓取新闻正文到
  data/covariates/news/text/articles_text_dedup.jsonl（status=ok 约 1.1 万篇）。
  GDELT 自带的 themes/tone 字段粒度太粗（如 NATURAL_DISASTER 一个标签占了 70% 的文章），
  无法区分"是不是某个具体市场的天气事件/供给冲击"。本脚本让 LLM 读正文，
  输出更细粒度、更贴近电价场景的结构化字段。

输出 schema（每篇文章一行 JSON）:
  url, source
  markets_llm        : LLM 重新判定的关联市场列表（可能比 GDELT 州映射更准，
                        因为 LLM 能读懂正文里的具体地名/电网名/公司名）
  event_type         : 事件主类型，枚举：
                        weather / supply_shock / demand_shock / policy_regulation /
                        fuel_price / market_operation / unrelated
  weather_subtype    : 若 event_type=weather，细分：heatwave/coldwave/storm/hurricane/
                        drought/wildfire/flood/none
  fuel_subtype       : 若 event_type=fuel_price，细分：natural_gas/coal/oil/uranium/carbon/none
                        （天然气机组多为边际机组，gas 价格几乎即时传导电价；
                         煤/铀传导慢且滞后，碳价仅部分市场适用，需分别聚合）
  sentiment          : 情绪极性，float，[-1, 1]，-1 极度看跌电价，+1 极度看涨电价，0 中性
  severity           : 事件严重度，float，[0, 1]，0 轻微/无实质影响，1 极端严重
                        （与 sentiment 解耦：severity 刻画"事件本身有多大"，
                         sentiment 刻画"对价格是利好还是利空"，避免两者混淆导致失真）
  price_relevance    : 与目标市场电价的关联度，float，[0, 1]，0 完全无关，1 高度相关
  confidence         : LLM 对本次判断的置信度，float，[0, 1]
  summary            : 一句话摘要（<=40 词，英文）
  raw_model          : 使用的模型名（用于追溯）

  注：v2 移除了 price_direction（与 sentiment 高度冗余，sentiment 的正负号
      已经表达方向，保留一个连续值更适合作为数值特征）。

用法:
  # 小样本测试（默认 100 篇，随机种子固定，便于复现）
  python scripts/covariates/12_llm_extract_news_features.py --sample 100

  # 全量跑
  python scripts/covariates/12_llm_extract_news_features.py --all

依赖:
  pip install openai   (DeepSeek 官方 API 兼容 OpenAI SDK)
  export DEEPSEEK_API_KEY=sk-xxxx

模型:
  默认使用 deepseek-v4-flash（原 deepseek-chat，旧名将于 2026-07-24 弃用）。
  Flash 版本价格远低于 Pro（输入1元/输出2元 vs 输入3元/输出6元，每百万tokens），
  且并发上限更高（2500 vs 500），结构化抽取这类中等难度任务用 Flash 即可，无需 Pro。
"""

import os
import sys
import json
import time
import random
import argparse
import concurrent.futures as cf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = ROOT / "data" / "covariates" / "news" / "text" / "articles_text_dedup.jsonl"
OUT_DIR = ROOT / "data" / "covariates" / "news" / "llm_extract"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

EVENT_TYPES = {
    "weather", "supply_shock", "demand_shock",
    "policy_regulation", "fuel_price", "market_operation", "unrelated",
}
WEATHER_SUBTYPES = {
    "heatwave", "coldwave", "storm", "hurricane",
    "drought", "wildfire", "flood", "none",
}
FUEL_SUBTYPES = {"natural_gas", "coal", "oil", "uranium", "carbon", "none"}
VALID_MARKETS = {"ERCOT", "PJM", "CAISO", "NYISO", "US_GENERAL"}

SYSTEM_PROMPT = """You are an energy-market analyst assistant. You read a news article \
and extract structured signals relevant to short-term wholesale electricity price \
forecasting in US power markets (ERCOT / PJM / CAISO / NYISO).

Always respond with a single valid JSON object, no markdown fences, no extra text, \
matching exactly this schema:

{
  "markets_llm": [<subset of ["ERCOT","PJM","CAISO","NYISO","US_GENERAL"]>],
  "event_type": <one of "weather","supply_shock","demand_shock","policy_regulation","fuel_price","market_operation","unrelated">,
  "weather_subtype": <one of "heatwave","coldwave","storm","hurricane","drought","wildfire","flood","none">,
  "fuel_subtype": <one of "natural_gas","coal","oil","uranium","carbon","none">,
  "sentiment": <float in [-1,1], negative=bearish for price, positive=bullish for price, 0=neutral>,
  "severity": <float in [0,1], 0=minor/no real-world impact, 1=extreme/severe event>,
  "price_relevance": <float in [0,1], 0=irrelevant to power prices, 1=highly relevant>,
  "confidence": <float in [0,1], your confidence in this judgment>,
  "summary": <one-sentence summary, <=40 words, English>
}

Guidance:
- markets_llm: infer from article content (place names, grid operators, utilities, states),
  not from any external tag. Use "US_GENERAL" only if no specific market is identifiable
  but the content is still energy/macro relevant. Can be empty list if truly unrelated.
- event_type "unrelated" should be used liberally for articles that have nothing to do
  with energy, weather-driven demand/supply, or power markets (e.g. celebrity news, sports,
  local crime, obituaries that happened to be scraped).
- weather_subtype only meaningful when event_type="weather", otherwise use "none".
- fuel_subtype only meaningful when event_type="fuel_price", otherwise use "none".
  Pick the dominant fuel the article is about. natural_gas is by far the most price-relevant
  for US power markets (marginal generator in many regions); coal/oil/uranium/carbon are slower
  or indirect transmission channels.
- severity is about the real-world magnitude of the event itself (e.g. a minor cold snap vs.
  a historic winter storm causing grid emergencies), independent of tone/wording. Do NOT
  conflate it with sentiment: a severe event can still have uncertain price direction.
- price_relevance should be low (<0.2) for most general news; reserve high values for
  articles that plausibly move short-term wholesale power prices (extreme weather forecasts,
  plant outages, grid emergency alerts, fuel price shocks, major policy changes).
- Be concise and decisive; do not hedge in the JSON fields themselves.
"""

USER_PROMPT_TMPL = """Source: {source}
URL: {url}

Article text (may be truncated):
---
{text}
---

Return only the JSON object described in the system prompt."""

MAX_CHARS = 6000  # 截断正文，控制 token 成本


def load_articles():
    """加载所有 status=ok 的文章"""
    arts = []
    with open(TEXT_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok" and (rec.get("text") or "").strip():
                arts.append(rec)
    return arts


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("请先安装依赖: pip install openai", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("请设置环境变量 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def validate_and_clean(d: dict) -> dict:
    """校验 LLM 输出，字段异常时做兜底修正，不抛异常。"""
    out = {}
    markets = d.get("markets_llm", [])
    if not isinstance(markets, list):
        markets = []
    out["markets_llm"] = [m for m in markets if m in VALID_MARKETS]

    et = d.get("event_type")
    out["event_type"] = et if et in EVENT_TYPES else "unrelated"

    ws = d.get("weather_subtype")
    out["weather_subtype"] = ws if ws in WEATHER_SUBTYPES else "none"

    fs = d.get("fuel_subtype")
    out["fuel_subtype"] = fs if fs in FUEL_SUBTYPES else "none"

    def _clip_float(v, lo, hi, default=0.0):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    out["sentiment"] = _clip_float(d.get("sentiment"), -1.0, 1.0, 0.0)
    out["severity"] = _clip_float(d.get("severity"), 0.0, 1.0, 0.0)
    out["price_relevance"] = _clip_float(d.get("price_relevance"), 0.0, 1.0, 0.0)
    out["confidence"] = _clip_float(d.get("confidence"), 0.0, 1.0, 0.0)

    summary = d.get("summary")
    out["summary"] = summary.strip() if isinstance(summary, str) else ""

    return out


def extract_one(client, art: dict, max_retries: int = 3) -> dict:
    text = art["text"][:MAX_CHARS]
    user_prompt = USER_PROMPT_TMPL.format(
        source=art.get("source", ""), url=art.get("url", ""), text=text
    )

    t0 = time.monotonic()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=60,
            )
            elapsed = time.monotonic() - t0
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            clean = validate_and_clean(parsed)
            usage = getattr(resp, "usage", None)
            clean.update({
                "url": art["url"],
                "source": art.get("source", ""),
                "markets_gdelt": art.get("markets", []),
                "dates": art.get("dates", []),
                "raw_model": MODEL_NAME,
                "llm_status": "ok",
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            })
            return clean
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    elapsed = time.monotonic() - t0
    return {
        "url": art["url"],
        "source": art.get("source", ""),
        "markets_gdelt": art.get("markets", []),
        "dates": art.get("dates", []),
        "raw_model": MODEL_NAME,
        "llm_status": f"error: {last_err}",
        "elapsed_sec": round(elapsed, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="随机抽样 N 篇做小样本测试")
    ap.add_argument("--all", action="store_true", help="跑全量")
    ap.add_argument("--workers", type=int, default=4, help="并发线程数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="输出文件名（默认按模式自动命名）")
    args = ap.parse_args()

    if not args.sample and not args.all:
        print("请指定 --sample N 或 --all")
        sys.exit(1)

    articles = load_articles()
    print(f"可用文章总数: {len(articles)}")

    if args.sample:
        random.seed(args.seed)
        articles = random.sample(articles, min(args.sample, len(articles)))
        out_name = args.out or f"llm_features_sample{args.sample}.jsonl"
    else:
        out_name = args.out or "llm_features_all.jsonl"

    out_path = OUT_DIR / out_name

    # 断点续传：已处理过的 url 跳过
    done_urls = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("llm_status") == "ok":
                        done_urls.add(r["url"])
                except Exception:
                    pass
    todo = [a for a in articles if a["url"] not in done_urls]
    print(f"本次待处理: {len(todo)}（已完成 {len(done_urls)}）")

    if not todo:
        print("全部已完成。")
        return

    client = get_client()

    n_ok, n_err = 0, 0
    per_article_times = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    lock_file = open(out_path, "a", encoding="utf-8")

    batch_t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(extract_one, client, a): a for a in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            result = fut.result()
            lock_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            lock_file.flush()
            if result.get("llm_status") == "ok":
                n_ok += 1
            else:
                n_err += 1
            if result.get("elapsed_sec") is not None:
                per_article_times.append(result["elapsed_sec"])
            if result.get("prompt_tokens"):
                total_prompt_tokens += result["prompt_tokens"]
            if result.get("completion_tokens"):
                total_completion_tokens += result["completion_tokens"]
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] ok={n_ok} err={n_err}")
    batch_wall_sec = time.monotonic() - batch_t0

    lock_file.close()
    print(f"完成。输出: {out_path}")
    print(f"成功: {n_ok}  失败: {n_err}")

    # ── 耗时统计 ─────────────────────────────────────────────────────
    print("\n=== 耗时统计 ===")
    print(f"总墙钟时间（{args.workers} 并发）: {batch_wall_sec:.1f} 秒 "
          f"({batch_wall_sec/60:.2f} 分钟)")
    if per_article_times:
        n = len(per_article_times)
        avg = sum(per_article_times) / n
        srt = sorted(per_article_times)
        print(f"单篇请求耗时: 平均 {avg:.2f}s | 最快 {srt[0]:.2f}s | "
              f"最慢 {srt[-1]:.2f}s | 中位数 {srt[n//2]:.2f}s")
        print(f"（单篇平均耗时之和 = {sum(per_article_times):.1f}s，"
              f"因并发 {args.workers} 路，实际墙钟时间约为其 1/{args.workers}）")
    if total_prompt_tokens or total_completion_tokens:
        print(f"\n=== Token 用量统计（本次 {len(todo)} 篇） ===")
        print(f"输入 tokens: {total_prompt_tokens}  输出 tokens: {total_completion_tokens}")
        if len(todo) > 0:
            print(f"平均每篇: 输入 {total_prompt_tokens/len(todo):.0f} tokens, "
                  f"输出 {total_completion_tokens/len(todo):.0f} tokens")
        # 按 deepseek-v4-flash 价格粗估成本（输入1元/输出2元，每百万tokens，缓存未命中口径）
        est_cost = total_prompt_tokens / 1e6 * 1.0 + total_completion_tokens / 1e6 * 2.0
        print(f"预估成本（按 deepseek-v4-flash 缓存未命中价）: ¥{est_cost:.4f}")


if __name__ == "__main__":
    main()
