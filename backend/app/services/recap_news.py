"""复盘消息催化 —— 收盘综述 + 热门板块涨停原因检索。

只做「检索 + 组织」, 不调用 LLM。任何失败都返回空结果, 由复盘主流程
降级为「无新闻数据」分支, 不阻断报告生成。

调用预算: 每次复盘最多 4 次 Search (综述 1 + 概念榜前 3 各 1)。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from app.market_time import cn_today
from app.services import web_search

logger = logging.getLogger(__name__)

MAX_QUERIES = 6
MAX_RESULTS = 10


async def collect_recap_news(overview: dict, as_of: date | None) -> list[dict]:
    """按当日盘面检索复盘所需新闻, 返回统一结构的结果列表。

    结果字段与 market_recap._build_user_prompt 的 news 约定一致:
    title / snippet / source / published_date / url。
    """
    cfg = web_search.current_config()
    if not cfg["enabled"] or not cfg["use_in"].get("review", True):
        return []
    if not web_search.key_status()["configured"]:
        return []

    day = as_of or cn_today()
    after = day - timedelta(days=1)  # 覆盖盘前与盘中消息
    queries: list[str] = [f"A股 收盘 综述 {day:%Y-%m-%d}"]

    # 概念榜前 3 的涨停原因
    concept = (overview.get("concept_rank") or {}).get("leading") or []
    for item in concept[:3]:
        name = (item.get("name") or "").strip()
        if name:
            queries.append(f"{name} 板块 涨停 原因")

    results: list[dict] = []
    seen_urls: set[str] = set()
    for query in queries[:MAX_QUERIES]:
        try:
            # 同步 HTTP + 限速 sleep 放到线程池, 不阻塞事件循环 (复盘流运行在 async 生成器里)
            hits = await web_search.asearch(
                query, count=5, after=after, purpose="recap_news",
            )
        except web_search.WebSearchError as e:
            logger.info("recap news search skipped (%s): %s", query, e)
            continue
        for hit in hits:
            url = hit.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append({
                "title": hit.get("title") or "",
                "snippet": hit.get("summary") or hit.get("snippet") or "",
                "source": hit.get("site_name") or "",
                "published_date": (hit.get("date") or "")[:10],
                "url": url,
            })
            if len(results) >= MAX_RESULTS:
                return results
    return results
