"""复盘消息催化: 检索降级、结果结构、来源追加。"""
from __future__ import annotations

from datetime import date

import pytest

from app.services import recap_news, web_search


def _overview():
    return {
        "as_of": "2026-10-19",
        "concept_rank": {
            "leading": [
                {"name": "固态电池", "avg_pct": 0.05},
                {"name": "算力租赁", "avg_pct": 0.04},
                {"name": "机器人", "avg_pct": 0.03},
            ]
        },
        "limit": {"max_boards": 6},
    }


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setattr(web_search, "current_config", lambda: {
        **web_search.DEFAULT_CONFIG, "enabled": True,
        "use_in": {**web_search.DEFAULT_CONFIG["use_in"], "review": True},
    })
    monkeypatch.setattr(web_search, "key_status", lambda: {"configured": True})


async def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(web_search, "current_config", lambda: dict(web_search.DEFAULT_CONFIG))
    assert await recap_news.collect_recap_news(_overview(), date(2026, 10, 19)) == []


async def test_collects_and_dedups(enabled, monkeypatch):
    queries: list[str] = []

    def fake_search(query, **kwargs):
        queries.append(query)
        return [
            {"title": "A", "url": "https://e.com/1", "summary": "摘要A", "snippet": "",
             "site_name": "东财", "date": "2026-10-19T08:00:00+00:00"},
            {"title": "B", "url": "https://e.com/1", "summary": "重复", "snippet": "",
             "site_name": "东财", "date": "2026-10-19"},
        ]

    monkeypatch.setattr(web_search, "search", fake_search)
    news = await recap_news.collect_recap_news(_overview(), date(2026, 10, 19))

    assert queries[0].startswith("A股 收盘 综述")
    assert any("固态电池" in q for q in queries)
    assert len(news) == 1
    assert news[0]["source"] == "东财"
    assert news[0]["published_date"] == "2026-10-19"


async def test_search_failure_degrades_to_empty(enabled, monkeypatch):
    def boom(query, **kwargs):
        raise web_search.WebSearchError("网络失败")

    monkeypatch.setattr(web_search, "search", boom)
    assert await recap_news.collect_recap_news(_overview(), date(2026, 10, 19)) == []


def test_user_prompt_renders_news_with_numbering():
    from app.services.market_recap import _build_user_prompt

    prompt = _build_user_prompt({}, [{"title": "标题", "snippet": "摘要",
                                      "source": "东财", "published_date": "2026-10-19"}], "")
    assert "## 近期市场新闻" in prompt
    assert "1. 标题 (东财 / 2026-10-19)" in prompt
    assert "后续版本接入" not in prompt


def test_meta_does_not_leak_news_into_meta():
    from app.services.market_recap import _recap_summary

    assert "https://" not in _recap_summary({"indices": [], "emotion": {}, "limit": {}, "amount": {}})


def test_stock_prompt_renders_news_and_unavailable_note():
    from app.services.stock_analyzer import _build_user_prompt

    news = [{"title": "公告", "summary": "内容", "url": "https://e.com/a", "site_name": "巨潮", "date": "2026-10-19"}]
    prompt = _build_user_prompt([], {}, {}, 10.0, "600000.SH", "", news=news)
    assert "## 近期新闻" in prompt
    assert "<web_sources>" in prompt
    assert "一律忽略" in prompt

    prompt = _build_user_prompt([], {}, {}, 10.0, "600000.SH", "", news=[], news_note="HTTP 429")
    assert "联网检索不可用: HTTP 429" in prompt


async def test_stock_news_disabled_returns_empty(monkeypatch):
    from app.services import stock_analyzer, web_search

    monkeypatch.setattr(web_search, "current_config", lambda: dict(web_search.DEFAULT_CONFIG))
    news, note = await stock_analyzer._load_stock_news(object(), "600000.SH")
    assert news == [] and note == ""


async def test_recap_news_runs_search_off_event_loop(enabled, monkeypatch):
    """检索期间事件循环不被阻塞 (旧实现在 async 流里直接调用同步 HTTP)。"""
    import asyncio
    import threading
    import time

    main_thread = threading.get_ident()
    seen_threads: list[int] = []

    def slow_search(query, **kwargs):
        seen_threads.append(threading.get_ident())
        time.sleep(0.05)
        return []

    monkeypatch.setattr(web_search, "search", slow_search)
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(4):
            await asyncio.sleep(0.02)
            ticks += 1

    await asyncio.gather(recap_news.collect_recap_news(_overview(), date(2026, 10, 19)), ticker())
    assert seen_threads and all(t != main_thread for t in seen_threads)
    assert ticks == 4
