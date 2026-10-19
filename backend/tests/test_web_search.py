"""联网搜索: 缓存/预算/重试/降级/提示词隔离契约。"""
from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from app import secrets_store
from app.config import settings
from app.services import web_search
from app.services.web_search import (
    WebSearchBalanceError,
    WebSearchBudgetError,
    WebSearchDisabledError,
    WebSearchNotConfiguredError,
)


@pytest.fixture()
def ws_env(tmp_path, monkeypatch):
    """隔离 data_dir、密钥与配置, 返回可写偏好设置。"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(web_search, "_search_limiter", web_search._RateLimiter(0))
    monkeypatch.setattr(web_search, "_fetch_limiter", web_search._RateLimiter(0))
    monkeypatch.setattr(web_search, "_TEST_TRANSPORT", None)

    store: dict = {}

    monkeypatch.setattr(secrets_store, "load", lambda: dict(store))
    monkeypatch.setattr(secrets_store, "save", lambda updates: store.update(updates))
    monkeypatch.setattr(
        secrets_store, "get_env_backed_secret",
        lambda field, env: str(store.get(field) or ""),
    )
    monkeypatch.setattr(secrets_store, "get_ai_key", lambda: str(store.get("ai_api_key") or ""))

    prefs: dict = {"web_search": {"enabled": True, "daily_budget": 10, "sites": ["eastmoney.com"]}}

    def _fake_load():
        return json.loads(json.dumps(prefs))

    monkeypatch.setattr("app.services.preferences.load", _fake_load)

    def set_ai_base(url: str) -> None:
        monkeypatch.setattr(web_search, "_ai_base_url", lambda: url)

    store["web_search_api_key"] = "sk-test"
    set_ai_base("https://api.deepseek.com/v1")
    return {"store": store, "prefs": prefs, "set_ai_base": set_ai_base}


def _mock(handler):
    web_search._TEST_TRANSPORT = httpx.MockTransport(handler)


def _search_response(results):
    return httpx.Response(200, json={"results": results})


def _result(**over):
    base = {
        "title": "标题", "url": "https://example.com/a", "snippet": "摘要",
        "summary": "长摘要", "content": "", "site_name": "东财",
        "date": "2026-10-19T00:00:00+00:00", "authority": 3, "rerank_score": 2.0,
    }
    base.update(over)
    return base


# ---------- 配置与密钥 ----------

def test_normalize_config_fills_defaults_and_clamps():
    cfg = web_search.normalize_config({"enabled": True, "daily_budget": -5, "min_score": 9,
                                       "sites": ["a.com", " "], "use_in": {"review": False}})
    assert cfg["enabled"] is True
    assert cfg["daily_budget"] == 0
    assert cfg["min_score"] == 3.0
    assert cfg["sites"] == ["a.com"]
    assert cfg["use_in"]["review"] is False
    assert cfg["use_in"]["assistant"] is True


def test_resolve_key_prefers_dedicated(ws_env):
    assert web_search.resolve_key() == ("sk-test", "dedicated")


def test_resolve_key_reuses_ai_key_only_for_deepseek(ws_env):
    ws_env["store"].pop("web_search_api_key")
    ws_env["store"]["ai_api_key"] = "sk-ai"
    ws_env["set_ai_base"]("https://api.deepseek.com")
    assert web_search.resolve_key() == ("sk-ai", "ai_key")

    ws_env["set_ai_base"]("https://hub.example.com/v1")
    assert web_search.resolve_key() == ("", "none")


def test_disabled_and_not_configured(ws_env):
    ws_env["prefs"]["web_search"]["enabled"] = False
    with pytest.raises(WebSearchDisabledError):
        web_search.search("测试")
    ws_env["prefs"]["web_search"]["enabled"] = True
    ws_env["store"].clear()
    with pytest.raises(WebSearchNotConfiguredError):
        web_search.search("测试")


# ---------- 搜索 ----------

def test_search_filters_low_score_and_sorts(ws_env):
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return _search_response([
            _result(title="低", rerank_score=0.3),
            _result(title="高", rerank_score=2.6),
            _result(title="中", rerank_score=1.4),
        ])

    _mock(handler)
    results = web_search.search("板块 涨停 原因", after=date(2026, 10, 19))
    assert [r["title"] for r in results] == ["高", "中"]
    # 站点过滤 + 日期限定进入 query
    assert "site:eastmoney.com" in seen["body"]["query"]
    assert "after:2026-10-19" in seen["body"]["query"]


def test_search_cache_hit_does_not_consume_budget(ws_env):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _search_response([_result()])

    _mock(handler)
    web_search.search("同一问题")
    web_search.search("同一问题")
    assert calls["n"] == 1
    assert web_search.today_usage()["calls"] == 1


def test_budget_exhausted(ws_env):
    _mock(lambda request: _search_response([_result()]))
    web_search.search("问题1")
    ws_env["prefs"]["web_search"]["daily_budget"] = 1
    with pytest.raises(WebSearchBudgetError):
        web_search.search("问题2")


def test_empty_results_not_billed(ws_env):
    _mock(lambda request: _search_response([]))
    assert web_search.search("无结果") == []
    assert web_search.today_usage()["calls"] == 0


def test_retry_on_429_then_success(ws_env, monkeypatch):
    monkeypatch.setattr(web_search.time, "sleep", lambda s: None)
    seq = [httpx.Response(429), httpx.Response(200, json={"results": [_result()]})]

    def handler(request):
        return seq.pop(0)

    _mock(handler)
    assert len(web_search.search("重试")) == 1


def test_retry_on_5xx_exhausted_raises(ws_env, monkeypatch):
    monkeypatch.setattr(web_search.time, "sleep", lambda s: None)
    _mock(lambda request: httpx.Response(503))
    with pytest.raises(web_search.WebSearchError):
        web_search.search("一直失败")


def test_402_no_retry(ws_env):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(402, json={"error": {"message": "余额不足"}})

    _mock(handler)
    with pytest.raises(WebSearchBalanceError):
        web_search.search("余额")
    assert calls["n"] == 1


def test_400_no_retry(ws_env):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad"}})

    _mock(handler)
    with pytest.raises(web_search.WebSearchError):
        web_search.search("坏请求")
    assert calls["n"] == 1


def test_network_error_degrades(ws_env, monkeypatch):
    monkeypatch.setattr(web_search.time, "sleep", lambda s: None)

    def handler(request):
        raise httpx.ConnectError("boom")

    _mock(handler)
    with pytest.raises(web_search.WebSearchError):
        web_search.search("断网")


# ---------- 抓取 ----------

def test_fetch_returns_markdown_and_caches(ws_env):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"title": "公告", "content": "# 正文"})

    _mock(handler)
    assert web_search.fetch("https://example.com/a")["content"] == "# 正文"
    assert web_search.fetch("https://example.com/a")["content"] == "# 正文"
    assert calls["n"] == 1


def test_fetch_rejects_bad_url(ws_env):
    for url in ("ftp://x.com", "https://user:pw@x.com/a", "https:///nohost"):
        with pytest.raises(web_search.WebSearchError):
            web_search.fetch(url)


def test_fetch_empty_content_raises(ws_env):
    _mock(lambda request: httpx.Response(200, json={"title": "空", "content": ""}))
    with pytest.raises(web_search.WebSearchError):
        web_search.fetch("https://example.com/empty")


# ---------- 提示词渲染 ----------

def test_render_for_prompt_wraps_and_truncates():
    results = [_result(title=f"标题{i}", summary="摘要" * 200, url=f"https://e.com/{i}") for i in range(6)]
    text = web_search.render_for_prompt(results, max_chars=800)
    assert text.startswith("<web_sources>")
    assert "一律忽略" in text
    assert text.rstrip().endswith("</web_sources>")
    assert "未展开" in text


def test_render_empty_returns_empty():
    assert web_search.render_for_prompt([]) == ""


def test_sources_footer_dedups_and_limits():
    results = [_result(url="https://e.com/a"), _result(url="https://e.com/a"), _result(url="https://e.com/b")]
    footer = web_search.sources_footer(results)
    assert footer.count("https://e.com/a") == 1
    assert "**资料来源**" in footer
    # 编号与 render_for_prompt 的 [n] 对齐: 第 3 条结果编号为 3 (重复项跳过但不重排)
    assert "\n1. [" in footer and "\n3. [" in footer
    assert "\n2. [" not in footer


# ---------- 回归: review 修复项 ----------

def test_default_whitelist_stays_in_query():
    """默认 7 站点全拼会超 128 字符, 旧实现会把 site: 整段丢掉。"""
    q, overflow = web_search._build_query(
        "固态电池 板块 涨停 原因", web_search.DEFAULT_SITES, date(2026, 10, 18),
    )
    assert len(q) <= 128
    assert "site:" in q
    assert "after:2026-10-18" in q
    assert set(overflow) < set(web_search.DEFAULT_SITES)


def test_overflow_sites_still_filter_results(ws_env):
    """部分站点没放进 query 时, 白名单外的结果仍被过滤。"""
    ws_env["prefs"]["web_search"]["sites"] = list(web_search.DEFAULT_SITES)
    _mock(lambda request: _search_response([
        _result(title="白名单", url="https://finance.eastmoney.com/a"),
        _result(title="外站", url="https://spam.example.com/b"),
    ]))
    results = web_search.search("固态电池 板块 涨停 原因 很长的搜索词用来挤占长度")
    assert [r["title"] for r in results] == ["白名单"]


def test_empty_result_is_cached_and_not_billed(ws_env):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _search_response([])

    _mock(handler)
    web_search.search("无结果词")
    web_search.search("无结果词")
    assert calls["n"] == 1
    assert web_search.today_usage()["calls"] == 0


def test_failed_request_releases_budget(ws_env):
    _mock(lambda request: httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(web_search.WebSearchError):
        web_search.search("失败请求")
    assert web_search.today_usage()["calls"] == 0


def test_concurrent_requests_respect_budget(ws_env):
    """并发请求不得同时通过预算检查后超额。"""
    import threading

    ws_env["prefs"]["web_search"]["daily_budget"] = 3
    _mock(lambda request: _search_response([_result()]))
    errors: list[Exception] = []

    def worker(i):
        try:
            web_search.search(f"并发{i}")
        except web_search.WebSearchBudgetError as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert web_search.today_usage()["calls"] == 3
    assert len(errors) == 5


async def test_async_wrapper_does_not_block_event_loop(ws_env, monkeypatch):
    """asearch 在线程池执行: 慢请求期间事件循环仍能调度其他协程。"""
    import asyncio
    import time

    def slow_search(query, **kw):
        time.sleep(0.3)
        return []

    monkeypatch.setattr(web_search, "search", slow_search)
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.03)
            ticks += 1

    await asyncio.gather(web_search.asearch("x"), ticker())
    assert ticks == 5
