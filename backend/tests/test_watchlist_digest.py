"""复盘·自选股事实包: 排序、截断、角色标签、降级与表格。"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from app.services import watchlist_digest as wd


def _df(rows: list[dict]) -> pl.DataFrame:
    base = {
        "close": 10.0, "ma5": 10.0, "ma20": 9.5, "ma60": 9.0,
        "change_pct": 0.02, "turnover_rate": 3.0, "amplitude": 4.0,
        "amount": 1e8, "consecutive_limit_ups": 0,
        "high_60d": 12.0, "low_60d": 8.0, "vol_ratio_5d": 1.2,
        "macd_hist": 0.1, "rsi_14": 55.0,
    }
    return pl.DataFrame([{**base, **r} for r in rows])


class _Repo:
    def __init__(self, df, latest=date(2026, 10, 19)):
        self._df = df
        self._latest = latest

    def get_enriched_latest(self):
        return self._df, self._latest

    def get_name_map(self, symbols):
        return {s: f"名称{s[-1]}" for s in symbols}

    def get_enriched_range(self, start, end, symbols=None, columns=None):
        return None


@pytest.fixture()
def patched(monkeypatch):
    entries = [
        {"symbol": "600001.SH", "group_ids": ["g1"]},
        {"symbol": "600002.SH", "group_ids": ["g2"]},
        {"symbol": "600003.SH", "group_ids": ["g1"]},
    ]
    monkeypatch.setattr("app.services.watchlist.list_symbols", lambda: list(entries))
    from app.services import web_search
    monkeypatch.setattr(web_search, "current_config", lambda: dict(web_search.DEFAULT_CONFIG))
    return entries


def test_disabled_returns_reason(patched):
    out = wd.build_watchlist_digest(_Repo(_df([])), date(2026, 10, 19), config={"enabled": False})
    assert out["enabled"] is False and out["rows"] == []


def test_group_filter(patched):
    df = _df([
        {"symbol": "600001.SH"}, {"symbol": "600002.SH"}, {"symbol": "600003.SH"},
    ])
    out = wd.build_watchlist_digest(
        _Repo(df), date(2026, 10, 19), config={"enabled": True, "group_ids": ["g1"]},
    )
    assert {r["symbol"] for r in out["rows"]} == {"600001.SH", "600003.SH"}


def test_sorted_by_alert_score_and_truncated(patched):
    df = _df([
        {"symbol": "600001.SH", "change_pct": 0.001},
        {"symbol": "600002.SH", "change_pct": 0.095, "signal_limit_up": True},
        {"symbol": "600003.SH", "change_pct": -0.02},
    ])
    out = wd.build_watchlist_digest(
        _Repo(df), date(2026, 10, 19),
        config={"enabled": True, "max_rows": 2, "max_detail": 1, "news_top": 0},
    )
    assert out["rows"][0]["symbol"] == "600002.SH"
    assert len(out["rows"]) == 2
    assert out["truncated"] == 1
    assert len(out["detail"]) == 1


def test_role_and_gate_labels(patched):
    df = _df([
        {"symbol": "600001.SH", "consecutive_limit_ups": 3},
        {"symbol": "600002.SH", "close": 11.9},  # 60日区间高位
    ])
    overview = {"industry_rank": {"leading": [], "lagging": []}}
    out = wd.build_watchlist_digest(
        _Repo(df), date(2026, 10, 19), overview=overview,
        config={"enabled": True, "news_top": 0},
    )
    roles = {r["symbol"]: r["role"] for r in out["rows"]}
    assert roles["600001.SH"] == "3连板"
    assert roles["600002.SH"] == "高位"


def test_consensus_states():
    assert wd._consensus_label({"change_pct": 0.1, "turnover_rate": 1.0, "consecutive_limit_ups": 1}) == "一致加速"
    # amplitude 为小数口径 (0.09 = 9%)
    assert wd._consensus_label({"change_pct": -0.03, "turnover_rate": 12.0, "amplitude": 0.09}) == "分歧转弱"
    assert wd._consensus_label({"change_pct": 0.02, "turnover_rate": 12.0, "amplitude": 0.09}) == "分歧换手"
    # 百分数口径的 9.0 若被误当成 9% 会让下一行也命中; 3% 振幅不应判为分歧
    assert wd._consensus_label({"change_pct": 0.02, "turnover_rate": 12.0, "amplitude": 0.03}) == "中性"
    assert wd._consensus_label({"change_pct": 0.01, "turnover_rate": 3.0, "amplitude": 3.0}) == "中性"


def test_empty_watchlist(monkeypatch):
    monkeypatch.setattr("app.services.watchlist.list_symbols", lambda: [])
    out = wd.build_watchlist_digest(_Repo(_df([])), date(2026, 10, 19), config={"enabled": True})
    assert out["rows"] == [] and out["detail"] == []


def test_table_contains_columns_and_truncation_note():
    digest = {
        "rows": [
            {"symbol": "600001.SH", "name": "名称1", "change_pct": 0.02, "rs5": 0.01,
             "strength_rating": "极强", "strength_tag": "逆势强势",
             "excess_index": {"20": 0.068}, "excess_sector": {"20": 0.031},
             "trend": "多头", "role": "首板·热门", "sector_gate": "板块强势",
             "consensus": "中性", "signals": ["放量"]},
        ],
        "truncated": 2,
    }
    table = wd.format_table(digest)
    assert "| 代码 | 名称 |" in table
    assert "+2.00%" in table
    # 保留 5 日列 + 新增相对强度与 20 日超额两列
    assert "5日相对强弱" in table and "相对强度" in table and "20日超额(指/板)" in table
    assert "极强·逆势强势" in table
    assert "+6.80%/+3.10%" in table
    assert "另有 2 只" in table
    assert wd.format_table({"rows": []}) == ""


def test_table_renders_missing_strength_as_dash():
    digest = {"rows": [{
        "symbol": "600001.SH", "name": "名称1", "change_pct": -0.01,
        "rs5": None, "strength_rating": "—", "excess_index": {}, "excess_sector": {},
    }]}
    table = wd.format_table(digest)
    assert "— | — | —/—" in table


def test_normalize_config_clamps():
    cfg = wd.normalize_config({"max_rows": 1000, "max_detail": 0, "news_top": 99})
    assert cfg["max_rows"] == 100
    assert cfg["max_detail"] == 1
    assert cfg["news_top"] == 10


@pytest.mark.asyncio
async def test_recap_second_stage_skips_when_no_watchlist(monkeypatch):
    """自选为空: 只追加一行提示, 不调用 LLM。"""
    from app.services import market_recap, preferences

    monkeypatch.setattr(preferences, "get_review_watchlist", lambda: {"enabled": True})
    monkeypatch.setattr(
        "app.services.watchlist_digest.build_watchlist_digest",
        lambda *a, **k: {"enabled": True, "rows": [], "reason": "自选股为空"},
    )
    called = {"n": 0}

    def fake_stream(*a, **k):
        called["n"] += 1
        raise AssertionError("不应调用 LLM")

    monkeypatch.setattr("app.services.ai_provider.stream_ai_text", fake_stream)
    events = [
        e async for e in market_recap._watchlist_section_stream(None, {"as_of": "2026-10-19"}, None, "2026-10-19")
    ]
    assert len(events) == 1
    assert "本次跳过" in events[0]
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_recap_second_stage_failure_appends_note(monkeypatch):
    from app.services import market_recap, preferences

    monkeypatch.setattr(preferences, "get_review_watchlist", lambda: {"enabled": True})

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.services.watchlist_digest.build_watchlist_digest", boom)
    events = [
        e async for e in market_recap._watchlist_section_stream(None, {"as_of": "2026-10-19"}, None, "2026-10-19")
    ]
    assert "自选股复盘生成失败" in events[0]


@pytest.mark.asyncio
async def test_recap_second_stage_streams_header_and_llm(monkeypatch):
    from app.services import market_recap, preferences

    monkeypatch.setattr(preferences, "get_review_watchlist", lambda: {"enabled": True})
    digest = {
        "enabled": True,
        "as_of": "2026-10-19",
        "total": 1,
        "rows": [{"symbol": "600001.SH", "name": "名称1", "change_pct": 0.02, "rs5": None,
                  "trend": "多头", "role": "—", "sector_gate": "—", "consensus": "中性",
                  "signals": []}],
        "detail": [{"symbol": "600001.SH", "name": "名称1", "change_pct": 0.02}],
        "news": {},
    }
    monkeypatch.setattr(
        "app.services.watchlist_digest.build_watchlist_digest", lambda *a, **k: digest,
    )

    async def fake_stream(messages, **kwargs):
        yield "解读正文"

    monkeypatch.setattr("app.services.ai_provider.stream_ai_text", fake_stream)
    events = [
        e async for e in market_recap._watchlist_section_stream(None, {"as_of": "2026-10-19"}, None, "2026-10-19")
    ]
    kinds = [__import__("json").loads(e)["type"] for e in events]
    assert kinds == ["section_start", "delta", "delta"]
    assert "九、自选股复盘" in events[1]
    assert "解读正文" in events[2]


def test_sector_gate_matches_engine_semantics():
    assert wd._sector_gate(None) == "—"
    assert wd._sector_gate({"rs_pct": 0.9, "phase": "冰点"}) == "板块强势·冰点"
    assert wd._sector_gate({"rs_pct": 0.5, "phase": "退潮"}) == "板块走弱·退潮"
    assert wd._sector_gate({"rs_pct": 0.1, "phase": None}) == "板块走弱"
    assert wd._sector_gate({"rs_pct": 0.5, "phase": "修复"}) == "板块中性·修复"
