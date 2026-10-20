"""自选股多周期相对强度: 收益口径、基准对齐、评级与降级。

核心回归: 短期回调但长周期大幅跑赢的标的 (行云科技 300209 那类) 不能被判成弱。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.services import watchlist_strength as ws


def _days(n: int, start: date = date(2026, 1, 1)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _stock_frame(symbol: str, closes: list[float], start: date = date(2026, 1, 1)) -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": [symbol] * len(closes),
        "date": _days(len(closes), start),
        "close": closes,
    }).with_columns(pl.col("date").cast(pl.Date))


def _index_frame(symbol: str, closes: list[float], start: date = date(2026, 1, 1)) -> pl.DataFrame:
    return _stock_frame(symbol, closes, start)


class _Repo:
    """最小仓库桩: 历史 span + 个股区间 + 指数日线。"""

    def __init__(self, stocks: dict[str, pl.DataFrame], indexes: dict[str, pl.DataFrame],
                 span=None, latest=None):
        self._stocks = stocks
        self._indexes = indexes
        self._span = span or (date(2026, 1, 1), date(2026, 6, 30))
        self._latest = latest or self._span[1]

    def get_enriched_history_span(self):
        return self._span

    def get_enriched_latest(self):
        return pl.DataFrame(), self._latest

    def get_enriched_range(self, start, end, symbols=None, columns=None):
        frames = []
        for sym, df in self._stocks.items():
            if symbols is not None and sym not in symbols:
                continue
            sub = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
            if not sub.is_empty():
                frames.append(sub.select(["symbol", "date", "close"]))
        if not frames:
            return None
        return pl.concat(frames)

    def get_index_daily(self, symbol, start, end, columns=None):
        df = self._indexes.get(symbol)
        if df is None:
            return pl.DataFrame()
        return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    store = staticmethod(lambda: None)


@pytest.fixture(autouse=True)
def _clear_cache():
    ws.invalidate_cache()
    yield
    ws.invalidate_cache()


# ---------- 收益 ----------

def test_window_return_exact():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
    assert ws._window_return(closes, 5) == pytest.approx(0.10)
    assert ws._window_return(closes, 90) is None      # 窗口不足


def test_benchmark_alignment_uses_stock_window_dates():
    # 基准日期比个股少一天: 应取 ≤ 起点的最近一根, 而不是错位
    bench = ([date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 6)],
             [100.0, 110.0, 120.0])
    # 起点 1/2 取 ≤ 该日的最近一根 (1/1 = 100), 终点 1/6 = 120 → +20%
    assert ws._benchmark_return(bench, date(2026, 1, 2), date(2026, 1, 6)) == pytest.approx(0.20)
    # 起点早于基准首日: 没有可用起点, 返回 None 而不是用首日强行外推
    assert ws._benchmark_return(bench, date(2025, 12, 1), date(2026, 1, 6)) is None


# ---------- 评级 ----------

def test_rating_boundaries():
    horizons = {20, 60, 120}
    assert ws._rating(0.15, horizons) == "极强"
    assert ws._rating(0.149, horizons) == "强"
    assert ws._rating(0.05, horizons) == "强"
    assert ws._rating(0.049, horizons) == "中性"
    assert ws._rating(-0.05, horizons) == "弱"        # 恰好 -5pp 归入弱 (区间左闭)
    assert ws._rating(-0.145, horizons) == "弱"
    assert ws._rating(-0.051, horizons) == "弱"
    assert ws._rating(-0.15, horizons) == "极弱"      # 恰好 -15pp 归入极弱 (区间左闭)
    assert ws._rating(-0.149, horizons) == "弱"
    assert ws._rating(None, horizons) == "—"


def test_rating_capped_when_only_short_window():
    """只有 5 日数据时不得给出极强/极弱 (短期噪声)。"""
    assert ws._rating(0.40, {5}) == "中性"
    assert ws._rating(-0.40, {5}) == "中性"


def test_tags():
    # 60 日绝对与超额都强 → 逆势强势
    assert ws._tag({60: 0.20, 20: -0.05}, {60: 0.20, 20: -0.02}) == "逆势强势"
    # 120 日更强时也命中
    assert ws._tag({120: 0.30}, {120: 0.25}) == "逆势强势"
    # 短期下跌但显著跑赢 → 相对抗跌
    assert ws._tag({20: -0.10}, {20: 0.12}) == "相对抗跌"
    # 绝对涨但没跑赢 → 无标签
    assert ws._tag({60: 0.05}, {60: -0.02}) == ""


def test_weighted_mean_renormalizes_missing_windows():
    value = ws._weighted_mean({5: None, 20: 0.10, 60: 0.20, 120: None})
    assert value == pytest.approx((0.10 * 0.3 + 0.20 * 0.3) / 0.6)


# ---------- 端到端 ----------

def _flat(n: int, price: float = 100.0) -> list[float]:
    return [price] * n


def test_reverse_strong_stock_is_not_weak():
    """当日下跌、20 日下跌, 但 60/120 日大幅跑赢 → 逆势强势 + 极强。"""
    n = 140
    # 个股: 120 天稳步上涨 (100 → ~400), 最近 20 天回落到 ~300
    closes = [100 * (1.012 ** i) for i in range(120)] + [400 * (0.986 ** i) for i in range(20)]
    stock = _stock_frame("300209.SZ", closes)
    index = _index_frame("399006.SZ", [3000 - i * 8 for i in range(n)])
    repo = _Repo({"300209.SZ": stock}, {"399006.SZ": index},
                 span=(date(2026, 1, 1), date(2026, 6, 1)), latest=date(2026, 6, 1))

    result = ws.build_strength(repo, date(2026, 6, 1), ["300209.SZ"], {})
    stock_out = result["stocks"]["300209.SZ"]
    assert stock_out["abs"]["120"] > 1.0
    assert stock_out["excess_index"]["120"] > 1.0
    assert stock_out["tag"] == "逆势强势"
    assert stock_out["rating"] == "极强"
    assert stock_out["index_symbol"] == "399006.SZ"


def test_cache_span_clamping_nulls_long_windows():
    """span 只有 40 个交易日: 60/120 返回 None, 不抛异常且带覆盖提示。"""
    closes = [100 + i for i in range(40)]
    stock = _stock_frame("600001.SH", closes)
    index = _index_frame("000300.SH", [3000 + i for i in range(40)])
    span = (date(2026, 1, 1), date(2026, 2, 9))
    repo = _Repo({"600001.SH": stock}, {"000300.SH": index}, span=span, latest=span[1])

    result = ws.build_strength(repo, span[1], ["600001.SH"], {})
    item = result["stocks"]["600001.SH"]
    assert item["abs"]["20"] is not None
    assert item["abs"]["60"] is None and item["abs"]["120"] is None
    assert result["coverage"]["horizons"]["60"] is False
    assert "60" in result["coverage"]["note"]


def test_no_sector_membership_yields_null_sector_excess(monkeypatch):
    stock = _stock_frame("600001.SH", [100 + i for i in range(40)])
    index = _index_frame("000300.SH", [3000 + i for i in range(40)])
    repo = _Repo({"600001.SH": stock}, {"000300.SH": index})

    monkeypatch.setattr(
        "app.services.market_overview_builder.sector_member_map",
        lambda *a, **k: {},
    )
    result = ws.build_strength(repo, date(2026, 2, 9), ["600001.SH"],
                               {"600001.SH": {"name": "冷门概念", "kind": "concept"}})
    item = result["stocks"]["600001.SH"]
    assert item["excess_sector"]["20"] is None
    assert item["sector_members"] is None
    # 对指数仍有数据 → 评级仍可用
    assert item["rating"] != "—"


def test_small_sector_is_null(monkeypatch):
    members = {f"{i:06d}.SH" for i in range(5)}
    frames = {f"{i:06d}.SH": _stock_frame(f"{i:06d}.SH", [100 + i for i in range(40)])
              for i in range(5)}
    index = _index_frame("000300.SH", [3000 + i for i in range(40)])
    repo = _Repo(frames, {"000300.SH": index})
    monkeypatch.setattr(
        "app.services.market_overview_builder.sector_member_map",
        lambda *a, **k: {"小板块": members},
    )
    result = ws.build_strength(repo, date(2026, 2, 9), ["000000.SH"],
                               {"000000.SH": {"name": "小板块", "kind": "concept"}})
    bench = result["sector_bench"]["小板块"]
    assert bench["members"] == 5
    assert bench["ret"]["20"] is None


def test_duplicate_sector_keys_not_double_counted(monkeypatch):
    """sector_member_map 同时给带后缀与裸代码, 去重后成员数与均值都不能翻倍。"""
    keys = {"600001.SH", "600001", "600002.SH", "600002"}
    monkeypatch.setattr(
        "app.services.market_overview_builder.sector_member_map",
        lambda *a, **k: {"板块A": keys},
    )
    deduped = ws._dedup_members(keys)
    assert deduped == ["600001.SH", "600002.SH"]


def test_multiple_sectors_both_computed(monkeypatch):
    """同一 kind 下多只自选分属不同板块时, 两个板块都要算 (旧实现会互相覆盖)。"""
    symbols = ["600001.SH", "600002.SH"]
    frames = {}
    for s in symbols:
        frames[s] = _stock_frame(s, [100 + i for i in range(40)])
    for prefix in ("6000",):
        for i in range(10, 20):
            frames[f"{prefix}{i:02d}.SH"] = _stock_frame(f"{prefix}{i:02d}.SH", [100 + i for i in range(40)])
    index = _index_frame("000300.SH", [3000 + i for i in range(40)])
    repo = _Repo(frames, {"000300.SH": index})
    monkeypatch.setattr(
        "app.services.market_overview_builder.sector_member_map",
        lambda *a, **k: {
            "板块A": {"600001.SH", "600010.SH", "600011.SH", "600012.SH", "600013.SH",
                    "600014.SH", "600015.SH", "600016.SH", "600017.SH", "600018.SH", "600019.SH"},
            "板块B": {"600002.SH", "600010.SH", "600011.SH", "600012.SH", "600013.SH",
                    "600014.SH", "600015.SH", "600016.SH", "600017.SH", "600018.SH", "600019.SH"},
        },
    )
    result = ws.build_strength(
        repo, date(2026, 2, 9), symbols,
        {s: {"name": name, "kind": "concept"} for s, name in zip(symbols, ("板块A", "板块B"), strict=True)},
    )
    assert set(result["sector_bench"]) == {"板块A", "板块B"}
    assert result["stocks"]["600001.SH"]["sector"] == "板块A"
    assert result["stocks"]["600002.SH"]["sector"] == "板块B"


def test_empty_inputs_return_empty_payload():
    repo = _Repo({}, {})
    payload = ws.build_strength(repo, None, [], {})
    assert payload["stocks"] == {}
    assert payload["coverage"]["note"] == "无历史数据"


# ---------- 与复盘/个股分析的接口契约 ----------

def test_digest_payload_carries_benchmarks_and_coverage():
    from app.services.market_recap import _digest_payload

    digest = {
        "as_of": "2026-10-20",
        "cycle_phase": "分歧",
        "total": 1,
        "detail": [{
            "symbol": "300209.SZ", "name": "行云科技", "change_pct": -0.0183,
            "abs": {"20": -0.068, "120": 1.63}, "excess_index": {"20": 0.098},
            "excess_sector": {"20": -0.03}, "rps": {"60": 93.9},
            "strength_rating": "极强", "strength_tag": "逆势强势",
            "board": "创业板", "index_symbol": "399006.SZ", "index_name": "创业板指",
        }],
        "benchmarks": {"399006.SZ": {"name": "创业板指", "ret": {"20": -0.17}}},
        "sector_bench": {"物联网": {"members": 473, "ret": {"20": -0.04}}},
        "strength_coverage": {"note": ""},
        "excluded": [{"symbol": "510300.SH", "reason": "ETF"}],
    }
    payload = _digest_payload(digest)
    assert payload["benchmarks"]["399006.SZ"]["name"] == "创业板指"
    assert payload["sector_bench"]["物联网"]["members"] == 473
    assert payload["excluded"][0]["symbol"] == "510300.SH"
    detail = payload["detail"][0]
    for key in ("abs", "excess_index", "excess_sector", "rps", "strength_rating",
                "strength_tag", "index_name"):
        assert key in detail


def test_prompt_forbids_short_window_weak_calls():
    from app.services.market_recap import _WATCHLIST_SYSTEM_PROMPT

    assert "禁止仅凭当日或 5 日涨跌" in _WATCHLIST_SYSTEM_PROMPT
    assert "逆势强势" in _WATCHLIST_SYSTEM_PROMPT
    assert "175pp" not in _WATCHLIST_SYSTEM_PROMPT   # 示例数字不是硬编码结论


def test_board_benchmark_mapping():
    from app.services.index_const import board_benchmark

    assert board_benchmark("300209.SZ")[:2] == ("399006.SZ", "创业板指")
    assert board_benchmark("301001.SZ")[:2] == ("399006.SZ", "创业板指")
    assert board_benchmark("688001.SH")[:2] == ("000680.SH", "科创综指")
    assert board_benchmark("600519.SH")[:2] == ("000300.SH", "沪深300")
    assert board_benchmark("920196.BJ")[:2] == ("000300.SH", "沪深300")
    assert board_benchmark("920196.BJ")[2] is False     # 北交所基准不同类


def test_stock_benchmark_uses_board_index(monkeypatch):
    """个股分析的基准随板块走, 不再是固定上证指数。"""
    from app.services import stock_analyzer as sa

    calls: list[str] = []

    class _Repo:
        def get_index_daily(self, symbol, start, end, columns=None):
            calls.append(symbol)
            if symbol != "399006.SZ":
                raise AssertionError(f"应使用创业板指, 实际 {symbol}")
            return pl.DataFrame({"date": _days(80), "close": [1000 + i for i in range(80)]})

    df = pl.DataFrame({"date": _days(80), "close": [10 + i * 0.1 for i in range(80)]})
    out = sa._benchmark_returns(_Repo(), df, "300209.SZ")
    assert "399006.SZ" in calls
    assert out["benchmark_name"] == "创业板指"
    assert out["ret20"] is not None and out["ret60"] is not None
