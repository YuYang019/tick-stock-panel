"""盘中递推状态 (live_agg) 的窗口必须真有足够的交易日 K 线。

_build_live_agg 按「自然日 180 天」切历史再 tail(119)/tail(59)/tail(60) 取部分和、N 日前收盘:

- 春节/国庆长假前后, 180 个自然日可能不足 120 个交易日, 全市场的盘中 MA120
  用残缺窗口算出, 与盘后全量 (rolling_mean(120)) 不一致;
- 次新股历史不足窗口长度时, tail(N) 取到的是全部已有 K 线, 盘中 MA120/MA60 等于
  (已有收盘价之和 + 今收) / N 这类残缺值, 而盘后全量在窗口不满时为空。
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from app.indicators.pipeline import compute_enriched_today, compute_indicators
from app.tickflow.repository import (
    _LIVE_AGG_CALENDAR_DAYS,
    _LIVE_AGG_WINDOW_BARS,
    DataStore,
    KlineRepository,
    _live_agg_window_start,
)

TODAY = date(2025, 3, 3)
# 2024 国庆 10/1~10/7 + 2025 元旦 + 春节休市 1/28~2/4:
# 2024-09-01 ~ 2025-02-28 (180 自然日) 只有 118 个交易日; 2024-11-30 起 90 自然日只有 58 个
CLOSED = (
    {date(2024, 10, 1) + timedelta(days=i) for i in range(7)}
    | {date(2025, 1, 1)}
    | {date(2025, 1, 28) + timedelta(days=i) for i in range(8)}
)

VALUE_COLUMNS = (
    "ma5", "ma10", "ma20", "ma30", "ma60", "ma120",
    "momentum_5d", "momentum_10d", "momentum_20d", "momentum_30d", "momentum_60d",
    "vol_ma5", "vol_ma10", "vol_ratio_5d",
)
# 以下列在窗口不满时全量为空; 数值口径另有独立问题, 这里只核对空与非空
NULLNESS_COLUMNS = ("high_60d", "low_60d", "boll_upper", "boll_lower", "annual_vol_20d")


def _history_days() -> list[date]:
    days: list[date] = []
    d = TODAY - timedelta(days=299)
    while d < TODAY:
        if d.weekday() < 5 and d not in CLOSED:
            days.append(d)
        d += timedelta(days=1)
    return days


def _bars() -> pl.DataFrame:
    days = _history_days()
    rows = []
    listings = {
        "600001.SH": days,       # 老股: 约 200 个历史交易日
        "000002.SZ": days,
        "001003.SZ": days[-12:],  # 次新股: 12 个历史交易日
        "001004.SZ": days[-40:],  # 次新股: 40 个历史交易日
        "001005.SZ": days[-119:],  # 119 根历史 + 今日 = 120: MA120 恰好满窗
        "001006.SZ": days[-118:],  # 118 根历史 + 今日 = 119: MA120 差一根, 为空
    }
    for n, (symbol, history) in enumerate(listings.items()):
        for i, day in enumerate([*history, TODAY]):
            close = round(10.0 + n + 0.6 * math.sin(i * 0.37 + n) + 0.004 * i, 2)
            volume = 1000.0 + 300.0 * math.cos(i * 0.51 + n)
            rows.append((symbol, day, close, close + 0.08, close - 0.08, close, volume))
    return pl.DataFrame(
        rows,
        schema=["symbol", "date", "open", "high", "low", "close", "volume"],
        orient="row",
    ).with_columns(
        (pl.col("close") * pl.col("volume")).alias("amount"),
        pl.col("close").alias("raw_close"),
        pl.col("high").alias("raw_high"),
        pl.col("low").alias("raw_low"),
    ).sort(["symbol", "date"])


def _incremental(repo: KlineRepository, bars: pl.DataFrame, latest: date) -> pl.DataFrame:
    history_indicators = compute_indicators(bars.filter(pl.col("date") < TODAY))
    repo._build_live_agg(latest)
    return compute_enriched_today(
        repo.get_live_agg(),
        history_indicators.filter(pl.col("date") == latest),
        bars.filter(pl.col("date") == TODAY).drop("raw_close", "raw_high", "raw_low"),
        None,
    ).sort("symbol")


def _assert_matches_full(incremental: pl.DataFrame, bars: pl.DataFrame) -> None:
    full = compute_indicators(bars).filter(pl.col("date") == TODAY).sort("symbol")
    assert incremental["symbol"].to_list() == full["symbol"].to_list()
    mismatches: list[str] = []
    for column in (*VALUE_COLUMNS, *NULLNESS_COLUMNS):
        pairs = zip(
            full["symbol"].to_list(), incremental[column].to_list(), full[column].to_list(),
            strict=True,
        )
        for symbol, got, want in pairs:
            if (got is None) != (want is None):
                mismatches.append(f"{symbol} {column}: 盘中={got} 全量={want}")
            elif column in VALUE_COLUMNS and got is not None and got != pytest.approx(want, rel=1e-9):
                mismatches.append(f"{symbol} {column}: 盘中={got:.6f} 全量={want:.6f}")
    assert mismatches == []


def test_fixture_holiday_window_has_fewer_than_120_trading_days():
    days = _history_days()
    latest = days[-1]
    in_window = [d for d in days if d >= latest - timedelta(days=_LIVE_AGG_CALENDAR_DAYS)]
    # 构造前提: 长假让 180 个自然日不足 120 个交易日
    assert len(in_window) < _LIVE_AGG_WINDOW_BARS == 120


def test_window_start_counts_trading_days():
    days = _history_days()
    latest = days[-1]
    calendar_start = latest - timedelta(days=_LIVE_AGG_CALENDAR_DAYS)
    dates = pl.Series("date", days)

    # 长假: 自然日起点只含 118 个交易日 → 前移到第 120 个交易日
    assert _live_agg_window_start(dates, latest, calendar_start) == days[-120]
    assert _live_agg_window_start(dates, latest, calendar_start) < calendar_start
    # 平常: 自然日起点已覆盖 120 个交易日 → 保持自然日起点 (窗口不缩小)
    assert _live_agg_window_start(dates, latest, days[-125]) == days[-125]
    # 本地不足 120 个交易日 → 退回自然日起点
    assert _live_agg_window_start(dates.tail(80), latest, calendar_start) == calendar_start


def test_live_agg_from_history_cache_matches_full(tmp_path):
    bars = _bars()
    history = bars.filter(pl.col("date") < TODAY)
    repo = KlineRepository(DataStore(tmp_path))
    repo._enriched_history_cache = compute_indicators(history)

    incremental = _incremental(repo, bars, history["date"].max())

    _assert_matches_full(incremental, bars)


def test_live_agg_from_parquet_fallback_matches_full(tmp_path):
    bars = _bars()
    history = bars.filter(pl.col("date") < TODAY)
    target = tmp_path / "kline_daily_enriched" / "all"
    target.mkdir(parents=True)
    history.write_parquet(target / "part.parquet")
    repo = KlineRepository(DataStore(tmp_path))
    assert repo._enriched_history_cache is None  # 走 _build_live_agg_from_parquet 降级路径

    incremental = _incremental(repo, bars, history["date"].max())

    _assert_matches_full(incremental, bars)


def test_ma120_boundary_listing_days(tmp_path):
    """次新股恰好 120 根 (含今日) 时 MA120 有值, 119 根时为空, 与全量 rolling_mean(120) 同口径。"""
    bars = _bars()
    history = bars.filter(pl.col("date") < TODAY)
    repo = KlineRepository(DataStore(tmp_path))
    repo._enriched_history_cache = compute_indicators(history)

    incremental = _incremental(repo, bars, history["date"].max())
    by_symbol = dict(zip(incremental["symbol"], incremental["ma120"], strict=True))

    full_closes = bars.filter(pl.col("symbol") == "001005.SZ")["close"]
    assert full_closes.len() == 120
    assert by_symbol["001005.SZ"] == pytest.approx(full_closes.mean(), rel=1e-12)
    assert by_symbol["001006.SZ"] is None
    assert by_symbol["001004.SZ"] is None
    assert by_symbol["600001.SH"] is not None
