"""enriched 历史缓存首段窗口指标不能为空。

缓存按 latest-300 自然日构建, get_daily / 策略按「缓存起点 ≤ 请求起点」直接复用缓存。
若只读 300 天就算指标, 缓存首段约 120 个交易日 ma120 (以及 ma60/60 日动量的首段)
为空 —— 本地明明有更早的历史, K 线图上 MA120 却缺一大段。构建时须多读一段预热再裁回。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.indicators.pipeline import compute_indicators
from app.tickflow.repository import DataStore, KlineRepository

LATEST = date(2026, 8, 3)


def _write_history(root, days: int = 600) -> pl.DataFrame:
    rows = []
    d = LATEST - timedelta(days=days)
    i = 0
    while d <= LATEST:
        if d.weekday() < 5:
            close = round(10.0 + 0.01 * i + 0.3 * ((i % 7) - 3), 4)
            rows.append(("600001.SH", d, close, close + 0.1, close - 0.1, close))
            i += 1
        d += timedelta(days=1)
    bars = pl.DataFrame(rows, schema=["symbol", "date", "open", "high", "low", "close"], orient="row").with_columns(
        pl.lit(1000.0).alias("volume"),
        (pl.col("close") * 1000.0).alias("amount"),
        pl.col("close").alias("raw_close"),
        pl.col("high").alias("raw_high"),
        pl.col("low").alias("raw_low"),
    )
    for day, part in bars.group_by("date"):
        target = root / "kline_daily_enriched" / f"date={day[0].isoformat()}"
        target.mkdir(parents=True, exist_ok=True)
        part.write_parquet(target / "part.parquet")
    return bars.sort("date")


def test_history_cache_has_warmed_up_ma120_from_its_first_row(tmp_path):
    bars = _write_history(tmp_path)
    repo = KlineRepository(DataStore(tmp_path))
    repo._refresh_enriched()

    cache = repo._enriched_history_cache
    assert cache is not None and not cache.is_empty()
    start = repo._enriched_history_start
    assert start >= LATEST - timedelta(days=300)  # 缓存范围不变, 只是预热更足

    full = compute_indicators(bars).filter(pl.col("date") >= start).select("date", "ma120")
    got = cache.filter(pl.col("symbol") == "600001.SH").sort("date").select("date", "ma120")
    assert got["ma120"].null_count() == 0
    assert got["date"].to_list() == full["date"].to_list()
    assert got["ma120"].to_list() == pytest.approx(full["ma120"].to_list(), rel=1e-9)
