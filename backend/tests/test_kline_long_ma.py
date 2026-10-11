"""日K MA120 来自 enriched 指标管道 (单一来源): /api/kline/daily 行内即 ma120 列。

口径同 rolling_mean(120) (前复权收盘价, 窗口不满为 null); 当日实时蜡烛的 ma120
取自盘中 enriched (live_agg 部分和递推), 与 MA5~MA60 一致。
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.kline as kline_module
from app.api.kline import router
from app.tickflow.repository import KlineRepository

SYM = "600001.SH"


def _raw_frame(days: int) -> pl.DataFrame:
    rows = []
    d = date(2025, 1, 6)
    while len(rows) < days:
        if d.weekday() < 5:
            close = round(10.0 + 0.5 * ((len(rows) % 9) - 4) / 4 + 0.01 * len(rows), 4)
            rows.append({
                "symbol": SYM, "date": d,
                "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                "volume": 1.0e4, "amount": 1.0e7,
                "raw_close": close, "raw_high": close * 1.01, "raw_low": close * 0.99,
            })
        d += timedelta(days=1)
    return pl.DataFrame(rows).sort(["symbol", "date"])


def _client(raw: pl.DataFrame, live_row: dict | None = None) -> TestClient:
    repo = KlineRepository.__new__(KlineRepository)
    repo._enriched_history_cache = None
    repo._enriched_history_start = None
    repo.get_instruments = lambda: pl.DataFrame()  # type: ignore[method-assign]
    repo.get_historical_shares = lambda: pl.DataFrame()  # type: ignore[method-assign]
    repo.get_enriched_latest = lambda: (pl.DataFrame(), None)  # type: ignore[method-assign]
    repo.resolve_asset_type = lambda symbol: "stock"  # type: ignore[method-assign]
    repo._scan_daily_symbol = (  # type: ignore[method-assign]
        lambda symbol, start, end, columns: raw.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )
    )

    app = FastAPI()
    app.include_router(router)
    app.state.repo = repo
    if live_row is not None:
        live_date = live_row["date"]
        app.state.quote_service = SimpleNamespace(
            get_enriched_today=lambda: (pl.DataFrame([live_row]), live_date),
        )
    return TestClient(app)


def test_daily_rows_ma120_matches_rolling_mean_with_warmup():
    raw = _raw_frame(300)
    dates = raw["date"].to_list()
    start, end = dates[200], dates[-1]

    resp = _client(raw).get(
        "/api/kline/daily",
        params={"symbol": SYM, "start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    expected = (
        raw.with_columns(pl.col("close").rolling_mean(120).alias("ma120"))
        .filter(pl.col("date") >= start)["ma120"].to_list()
    )
    assert len(rows) == len(expected)
    assert [r["ma120"] for r in rows] == pytest.approx(expected, rel=1e-12)


def test_daily_rows_ma120_null_until_window_full():
    raw = _raw_frame(100)
    dates = raw["date"].to_list()

    resp = _client(raw).get(
        "/api/kline/daily",
        params={"symbol": SYM, "start_date": dates[0].isoformat(), "end_date": dates[-1].isoformat()},
    )
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert rows and all(r["ma120"] is None for r in rows)


def test_live_candle_carries_intraday_ma120(monkeypatch):
    raw = _raw_frame(200)
    today = raw["date"].max() + timedelta(days=1)
    monkeypatch.setattr(kline_module, "cn_today", lambda: today)
    live_row = {
        "symbol": SYM, "date": today,
        "open": 12.0, "high": 12.2, "low": 11.9, "close": 12.1,
        "volume": 1.0e4, "amount": 1.2e7, "change_pct": 0.01,
        "ma60": 11.5, "ma120": 11.25,
    }

    resp = _client(raw, live_row).get(
        "/api/kline/daily",
        params={"symbol": SYM, "start_date": raw["date"][150].isoformat()},
    )
    assert resp.status_code == 200
    last = resp.json()["rows"][-1]
    assert last["is_live"] is True
    assert last["ma120"] == 11.25
