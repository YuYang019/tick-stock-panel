"""ma120 (120 日简单均线, 前复权收盘价) 作为一等 enriched 指标列。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.indicators.pipeline import (
    ENRICHED_COLUMNS,
    ENRICHED_COLUMNS_BY_CATEGORY,
    INDICATOR_COLUMNS,
    compute_indicators,
    indicator_warmup_bars,
)
from app.strategy import custom_signals


def _bars(lengths: dict[str, int]) -> pl.DataFrame:
    rows = []
    for n, (symbol, length) in enumerate(lengths.items()):
        d = date(2025, 1, 1)
        for i in range(length):
            close = 10.0 + n + 0.3 * ((i * 7) % 11) + 0.01 * i
            rows.append((symbol, d + timedelta(days=i), close, close + 0.1, close - 0.1, close, 1000.0))
    return pl.DataFrame(
        rows, schema=["symbol", "date", "open", "high", "low", "close", "volume"], orient="row",
    )


def test_ma120_registered_as_enriched_indicator():
    assert "ma120" in ENRICHED_COLUMNS
    assert "ma120" in ENRICHED_COLUMNS_BY_CATEGORY["ma"]
    assert "ma120" in INDICATOR_COLUMNS
    assert "ma120" in custom_signals.ALLOWED_FIELDS
    assert indicator_warmup_bars({"ma60", "ma120"}) == 120
    assert indicator_warmup_bars({"ma60"}) == 1


def test_ma120_equals_rolling_mean_per_symbol_and_null_before_120_bars():
    bars = _bars({"600001.SH": 200, "000002.SZ": 119})
    out = compute_indicators(bars)
    for symbol, frame in out.partition_by("symbol", as_dict=True).items():
        frame = frame.sort("date")
        expected = frame["close"].rolling_mean(120).to_list()
        got = frame["ma120"].to_list()
        assert got[:119] == [None] * 119, symbol
        assert got == pytest.approx(expected, rel=1e-12, nan_ok=True), symbol
    short = out.filter(pl.col("symbol") == "000002.SZ")
    assert short["ma120"].null_count() == short.height  # 不足 120 根全为空


def test_ma120_needed_subset_matches_full():
    bars = _bars({"600001.SH": 150})
    full = compute_indicators(bars)["ma120"]
    subset = compute_indicators(bars, needed={"ma120"})
    assert "ma120" in subset.columns
    assert subset["ma120"].to_list() == full.to_list()


def test_custom_signal_can_reference_ma120():
    sig = {
        "id": "above_ma120", "name": "站上半年线", "kind": "entry", "enabled": True,
        "conditions": [
            {"left": "close", "op": ">", "right": "field:ma120", "leftDays": 0, "rightDays": 0},
        ],
    }
    custom_signals.validate(sig)
    exprs = custom_signals.build_expressions([sig])
    deps = custom_signals.expression_dependencies(exprs)
    assert any("ma120" in cols for cols in deps.values())

    out = compute_indicators(_bars({"600001.SH": 130})).with_columns(**exprs)
    column = next(iter(exprs))
    assert out[column][:119].drop_nulls().sum() == 0  # ma120 为空的前段不触发
    assert out[column][119:].null_count() == 0
