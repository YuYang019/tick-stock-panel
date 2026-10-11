"""自定义信号引用通达信指标输出 (ind_*) — 历史路径物化正确, 盘中单日路径优雅跳过。

参照 test_custom_signals_factor_conditions.py: 指标输出注册为 kind=indicator 因子,
materialize_factor_columns → materialize_scoring_columns → materialize_indicator_columns。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# 模块级导入: app.backtest.factor 在导入时冻结 FACTOR_COLUMNS 快照, 必须早于本文件注册
# 测试指标, 否则快照混入 ind_* 条目, 污染 test_factor_registry 的基线快照。
from app.backtest.factor import FactorBacktestService
from app.factors.registry import get_factor
from app.indicators.tdx import factors as indicator_factors
from app.strategy import custom_signals
from app.strategy.scoring import materialize_scoring_columns, scoring_warmup_bars

DEFINITION = {"id": "t_ma2", "name": "双均线", "formula": "FAST:MA(C,3);SLOW:EMA(C,6),NODRAW;", "pane": "main"}
FAST, SLOW = "ind_t_ma2_fast", "ind_t_ma2_slow"


@pytest.fixture(autouse=True)
def registered():
    indicator_factors.register_indicator(DEFINITION)
    yield
    indicator_factors.unregister_indicator(DEFINITION["id"])


def _frame(n: int = 40) -> pl.DataFrame:
    rows = []
    end = date(2026, 8, 31)
    for symbol, base, sign in (("A", 10.0, 1.0), ("B", 50.0, -1.0)):
        for i in range(n):
            # A 先跌后涨 (上穿), B 先涨后跌 (下穿)
            wave = sign * (abs(i - n / 2) - n / 4) * 0.2
            rows.append({"symbol": symbol, "date": end - timedelta(days=n - 1 - i), "close": base + wave})
    return pl.DataFrame(rows).sort(["symbol", "date"])


def _ma(values, n):
    return [None if i < n - 1 else float(np.mean(values[i - n + 1:i + 1])) for i in range(len(values))]


def _ema(values, n):
    out, prev = [], None
    for v in values:
        prev = v if prev is None else (2 * v + (n - 1) * prev) / (n + 1)
        out.append(prev)
    return out


def test_registered_specs() -> None:
    spec = get_factor(FAST)
    assert spec is not None and spec.kind == "indicator"
    assert spec.indicator_ref == ("t_ma2", "fast") and spec.label == "双均线.FAST"
    assert spec.warmup_bars == compile_warmup()
    assert get_factor(SLOW) is not None  # NODRAW 输出仍可被引用
    assert FAST in custom_signals.allowed_fields()
    assert scoring_warmup_bars({FAST: 1.0}) == spec.warmup_bars


def compile_warmup() -> int:
    from app.indicators.tdx import compile_indicator

    return compile_indicator(DEFINITION["formula"]).warmup_bars


def test_materialize_values_per_symbol() -> None:
    df = _frame()
    out = materialize_scoring_columns(df, [FAST, SLOW])
    for symbol in ("A", "B"):
        part = out.filter(pl.col("symbol") == symbol)
        close = part["close"].to_list()
        expected_fast = _ma(close, 3)
        actual_fast = part[FAST].to_list()
        assert actual_fast[:2] == [None, None]
        np.testing.assert_allclose(actual_fast[2:], expected_fast[2:], rtol=1e-10)
        np.testing.assert_allclose(part[SLOW].to_numpy(), _ema(close, 6), rtol=1e-10)
    assert not any(c.startswith("__ind_") for c in out.columns)


def test_signal_cross_and_greater_history_path() -> None:
    above = {
        "id": "t_fast_above", "name": "快线在上", "kind": "entry", "enabled": True,
        "conditions": [{"left": FAST, "op": ">", "right": f"field:{SLOW}", "leftDays": 0, "rightDays": 0}],
    }
    # 上穿 = 今日 FAST>SLOW 且 昨日 FAST<=SLOW
    cross = {
        "id": "t_fast_cross", "name": "快线上穿", "kind": "entry", "enabled": True,
        "conditions": [
            {"left": FAST, "op": ">", "right": f"field:{SLOW}", "leftDays": 0, "rightDays": 0},
            {"left": FAST, "op": "<=", "right": f"field:{SLOW}", "leftDays": 1, "rightDays": 1},
        ],
    }
    for sig in (above, cross):
        custom_signals.validate(sig)
    exprs = custom_signals.build_expressions([above, cross])
    df = custom_signals.materialize_factor_columns(_frame(), exprs)
    assert FAST in df.columns and SLOW in df.columns
    injected = custom_signals.inject(df, exprs)
    col_above = custom_signals.column_name("t_fast_above")
    col_cross = custom_signals.column_name("t_fast_cross")
    for symbol in ("A", "B"):
        part = injected.filter(pl.col("symbol") == symbol)
        fast, slow = part[FAST].to_list(), part[SLOW].to_list()
        expect_above = [None if f is None else f > s for f, s in zip(fast, slow, strict=True)]
        assert part[col_above].to_list() == expect_above
        def kleene_and(a, b):  # Polars 布尔 & 为三值逻辑: False 优先于 null
            if a is False or b is False:
                return False
            return None if a is None or b is None else True

        prev = [None] + [
            None if fast[i - 1] is None else fast[i - 1] <= slow[i - 1] for i in range(1, len(fast))
        ]
        expect_cross = [kleene_and(expect_above[i], prev[i]) for i in range(len(fast))]
        assert part[col_cross].to_list() == expect_cross
    a_hits = injected.filter((pl.col("symbol") == "A") & pl.col(col_cross).fill_null(False)).height
    assert a_hits >= 1  # A 先跌后涨, 必有上穿


def test_intraday_single_day_snapshot_skips_gracefully(caplog) -> None:
    sig = {
        "id": "t_fast_above", "name": "快线在上", "kind": "entry", "enabled": True,
        "conditions": [{"left": FAST, "op": ">", "right": f"field:{SLOW}", "leftDays": 0, "rightDays": 0}],
    }
    exprs = custom_signals.build_expressions([sig])
    frame = _frame()
    snapshot = frame.filter(pl.col("date") == frame["date"].max())  # 每 symbol 1 行
    materialized = custom_signals.materialize_factor_columns(snapshot, exprs)
    assert FAST not in materialized.columns  # 单日快照不计算时序指标
    injected = custom_signals.inject(materialized, exprs)
    assert custom_signals.column_name("t_fast_above") not in injected.columns  # 缺列告警跳过


def test_missing_dependency_column_skips() -> None:
    frame = _frame().rename({"close": "px"})
    out = materialize_scoring_columns(frame, [FAST])
    assert FAST not in out.columns


def test_backtest_compute_missing_factors_routes_indicator() -> None:
    panel = _frame().with_columns(
        pl.col("close").alias("open"), pl.col("close").alias("high"), pl.col("close").alias("low"),
        pl.lit(1.0).alias("volume"),
    )
    out = FactorBacktestService._compute_missing_factors(panel, {FAST})
    assert FAST in out.columns and out[FAST].drop_nulls().len() > 0
