"""市场风格: 分组、价差分位、四态判定与切换识别。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.services import market_style as ms


def _frame(days: int, inst_pct: float, small_pct: float, n: int = 20,
           ramp: float = 0.0) -> pl.DataFrame:
    """构造 n 只股 × days 日: 前 2 只为大市值(inst), 后 6 只为小市值(small)。

    ramp 让小市值收益逐日递增, 使价差分位不因全部相同而并列在 0.5。
    """
    rows = []
    start = date(2026, 6, 1)
    for d in range(days):
        for i in range(n):
            is_inst = i < max(1, int(n * 0.1))
            is_small = i >= n - max(1, int(n * 0.3))
            pct = inst_pct if is_inst else (
                small_pct + ramp * d if is_small else 0.0
            )
            rows.append({
                "symbol": f"S{i:02d}",
                "date": start + timedelta(days=d),
                "name": "测试股",
                "close": 10.0,
                "raw_close": 10.0,
                # 每只股唯一市值, 避免并列时名次分组不确定
                "float_shares": 1e9 - i * 1e6 if is_inst else 1e7 - i * 1e4,
                "ma20": 10.0,
                "change_pct": pct,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_spread_sign_and_groups():
    daily = ms.compute_daily_style(_frame(5, inst_pct=-0.01, small_pct=0.03))
    row = daily.to_dicts()[-1]
    assert row["inst_ret"] == pytest.approx(-0.01)
    assert row["small_ret"] == pytest.approx(0.03)
    assert row["style_spread"] == pytest.approx(0.04)


def test_theme_dominant_when_small_outperforms_and_inst_weak():
    # 小市值连续跑赢, 且大市值全部跌破 MA20 → 题材主导
    df = _frame(40, inst_pct=-0.01, small_pct=0.0, ramp=0.001).with_columns(
        pl.when(pl.col("float_shares") > 1e8)
        .then(pl.lit(11.0)).otherwise(pl.col("ma20")).alias("ma20")
    )
    result = ms.classify_style(ms.compute_daily_style(df))
    assert result["style"] == "题材主导"
    assert result["spread"] > 0


def test_institution_dominant():
    df = _frame(40, inst_pct=0.02, small_pct=0.0, ramp=-0.0005)
    result = ms.classify_style(ms.compute_daily_style(df))
    assert result["style"] == "机构主导"


def test_systemic_up():
    df = _frame(40, inst_pct=0.02, small_pct=0.019)
    result = ms.classify_style(ms.compute_daily_style(df))
    assert result["style"] == "普涨"


def test_systemic_down():
    df = _frame(40, inst_pct=-0.02, small_pct=-0.019)
    result = ms.classify_style(ms.compute_daily_style(df))
    assert result["style"] == "普跌"


def test_switching_when_spread_trend_reverses():
    # 前半段小市值强 (题材主导), 最近 3 日价差明显收敛但仍为正 → 切换中
    df = _frame(40, inst_pct=0.0, small_pct=0.02)
    recent = df["date"].unique().sort()[-3:]
    df = df.with_columns(
        pl.when(pl.col("date").is_in(recent.implode()) & (pl.col("float_shares") < 1e8))
        .then(pl.lit(0.002)).otherwise(pl.col("change_pct")).alias("change_pct")
    )
    result = ms.classify_style(ms.compute_daily_style(df))
    assert result["style"] in {"切换中", "题材主导"}


def test_missing_float_shares_returns_empty():
    df = _frame(5, 0.01, 0.01).drop("float_shares")
    assert ms.compute_daily_style(df).is_empty()


def test_merge_config_clamps():
    cfg = ms.merge_config({"lookback": 5, "inst_quantile": 2, "spread_high": 9})
    assert cfg["lookback"] == 20
    assert cfg["inst_quantile"] == 0.9
    assert cfg["spread_high"] == 1.0



def test_insignificant_spread_is_balanced():
    """价差为正但不显著 (分位未达 spread_high) 时不应判为题材主导。"""
    import random

    random.seed(11)
    rows = []
    start = date(2026, 6, 1)
    for d in range(40):
        small = random.uniform(-0.01, 0.01)
        for i in range(20):
            is_inst = i < 2
            is_small = i >= 14
            rows.append({
                "symbol": f"S{i:02d}", "date": start + timedelta(days=d), "name": "测试股",
                "close": 10.0, "raw_close": 10.0,
                "float_shares": 1e9 - i * 1e6 if is_inst else 1e7 - i * 1e4,
                "ma20": 10.0, "change_pct": 0.0 if is_inst else (small if is_small else 0.0),
            })
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    # 最后一天价差设为窗口中位附近的小正值
    last = df["date"].max()
    df = df.with_columns(
        pl.when((pl.col("date") == last) & (pl.col("float_shares") < 1e8))
        .then(pl.lit(0.0005)).otherwise(pl.col("change_pct")).alias("change_pct")
    )
    result = ms.classify_style(ms.compute_daily_style(df))
    assert result["spread"] > 0
    assert result["spread_pct"] is not None and result["spread_pct"] < 0.7
    assert result["style"] != "题材主导"


def test_spread_pct_uses_only_history():
    df = _frame(40, inst_pct=0.0, small_pct=0.01, ramp=0.0005)
    full = ms.classify_style(ms.compute_daily_style(df))
    # 单调递增价差: 最新一日是窗口最大值 → 分位 1.0
    assert full["spread_pct"] == pytest.approx(1.0)
