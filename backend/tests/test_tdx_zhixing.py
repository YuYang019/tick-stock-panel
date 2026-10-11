"""知行指标: 用户原文 (含 DRAWTEXT_FIX/STRCAT) 容错编译 + 数值与独立 numpy 实现一致。"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from app.indicators.tdx import compile_indicator, evaluate
from app.indicators.tdx.store import ZHIXING_FORMULA

ORIGINAL = """知行短期趋势线:EMA(EMA(C,10),10),COLORBLACK,LINETHICK1;
MA1:=MA(CLOSE,60);
MA2:=EMA(CLOSE,13);
Z1:=STRCAT(HYBLOCK,' ');
Z2:=STRCAT(Z1,DYBLOCK);
Z3:=STRCAT(Z2,' ');
DRAWTEXT_FIX(ISLASTBAR,0,0,0,STRCAT(Z3,GNBLOCK)),COLOR00C0C0;
知行多空线:(MA(CLOSE,14)+MA(CLOSE,28)+MA(CLOSE,57)+MA(CLOSE,114))/4;
"""


def _ema(values: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (2 * values[i] + (n - 1) * out[i - 1]) / (n + 1)
    return out


def _ma(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(n - 1, len(values)):
        out[i] = values[i - n + 1:i + 1].mean()
    return out


def _closes(seed: int, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))


def test_original_compiles_with_skip_warnings() -> None:
    compiled = compile_indicator(ORIGINAL)
    assert compiled.ok, compiled.errors
    assert [o.name for o in compiled.outputs] == ["知行短期趋势线", "知行多空线"]
    assert compiled.outputs[0].color == "#000000"
    assert all(w.code == "W001" for w in compiled.warnings)
    assert any("DRAWTEXT_FIX" in w.message for w in compiled.warnings)
    assert {w.line for w in compiled.warnings} == {4, 5, 6, 7}
    assert compiled.dependencies == frozenset({"close"})
    assert compiled.warmup_bars >= 114


def test_seed_formula_same_keys_as_original_and_visible_colors() -> None:
    seed = compile_indicator(ZHIXING_FORMULA)
    original = compile_indicator(ORIGINAL)
    assert seed.ok and not seed.warnings
    assert [o.key for o in seed.outputs] == [o.key for o in original.outputs]
    assert [o.color for o in seed.outputs] == ["#FFFFFF", "#FFFF00"]


@pytest.mark.parametrize("formula", [ORIGINAL, ZHIXING_FORMULA])
def test_values_match_numpy_reference(formula: str) -> None:
    compiled = compile_indicator(formula)
    series = {"A": _closes(1), "B": _closes(2) * 7}
    frame = pl.DataFrame({
        "symbol": [s for s, v in series.items() for _ in v],
        "date": [f"d{i:04d}" for v in series.values() for i in range(len(v))],
        "close": np.concatenate(list(series.values())),
    })
    out = evaluate(compiled, frame)
    short_col, long_col = (o.column for o in compiled.outputs)
    for symbol, close in series.items():
        part = out.filter(pl.col("symbol") == symbol).sort("date")
        short = part[short_col].to_numpy()
        np.testing.assert_allclose(short, _ema(_ema(close, 10), 10), rtol=1e-10)
        expected_long = (_ma(close, 14) + _ma(close, 28) + _ma(close, 57) + _ma(close, 114)) / 4
        actual_long = part[long_col].to_numpy().astype(float)
        assert np.isnan(actual_long[:113]).all()
        np.testing.assert_allclose(actual_long[113:], expected_long[113:], rtol=1e-10)
