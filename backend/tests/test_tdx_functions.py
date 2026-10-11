"""通达信函数数值: 固定样本 + 手算递推; EMA/SMA 首值=首个 X; 多 symbol 不串号。"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from app.indicators.tdx import compile_indicator, evaluate

X = [10.0, 11.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0, 18.0, 16.0]
Y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]


def _frame(close=X, open_=Y, symbols=("A",)) -> pl.DataFrame:
    rows = []
    for k, symbol in enumerate(symbols):
        scale = 1.0 + k * 10.0  # 不同 symbol 量级不同, 串号必然暴露
        for i, (c, o) in enumerate(zip(close, open_, strict=True)):
            rows.append({"symbol": symbol, "date": f"2026-01-{i + 1:02d}", "open": o * scale, "high": max(c, o) * scale,
                         "low": min(c, o) * scale, "close": c * scale, "volume": 100.0 + i, "amount": 1000.0 + i})
    return pl.DataFrame(rows)


def _run(formula: str, frame: pl.DataFrame | None = None, symbol: str = "A") -> list:
    compiled = compile_indicator(formula)
    assert compiled.ok, compiled.errors
    out = evaluate(compiled, frame if frame is not None else _frame())
    out = out.filter(pl.col("symbol") == symbol).sort("date")
    return out[compiled.outputs[0].column].to_list()


def _approx(actual: list, expected: list) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert a is None, (actual, expected)
        else:
            assert a == pytest.approx(e, rel=1e-9, abs=1e-12), (actual, expected)


def _ema_ref(values, n):
    out, prev = [], None
    for v in values:
        prev = v if prev is None else (2 * v + (n - 1) * prev) / (n + 1)
        out.append(prev)
    return out


def _sma_ref(values, n, m):
    out, prev = [], None
    for v in values:
        prev = v if prev is None else (m * v + (n - m) * prev) / n
        out.append(prev)
    return out


def _rolling(values, n, fn):
    return [None if i < n - 1 else fn(np.array(values[i - n + 1:i + 1])) for i in range(len(values))]


def test_ma_nulls_until_n_bars() -> None:
    _approx(_run("A:MA(C,3);"), _rolling(X, 3, np.mean))


def test_ema_matches_tdx_recursion_seeded_with_first_value() -> None:
    result = _run("A:EMA(C,5);")
    assert result[0] == X[0]
    _approx(result, _ema_ref(X, 5))


def test_sma_matches_tdx_recursion() -> None:
    result = _run("A:SMA(C,6,2);")
    assert result[0] == X[0]
    _approx(result, _sma_ref(X, 6, 2))
    _approx(_run("A:SMA(C,3,1);"), _sma_ref(X, 3, 1))


def test_sma_m_greater_than_n_rejected() -> None:
    assert compile_indicator("A:SMA(C,3,4);").errors[0].code == "T006"


def test_nested_ema_matches_reference() -> None:
    _approx(_run("A:EMA(EMA(C,3),4);"), _ema_ref(_ema_ref(X, 3), 4))


def test_wma() -> None:
    weights = np.arange(1, 5, dtype=float)
    _approx(_run("A:WMA(C,4);"), _rolling(X, 4, lambda w: float((w * weights).sum() / weights.sum())))


def test_ref_hhv_llv_sum_std_avedev() -> None:
    _approx(_run("A:REF(C,2);"), [None, None, *X[:-2]])
    _approx(_run("A:HHV(C,3);"), _rolling(X, 3, np.max))
    _approx(_run("A:LLV(C,3);"), _rolling(X, 3, np.min))
    _approx(_run("A:SUM(C,4);"), _rolling(X, 4, np.sum))
    _approx(_run("A:SUM(C,0);"), list(np.cumsum(X)))
    _approx(_run("A:HHV(C,0);"), list(np.maximum.accumulate(X)))
    _approx(_run("A:STD(C,5);"), _rolling(X, 5, lambda w: float(np.std(w, ddof=1))))
    _approx(_run("A:AVEDEV(C,4);"), _rolling(X, 4, lambda w: float(np.mean(np.abs(w - w.mean())))))


def test_count_every_exist_boundaries() -> None:
    up = [None] + [1.0 if X[i] > X[i - 1] else 0.0 for i in range(1, len(X))]
    # C>REF(C,1) 首行为 null → COUNT 窗口含 null 即为 null (fail-closed)
    count = _run("A:COUNT(C>REF(C,1),3);")
    expected = [None, None, None] + [sum(up[i - 2:i + 1]) for i in range(3, len(X))]
    _approx(count, expected)
    every = _run("A:EVERY(C>REF(C,1),2);")
    _approx(every, [None, None] + [1.0 if up[i] and up[i - 1] else 0.0 for i in range(2, len(X))])
    exist = _run("A:EXIST(C>15,3);")
    _approx(exist, [None, None] + [1.0 if max(X[i - 2:i + 1]) > 15 else 0.0 for i in range(2, len(X))])


def test_cross() -> None:
    result = _run("A:CROSS(C,14);")
    expected = [None] + [1.0 if X[i] > 14 and X[i - 1] <= 14 else 0.0 for i in range(1, len(X))]
    _approx(result, expected)
    # 两条序列: C 上穿 O
    result = _run("A:CROSS(C,O);")
    expected = [None] + [1.0 if X[i] > Y[i] and X[i - 1] <= Y[i - 1] else 0.0 for i in range(1, len(X))]
    _approx(result, expected)


def test_barslast_and_barscount() -> None:
    result = _run("A:BARSLAST(C>=15);")
    expected, last = [], None
    for i, value in enumerate(X):
        if value >= 15:
            last = i
        expected.append(None if last is None else float(i - last))
    _approx(result, expected)
    _approx(_run("A:BARSCOUNT(C);"), [float(i + 1) for i in range(len(X))])
    _approx(_run("A:BARSCOUNT(MA(C,3));"), [None, None] + [float(i + 1) for i in range(len(X) - 2)])


def test_if_max_min_abs_round_sqrt_pow_log_between() -> None:
    _approx(_run("A:IF(C>O,C,O);"), [max(c, o) if c > o else o for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:IFF(C>O,1,0);"), [1.0 if c > o else 0.0 for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:MAX(C,O*3);"), [max(c, o * 3) for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:MIN(C,O*3);"), [min(c, o * 3) for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:ABS(O-C);"), [abs(o - c) for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:ROUND(C/4);"), [math.floor(c / 4 + 0.5) for c in X])
    _approx(_run("A:SQRT(O-4);"), [math.sqrt(o - 4) if o >= 4 else None for o in Y])
    _approx(_run("A:POW(O,2);"), [o ** 2 for o in Y])
    _approx(_run("A:LOG(C);"), [math.log10(c) for c in X])
    _approx(_run("A:LN(C);"), [math.log(c) for c in X])
    _approx(_run("A:BETWEEN(C,12,15);"), [1.0 if 12 <= c <= 15 else 0.0 for c in X])
    _approx(_run("A:BETWEEN(C,15,12);"), [1.0 if 12 <= c <= 15 else 0.0 for c in X])


def test_arithmetic_logic_and_division_by_zero() -> None:
    _approx(_run("A:(C-O)/(O-O);"), [None] * len(X))
    _approx(_run("A:C>O AND O>2;"), [1.0 if c > o and o > 2 else 0.0 for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:NOT(C>O) OR O=1;"), [1.0 if (not c > o) or o == 1 else 0.0 for c, o in zip(X, Y, strict=True)])
    _approx(_run("A:-C+2*3;"), [-c + 6 for c in X])
    _approx(_run("A:5;"), [5.0] * len(X))


def test_intermediate_variables_inline() -> None:
    _approx(_run("M:=MA(C,3);D:=C-M;A:D*2;"), [None if v is None else (c - v) * 2 for c, v in zip(X, _rolling(X, 3, np.mean), strict=True)])


def test_multi_symbol_no_leakage_and_row_order_preserved() -> None:
    frame = _frame(symbols=("A", "B"))
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)
    compiled = compile_indicator("A:EMA(C,4);B:REF(C,1);M:MA(C,3);")
    out = evaluate(compiled, shuffled)
    assert out["date"].to_list() == shuffled["date"].to_list()  # 行序不变
    for symbol, scale in (("A", 1.0), ("B", 11.0)):
        part = out.filter(pl.col("symbol") == symbol).sort("date")
        series = [x * scale for x in X]
        _approx(part[compiled.outputs[0].column].to_list(), _ema_ref(series, 4))
        _approx(part[compiled.outputs[1].column].to_list(), [None, *series[:-1]])
        _approx(part[compiled.outputs[2].column].to_list(), _rolling(series, 3, np.mean))


def test_stickline_columns() -> None:
    compiled = compile_indicator("STICKLINE(C>O,O,C,6,1),COLORRED;")
    assert compiled.ok
    output = compiled.outputs[0]
    assert output.kind == "stick" and output.expr is None
    assert output.stick[3:] == (6.0, True)
    out = evaluate(compiled, _frame())
    assert out[f"{output.column}__cond"].to_list() == [1.0 if c > o else 0.0 for c, o in zip(X, Y, strict=True)]


def test_nan_inf_become_null() -> None:
    frame = _frame().with_columns(pl.when(pl.col("date") == "2026-01-03").then(float("nan")).otherwise(pl.col("close")).alias("close"))
    result = _run("A:C*1;", frame)
    assert result[2] is None


def test_evaluate_missing_columns_raises() -> None:
    compiled = compile_indicator("A:MA(C,3);")
    with pytest.raises(ValueError):
        evaluate(compiled, pl.DataFrame({"symbol": ["A"], "date": ["2026-01-01"]}))


def test_warmup_estimate() -> None:
    assert compile_indicator("A:MA(C,20);").warmup_bars == 20
    assert compile_indicator("A:EMA(C,10);").warmup_bars == 41
    assert compile_indicator("A:MA(REF(C,5),10);").warmup_bars == 15
    assert compile_indicator("A:C;").warmup_bars == 1
