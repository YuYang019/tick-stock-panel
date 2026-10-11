"""通达信函数 → Polars 表达式构造。

约定:
- 所有值为 Float64; 条件 = 1.0/0.0 (null 传播, fail-closed: 数据不足不算成立)。
- 这里构造的表达式**不挂 .over()**: 嵌套窗口在 Polars 中会静默产出全 null,
  编译器在输出层统一挂一次 .over("symbol") (帧按 symbol,date 排序), 窗内
  shift/rolling/ewm/cum 均按 symbol 分组生效。
- 常量用 pl.repeat(c, pl.len()) 表示 (组内满长, shift/rolling 语义正确)。
- 窗口 N 必须是常数整数 1..512 (部分函数允许 N=0 = 全部历史, 与通达信一致)。
- 预热 (warmup): 值首次有效前需要的 bar 数; EMA/SMA 按 4×N 计以收敛。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import polars as pl

WINDOW_MAX = 512
UNBOUNDED_WARMUP = 250  # BARSLAST/BARSCOUNT/N=0 累计类: 依赖全部历史, 预热按一年估计


class CompileError(Exception):
    def __init__(self, code: str, message: str, line: int = 0, col: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.line = line
        self.col = col


@dataclass(frozen=True, eq=False)
class Value:
    expr: pl.Expr
    warmup: int = 0
    const: float | None = None
    deps: frozenset[str] = frozenset()
    size: int = 1
    depth: int = 1
    history: bool = False  # 结果依赖已加载历史长度 (BARSCOUNT/SUM(X,0) 等)


def constant(value: float) -> Value:
    return Value(expr=pl.repeat(float(value), pl.len(), dtype=pl.Float64), const=float(value))


def derive(expr: pl.Expr, args: list[Value], extra_warmup: int = 0, *, history: bool = False) -> Value:
    return Value(
        expr=expr,
        warmup=max((a.warmup for a in args), default=0) + extra_warmup,
        deps=frozenset().union(*(a.deps for a in args)) if args else frozenset(),
        size=1 + sum(a.size for a in args),
        depth=1 + max((a.depth for a in args), default=0),
        history=history or any(a.history for a in args),
    )


def truthy(expr: pl.Expr) -> pl.Expr:
    return expr != 0


def as_float(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Float64)


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator.is_not_null() & (denominator != 0)).then(numerator / denominator).otherwise(None)


def _all_not_null(*exprs: pl.Expr) -> pl.Expr:
    cond = exprs[0].is_not_null()
    for expr in exprs[1:]:
        cond = cond & expr.is_not_null()
    return cond


def _window(fname: str, arg: Value, node, label: str = "N", lo: int = 1, hi: int = WINDOW_MAX) -> int:
    if arg.const is None:
        raise CompileError("T006", f"{fname} 的参数 {label} 必须是常数", node.line, node.col)
    value = arg.const
    if value != int(value) or not (lo <= int(value) <= hi):
        raise CompileError("T006", f"{fname} 的参数 {label} 必须是 {lo}..{hi} 的整数, 实际为 {value:g}", node.line, node.col)
    return int(value)


# ---- 各函数 -----------------------------------------------------------------


def _ma(node, a):
    n = _window("MA", a[1], node)
    return derive(a[0].expr.rolling_mean(n, min_samples=n), a[:1], n - 1)


def _ema(node, a):
    n = _window("EMA", a[1], node)
    # 通达信 EMA: Y=(2X+(N-1)Y')/(N+1), 首值=首个有效 X ⇔ ewm(span=N, adjust=False)
    return derive(a[0].expr.ewm_mean(span=n, adjust=False, ignore_nulls=True), a[:1], 4 * n)


def _sma(node, a):
    n = _window("SMA", a[1], node)
    m = _window("SMA", a[2], node, "M", 1, n)
    # 通达信 SMA: Y=(M×X+(N-M)×Y')/N, 首值=首个有效 X ⇔ ewm(alpha=M/N, adjust=False)
    return derive(a[0].expr.ewm_mean(alpha=m / n, adjust=False, ignore_nulls=True), a[:1], 4 * n)


def _wma(node, a):
    n = _window("WMA", a[1], node)
    weights = [float(i) for i in range(1, n + 1)]  # 窗口末尾 (最新) 权重最大
    total = sum(weights)
    return derive(a[0].expr.rolling_sum(n, weights=weights, min_samples=n) / total, a[:1], n - 1)


def _ref(node, a):
    n = _window("REF", a[1], node, lo=0)
    return derive(a[0].expr.shift(n) if n else a[0].expr, a[:1], n)


def _hhv(node, a):
    n = _window("HHV", a[1], node, lo=0)
    if n == 0:
        return derive(a[0].expr.cum_max(), a[:1], UNBOUNDED_WARMUP, history=True)
    return derive(a[0].expr.rolling_max(n, min_samples=n), a[:1], n - 1)


def _llv(node, a):
    n = _window("LLV", a[1], node, lo=0)
    if n == 0:
        return derive(a[0].expr.cum_min(), a[:1], UNBOUNDED_WARMUP, history=True)
    return derive(a[0].expr.rolling_min(n, min_samples=n), a[:1], n - 1)


def _sum(node, a):
    n = _window("SUM", a[1], node, lo=0)
    if n == 0:
        return derive(a[0].expr.cum_sum(), a[:1], UNBOUNDED_WARMUP, history=True)
    return derive(a[0].expr.rolling_sum(n, min_samples=n), a[:1], n - 1)


def _std(node, a):
    n = _window("STD", a[1], node, lo=2)
    return derive(a[0].expr.rolling_std(n, min_samples=n, ddof=1), a[:1], n - 1)


def _avedev(node, a):
    n = _window("AVEDEV", a[1], node)
    x = a[0].expr
    mean = x.rolling_mean(n, min_samples=n)
    total = pl.sum_horizontal([(x.shift(k) - mean).abs() for k in range(n)])
    return derive(pl.when(mean.is_not_null()).then(total / n).otherwise(None), a[:1], n - 1)


def _cond_count(fname: str, node, a):
    n = _window(fname, a[1], node, lo=0 if fname == "COUNT" else 1)
    hits = as_float(truthy(a[0].expr))
    if n == 0:
        return n, derive(hits.cum_sum(), a[:1], UNBOUNDED_WARMUP, history=True)
    return n, derive(hits.rolling_sum(n, min_samples=n), a[:1], n - 1)


def _count(node, a):
    return _cond_count("COUNT", node, a)[1]


def _every(node, a):
    n, counted = _cond_count("EVERY", node, a)
    return replace(counted, expr=as_float(counted.expr == n))


def _exist(node, a):
    _, counted = _cond_count("EXIST", node, a)
    return replace(counted, expr=as_float(counted.expr > 0))


def _cross(node, a):
    left, right = a[0].expr, a[1].expr
    prev_left, prev_right = left.shift(1), right.shift(1)
    # 任一侧当前/前值缺失 → null (首根 bar 恒为 null, 不依赖 Kleene 逻辑的偶然短路)
    expr = pl.when(_all_not_null(left, right, prev_left, prev_right)).then(
        as_float((left > right) & (prev_left <= prev_right))
    ).otherwise(None)
    return derive(expr, a, 1)


def _if(node, a):
    cond = a[0].expr
    expr = pl.when(cond.is_null()).then(None).when(truthy(cond)).then(a[1].expr).otherwise(a[2].expr)
    return derive(expr, a)


def _max(node, a):
    left, right = a[0].expr, a[1].expr
    return derive(pl.when(_all_not_null(left, right)).then(pl.max_horizontal(left, right)).otherwise(None), a)


def _min(node, a):
    left, right = a[0].expr, a[1].expr
    return derive(pl.when(_all_not_null(left, right)).then(pl.min_horizontal(left, right)).otherwise(None), a)


def _abs(node, a):
    return derive(a[0].expr.abs(), a)


def _round(node, a):
    return derive(a[0].expr.round(0, mode="half_away_from_zero"), a)


def _sqrt(node, a):
    x = a[0].expr
    return derive(pl.when(x >= 0).then(x.sqrt()).otherwise(None), a)


def _pow(node, a):
    x, y = a[0].expr, a[1].expr
    result = x.pow(y)
    return derive(pl.when(result.is_finite()).then(result).otherwise(None), a)


def _log(node, a):
    x = a[0].expr
    return derive(pl.when(x > 0).then(x.log10()).otherwise(None), a)


def _ln(node, a):
    x = a[0].expr
    return derive(pl.when(x > 0).then(x.log()).otherwise(None), a)


def _barslast(node, a):
    index = pl.int_range(pl.len()).cast(pl.Float64)
    hit = truthy(a[0].expr).fill_null(False)
    last = pl.when(hit).then(index).otherwise(None).forward_fill()
    return derive(index - last, a, UNBOUNDED_WARMUP, history=True)


def _barscount(node, a):
    index = pl.int_range(pl.len()).cast(pl.Float64)
    first = pl.when(a[0].expr.is_not_null()).then(index).otherwise(None).cum_min().forward_fill()
    return derive(index - first + 1, a, UNBOUNDED_WARMUP, history=True)


def _between(node, a):
    x, lo, hi = a[0].expr, a[1].expr, a[2].expr
    lower = pl.min_horizontal(lo, hi)
    upper = pl.max_horizontal(lo, hi)
    expr = pl.when(_all_not_null(x, lo, hi)).then(as_float((x >= lower) & (x <= upper))).otherwise(None)
    return derive(expr, a)


# 名称 → (最少参数, 最多参数, 构造器)
FUNCTIONS: dict[str, tuple[int, int, Callable]] = {
    "MA": (2, 2, _ma), "EMA": (2, 2, _ema), "SMA": (3, 3, _sma), "WMA": (2, 2, _wma),
    "REF": (2, 2, _ref), "HHV": (2, 2, _hhv), "LLV": (2, 2, _llv), "SUM": (2, 2, _sum),
    "STD": (2, 2, _std), "AVEDEV": (2, 2, _avedev),
    "COUNT": (2, 2, _count), "EVERY": (2, 2, _every), "EXIST": (2, 2, _exist),
    "CROSS": (2, 2, _cross), "IF": (3, 3, _if), "IFF": (3, 3, _if),
    "MAX": (2, 2, _max), "MIN": (2, 2, _min), "ABS": (1, 1, _abs), "ROUND": (1, 1, _round),
    "SQRT": (1, 1, _sqrt), "POW": (2, 2, _pow), "LOG": (1, 1, _log), "LN": (1, 1, _ln),
    "BARSLAST": (1, 1, _barslast), "BARSCOUNT": (1, 1, _barscount), "BETWEEN": (3, 3, _between),
}
