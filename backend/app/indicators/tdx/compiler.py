"""通达信公式编译器: AST → pl.Expr。永不抛异常, 失败以结构化 errors 表达。

窗口纪律: 函数层表达式不挂 .over(); 每条输出在此统一挂一次 .over("symbol")
(evaluate 先按 symbol,date 排序), 嵌套窗口 (EMA(EMA(C,10),10)) 因此正确。
中间变量 (:=) 以 CTE 方式内联为表达式 (不物化列)。

错误码:
  T001 词法  T002 语法  T003 未知标识符  T004 未知函数  T005 不支持的函数
  T006 参数非法  T007 参数个数  T008 名称重复/冲突  T009 规模超限
  T010 引用了被跳过语句的变量  T011 字符串只能用于绘图函数  T012 无输出
  T013 STICKLINE 用法
警告码:
  W001 语句被跳过 (不可表达的绘图/文本函数)  W002 未知修饰符已忽略
  W003 结果依赖已加载历史长度
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import polars as pl

from app.indicators.tdx import catalog
from app.indicators.tdx.functions import (
    FUNCTIONS,
    CompileError,
    Value,
    as_float,
    constant,
    derive,
    safe_div,
)
from app.indicators.tdx.lexer import LexError, Token, tokenize
from app.indicators.tdx.parser import Node, ParseError, Statement, parse_statement, split_statements

MAX_FORMULA_LENGTH = 8000
MAX_STATEMENTS = 64
MAX_EXPANDED_SIZE = 20000  # 中间变量内联后的节点总数 (防指数级膨胀)
MAX_EXPANDED_DEPTH = 256  # 内联后的表达式深度 (防 Polars 递归栈溢出)
MAX_WARMUP = 2000
OUTPUT_PREFIX = "__ind_"

OutputKind = Literal["line", "stick", "colorstick", "volstick", "dot"]
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


@dataclass(frozen=True)
class TdxError:
    code: str
    message: str
    line: int = 0
    col: int = 0

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "line": self.line, "col": self.col}


@dataclass(frozen=True, eq=False)
class IndicatorOutput:
    name: str
    key: str
    expr: pl.Expr | None
    kind: OutputKind = "line"
    color: str | None = None
    thick: int = 1
    dashed: bool = False
    visible: bool = True
    # STICKLINE: (cond, p1, p2, width, empty)
    stick: tuple[pl.Expr, pl.Expr, pl.Expr, float, bool] | None = None

    @property
    def column(self) -> str:
        return f"{OUTPUT_PREFIX}{self.key}"

    def meta(self) -> dict:
        return {
            "name": self.name, "key": self.key, "kind": self.kind, "color": self.color,
            "thick": self.thick, "dashed": self.dashed, "visible": self.visible,
            "stickline": self.stick is not None,
        }


@dataclass(frozen=True, eq=False)
class CompiledIndicator:
    ok: bool
    errors: tuple[TdxError, ...] = ()
    warnings: tuple[TdxError, ...] = ()
    outputs: tuple[IndicatorOutput, ...] = ()
    warmup_bars: int = 1
    dependencies: frozenset[str] = frozenset()


@dataclass
class _Binding:
    value: Value | None = None  # None = 被跳过/编译失败的语句
    skipped: bool = False


@dataclass
class _State:
    env: dict[str, _Binding] = field(default_factory=dict)
    errors: list[TdxError] = field(default_factory=list)
    warnings: list[TdxError] = field(default_factory=list)
    outputs: list[IndicatorOutput] = field(default_factory=list)
    output_values: list[Value] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)
    anon_index: int = 0


def _fail(errors: list[TdxError]) -> CompiledIndicator:
    return CompiledIndicator(ok=False, errors=tuple(errors))


def output_key(name: str, used: set[str]) -> str:
    """输出名 → 列安全键: ascii 名取小写; 否则 'o_' + sha1(name)[:8]。

    只由输出名决定 (与语句顺序无关), 编辑公式时输出名不变 → 键不变 →
    因子 id ind_{indicator}_{key} 稳定, 已引用的信号/策略不断链。
    """
    lowered = name.lower()
    base = lowered if _KEY_RE.match(lowered) else "o_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    key, suffix = base, 2
    while key in used:
        key, suffix = f"{base}_{suffix}", suffix + 1
    used.add(key)
    return key


# ---- 表达式编译 -------------------------------------------------------------

_ARITH = {"+", "-", "*", "/"}
_COMPARE = {">", "<", ">=", "<=", "=", "<>"}


def _fold(op: str, left: float, right: float) -> float | None:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right if right != 0 else None
    return None


def _binary(op: str, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return safe_div(left, right)
    if op == "AND":
        return as_float((left != 0) & (right != 0))
    if op == "OR":
        return as_float((left != 0) | (right != 0))
    comparisons = {
        ">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right,
        "=": left == right, "<>": left != right,
    }
    return as_float(comparisons[op])


def _check_limits(value: Value, node: Node) -> Value:
    if value.size > MAX_EXPANDED_SIZE:
        raise CompileError("T009", f"表达式展开后规模超限 (> {MAX_EXPANDED_SIZE} 节点)", node.line, node.col)
    if value.depth > MAX_EXPANDED_DEPTH:
        raise CompileError("T009", f"表达式展开后嵌套深度超限 (> {MAX_EXPANDED_DEPTH})", node.line, node.col)
    return value


def _compile_node(node: Node, env: dict[str, _Binding]) -> Value:
    if node.kind == "num":
        return constant(float(node.value))
    if node.kind == "str":
        raise CompileError("T011", "字符串只能用于绘图/文本函数 (所在语句会被跳过)", node.line, node.col)
    if node.kind == "name":
        binding = env.get(node.value)
        if binding is not None:
            if binding.value is None:
                reason = "被跳过的语句" if binding.skipped else "编译失败的语句"
                raise CompileError("T010", f"引用了{reason}定义的变量 {node.value}", node.line, node.col)
            return binding.value
        column = catalog.VARIABLES.get(node.value)
        if column is not None:
            return Value(expr=pl.col(column).cast(pl.Float64), deps=frozenset({column}))
        if node.value in catalog.FUNCTION_NAMES:
            raise CompileError("T007", f"函数 {node.value} 需要参数", node.line, node.col)
        raise CompileError("T003", f"未知标识符 {node.value}", node.line, node.col)
    if node.kind == "unary":
        operand = _compile_node(node.children[0], env)
        if node.value == "-":
            if operand.const is not None:
                return constant(-operand.const)
            return _check_limits(derive(-operand.expr, [operand]), node)
        return _check_limits(derive(as_float(operand.expr == 0), [operand]), node)  # NOT
    if node.kind == "binary":
        left = _compile_node(node.children[0], env)
        right = _compile_node(node.children[1], env)
        if node.value in _ARITH and left.const is not None and right.const is not None:
            folded = _fold(node.value, left.const, right.const)
            if folded is not None and math.isfinite(folded):
                return constant(folded)
        return _check_limits(derive(_binary(node.value, left.expr, right.expr), [left, right]), node)
    # call
    name = node.value
    if name in catalog.UNSUPPORTED_FUNCTIONS:
        raise CompileError("T005", catalog.UNSUPPORTED_FUNCTIONS[name], node.line, node.col)
    if name == "STICKLINE":
        raise CompileError("T013", "STICKLINE 只能作为独立输出语句使用", node.line, node.col)
    spec = FUNCTIONS.get(name)
    if spec is None:
        if name in env or name in catalog.VARIABLES:
            raise CompileError("T004", f"{name} 不是函数", node.line, node.col)
        raise CompileError("T004", f"未知函数 {name}", node.line, node.col)
    lo, hi, builder = spec
    if not (lo <= len(node.children) <= hi):
        expect = str(lo) if lo == hi else f"{lo}~{hi}"
        raise CompileError("T007", f"{name} 需要 {expect} 个参数, 实际 {len(node.children)} 个", node.line, node.col)
    args = [_compile_node(child, env) for child in node.children]
    return _check_limits(builder(node, args), node)


# ---- 语句与修饰符 -----------------------------------------------------------

_HEX_COLOR = re.compile(r"^COLOR([0-9A-F]{6})$")
_THICK = re.compile(r"^LINETHICK([0-9])$")


def _apply_modifiers(modifiers: list[Token], state: _State) -> dict:
    style: dict = {"kind": "line", "color": None, "thick": 1, "dashed": False, "visible": True}
    for token in modifiers:
        name = token.value
        if name.startswith("COLOR") and name[5:] in catalog.NAMED_COLORS:
            style["color"] = catalog.NAMED_COLORS[name[5:]]
        elif (match := _HEX_COLOR.match(name)) is not None:
            bgr = match.group(1)  # 通达信 BBGGRR
            style["color"] = f"#{bgr[4:6]}{bgr[2:4]}{bgr[0:2]}"
        elif (match := _THICK.match(name)) is not None:
            style["thick"] = int(match.group(1))
        elif name == "NODRAW":
            style["visible"] = False
        elif name == "DOTLINE":
            style["dashed"] = True
        elif name in catalog.KIND_MODIFIERS:
            style["kind"] = catalog.KIND_MODIFIERS[name]
        else:
            state.warnings.append(TdxError("W002", f"未知修饰符 {token.raw or name} 已忽略", token.line, token.col))
    return style


def _referenced_idents(tokens: list[Token]) -> set[str]:
    return {token.value for token in tokens if token.kind == "IDENT"}


def _defined_name(tokens: list[Token]) -> str | None:
    if len(tokens) >= 2 and tokens[0].kind == "IDENT" and tokens[1].kind == "OP" and tokens[1].value in (":", ":="):
        return tokens[0].value
    return None


def _check_name(stmt: Statement, state: _State) -> None:
    name = stmt.name
    if name is None:
        return
    if name in state.env:
        raise CompileError("T008", f"名称 {stmt.raw_name} 重复定义", stmt.line, stmt.col)
    if name in catalog.VARIABLES or name in catalog.FUNCTION_NAMES or name in catalog.UNDRAWABLE:
        raise CompileError("T008", f"名称 {stmt.raw_name} 与内置变量/函数冲突", stmt.line, stmt.col)


def _output_name(stmt: Statement, state: _State) -> str:
    if stmt.raw_name:
        return stmt.raw_name
    state.anon_index += 1
    return f"OUT{state.anon_index}"


def _compile_stickline(stmt: Statement, state: _State) -> None:
    node = stmt.expr
    assert node is not None
    if stmt.kind != "output":
        raise CompileError("T013", "STICKLINE 不能赋值给中间变量 (:=)", node.line, node.col)
    if len(node.children) != 5:
        raise CompileError("T007", f"STICKLINE 需要 5 个参数, 实际 {len(node.children)} 个", node.line, node.col)
    cond, p1, p2, width, empty = (_compile_node(child, state.env) for child in node.children)
    if width.const is None or empty.const is None:
        raise CompileError("T006", "STICKLINE 的宽度与空心参数必须是常数", node.line, node.col)
    style = _apply_modifiers(stmt.modifiers, state)
    name = _output_name(stmt, state)
    value = derive(cond.expr, [cond, p1, p2])
    output = IndicatorOutput(
        name=name, key=output_key(name, state.keys), expr=None, kind="stick",
        color=style["color"], thick=style["thick"], dashed=style["dashed"], visible=style["visible"],
        stick=(
            cond.expr.over("symbol"), p1.expr.over("symbol"), p2.expr.over("symbol"),
            float(width.const), bool(empty.const),
        ),
    )
    state.outputs.append(output)
    state.output_values.append(value)
    if stmt.name:
        state.env[stmt.name] = _Binding(value=None, skipped=False)


def _compile_statement(stmt: Statement, state: _State) -> None:
    _check_name(stmt, state)
    node = stmt.expr
    assert node is not None
    if node.kind == "call" and node.value == "STICKLINE":
        _compile_stickline(stmt, state)
        return
    value = _compile_node(node, state.env)
    if stmt.kind == "assign":
        state.env[stmt.name] = _Binding(value=value)  # type: ignore[index]
        return
    style = _apply_modifiers(stmt.modifiers, state)
    name = _output_name(stmt, state)
    output = IndicatorOutput(
        name=name, key=output_key(name, state.keys), expr=value.expr.over("symbol"),
        kind=style["kind"], color=style["color"], thick=style["thick"],
        dashed=style["dashed"], visible=style["visible"],
    )
    state.outputs.append(output)
    state.output_values.append(value)
    if stmt.name:
        state.env[stmt.name] = _Binding(value=value)


# ---- 入口 -------------------------------------------------------------------


def compile_indicator(text: str) -> CompiledIndicator:
    """编译通达信公式; 永不抛异常 (fail-closed: ok=False + errors)。"""
    try:
        return _compile(text)
    except RecursionError:
        return _fail([TdxError("T009", "公式过于复杂 (递归过深)")])
    except Exception as exc:  # noqa: BLE001 — 编译器契约: 不向调用方抛异常
        return _fail([TdxError("T002", f"编译失败: {exc}")])


def _compile(text: str) -> CompiledIndicator:
    if not isinstance(text, str) or not text.strip():
        return _fail([TdxError("T012", "公式为空")])
    if len(text) > MAX_FORMULA_LENGTH:
        return _fail([TdxError("T009", f"公式过长 ({len(text)} > {MAX_FORMULA_LENGTH} 字符)")])
    try:
        tokens = tokenize(text)
    except LexError as exc:
        return _fail([TdxError("T001", exc.message, exc.line, exc.col)])
    statements = split_statements(tokens)
    if len(statements) > MAX_STATEMENTS:
        return _fail([TdxError("T009", f"语句数超限 ({len(statements)} > {MAX_STATEMENTS})")])

    state = _State()
    for stmt_tokens in statements:
        first = stmt_tokens[0]
        defined = _defined_name(stmt_tokens)
        undrawable = sorted(_referenced_idents(stmt_tokens) & catalog.UNDRAWABLE)
        if undrawable:
            state.warnings.append(TdxError(
                "W001", f"已跳过不可表达的语句 (含 {', '.join(undrawable)})", first.line, first.col,
            ))
            if defined and defined not in state.env:
                state.env[defined] = _Binding(value=None, skipped=True)
            continue
        try:
            stmt = parse_statement(stmt_tokens)
        except ParseError as exc:
            state.errors.append(TdxError("T002", exc.message, exc.line, exc.col))
            if defined and defined not in state.env:
                state.env[defined] = _Binding(value=None)
            continue
        try:
            _compile_statement(stmt, state)
        except CompileError as exc:
            state.errors.append(TdxError(exc.code, exc.message, exc.line, exc.col))
            if stmt.name and stmt.name not in state.env:
                state.env[stmt.name] = _Binding(value=None)

    if not state.errors and not state.outputs:
        state.errors.append(TdxError("T012", "公式没有输出线 (至少需要一条 NAME:表达式 或匿名输出)"))
    if any(value.history for value in state.output_values):
        state.warnings.append(TdxError(
            "W003", "BARSLAST/BARSCOUNT/N=0 累计类函数的结果依赖已加载的历史长度, 与通达信全历史计算可能略有差异",
        ))
    if state.errors:
        return CompiledIndicator(ok=False, errors=tuple(state.errors), warnings=tuple(state.warnings))
    warmup = min(MAX_WARMUP, max((value.warmup for value in state.output_values), default=0) + 1)
    dependencies = frozenset().union(*(value.deps for value in state.output_values))
    return CompiledIndicator(
        ok=True, warnings=tuple(state.warnings), outputs=tuple(state.outputs),
        warmup_bars=warmup, dependencies=dependencies,
    )


@lru_cache(maxsize=256)
def compile_indicator_cached(text: str) -> CompiledIndicator:
    """带 LRU 缓存的编译入口; CompiledIndicator 不可变, 缓存共享安全。"""
    return compile_indicator(text)


def stick_columns(output: IndicatorOutput) -> tuple[str, str, str]:
    base = output.column
    return f"{base}__cond", f"{base}__p1", f"{base}__p2"


def _finite(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def evaluate(compiled: CompiledIndicator, frame: pl.DataFrame) -> pl.DataFrame:
    """在帧上求值全部输出, 追加 __ind_{key} 列 (STICKLINE 追加 __cond/__p1/__p2 三列)。

    帧须含 symbol/date 与依赖列 (缺列抛 ValueError, 调用方决定降级); 内部按
    symbol,date 排序求值后恢复原行序。NaN/inf → null。
    """
    if not compiled.ok:
        raise ValueError("指标未编译通过")
    required = {"symbol", "date"} | set(compiled.dependencies)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"指标依赖列缺失: {missing}")
    expressions: list[pl.Expr] = []
    for output in compiled.outputs:
        if output.stick is not None:
            cond_col, p1_col, p2_col = stick_columns(output)
            cond, p1, p2, _width, _empty = output.stick
            expressions.append(_finite(cond).alias(cond_col))
            expressions.append(_finite(p1).alias(p1_col))
            expressions.append(_finite(p2).alias(p2_col))
        elif output.expr is not None:
            expressions.append(_finite(output.expr).alias(output.column))
    if frame.is_empty() or not expressions:
        return frame.with_columns([pl.lit(None, dtype=pl.Float64).alias(e.meta.output_name()) for e in expressions])
    order = "__ind_row__"
    return (
        frame.lazy()
        .with_row_index(order)
        .sort(["symbol", "date"])
        .with_columns(expressions)
        .sort(order)
        .drop(order)
        .collect()
    )
