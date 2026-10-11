"""通达信公式语法分析 (递归下降, 纯 AST, 不执行任何代码)。

语句:
  NAME : expr [, modifier]* ;     具名输出
  NAME := expr ;                  中间变量
  expr [, modifier]* ;            匿名输出 (名称由编译器分配 OUT1..)
优先级: OR < AND < 比较 < 加减 < 乘除 < 一元(-, NOT) < 调用/原子。
最后一条语句的分号可省略。按语句切分并逐条解析: 单条语句出错不影响其余
语句的诊断 (编辑器一次看到全部错误)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.indicators.tdx.lexer import Token

MAX_DEPTH = 32


@dataclass(frozen=True)
class Node:
    """AST 节点。kind: num / str / name / call / unary / binary。"""

    kind: str
    value: str
    children: tuple[Node, ...]
    line: int
    col: int
    depth: int = 1


@dataclass
class Statement:
    kind: str  # "assign" / "output"
    name: str | None  # 大写名 (匿名输出为 None)
    raw_name: str | None  # 原文名 (显示用)
    expr: Node | None
    modifiers: list[Token] = field(default_factory=list)
    line: int = 1
    col: int = 1
    tokens: list[Token] = field(default_factory=list)


class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


def split_statements(tokens: list[Token]) -> list[list[Token]]:
    """按顶层分号切分语句 (不含分号与 EOF); 空语句丢弃。"""
    statements: list[list[Token]] = []
    current: list[Token] = []
    for token in tokens:
        if token.kind == "EOF":
            break
        if token.kind == "OP" and token.value == ";":
            if current:
                statements.append(current)
            current = []
            continue
        current.append(token)
    if current:
        statements.append(current)
    return statements


_CMP = frozenset({">", "<", ">=", "<=", "=", "<>"})


class _StatementParser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        last = tokens[-1]
        self._end = Token("EOF", "", last.line, last.col + max(len(last.value), 1))

    def peek(self, offset: int = 0) -> Token:
        index = self.pos + offset
        return self.tokens[index] if index < len(self.tokens) else self._end

    def take(self) -> Token:
        token = self.peek()
        self.pos += 1
        return token

    def is_op(self, value: str, offset: int = 0) -> bool:
        token = self.peek(offset)
        return token.kind == "OP" and token.value == value

    def expect_op(self, value: str) -> Token:
        token = self.peek()
        if not (token.kind == "OP" and token.value == value):
            raise ParseError(f"期望 '{value}', 实际为 {_describe(token)}", token.line, token.col)
        return self.take()

    # ---- 语句 -------------------------------------------------------

    def parse_statement(self) -> Statement:
        first = self.peek()
        if first.kind == "IDENT" and (self.is_op(":=", 1) or self.is_op(":", 1)):
            name_token = self.take()
            op = self.take().value
            expr = self.parse_expr()
            modifiers = self.parse_modifiers()
            if op == ":=":
                if modifiers:
                    mod = modifiers[0]
                    raise ParseError("中间变量 (:=) 不能带绘图修饰符", mod.line, mod.col)
                kind = "assign"
            else:
                kind = "output"
            return Statement(
                kind=kind, name=name_token.value, raw_name=name_token.raw or name_token.value,
                expr=expr, modifiers=modifiers, line=first.line, col=first.col, tokens=self.tokens,
            )
        expr = self.parse_expr()
        modifiers = self.parse_modifiers()
        return Statement(
            kind="output", name=None, raw_name=None, expr=expr, modifiers=modifiers,
            line=first.line, col=first.col, tokens=self.tokens,
        )

    def parse_modifiers(self) -> list[Token]:
        modifiers: list[Token] = []
        while self.is_op(","):
            self.take()
            token = self.peek()
            if token.kind != "IDENT" or self.is_op("(", 1):
                raise ParseError(
                    f"输出表达式后只能跟绘图修饰符 (如 COLORRED), 实际为 {_describe(token)}",
                    token.line, token.col,
                )
            modifiers.append(self.take())
        token = self.peek()
        if token.kind != "EOF":
            raise ParseError(f"多余的内容 {_describe(token)} (缺少分号?)", token.line, token.col)
        return modifiers

    # ---- 表达式 -----------------------------------------------------

    def parse_expr(self) -> Node:
        return self.parse_or()

    def _binary(self, op: Token, left: Node, right: Node) -> Node:
        depth = max(left.depth, right.depth) + 1
        if depth > MAX_DEPTH:
            raise ParseError(f"表达式嵌套深度超限 (> {MAX_DEPTH})", op.line, op.col)
        return Node("binary", op.value, (left, right), op.line, op.col, depth)

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.is_op("OR"):
            op = self.take()
            left = self._binary(op, left, self.parse_and())
        return left

    def parse_and(self) -> Node:
        left = self.parse_cmp()
        while self.is_op("AND"):
            op = self.take()
            left = self._binary(op, left, self.parse_cmp())
        return left

    def parse_cmp(self) -> Node:
        left = self.parse_add()
        while self.peek().kind == "OP" and self.peek().value in _CMP:
            op = self.take()
            left = self._binary(op, left, self.parse_add())
        return left

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self.is_op("+") or self.is_op("-"):
            op = self.take()
            left = self._binary(op, left, self.parse_mul())
        return left

    def parse_mul(self) -> Node:
        left = self.parse_unary()
        while self.is_op("*") or self.is_op("/"):
            op = self.take()
            left = self._binary(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Node:
        if self.is_op("-") or self.is_op("NOT") or self.is_op("+"):
            op = self.take()
            operand = self.parse_unary()
            if op.value == "+":
                return operand
            depth = operand.depth + 1
            if depth > MAX_DEPTH:
                raise ParseError(f"表达式嵌套深度超限 (> {MAX_DEPTH})", op.line, op.col)
            return Node("unary", op.value, (operand,), op.line, op.col, depth)
        return self.parse_atom()

    def parse_atom(self) -> Node:
        token = self.peek()
        if token.kind == "NUM":
            self.take()
            return Node("num", token.value, (), token.line, token.col)
        if token.kind == "STR":
            self.take()
            return Node("str", token.value, (), token.line, token.col)
        if token.kind == "IDENT":
            self.take()
            if self.is_op("("):
                self.take()
                args: list[Node] = []
                if not self.is_op(")"):
                    args.append(self.parse_expr())
                    while self.is_op(","):
                        self.take()
                        args.append(self.parse_expr())
                self.expect_op(")")
                depth = max((arg.depth for arg in args), default=0) + 1
                if depth > MAX_DEPTH:
                    raise ParseError(f"表达式嵌套深度超限 (> {MAX_DEPTH})", token.line, token.col)
                return Node("call", token.value, tuple(args), token.line, token.col, depth)
            return Node("name", token.value, (), token.line, token.col)
        if self.is_op("("):
            self.take()
            inner = self.parse_expr()
            self.expect_op(")")
            return inner
        raise ParseError(f"此处需要表达式, 实际为 {_describe(token)}", token.line, token.col)


def _describe(token: Token) -> str:
    if token.kind == "EOF":
        return "语句结尾"
    if token.kind == "STR":
        return f"字符串 '{token.value}'"
    return f"'{token.raw or token.value}'"


def parse_statement(tokens: list[Token]) -> Statement:
    """解析单条语句 (tokens 不含分号); 失败抛 ParseError。"""
    return _StatementParser(tokens).parse_statement()
