"""通达信公式词法分析。

- 标识符含中文, 统一大写 (raw 保留原文供显示);
- 全角 ；：，（） 等归一为半角;
- {...} 注释跳过 (未闭合报错);
- 未知字符报错 (带行列), 不静默丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass

# 全角 → 半角归一 (标点/运算符/引号)
_FULLWIDTH = str.maketrans({
    "；": ";", "：": ":", "，": ",", "（": "(", "）": ")",
    "＋": "+", "－": "-", "＊": "*", "／": "/", "＞": ">", "＜": "<", "＝": "=",
    "｛": "{", "｝": "}", "‘": "'", "’": "'", "“": "'", "”": "'", "　": " ",
    "！": "!", "＆": "&", "｜": "|",
})

KEYWORD_OPS = {"AND": "AND", "OR": "OR", "NOT": "NOT"}

# 多字符运算符优先匹配 (最长匹配)
_OPERATORS = (
    ":=", ">=", "<=", "<>", "!=", "&&", "||",
    "+", "-", "*", "/", ">", "<", "=", "(", ")", ",", ";", ":",
)
_OP_ALIASES = {"&&": "AND", "||": "OR", "!=": "<>"}


@dataclass(frozen=True)
class Token:
    kind: str  # NUM / STR / IDENT / OP / EOF
    value: str  # IDENT 已大写; OP 已归一 (AND/OR/NOT/<>); NUM 原文; STR 内容
    line: int
    col: int
    raw: str = ""  # IDENT 原文 (保留大小写)


class LexError(Exception):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()  # isalpha 覆盖中文


def _is_ident_char(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def tokenize(text: str) -> list[Token]:
    """文本 → token 列表 (末尾带 EOF); 失败抛 LexError (由 compiler 转结构化错误)。"""
    src = text.translate(_FULLWIDTH)
    tokens: list[Token] = []
    i = 0
    line, col = 1, 1
    n = len(src)

    def advance(count: int) -> None:
        nonlocal i, line, col
        for _ in range(count):
            if src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            advance(1)
            continue
        if ch == "{":
            start_line, start_col = line, col
            end = src.find("}", i + 1)
            if end < 0:
                raise LexError("注释未闭合: 缺少 '}'", start_line, start_col)
            advance(end - i + 1)
            continue
        if ch == "}":
            raise LexError("多余的 '}'", line, col)
        if ch == "'":
            start_line, start_col = line, col
            end = src.find("'", i + 1)
            if end < 0:
                raise LexError("字符串未闭合: 缺少 '''", start_line, start_col)
            content = src[i + 1:end]
            if "\n" in content:
                raise LexError("字符串不能跨行", start_line, start_col)
            tokens.append(Token("STR", content, start_line, start_col))
            advance(end - i + 1)
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == ".":
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            tokens.append(Token("NUM", src[i:j], line, col))
            advance(j - i)
            continue
        if _is_ident_start(ch):
            j = i + 1
            while j < n and _is_ident_char(src[j]):
                j += 1
            raw = src[i:j]
            upper = raw.upper()
            if upper in KEYWORD_OPS:
                tokens.append(Token("OP", KEYWORD_OPS[upper], line, col, raw))
            else:
                tokens.append(Token("IDENT", upper, line, col, raw))
            advance(j - i)
            continue
        matched = next((op for op in _OPERATORS if src.startswith(op, i)), None)
        if matched is None:
            raise LexError(f"无法识别的字符 '{ch}'", line, col)
        tokens.append(Token("OP", _OP_ALIASES.get(matched, matched), line, col))
        advance(len(matched))
    tokens.append(Token("EOF", "", line, col))
    return tokens
