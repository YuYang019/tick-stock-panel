"""通达信公式词法/语法: 语句类型、修饰符、全角、注释、中文名、错误行列、一元负号。"""
from __future__ import annotations

import pytest

from app.indicators.tdx import compile_indicator
from app.indicators.tdx.lexer import LexError, tokenize
from app.indicators.tdx.parser import parse_statement, split_statements


def _stmts(text: str):
    return [parse_statement(tokens) for tokens in split_statements(tokenize(text))]


def test_statement_kinds() -> None:
    named, assign, anon = _stmts("A:C;B:=O;C+1;")
    assert (named.kind, named.name) == ("output", "A")
    assert (assign.kind, assign.name) == ("assign", "B")
    assert (anon.kind, anon.name) == ("output", None)


def test_last_semicolon_optional_and_anonymous_names() -> None:
    compiled = compile_indicator("C; O")
    assert compiled.ok
    assert [o.name for o in compiled.outputs] == ["OUT1", "OUT2"]


def test_fullwidth_punctuation_and_comments() -> None:
    compiled = compile_indicator("{均线} 短线：MA（C，5）；{另一条}\n长线:MA(C,10)")
    assert compiled.ok, compiled.errors
    assert [o.name for o in compiled.outputs] == ["短线", "长线"]


def test_identifiers_case_insensitive_and_chinese_names_kept() -> None:
    compiled = compile_indicator("趋势Line:ema(close,5);")
    assert compiled.ok
    assert compiled.outputs[0].name == "趋势Line"


def test_keyword_and_symbol_logic_operators() -> None:
    tokens = tokenize("C>O AND C>1 && NOT C<0 || C<>1 != 2")
    ops = [t.value for t in tokens if t.kind == "OP"]
    assert ops == [">", "AND", ">", "AND", "NOT", "<", "OR", "<>", "<>"]


def test_precedence_or_lowest_mul_over_add() -> None:
    (stmt,) = _stmts("1+2*3>6 OR 0;")
    root = stmt.expr
    assert root.kind == "binary" and root.value == "OR"
    cmp_node = root.children[0]
    assert cmp_node.value == ">"
    add = cmp_node.children[0]
    assert add.value == "+" and add.children[1].value == "*"


def test_unary_minus() -> None:
    (stmt,) = _stmts("-C+-2;")
    assert stmt.expr.value == "+"
    assert stmt.expr.children[0].kind == "unary"


def test_modifiers_color_thick_flags() -> None:
    compiled = compile_indicator(
        "A:C,COLORRED,LINETHICK3,DOTLINE;B:C,COLOR00C0C0;D:C,NODRAW;E:C,COLORSTICK;F:C,POINTDOT;G:C,COLORLIGRAY;"
    )
    assert compiled.ok
    a, b, d, e, f, g = compiled.outputs
    assert (a.color, a.thick, a.dashed) == ("#FF0000", 3, True)
    assert b.color == "#C0C000"  # BBGGRR=00C0C0 → R=C0 G=C0 B=00
    assert d.visible is False
    assert e.kind == "colorstick"
    assert f.kind == "dot"
    assert g.color == "#C0C0C0"


def test_unknown_modifier_warns() -> None:
    compiled = compile_indicator("A:C,FOOBAR;")
    assert compiled.ok
    assert [w.code for w in compiled.warnings] == ["W002"]


def test_lex_error_line_col() -> None:
    with pytest.raises(LexError) as info:
        tokenize("A:C;\nB:O $ 1;")
    assert (info.value.line, info.value.col) == (2, 5)
    compiled = compile_indicator("A:C;\nB:O $ 1;")
    assert not compiled.ok
    assert compiled.errors[0].to_dict() == {"code": "T001", "message": "无法识别的字符 '$'", "line": 2, "col": 5}


def test_unclosed_comment_and_string() -> None:
    assert compile_indicator("A:C; {注释").errors[0].code == "T001"
    assert compile_indicator("A:'abc").errors[0].code == "T001"


def test_parse_error_position_and_all_statements_reported() -> None:
    compiled = compile_indicator("A:C+;\nB:MA(C,5;\nD:C;")
    assert not compiled.ok
    codes = [(e.code, e.line) for e in compiled.errors]
    assert codes == [("T002", 1), ("T002", 2)]


def test_semantic_errors() -> None:
    cases = {
        "A:FOO;": "T003",
        "A:FOO(C);": "T004",
        "A:MA(C);": "T007",
        "A:MA(C,N1);": "T003",
        "A:MA(C,C);": "T006",
        "A:MA(C,513);": "T006",
        "A:MA(C,2.5);": "T006",
        "A:FILTER(C>O,5);": "T005",
        "A:LONGCROSS(C,O,3);": "T005",
        "A:C;A:O;": "T008",
        "MA:C;": "T008",
        "A:='x';": "T011",
        "X:=1;": "T012",
        "": "T012",
    }
    for text, code in cases.items():
        compiled = compile_indicator(text)
        assert not compiled.ok, text
        assert compiled.errors[0].code == code, (text, compiled.errors)


def test_skipped_statement_reference_is_error() -> None:
    compiled = compile_indicator("Z:=STRCAT(HYBLOCK,' ');A:C+Z;")
    assert not compiled.ok
    assert compiled.errors[0].code == "T010"
    assert [w.code for w in compiled.warnings] == ["W001"]


def test_limits() -> None:
    assert compile_indicator("A:C;" + " " * 8000).errors[0].code == "T009"
    many = "".join(f"A{i}:C;" for i in range(65))
    assert compile_indicator(many).errors[0].code == "T009"
    deep = "A:" + "(" * 40 + "C" + ")" * 40 + ";"
    assert compile_indicator(deep).ok  # 括号不计深度
    nested = "A:" + "ABS(" * 33 + "C" + ")" * 33 + ";"
    assert compile_indicator(nested).errors[0].code == "T002"
    # 中间变量内联后指数膨胀
    chain = "X0:=C;" + "".join(f"X{i}:=X{i-1}+X{i-1};" for i in range(1, 20)) + "A:X19;"
    assert compile_indicator(chain).errors[0].code == "T009"


def test_never_raises_on_garbage() -> None:
    for text in ["(((", ")))", ":=:=", ",,,", "A:,;", "'", "}", "A:C,,", None]:
        compiled = compile_indicator(text)  # type: ignore[arg-type]
        assert compiled.ok is False
