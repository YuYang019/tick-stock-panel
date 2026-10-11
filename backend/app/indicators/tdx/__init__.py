"""通达信公式指标 (单只股票逐 bar 时间序列)。

text → lexer → parser (递归下降, 纯 AST, 无 eval) → compiler (AST → pl.Expr)。
与因子 DSL (截面因子, app/factors/dsl.py) 并存; 已保存指标的输出线注册为
kind="indicator" 因子, 经评分物化路径供信号/策略/回测引用。
"""
from app.indicators.tdx.compiler import (
    CompiledIndicator,
    IndicatorOutput,
    TdxError,
    compile_indicator,
    compile_indicator_cached,
    evaluate,
)

__all__ = [
    "CompiledIndicator",
    "IndicatorOutput",
    "TdxError",
    "compile_indicator",
    "compile_indicator_cached",
    "evaluate",
]
