"""实时指数核心清单 — 产品级固定契约 (单一权威)。

指数展示层 (侧栏指数条 / 市场总览) 固定四只核心指数, 不开放配置:
- 数据源边界: TickFlow 与 fuyao 指数快照双源均完整覆盖, 无降级分歧;
- 后端消费方 (quote_service / overview / sector_monitor) 与前端 Layout
  统一引用此处, 不得各自维护副本。

监控规则的指数标的不受此限 — quote_service 会把启用规则的指数并入显式拉取。
"""

CORE_INDEX_NAMES: dict[str, str] = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000680.SH": "科创综指",
}

CORE_INDEX_SYMBOLS: tuple[str, ...] = tuple(CORE_INDEX_NAMES.keys())

# 个股相对强度的板块基准 — 与 abnormal_moves.board_of 的板块划分一一对应。
# 复盘、个股分析共用此映射, 避免两处各写一份 (口径漂移会让同一只股票在不同
# 报告里强弱结论相反)。北交所没有对应宽基指数, 退化为沪深300 并标记未匹配,
# 由调用方在结论里注明基准不完全对应。
BOARD_BENCHMARK: dict[str, tuple[str, str]] = {
    "创业板": ("399006.SZ", "创业板指"),
    "科创板": ("000680.SH", "科创综指"),
    "主板": ("000300.SH", "沪深300"),
    "北交所": ("000300.SH", "沪深300"),
}

# 北交所等无专属基准的板块: 基准可用但并非同类, 结论需降置信度
_BOARD_BENCHMARK_MISMATCHED = frozenset({"北交所"})


def board_benchmark(symbol: str) -> tuple[str, str, bool]:
    """个股 → (基准指数代码, 名称, 是否同类板块基准)。"""
    from app.services.abnormal_moves import board_of

    board = board_of(symbol)
    index_symbol, name = BOARD_BENCHMARK.get(board, BOARD_BENCHMARK["主板"])
    return index_symbol, name, board not in _BOARD_BENCHMARK_MISMATCHED
