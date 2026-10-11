"""通达信公式的变量/函数/修饰符清单 (编辑器函数面板 + 编译器名字解析的单一来源)。"""
from __future__ import annotations

# 行情变量 → enriched 列 (前复权价, 与图表一致; CONTRIBUTING §3.2)
VARIABLES: dict[str, str] = {
    "O": "open", "OPEN": "open",
    "H": "high", "HIGH": "high",
    "L": "low", "LOW": "low",
    "C": "close", "CLOSE": "close",
    "V": "volume", "VOL": "volume", "VOLUME": "volume",
    "AMOUNT": "amount", "AMO": "amount",
}

VARIABLE_DOCS: list[dict[str, str]] = [
    {"name": "O / OPEN", "desc": "开盘价 (前复权)"},
    {"name": "H / HIGH", "desc": "最高价 (前复权)"},
    {"name": "L / LOW", "desc": "最低价 (前复权)"},
    {"name": "C / CLOSE", "desc": "收盘价 (前复权)"},
    {"name": "V / VOL / VOLUME", "desc": "成交量 (手)"},
    {"name": "AMOUNT / AMO", "desc": "成交额 (元)"},
]

# (名称, 签名, 说明) — 顺序即面板展示顺序
FUNCTION_DOCS: list[dict[str, str]] = [
    {"name": "MA", "signature": "MA(X,N)", "desc": "N 日简单移动平均, 不足 N 根为空"},
    {"name": "EMA", "signature": "EMA(X,N)", "desc": "N 日指数移动平均, α=2/(N+1), 首值=首个 X"},
    {"name": "SMA", "signature": "SMA(X,N,M)", "desc": "移动平均 Y=(M×X+(N-M)×Y')/N, 首值=首个 X, M≤N"},
    {"name": "WMA", "signature": "WMA(X,N)", "desc": "N 日加权移动平均 (权重 1..N, 近期权重大)"},
    {"name": "REF", "signature": "REF(X,N)", "desc": "N 周期前的 X"},
    {"name": "HHV", "signature": "HHV(X,N)", "desc": "N 周期内最高值; N=0 为全部历史"},
    {"name": "LLV", "signature": "LLV(X,N)", "desc": "N 周期内最低值; N=0 为全部历史"},
    {"name": "SUM", "signature": "SUM(X,N)", "desc": "N 周期累加; N=0 为从第一个有效值起累加"},
    {"name": "STD", "signature": "STD(X,N)", "desc": "N 周期样本标准差"},
    {"name": "AVEDEV", "signature": "AVEDEV(X,N)", "desc": "N 周期平均绝对偏差"},
    {"name": "COUNT", "signature": "COUNT(COND,N)", "desc": "N 周期内条件成立的次数; N=0 为全部历史"},
    {"name": "EVERY", "signature": "EVERY(COND,N)", "desc": "N 周期内条件一直成立"},
    {"name": "EXIST", "signature": "EXIST(COND,N)", "desc": "N 周期内条件至少成立一次"},
    {"name": "CROSS", "signature": "CROSS(A,B)", "desc": "A 上穿 B: A>B 且前一周期 A<=B"},
    {"name": "IF", "signature": "IF(COND,A,B)", "desc": "条件成立取 A, 否则取 B"},
    {"name": "IFF", "signature": "IFF(COND,A,B)", "desc": "同 IF"},
    {"name": "MAX", "signature": "MAX(A,B)", "desc": "较大值"},
    {"name": "MIN", "signature": "MIN(A,B)", "desc": "较小值"},
    {"name": "ABS", "signature": "ABS(X)", "desc": "绝对值"},
    {"name": "ROUND", "signature": "ROUND(X)", "desc": "四舍五入取整"},
    {"name": "SQRT", "signature": "SQRT(X)", "desc": "平方根 (负数为空)"},
    {"name": "POW", "signature": "POW(X,Y)", "desc": "X 的 Y 次幂"},
    {"name": "LOG", "signature": "LOG(X)", "desc": "常用对数 (以 10 为底, 非正数为空)"},
    {"name": "LN", "signature": "LN(X)", "desc": "自然对数 (非正数为空)"},
    {"name": "BARSLAST", "signature": "BARSLAST(COND)", "desc": "上一次条件成立到当前的周期数 (结果依赖已加载历史长度)"},
    {"name": "BARSCOUNT", "signature": "BARSCOUNT(X)", "desc": "第一个有效值到当前的周期数 (结果依赖已加载历史长度)"},
    {"name": "BETWEEN", "signature": "BETWEEN(A,B,C)", "desc": "A 介于 B、C 之间 (含端点)"},
    {"name": "STICKLINE", "signature": "STICKLINE(COND,P1,P2,W,EMPTY)", "desc": "条件成立时在 P1~P2 间画柱, W 宽度, EMPTY≠0 为空心 (仅作为独立语句)"},
]

FUNCTION_NAMES: frozenset[str] = frozenset(item["name"] for item in FUNCTION_DOCS)

# 明确不支持且无法容错跳过的函数 (语义依赖未来数据或第一期不做)
UNSUPPORTED_FUNCTIONS: dict[str, str] = {
    "LONGCROSS": "LONGCROSS 暂不支持",
    "FILTER": "FILTER 含未来函数语义, 不支持",
    "ZIG": "ZIG 为未来函数, 不支持",
    "PEAK": "PEAK 为未来函数, 不支持",
    "TROUGH": "TROUGH 为未来函数, 不支持",
    "REFX": "REFX 引用未来数据, 不支持",
}

# 只用于绘图/文本/板块信息、不可表达为数值序列的标识符: 所在语句跳过并警告
UNDRAWABLE: frozenset[str] = frozenset({
    "DRAWTEXT", "DRAWTEXT_FIX", "DRAWTEXTABS", "DRAWTEXTREL", "DRAWICON", "DRAWNUMBER",
    "DRAWNUMBER_FIX", "DRAWBAND", "DRAWKLINE", "DRAWLINE", "DRAWSL", "DRAWGBK", "DRAWBMP",
    "DRAWNULL", "DRAWRECTREL", "PARTLINE", "POLYLINE", "PLOYLINE",
    "STRCAT", "VAR2STR", "CON2STR", "STR2CON", "STRLEFT", "STRRIGHT", "STRMID", "STRSPACE",
    "HYBLOCK", "DYBLOCK", "GNBLOCK", "FGBLOCK", "ZSBLOCK", "ZHBLOCK",
    "ISLASTBAR", "ISFIRSTBAR", "BACKSET",
})

NAMED_COLORS: dict[str, str] = {
    "RED": "#FF0000", "GREEN": "#00FF00", "BLUE": "#0000FF", "YELLOW": "#FFFF00",
    "WHITE": "#FFFFFF", "BLACK": "#000000", "MAGENTA": "#FF00FF", "CYAN": "#00FFFF",
    "GRAY": "#808080", "LIGRAY": "#C0C0C0", "LIRED": "#FF8080", "LIGREEN": "#80FF80",
    "LIBLUE": "#8080FF", "BROWN": "#804000",
}

# 修饰符 → 输出 kind
KIND_MODIFIERS: dict[str, str] = {
    "STICK": "stick", "COLORSTICK": "colorstick", "VOLSTICK": "volstick",
    "CIRCLEDOT": "dot", "POINTDOT": "dot",
}

MODIFIER_DOCS: list[dict[str, str]] = [
    {"name": "COLORRED / COLORGREEN / ...", "desc": "具名颜色: " + "/".join(NAMED_COLORS)},
    {"name": "COLORBBGGRR", "desc": "6 位十六进制颜色, 通达信 BGR 顺序 (如 COLOR00C0C0)"},
    {"name": "LINETHICK0-9", "desc": "线宽"},
    {"name": "NODRAW", "desc": "不绘制 (仍可被信号/策略引用)"},
    {"name": "DOTLINE", "desc": "虚线"},
    {"name": "STICK / COLORSTICK / VOLSTICK", "desc": "柱状线 / 红绿柱 / 成交量柱"},
    {"name": "CIRCLEDOT / POINTDOT", "desc": "点状线"},
]
