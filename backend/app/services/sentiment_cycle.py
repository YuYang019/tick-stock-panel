"""A 股情绪周期 —— 六阶段状态机。

冰点 → 修复 → 发酵 → 高潮 → 分歧 → 退潮 → 冰点

定位是「市场环境层」: 只判定全市场所处阶段, 不给个股打分。阶段由每日
情绪指标经状态机逐日推进得出, 阈值与转移规则全部来自配置 (出厂默认
见 DEFAULT_CONFIG), 用户可覆盖。

指标口径 (全部基于 enriched 历史, 单位见注释):
- 涨停/跌停/炸板家数、首板数、最高连板;
- 晋级率: 昨日 n 板今日 n+1 板占比 (一字板不计入分母);
- 涨停溢价: 昨日涨停股今日平均涨跌幅 (小数);
- 大面: 昨日涨停今日跌幅 <= big_loss_pct 的家数;
- 高位核按钮: 昨日 >=3 板今日跌幅 <= -7% 的家数;
- 封板率、上涨占比、单日情绪分。

阶段判定是「合法转移表 + 加权条件 + 防抖」: 只有当前阶段允许的候选
阶段参与打分, 候选需连续 confirm_days 日满足才切换。
"""
from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from app.price_limits import is_no_limit_day, is_risk_warning_name

logger = logging.getLogger(__name__)

PHASES: tuple[str, ...] = ("冰点", "修复", "发酵", "高潮", "分歧", "退潮")

# 合法转移: 当前阶段 → 允许进入的候选阶段 (不含自身)。
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "冰点": ("修复", "退潮"),
    "修复": ("发酵", "退潮", "冰点"),
    "发酵": ("高潮", "分歧", "退潮", "冰点"),
    "高潮": ("分歧", "退潮", "冰点"),
    "分歧": ("发酵", "退潮", "冰点"),
    "退潮": ("冰点", "修复"),
}

LOOKBACK_DEFAULT = 60
CONFIRM_DAYS_DEFAULT = 2
BIG_LOSS_PCT_DEFAULT = -5.0
HIGH_NUKE_PCT_DEFAULT = -7.0
SCORE_THRESHOLD = 0.6
MIN_HISTORY_DAYS = 10


@dataclass(frozen=True)
class PhaseRule:
    """单个阶段的进入条件。

    conditions: [(metric_key, comparator, threshold, weight)]
    comparator: "gte" | "lte" | "gt" | "lt" | "pct_gte" | "pct_lte"
                pct_* 比较的是该指标在窗口内的分位 (0..1)。
    """

    conditions: tuple[tuple[str, str, float, float], ...]


DEFAULT_RULES: dict[str, PhaseRule] = {
    "冰点": PhaseRule((
        ("limit_up", "pct_lte", 0.20, 1.0),
        ("max_height", "lte", 3, 0.8),
        ("premium_lu", "lte", 0.0, 1.0),
        ("big_loss", "pct_gte", 0.60, 0.6),
    )),
    "修复": PhaseRule((
        ("premium_lu", "gt", 0.0, 1.2),
        ("first_board", "trend3_gt", 0, 1.0),
        ("big_loss", "trend3_lt", 0, 0.8),
        ("up_ratio", "gt", 0.45, 0.5),
    )),
    "发酵": PhaseRule((
        ("max_height", "trend3_gt", 0, 1.0),
        ("promo_1_2", "gte", 25.0, 1.0),
        ("seal_rate", "gte", 70.0, 0.8),
        ("premium_multi", "gt", 0.02, 0.8),
    )),
    "高潮": PhaseRule((
        ("limit_up", "pct_gte", 0.80, 1.0),
        ("max_height", "pct_gte", 0.80, 1.0),
        ("seal_rate", "gte", 80.0, 0.6),
        ("up_ratio", "gt", 0.60, 0.5),
    )),
    "分歧": PhaseRule((
        ("max_height", "trend3_lte", 0, 1.0),
        ("seal_rate_drop", "gte", 10.0, 1.0),
        ("premium_lu", "trend3_lt", 0, 0.8),
        ("broken", "pct_gte", 0.70, 0.6),
    )),
    "退潮": PhaseRule((
        ("height_broken", "gte", 1, 1.2),
        ("premium_multi", "lt", 0.0, 1.0),
        ("big_loss", "trend3_gt", 0, 0.8),
        ("high_nuke", "trend3_gt", 0, 0.8),
    )),
}


# 板块规则: 板块样本小、连板高度普遍低, 全市场的绝对阈值 (最高板≤3、封板率≥70%)
# 会让绝大多数板块落入冰点。这里改用板块自身历史分位 + 板块平均涨跌 + 上涨占比。
SECTOR_RULES: dict[str, PhaseRule] = {
    "冰点": PhaseRule((
        ("avg_pct", "pct_lte", 0.20, 1.0),
        ("limit_up", "pct_lte", 0.30, 0.8),
        ("up_ratio", "lt", 0.35, 1.0),
        ("premium_lu", "lte", 0.0, 0.5),
    )),
    "修复": PhaseRule((
        ("avg_pct", "trend3_gt", 0, 1.0),
        ("up_ratio", "gt", 0.50, 1.0),
        ("limit_up", "trend3_gt", 0, 0.8),
        ("premium_lu", "gt", 0.0, 0.5),
    )),
    "发酵": PhaseRule((
        ("avg_pct", "pct_gte", 0.70, 1.0),
        ("limit_up", "pct_gte", 0.70, 1.0),
        ("up_ratio", "gt", 0.60, 0.8),
        ("max_height", "trend3_gte", 0, 0.5),
    )),
    "高潮": PhaseRule((
        ("limit_up", "pct_gte", 0.90, 1.0),
        ("avg_pct", "pct_gte", 0.90, 1.0),
        ("up_ratio", "gt", 0.75, 0.8),
    )),
    "分歧": PhaseRule((
        ("avg_pct", "trend3_lt", 0, 1.0),
        ("broken", "pct_gte", 0.70, 1.0),
        ("premium_lu", "trend3_lt", 0, 0.8),
        ("up_ratio", "lt", 0.55, 0.5),
    )),
    "退潮": PhaseRule((
        ("avg_pct", "pct_lte", 0.30, 1.0),
        ("premium_lu", "lt", 0.0, 0.8),
        ("up_ratio", "lt", 0.40, 1.0),
        ("limit_up", "trend3_lt", 0, 0.8),
    )),
}


def _rules_to_config(rules: dict[str, PhaseRule]) -> dict:
    return {
        phase: {
            "conditions": [
                {"key": k, "cmp": c, "value": v, "weight": w}
                for k, c, v, w in rule.conditions
            ]
        }
        for phase, rule in rules.items()
    }


def _default_config() -> dict:
    return {
        "lookback": LOOKBACK_DEFAULT,
        "confirm_days": CONFIRM_DAYS_DEFAULT,
        "big_loss_pct": BIG_LOSS_PCT_DEFAULT,
        "high_nuke_pct": HIGH_NUKE_PCT_DEFAULT,
        "include_bj": False,
        "score_threshold": SCORE_THRESHOLD,
        # 市场风格联动 (心法「机构股熄火, 炒题材本质是做空」): 风格为机构主导/切换中时,
        # 退潮的进入阈值下调, 更早识别题材被抽血。0 = 关闭联动。
        "style_retreat_discount": 0.1,
        "phases": _rules_to_config(DEFAULT_RULES),
        "sector_phases": _rules_to_config(SECTOR_RULES),
    }


DEFAULT_CONFIG: dict = _default_config()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def merge_config(raw: dict | None) -> dict:
    """用户配置深合并到出厂默认; 非法值忽略, 未知阶段丢弃。"""
    merged = _default_config()
    if not isinstance(raw, dict):
        return merged

    for key in ("lookback", "confirm_days"):
        v = raw.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            merged[key] = max(1, min(250, v)) if key == "confirm_days" else max(2, min(250, v))
    for key in ("big_loss_pct", "high_nuke_pct", "score_threshold"):
        v = _finite(raw.get(key))
        if v is not None:
            merged[key] = v
    if isinstance(raw.get("include_bj"), bool):
        merged["include_bj"] = raw["include_bj"]

    v = _finite(raw.get("style_retreat_discount"))
    if v is not None:
        merged["style_retreat_discount"] = max(0.0, min(0.5, v))
    for section in ("phases", "sector_phases"):
        _merge_phase_rules(merged[section], raw.get(section))
    return merged


def _merge_phase_rules(target: dict, raw_phases) -> None:
    """按 (阶段, 指标 key) 合并用户覆盖; 未知阶段/指标丢弃, 非法比较符保留出厂值。"""
    if not isinstance(raw_phases, dict):
        return
    for phase, rule in target.items():
        user = raw_phases.get(phase)
        if not isinstance(user, dict):
            continue
        user_conds = user.get("conditions")
        if not isinstance(user_conds, list):
            continue
        by_key = {c.get("key"): c for c in user_conds if isinstance(c, dict)}
        for cond in rule["conditions"]:
            u = by_key.get(cond["key"])
            if not isinstance(u, dict):
                continue
            if u.get("cmp") in _ALLOWED_CMP:
                cond["cmp"] = u["cmp"]
            val = _finite(u.get("value"))
            if val is not None:
                cond["value"] = val
            weight = _finite(u.get("weight"))
            if weight is not None and weight >= 0:
                cond["weight"] = weight


_ALLOWED_CMP = frozenset({
    "gte", "lte", "gt", "lt",
    "pct_gte", "pct_lte",
    "trend3_gt", "trend3_lt", "trend3_gte", "trend3_lte",
})


def rules_from_config(cfg: dict, section: str = "phases") -> dict[str, PhaseRule]:
    return {
        phase: PhaseRule(tuple(
            (c["key"], c["cmp"], float(c["value"]), float(c["weight"]))
            for c in rule["conditions"]
        ))
        for phase, rule in cfg[section].items()
    }


# ================================================================
# 每日指标
# ================================================================

_METRIC_KEYS: tuple[str, ...] = (
    "limit_up", "limit_down", "broken", "first_board", "max_height",
    "height_broken", "seal_rate", "seal_rate_drop", "promo_1_2", "promo_2_3",
    "promo_3p", "premium_lu", "premium_multi", "premium_broken", "big_loss",
    "high_nuke", "up_ratio", "avg_pct",
)


def _filter_universe(df: pl.DataFrame, include_bj: bool) -> pl.DataFrame:
    """剔除 ST、新股无涨跌幅期、北交所 (可配)。"""
    if df.is_empty():
        return df
    keep = []
    listing = None
    if "listing_date" in df.columns:
        listing = df["listing_date"].to_list()
    names = df["name"].to_list() if "name" in df.columns else [None] * len(df)
    symbols = df["symbol"].to_list()
    dates = df["date"].to_list()
    for i, (sym, name, d) in enumerate(zip(symbols, names, dates, strict=True)):
        excluded = (
            is_risk_warning_name(name)
            or (not include_bj and str(sym).endswith(".BJ"))
            or (listing is not None and is_no_limit_day(sym, listing[i], d))
        )
        keep.append(not excluded)
    return df.filter(pl.Series(keep, dtype=pl.Boolean))


def compute_daily_metrics(df: pl.DataFrame, *, include_bj: bool = False,
                          big_loss_pct: float = BIG_LOSS_PCT_DEFAULT,
                          high_nuke_pct: float = HIGH_NUKE_PCT_DEFAULT) -> pl.DataFrame:
    """由 enriched 历史 (单股多日) 聚合出每日市场指标。

    输入列: symbol, date, change_pct, raw_close, raw_high, raw_low, turnover_rate,
            consecutive_limit_ups, limit_up_price, limit_down_price,
            (可选) name, listing_date
    输出: 每日一行, 列为 _METRIC_KEYS + date。
    """
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})
    if "consecutive_limit_ups" not in df.columns:
        # instruments 维表未加载时 enriched 缓存不含连板/涨跌停判定价, 无法计算情绪指标:
        # 明确返回空 (调用方降级为「数据不足」), 不能用其他列近似。
        logger.info("sentiment cycle: consecutive_limit_ups missing, skip")
        return pl.DataFrame(schema={"date": pl.Date})

    if "name" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("name"))
    if "listing_date" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Date).alias("listing_date"))
    df = _filter_universe(df, include_bj)
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})

    has_limit_price = {"limit_up_price", "limit_down_price"}.issubset(df.columns)
    if not has_limit_price:
        logger.info("sentiment cycle: limit prices missing, falling back to consecutive_limit_ups")
    if has_limit_price:
        df = df.with_columns([
            (pl.col("raw_close") >= pl.col("limit_up_price")).fill_null(False).alias("_lu"),
            (pl.col("raw_close") <= pl.col("limit_down_price")).fill_null(False).alias("_ld"),
            (
                (pl.col("raw_high") >= pl.col("limit_up_price"))
                & (pl.col("raw_close") < pl.col("limit_up_price"))
            ).fill_null(False).alias("_broken"),
            # 一字板: 最高价 = 最低价 且涨停 (无法换手, 不计入晋级率分母)
            (
                (pl.col("raw_close") >= pl.col("limit_up_price"))
                & (pl.col("raw_high") <= pl.col("raw_low") + 1e-6)
            ).fill_null(False).alias("_one_word"),
        ])
    else:
        df = df.with_columns([
            pl.col("consecutive_limit_ups").fill_null(0).gt(0).alias("_lu"),
            pl.lit(False).alias("_ld"),
            pl.lit(False).alias("_broken"),
            pl.lit(False).alias("_one_word"),
        ])

    df = df.with_columns([
        pl.col("consecutive_limit_ups").fill_null(0).cast(pl.Int32).alias("_boards"),
        pl.col("change_pct").cast(pl.Float64).alias("_pct"),
    ])

    # 昨日状态 (逐股平移), 用于晋级率/溢价/断板判定
    df = df.sort(["symbol", "date"]).with_columns([
        pl.col("_lu").shift(1).over("symbol").alias("_prev_lu"),
        pl.col("_boards").shift(1).over("symbol").alias("_prev_boards"),
        pl.col("_one_word").shift(1).over("symbol").alias("_prev_one_word"),
        pl.col("_broken").shift(1).over("symbol").alias("_prev_broken"),
    ])

    df = df.with_columns([
        (pl.col("_lu") & (pl.col("_boards") == 1)).alias("_first"),
        (pl.col("_lu") & (pl.col("_boards") >= 2)).alias("_multi"),
        # 大面: 昨日涨停今日跌幅 <= big_loss_pct
        (pl.col("_prev_lu").fill_null(False) & (pl.col("_pct") * 100 <= big_loss_pct)).alias("_big_loss"),
        # 高位核按钮: 昨日 >=3 板今日跌幅 <= high_nuke_pct
        (
            (pl.col("_prev_boards").fill_null(0) >= 3)
            & (pl.col("_pct") * 100 <= high_nuke_pct)
        ).alias("_high_nuke"),
        # 晋级: 昨日 n 板且非一字, 今日 n+1 板
        (
            pl.col("_prev_lu").fill_null(False)
            & (~pl.col("_prev_one_word").fill_null(False))
            & (pl.col("_prev_boards") == 1)
            & pl.col("_lu")
        ).alias("_promo_1_2"),
        (
            pl.col("_prev_lu").fill_null(False)
            & (~pl.col("_prev_one_word").fill_null(False))
            & (pl.col("_prev_boards") == 2)
            & pl.col("_lu")
        ).alias("_promo_2_3"),
        (
            pl.col("_prev_lu").fill_null(False)
            & (~pl.col("_prev_one_word").fill_null(False))
            & (pl.col("_prev_boards") >= 3)
            & pl.col("_lu")
        ).alias("_promo_3p"),
        # 晋级率分母: 昨日 n 板且非一字
        (
            pl.col("_prev_lu").fill_null(False)
            & (~pl.col("_prev_one_word").fill_null(False))
            & (pl.col("_prev_boards") == 1)
        ).alias("_denom_1_2"),
        (
            pl.col("_prev_lu").fill_null(False)
            & (~pl.col("_prev_one_word").fill_null(False))
            & (pl.col("_prev_boards") == 2)
        ).alias("_denom_2_3"),
        (
            pl.col("_prev_lu").fill_null(False)
            & (~pl.col("_prev_one_word").fill_null(False))
            & (pl.col("_prev_boards") >= 3)
        ).alias("_denom_3p"),
    ])

    grouped = df.group_by("date").agg([
        pl.col("_lu").sum().alias("limit_up"),
        pl.col("_ld").sum().alias("limit_down"),
        pl.col("_broken").sum().alias("broken"),
        pl.col("_first").sum().alias("first_board"),
        pl.col("_boards").max().fill_null(0).alias("max_height"),
        pl.col("_big_loss").sum().alias("big_loss"),
        pl.col("_high_nuke").sum().alias("high_nuke"),
        pl.col("_promo_1_2").sum().alias("_p12"),
        pl.col("_promo_2_3").sum().alias("_p23"),
        pl.col("_promo_3p").sum().alias("_p3p"),
        pl.col("_denom_1_2").sum().alias("_d12"),
        pl.col("_denom_2_3").sum().alias("_d23"),
        pl.col("_denom_3p").sum().alias("_d3p"),
        pl.col("_pct").mean().alias("avg_pct"),
        (pl.col("_pct") > 0).mean().alias("up_ratio"),
        # 溢价: 昨日涨停股今日平均涨幅
        pl.col("_pct").filter(pl.col("_prev_lu").fill_null(False)).mean().alias("premium_lu"),
        pl.col("_pct").filter(
            pl.col("_prev_boards").fill_null(0) >= 2
        ).mean().alias("premium_multi"),
        pl.col("_pct").filter(pl.col("_prev_broken").fill_null(False)).mean().alias("premium_broken"),
        # 昨日最高板今日未涨停 (断板)。昨日最高板 < 2 时没有「高度」可断:
        # 否则无连板日 max=0, 全市场都会被当成「昨日最高板」而误报断板。
        pl.col("_prev_boards").fill_null(0).max().alias("_prev_top"),
        pl.col("_lu").filter(
            pl.col("_prev_boards").fill_null(0) == pl.col("_prev_boards").fill_null(0).max()
        ).len().alias("_top_n"),
        pl.col("_lu").filter(
            pl.col("_prev_boards").fill_null(0) == pl.col("_prev_boards").fill_null(0).max()
        ).sum().alias("_top_sealed"),
    ]).sort("date")

    grouped = grouped.with_columns([
        pl.when((pl.col("limit_up") + pl.col("broken")) > 0)
        .then(pl.col("limit_up") / (pl.col("limit_up") + pl.col("broken")) * 100)
        .otherwise(0.0).alias("seal_rate"),
        (pl.col("_p12") / pl.col("_d12").replace(0, None) * 100).fill_null(0.0).alias("promo_1_2"),
        (pl.col("_p23") / pl.col("_d23").replace(0, None) * 100).fill_null(0.0).alias("promo_2_3"),
        (pl.col("_p3p") / pl.col("_d3p").replace(0, None) * 100).fill_null(0.0).alias("promo_3p"),
        # 昨日最高板 (>=2 板) 今日全部未封 → 1; 昨日无连板时不判定 (0)
        pl.when((pl.col("_prev_top") >= 2) & (pl.col("_top_n") > 0) & (pl.col("_top_sealed") == 0))
        .then(pl.lit(1)).otherwise(pl.lit(0)).alias("height_broken"),
    ]).with_columns(
        # 封板率回落幅度 (百分点): 与前一交易日比较
        pl.col("seal_rate").diff().fill_null(0.0).alias("seal_rate_drop"),
    )

    return grouped.select(["date", *_METRIC_KEYS]).sort("date")


def _add_derived(grouped: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """分位与 3 日斜率: 每个指标一个 _pct_ 与 _t3_ 辅助列。

    分位是滚动的: 第 i 日只与自身及之前最多 lookback 日比较 (含当日),
    不得使用之后的数据 (CONTRIBUTING §5.3, 历史阶段判定否则会引入未来函数)。
    样本不足 MIN_HISTORY_DAYS 时分位为 null, 依赖分位的条件不成立。
    取值 = 窗口内严格小于当日值的个数 + 相等个数的一半, 再除以 (窗口长度 - 1),
    使窗口最大值为 1.0、最小值为 0.0, 极值只在真正创新高/新低时出现。
    """
    if grouped.is_empty():
        return grouped
    window = max(int(lookback), 2)
    out = grouped
    for key in _METRIC_KEYS:
        values = [None if v is None else float(v) for v in grouped[key].to_list()]
        pcts: list[float | None] = []
        for i, current in enumerate(values):
            hist = [v for v in values[max(0, i - window + 1):i + 1] if v is not None]
            if current is None or len(hist) < MIN_HISTORY_DAYS:
                pcts.append(None)
                continue
            less = sum(1 for v in hist if v < current)
            equal = sum(1 for v in hist if v == current) - 1   # 不计自身
            pcts.append((less + equal / 2) / (len(hist) - 1))
        col = pl.col(key).cast(pl.Float64)
        out = out.with_columns([
            pl.Series(f"_pct_{key}", pcts, dtype=pl.Float64),
            (col - col.shift(3)).alias(f"_t3_{key}"),
        ])
    return out


# ================================================================
# 阶段判定
# ================================================================

@dataclass
class CycleState:
    as_of: date
    phase: str
    pending: str | None
    confidence: float
    days_in_phase: int
    evidence: list[str] = field(default_factory=list)
    next_watch: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    extreme: str = ""          # "" | "赚钱" | "亏钱"
    extreme_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "phase": self.phase,
            "pending": self.pending,
            "confidence": round(self.confidence, 2),
            "days_in_phase": self.days_in_phase,
            "evidence": self.evidence,
            "next_watch": self.next_watch,
            "metrics": self.metrics,
            "extreme": self.extreme,
            "extreme_items": self.extreme_items,
        }


_EVIDENCE_FMT: dict[str, str] = {
    "limit_up": "涨停 {v:.0f} 家",
    "limit_down": "跌停 {v:.0f} 家",
    "broken": "炸板 {v:.0f} 家",
    "first_board": "首板 {v:.0f} 家",
    "max_height": "最高连板 {v:.0f} 板",
    "seal_rate": "封板率 {v:.0f}%",
    "premium_lu": "昨日涨停溢价 {v:+.2f}%",
    "premium_multi": "连板溢价 {v:+.2f}%",
    "big_loss": "大面 {v:.0f} 家",
    "high_nuke": "高位核按钮 {v:.0f} 家",
    "up_ratio": "上涨占比 {v:.0f}%",
    "promo_1_2": "1进2 {v:.0f}%",
    "avg_pct": "平均涨跌 {v:+.2f}%",
}


def _cond_holds(row: dict, key: str, cmp: str, value: float) -> bool:
    if cmp.startswith("pct_"):
        actual = row.get(f"_pct_{key}")
        op = cmp[4:]
    elif cmp.startswith("trend3_"):
        actual = row.get(f"_t3_{key}")
        op = cmp[7:]
    else:
        actual = row.get(key)
        op = cmp
    if actual is None:
        return False
    actual = float(actual)
    if op == "gte":
        return actual >= value
    if op == "lte":
        return actual <= value
    if op == "gt":
        return actual > value
    if op == "lt":
        return actual < value
    return False


def _score_phase(row: dict, rule: PhaseRule) -> tuple[float, list[str]]:
    total = 0.0
    hit = 0.0
    evidence: list[str] = []
    for key, cmp, value, weight in rule.conditions:
        total += weight
        if _cond_holds(row, key, cmp, value):
            hit += weight
            fmt = _EVIDENCE_FMT.get(key)
            if fmt:
                raw = row.get(key)
                shown = raw * 100 if key.startswith("premium_") or key in {"up_ratio", "avg_pct"} else raw
                if shown is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        evidence.append(fmt.format(v=float(shown)))
    return (hit / total if total > 0 else 0.0), evidence


_NEXT_WATCH: dict[str, str] = {
    "冰点": "涨停家数与连板高度能否止跌回升、是否出现新题材首板",
    "修复": "涨停溢价能否持续为正、首板数量是否继续增加",
    "发酵": "最高板能否继续抬升、晋级率与封板率是否维持",
    "高潮": "封板率是否见顶回落、高位股是否出现放量分歧",
    "分歧": "最高板是否断板、连板溢价是否转负",
    "退潮": "亏钱效应是否收敛、是否出现冰点后的首个修复信号",
}


def classify(
    metrics: pl.DataFrame,
    config: dict | None = None,
    *,
    section: str = "phases",
    style: str | None = None,
) -> list[CycleState]:
    """逐日推进状态机, 返回每日状态 (按日期升序)。

    section: "phases" 为全市场规则, "sector_phases" 为板块规则。
    style: 当前市场风格; 机构主导/切换中时「退潮」进入阈值下调
           style_retreat_discount (只影响最新一日, 历史风格未知不追溯)。
    """
    cfg = merge_config(config)
    rules = rules_from_config(cfg, section)
    confirm_days = int(cfg["confirm_days"])
    threshold = float(cfg["score_threshold"])

    enriched = _add_derived(metrics, int(cfg["lookback"]))
    rows = enriched.to_dicts()
    if not rows:
        return []

    states: list[CycleState] = []
    phase = _initial_phase(rows[0], rules, threshold)
    days_in_phase = 1
    candidate: str | None = None
    candidate_days = 0
    pending: str | None = None

    for i, row in enumerate(rows):
        if i > 0:
            best_phase = None
            best_score = 0.0
            is_last = i == len(rows) - 1
            for target in TRANSITIONS.get(phase, ()):
                score, _evidence = _score_phase(row, rules[target])
                need = threshold
                if target == "退潮" and is_last and style in {"机构主导", "切换中"}:
                    need = max(0.0, threshold - float(cfg.get("style_retreat_discount") or 0))
                if score >= need and score > best_score:
                    best_phase, best_score = target, score
            if best_phase is None:
                candidate = None
                candidate_days = 0
                pending = None
            elif best_phase == candidate:
                candidate_days += 1
            else:
                candidate = best_phase
                candidate_days = 1

            if candidate and candidate_days >= confirm_days:
                phase = candidate
                days_in_phase = 1
                candidate = None
                candidate_days = 0
                pending = None
            else:
                days_in_phase += 1
                pending = candidate if candidate_days > 0 else None

            confidence, evidence = _score_phase(row, rules[phase])
        else:
            confidence, evidence = _score_phase(row, rules[phase])

        states.append(CycleState(
            as_of=row["date"],
            phase=phase,
            pending=pending,
            confidence=confidence,
            days_in_phase=days_in_phase,
            evidence=evidence[:5],
            next_watch={phase: _NEXT_WATCH[phase]},
            metrics=_public_metrics(row),
            **_extreme_of(row),
        ))
    return states


_EXTREME_EUPHORIA = ("limit_up", "max_height", "premium_lu", "seal_rate")
_EXTREME_PANIC = ("big_loss", "limit_down", "high_nuke")
_EXTREME_LABELS = {
    "limit_up": "涨停家数", "max_height": "最高连板", "premium_lu": "涨停溢价",
    "seal_rate": "封板率", "big_loss": "大面数", "limit_down": "跌停家数",
    "high_nuke": "高位核按钮",
}


def _extreme_of(row: dict) -> dict[str, Any]:
    """极致赚钱/亏钱效应预警 (心法: 极致赚钱后常有极致亏钱)。

    只做标注, 不参与阶段判定, 避免与状态机冲突。
    """
    def pct(key: str) -> float | None:
        v = row.get(f"_pct_{key}")
        return None if v is None else float(v)

    euphoria = [k for k in _EXTREME_EUPHORIA if (pct(k) or 0) >= 0.95]
    panic = [k for k in _EXTREME_PANIC if (pct(k) or 0) >= 0.95]
    if len(euphoria) >= 3:
        return {"extreme": "赚钱", "extreme_items": [_EXTREME_LABELS[k] for k in euphoria]}
    if len(panic) >= 2 or (pct("premium_lu") is not None and pct("premium_lu") <= 0.05):
        items = [_EXTREME_LABELS[k] for k in panic]
        if pct("premium_lu") is not None and pct("premium_lu") <= 0.05:
            items.append("涨停溢价处于低位")
        return {"extreme": "亏钱", "extreme_items": items}
    return {"extreme": "", "extreme_items": []}


def _initial_phase(row: dict, rules: dict[str, PhaseRule], threshold: float) -> str:
    """冷启动: 取命中度最高的阶段 (平手按 PHASES 顺序)。"""
    best, best_score = PHASES[0], -1.0
    for phase in PHASES:
        score, _ = _score_phase(row, rules[phase])
        if score > best_score:
            best, best_score = phase, score
    return best


def _public_metrics(row: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _METRIC_KEYS:
        val = row.get(key)
        if isinstance(val, float):
            out[key] = round(val, 4)
        elif val is not None:
            out[key] = val
    for key in ("premium_lu", "premium_multi", "premium_broken", "avg_pct"):
        if isinstance(out.get(key), float):
            out[key] = round(out[key] * 100, 2)  # 对外统一为百分数
    return out


# ================================================================
# 对外入口 (带进程内缓存)
# ================================================================

_CACHE_TTL_SECONDS = 60
_cache: dict[str, tuple[float, dict]] = {}


def get_cycle(repo, as_of: date | None = None, config: dict | None = None,
              style: str | None = None) -> dict | None:
    """返回指定日期的情绪周期状态; 无数据返回 None。

    结果按 (as_of, lookback) 缓存 60 秒; enriched 数据刷新后日期变化自然失效。
    """
    import time

    cfg = merge_config(config)
    lookback = int(cfg["lookback"])
    cache_key = f"{as_of or 'latest'}:{lookback}:{cfg['include_bj']}:{style or ''}"
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    metrics = load_daily_metrics(repo, as_of, lookback, cfg)
    if metrics is None or metrics.is_empty():
        return None
    states = classify(metrics, cfg, style=style)
    if not states:
        return None
    target = states[-1]
    payload = {
        **target.to_dict(),
        "history": [
            {"date": s.as_of.isoformat(), "phase": s.phase} for s in states[-20:]
        ],
    }
    _cache[cache_key] = (now, payload)
    return payload


def invalidate_cache() -> None:
    _cache.clear()
    _sector_cache.clear()


def load_daily_metrics(repo, as_of: date | None, lookback: int, cfg: dict) -> pl.DataFrame | None:
    """从 enriched 历史缓存取窗口数据并聚合为每日指标。

    窗口需要 lookback + 5 个交易日: 晋级率/溢价依赖前一日, 状态机初始阶段也
    需要若干天。缓存不覆盖时返回 None (由调用方降级)。
    """
    try:
        target = as_of
        if target is None:
            _df_latest, latest_date = repo.get_enriched_latest()
            if latest_date is None:
                span = repo.get_enriched_history_span()
                latest_date = span[1] if span else None
            if latest_date is None:
                return None
            target = latest_date
        df = repo.get_enriched_history(target, lookback + 10)
    except Exception as e:  # noqa: BLE001 — 周期只影响增强展示, 失败不阻断主流程
        logger.warning("sentiment cycle: enriched history unavailable: %s", e)
        return None
    if df is None or df.is_empty():
        return None
    keep = [
        c for c in (
            "symbol", "date", "name", "change_pct",
            "raw_close", "raw_high", "raw_low", "turnover_rate",
            "consecutive_limit_ups", "limit_up_price", "limit_down_price",
        ) if c in df.columns
    ]
    return compute_daily_metrics(
        df.select(keep),
        include_bj=bool(cfg["include_bj"]),
        big_loss_pct=float(cfg["big_loss_pct"]),
        high_nuke_pct=float(cfg["high_nuke_pct"]),
    )


def public_config(raw: dict | None) -> dict:
    """供 API 返回: 合并后的配置 + 阶段顺序 + 合法转移。"""
    return {
        "config": merge_config(raw),
        "phases": list(PHASES),
        "transitions": {k: list(v) for k, v in TRANSITIONS.items()},
        "factory": _default_config(),
    }


def recap_block(cycle: dict | None) -> str:
    """复盘提示词用的精简文本块。"""
    if not cycle:
        return ""
    lines = [
        f"- 当前阶段: {cycle['phase']}"
        + (f" (候选: {cycle['pending']})" if cycle.get("pending") else ""),
        f"- 连续天数: {cycle.get('days_in_phase', 0)} 天, 判定置信度 {cycle.get('confidence', 0):.2f}",
    ]
    if cycle.get("evidence"):
        lines.append("- 判定依据: " + "; ".join(cycle["evidence"]))
    for phase, watch in (cycle.get("next_watch") or {}).items():
        lines.append(f"- 观察要点({phase}): {watch}")
    if cycle.get("extreme"):
        lines.append(
            f"- 极致效应预警: {cycle['extreme']}效应处于极值 ("
            + "、".join(cycle.get("extreme_items") or [])
            + "); 极致赚钱效应之后常伴随极致亏钱效应, 反之亦然"
        )
    history = cycle.get("history") or []
    if history:
        lines.append("- 近 10 日阶段: " + " → ".join(f"{h['date'][5:]}{h['phase']}" for h in history[-10:]))
    return "\n".join(lines)


# ================================================================
# 板块周期 (心法「万物皆周期」在板块层的落地)
# ================================================================
#
# 复用全市场的指标口径与状态机, 只把「宇宙」换成单个板块的成分股。板块样本
# 远小于全市场, 绝对家数阈值 (如冰点「涨停分位≤0.2」) 仍按板块自身历史分位
# 判定, 因此同一套出厂规则可直接复用; 成分股过少的板块统计噪声大, 不计算。

SECTOR_MIN_MEMBERS = 10


def tradable_sector(name: str, member_keys: set[str] | None, filter_cfg: dict | None = None) -> bool:
    """是否为可操作的题材/行业板块 (复用市场主线的过滤口径)。

    融资融券/深股通/国企改革这类宽基与风格标签成分股数千只, 既不能代表个股的
    「树干」, 也没有板块周期可言; 按主线过滤的成员数上限与黑名单排除。
    member_keys 同时含带后缀与裸代码两种键, 成员数按一半估算。
    """
    from app.services.market_mainline import _resolve_filter_config

    cfg = _resolve_filter_config(filter_cfg)
    if name in cfg["blacklist"]:
        return False
    if member_keys is None:
        return True
    approx = len({k.split(".", 1)[0] for k in member_keys})
    return cfg["min_members"] <= approx <= cfg["max_members"]
SECTOR_MAX_COUNT = 40
_SECTOR_CACHE_TTL_SECONDS = 300
_sector_cache: dict[str, tuple[float, dict]] = {}


def _sector_states(
    df: pl.DataFrame,
    members: dict[str, set[str]],
    sectors: list[str],
    cfg: dict,
) -> dict[str, dict]:
    """逐板块计算每日指标并跑状态机, 返回 {板块: 最新状态 dict}。"""
    out: dict[str, dict] = {}
    if df.is_empty():
        return out
    bare = df.with_columns(pl.col("symbol").str.split(".").list.first().alias("_bare"))
    for sector in sectors:
        keys = members.get(sector) or set()
        if not keys:
            continue
        sub = bare.filter(pl.col("symbol").is_in(list(keys)) | pl.col("_bare").is_in(list(keys)))
        if sub["symbol"].n_unique() < SECTOR_MIN_MEMBERS:
            continue
        metrics = compute_daily_metrics(
            sub.drop("_bare"),
            include_bj=bool(cfg["include_bj"]),
            big_loss_pct=float(cfg["big_loss_pct"]),
            high_nuke_pct=float(cfg["high_nuke_pct"]),
        )
        if metrics.is_empty():
            continue
        states = classify(metrics, cfg, section="sector_phases")
        if not states:
            continue
        last = states[-1]
        out[sector] = {
            "phase": last.phase,
            "pending": last.pending,
            "confidence": round(last.confidence, 2),
            "days_in_phase": last.days_in_phase,
            "members": int(sub["symbol"].n_unique()),
            "limit_up": last.metrics.get("limit_up"),
            "max_height": last.metrics.get("max_height"),
            "avg_pct": last.metrics.get("avg_pct"),
            "evidence": last.evidence[:3],
            "history": [s.phase for s in states[-10:]],
        }
    return out


def get_sector_cycles(
    repo,
    as_of: date | None = None,
    config: dict | None = None,
    *,
    sectors: list[str] | None = None,
    kind: str = "concept",
) -> dict[str, dict]:
    """板块周期: {板块名: {phase, pending, confidence, members, ...}}。

    sectors 为空时只算当日涨幅前 SECTOR_MAX_COUNT 的板块 (控制开销);
    显式传入时按给定顺序计算, 同样截断到上限。结果缓存 5 分钟。
    归属映射缺失或历史数据不足时返回空字典。
    """
    import time

    from app.services.market_overview_builder import dimension_rank_all, sector_member_map

    cfg = merge_config(config)
    level = 2 if kind == "industry" else None
    wanted = tuple(sectors or ())
    cache_key = f"{as_of or 'latest'}:{kind}:{cfg['lookback']}:{','.join(wanted)}"
    now = time.monotonic()
    cached = _sector_cache.get(cache_key)
    if cached and now - cached[0] < _SECTOR_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        members = sector_member_map(repo, kind, level)
    except Exception as e:  # noqa: BLE001 — 板块周期是增强信息, 失败不阻断
        logger.warning("sector cycle: membership unavailable: %s", e)
        return {}
    if not members:
        return {}

    target_list = list(wanted)
    if not target_list:
        try:
            df_latest, _ = repo.get_enriched_latest()
            cols = [c for c in ("symbol", "name", "change_pct", "amount") if c in df_latest.columns]
            ranked = dimension_rank_all(df_latest.select(cols).to_dicts(), repo, kind, level=level)
            target_list = [str(it["name"]) for it in ranked]
        except Exception as e:  # noqa: BLE001
            logger.warning("sector cycle: rank unavailable: %s", e)
            return {}
    target_list = [
        s for s in target_list if s in members and tradable_sector(s, members[s])
    ][:SECTOR_MAX_COUNT]
    if not target_list:
        return {}

    try:
        target = as_of
        if target is None:
            _df, target = repo.get_enriched_latest()
        if target is None:
            return {}
        df = repo.get_enriched_history(target, int(cfg["lookback"]) + 10)
    except Exception as e:  # noqa: BLE001
        logger.warning("sector cycle: enriched history unavailable: %s", e)
        return {}
    if df is None or df.is_empty():
        return {}
    keep = [c for c in (
        "symbol", "date", "name", "change_pct", "raw_close", "raw_high", "raw_low",
        "turnover_rate", "consecutive_limit_ups", "limit_up_price", "limit_down_price",
    ) if c in df.columns]
    result = _sector_states(df.select(keep), members, target_list, cfg)
    _sector_cache[cache_key] = (now, result)
    return result


def sector_recap_block(sector_cycles: dict[str, dict], limit: int = 8) -> str:
    """复盘提示词用的板块周期摘要 (按传入顺序, 即涨幅排名顺序)。"""
    if not sector_cycles:
        return ""
    lines = []
    for name, st in list(sector_cycles.items())[:limit]:
        extra = f", 候选 {st['pending']}" if st.get("pending") else ""
        avg = st.get("avg_pct")
        avg_s = f", 平均 {avg:+.2f}%" if isinstance(avg, (int, float)) else ""
        lines.append(
            f"- {name}: {st['phase']}(第 {st['days_in_phase']} 天{extra}), "
            f"成分 {st['members']} 只{avg_s}, 涨停 {st.get('limit_up', 0)}, 最高 {st.get('max_height', 0)} 板"
        )
    return "\n".join(lines)
