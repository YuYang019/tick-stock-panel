"""市场风格 —— 大市值(机构) vs 小市值(题材) 的跷跷板。

心法「机构股熄火, 在 A 股炒题材的本质是做空」的规则化:
存量博弈下, 权重走强时资金从题材流出; 权重熄火时资金才回流题材博弈。
这里只客观判定当前风格与切换方向, 不做任何方向性建议。

口径:
- 分组按当日流通市值 (float_shares × raw_close) 排序: 前 10% 为机构组,
  后 30% 为题材组; 等权平均收益。
- style_spread = 小市值收益 - 大市值收益 (小数), 并给出 60 日分位。
- inst_breadth = 大市值组收盘站上 MA20 的占比。
- 风格四态 + 切换中。
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

import polars as pl

from app.price_limits import is_risk_warning_name

logger = logging.getLogger(__name__)

STYLES: tuple[str, ...] = ("题材主导", "机构主导", "普涨", "普跌", "切换中", "均衡")
LOOKBACK_DEFAULT = 60

DEFAULT_CONFIG: dict = {
    "lookback": LOOKBACK_DEFAULT,
    "inst_quantile": 0.10,
    "small_quantile": 0.30,
    "spread_high": 0.70,
    "spread_low": 0.30,
    "inst_breadth_low": 40.0,
    "inst_breadth_high": 60.0,
    "systemic_pct": 1.0,
}


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def merge_config(raw: dict | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return cfg
    for key in ("lookback",):
        v = raw.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            cfg[key] = max(20, min(250, v))
    for key in ("inst_quantile", "small_quantile"):
        v = _finite(raw.get(key))
        if v is not None:
            cfg[key] = max(0.01, min(0.9, v))
    for key in ("spread_high", "spread_low"):
        v = _finite(raw.get(key))
        if v is not None:
            cfg[key] = max(0.0, min(1.0, v))
    for key in ("inst_breadth_low", "inst_breadth_high", "systemic_pct"):
        v = _finite(raw.get(key))
        if v is not None:
            cfg[key] = v
    return cfg


def compute_daily_style(df: pl.DataFrame, config: dict | None = None) -> pl.DataFrame:
    """由 enriched 历史计算每日风格指标。

    输入列: symbol, date, close, raw_close, float_shares, ma20, change_pct, name
    输出: date, inst_ret, small_ret, style_spread, inst_breadth, small_breadth
    """
    cfg = merge_config(config)
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})
    if "name" in df.columns:
        df = df.filter(~pl.col("name").map_elements(is_risk_warning_name, return_dtype=pl.Boolean))
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})
    if "float_shares" not in df.columns:
        logger.info("market style: float_shares missing, skip")
        return pl.DataFrame(schema={"date": pl.Date})

    df = df.with_columns(
        (pl.col("float_shares").cast(pl.Float64) * pl.col("raw_close").cast(pl.Float64))
        .alias("_cap")
    ).filter(pl.col("_cap") > 0)

    # 每日按市值名次分组: 前 k 只为机构组, 后 m 只为题材组 (k/m 由股票数
    # 与分位比例四舍五入得到, 至少 1 只)。按名次而非市值阈值切分, 避免市值
    # 高度集中时阈值退化导致某一组为空。
    df = df.with_columns([
        pl.col("_cap").rank(method="ordinal").over("date").alias("_cap_rank"),
        pl.len().over("date").alias("_n"),
    ]).with_columns([
        (pl.col("_n") * float(cfg["inst_quantile"])).round().clip(1, None).alias("_k"),
        (pl.col("_n") * float(cfg["small_quantile"])).round().clip(1, None).alias("_m"),
    ])

    has_ma = "ma20" in df.columns
    df = df.with_columns([
        pl.when(pl.col("_cap_rank") > pl.col("_n") - pl.col("_k")).then(pl.lit("inst"))
        .when(pl.col("_cap_rank") <= pl.col("_m")).then(pl.lit("small"))
        .otherwise(pl.lit("mid"))
        .alias("_group"),
    ] + ([
        (pl.col("close") >= pl.col("ma20")).fill_null(False).alias("_above_ma20")
    ] if has_ma else []))

    aggs = [
        pl.col("change_pct").filter(pl.col("_group") == "inst").mean().alias("inst_ret"),
        pl.col("change_pct").filter(pl.col("_group") == "small").mean().alias("small_ret"),
        pl.col("change_pct").mean().alias("market_ret"),
        pl.len().alias("count"),
    ]
    if has_ma:
        aggs += [
            pl.col("_above_ma20").filter(pl.col("_group") == "inst").mean().alias("_inst_breadth"),
            pl.col("_above_ma20").filter(pl.col("_group") == "small").mean().alias("_small_breadth"),
        ]
    out = df.group_by("date").agg(aggs).sort("date")
    out = out.with_columns([
        (pl.col("small_ret") - pl.col("inst_ret")).alias("style_spread"),
    ])
    if has_ma:
        out = out.with_columns([
            (pl.col("_inst_breadth") * 100).alias("inst_breadth"),
            (pl.col("_small_breadth") * 100).alias("small_breadth"),
        ]).drop(["_inst_breadth", "_small_breadth"])
    else:
        out = out.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("inst_breadth"),
            pl.lit(None, dtype=pl.Float64).alias("small_breadth"),
        ])
    return out


def classify_style(daily: pl.DataFrame, config: dict | None = None) -> dict | None:
    """判定最新一日风格, 返回 {style, spread, spread_pct, inst_breadth, ...}。"""
    cfg = merge_config(config)
    if daily.is_empty():
        return None
    rows = daily.to_dicts()
    last = rows[-1]
    spread = _finite(last.get("style_spread"))
    # 最新日价差在近 lookback 日中的分位 (只用当日及之前的数据)
    history = [_finite(r.get("style_spread")) for r in rows[-int(cfg["lookback"]):]]
    history = [v for v in history if v is not None]
    spread_pct = None
    if spread is not None and len(history) >= 10:
        less = sum(1 for v in history if v < spread)
        equal = sum(1 for v in history if v == spread) - 1
        spread_pct = (less + equal / 2) / (len(history) - 1)
    inst_b = _finite(last.get("inst_breadth"))
    small_b = _finite(last.get("small_breadth"))
    market_ret = _finite(last.get("market_ret"))

    style = "均衡"
    inst_ret = _finite(last.get("inst_ret")) or 0.0
    small_ret = _finite(last.get("small_ret")) or 0.0
    if spread is not None:
        # 系统性行情优先: 两组同向且都超过阈值。
        systemic = (
            abs(inst_ret) * 100 >= float(cfg["systemic_pct"])
            and abs(small_ret) * 100 >= float(cfg["systemic_pct"])
        )
        if systemic and inst_ret > 0 and small_ret > 0:
            style = "普涨"
        elif systemic and inst_ret < 0 and small_ret < 0:
            style = "普跌"
        # 风格需要「方向 + 显著性」同时满足: 价差方向决定谁强, 价差在窗口内的
        # 分位达到 spread_high/spread_low 才算显著; 分位缺失 (样本不足) 时只看方向。
        # 机构组宽度作为二次确认: 题材主导要求机构组不强, 机构主导要求机构组不弱。
        significant_up = spread_pct is None or spread_pct >= float(cfg["spread_high"])
        significant_down = spread_pct is None or spread_pct <= float(cfg["spread_low"])
        if spread > 0 and significant_up and (
            inst_b is None or inst_b < float(cfg["inst_breadth_high"])
        ):
            style = "题材主导"
        elif spread < 0 and significant_down and (
            inst_b is None or inst_b > float(cfg["inst_breadth_low"])
        ):
            style = "机构主导"

    # 切换中: 价差 3 日变化方向与近 10 日均值相反, 且幅度足够大。
    # 不用分位判定风格, 因为价差分布很窄时分位会失真 (连续多日同值时尤其明显)。
    trend3 = None
    mean10 = None
    if len(rows) >= 4:
        prev = _finite(rows[-4].get("style_spread"))
        if prev is not None and spread is not None:
            trend3 = spread - prev
    window = [_finite(r.get("style_spread")) for r in rows[-10:]]
    window = [v for v in window if v is not None]
    if window:
        mean10 = sum(window) / len(window)
    if (
        style in {"题材主导", "机构主导", "均衡"}
        and trend3 is not None and mean10 is not None
        and trend3 * mean10 < 0
        and abs(trend3) >= 0.3 * abs(mean10)
    ):
        style = "切换中"

    return {
        "as_of": last["date"].isoformat() if hasattr(last["date"], "isoformat") else str(last["date"]),
        "style": style,
        "style_label": style,
        "spread": round((spread or 0) * 100, 2),
        "spread_pct": None if spread_pct is None else round(spread_pct, 3),
        "spread_trend3": round((trend3 or 0) * 100, 2) if trend3 is not None else None,
        "inst_ret": round((_finite(last.get("inst_ret")) or 0) * 100, 2),
        "small_ret": round((_finite(last.get("small_ret")) or 0) * 100, 2),
        "inst_breadth": None if inst_b is None else round(inst_b, 1),
        "small_breadth": None if small_b is None else round(small_b, 1),
        "market_ret": round((market_ret or 0) * 100, 2),
        "note": _STYLE_NOTES.get(style, ""),
    }


_STYLE_NOTES: dict[str, str] = {
    "题材主导": "机构熄火, 存量资金博弈, 短线情绪与题材逻辑优先",
    "机构主导": "资金回流权重, 题材易被抽血, 情绪周期退潮风险上升",
    "普涨": "系统性上涨, 风格差异被掩盖",
    "普跌": "系统性下跌, 个股 alpha 空间受限",
    "切换中": "风格价差正在反向收敛, 属切换窗口, 双向风险上升",
    "均衡": "大小市值分化不明显",
}


_CACHE_TTL_SECONDS = 60
_cache: dict[str, tuple[float, dict]] = {}


def get_style(repo, as_of: date | None = None, config: dict | None = None) -> dict | None:
    """对外入口: 取窗口数据 → 每日风格 → 最新一日判定 (缓存 60s)。"""
    import time

    cfg = merge_config(config)
    key = f"{as_of or 'latest'}:{cfg['lookback']}"
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        target = as_of
        if target is None:
            _df, latest = repo.get_enriched_latest()
            if latest is None:
                return None
            target = latest
        df = repo.get_enriched_history(target, int(cfg["lookback"]) + 5)
    except Exception as e:  # noqa: BLE001 — 风格只影响增强展示
        logger.warning("market style: enriched history unavailable: %s", e)
        return None
    if df is None or df.is_empty():
        return None
    keep = [c for c in (
        "symbol", "date", "name", "close", "raw_close", "float_shares",
        "ma20", "change_pct",
    ) if c in df.columns]
    daily = compute_daily_style(df.select(keep), cfg)
    result = classify_style(daily, cfg)
    if result:
        _cache[key] = (now, result)
    return result


def invalidate_cache() -> None:
    _cache.clear()
