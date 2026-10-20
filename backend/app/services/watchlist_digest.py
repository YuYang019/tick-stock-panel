"""复盘·自选股事实包 —— 规则层先算好, 模型只解读。

输出两类内容:
- rows: 总表行 (代码/名称/涨跌/相对强弱/趋势/周期角色/树干/分歧状态/信号), 由代码
  直接拼进 Markdown, 不让模型重排数字;
- detail: 异动分最高的若干只, 附支撑压力位、板块阶段、近期新闻 (可选)。

数据来源与 /api/watchlist/enriched 同源 (enriched 最新日), 历史窗口只取自选标的,
避免全市场扫描。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from app.market_time import cn_today
from app.services import web_search

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "enabled": True,
    "group_ids": [],
    "max_rows": 30,
    "max_detail": 10,
    "news_top": 5,
}

_SIGNAL_LABELS = {
    "signal_limit_up": "涨停",
    "signal_broken_limit_up": "炸板",
    "signal_limit_down": "跌停",
    "signal_volume_surge": "放量",
    "signal_ma_golden_5_20": "均线金叉",
    "signal_ma_dead_5_20": "均线死叉",
    "signal_macd_golden": "MACD金叉",
    "signal_macd_dead": "MACD死叉",
    "signal_n_day_high": "N日新高",
    "signal_n_day_low": "N日新低",
    "signal_boll_breakout_upper": "突破布林上轨",
    "signal_boll_breakdown_lower": "跌破布林下轨",
    "signal_ma20_breakout": "站上MA20",
    "signal_ma20_breakdown": "跌破MA20",
}


def normalize_config(raw: dict | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return cfg
    if isinstance(raw.get("enabled"), bool):
        cfg["enabled"] = raw["enabled"]
    gids = raw.get("group_ids")
    if isinstance(gids, list):
        cfg["group_ids"] = [str(g) for g in gids if str(g).strip()][:50]
    for key, low, high in (("max_rows", 1, 100), ("max_detail", 1, 30), ("news_top", 0, 10)):
        v = raw.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            cfg[key] = max(low, min(high, v))
    return cfg


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:+.2f}%"


def _fmt_pp(v: float | None) -> str:
    """超额收益 (百分点) 展示: +6.79% 表示跑赢基准 6.79 个百分点。"""
    return "—" if v is None else f"{v * 100:+.2f}%"


def _trend_label(row: dict) -> str:
    close, ma5, ma20, ma60 = (_num(row.get(k)) for k in ("close", "ma5", "ma20", "ma60"))
    if None in (close, ma5, ma20, ma60):
        return "—"
    if ma5 > ma20 > ma60 and close >= ma20:
        return "多头"
    if ma5 < ma20 < ma60 and close <= ma20:
        return "空头"
    return "缠绕"


def _consensus_label(row: dict) -> str:
    """分歧/一致状态 (简化规则, P4 由心法特征层接管)。"""
    pct = _num(row.get("change_pct")) or 0.0
    turnover = _num(row.get("turnover_rate"))
    amplitude = _num(row.get("amplitude")) or 0.0
    boards = int(_num(row.get("consecutive_limit_ups")) or 0)
    if boards > 0 and (turnover is not None and turnover < 3):
        return "一致加速"
    # 口径: enriched 的 amplitude 为小数 (0.07 = 7%), turnover_rate 为百分数 (8 = 8%)
    if turnover is not None and turnover >= 8 and amplitude >= 0.07:
        return "分歧换手" if pct >= 0 else "分歧转弱"
    return "中性"


def _cycle_role(row: dict, hot_concepts: set[str], top_concepts: set[str]) -> str:
    boards = int(_num(row.get("consecutive_limit_ups")) or 0)
    high60 = _num(row.get("high_60d"))
    low60 = _num(row.get("low_60d"))
    close = _num(row.get("close"))
    pos = None
    if close is not None and high60 is not None and low60 is not None and high60 > low60:
        pos = (close - low60) / (high60 - low60)
    name = str(row.get("name") or "")
    if boards >= 2:
        return f"{boards}连板"
    if boards == 1 and name in hot_concepts:
        return "首板·热门"
    if pos is not None and pos >= 0.8:
        return "高位"
    if pos is not None and pos <= 0.2:
        return "低位"
    return "—"


def _sector_gate(sector: dict | None) -> str:
    """树干标签, 与心法引擎 sector_gate 同口径 (出厂: 退潮或强弱分位 < 0.2 为走弱)。

    冰点不算走弱: 冰点板块当日可能已是全市场最强 (等待修复确认), 判走弱会与
    个股分析的树干结论相互矛盾。阶段作为后缀如实展示。
    """
    if not sector:
        return "—"
    pct = sector.get("rs_pct")
    phase = sector.get("phase")
    if phase == "退潮" or (pct is not None and pct < 0.2):
        label = "板块走弱"
    elif phase in {"发酵", "高潮"} or (pct is not None and pct >= 0.8):
        label = "板块强势"
    else:
        label = "板块中性"
    return f"{label}·{phase}" if phase else label


def _sector_context_map(repo, symbols: list[str], cycle_cfg: dict | None) -> dict[str, dict]:
    """为每只自选确定主板块 (归属板块中当日排名最高的一个) 与其强弱分位/周期阶段。

    扩展数据一次性反查 + 全量排名一次 + 只对涉及到的板块算板块周期,
    避免按股循环重复扫描。归属缺失的标的不出现在结果里。
    """
    from app.services import sentiment_cycle
    from app.services.market_overview_builder import dimension_rank_all, symbol_sector_map

    df_latest, _ = repo.get_enriched_latest()
    if df_latest is None or df_latest.is_empty():
        return {}
    cols = [c for c in ("symbol", "name", "change_pct", "amount") if c in df_latest.columns]
    rows = df_latest.select(cols).to_dicts()

    best: dict[str, tuple[float, str, str]] = {}
    for kind, level in (("concept", None), ("industry", 2)):
        mapping = symbol_sector_map(repo, kind, level)
        if not mapping:
            continue
        from app.services.stock_analyzer import _member_count_ok

        ranked = [
            it for it in dimension_rank_all(rows, repo, kind, level=level)
            if sentiment_cycle.tradable_sector(str(it.get("name")), None) and _member_count_ok(it)
        ]
        n = len(ranked)
        if n < 2:
            continue
        pos = {str(it["name"]): 1 - idx / (n - 1) for idx, it in enumerate(ranked)}
        for sym in symbols:
            names = mapping.get(sym) or mapping.get(sym.split(".", 1)[0]) or []
            scored = [(pos[nm], nm) for nm in names if nm in pos]
            if not scored:
                continue
            pct, name = max(scored)
            if sym not in best or pct > best[sym][0]:
                best[sym] = (pct, name, kind)

    out: dict[str, dict] = {}
    for kind in ("concept", "industry"):
        wanted = sorted({name for pct, name, k in best.values() if k == kind})
        if not wanted:
            continue
        cycles = sentiment_cycle.get_sector_cycles(repo, None, cycle_cfg, sectors=wanted, kind=kind)
        for sym, (pct, name, k) in best.items():
            if k != kind:
                continue
            out[sym] = {
                "name": name,
                "kind": kind,
                "rs_pct": round(pct, 3),
                "phase": (cycles.get(name) or {}).get("phase"),
            }
    return out


def build_watchlist_digest(
    repo,
    as_of: date | None,
    *,
    overview: dict | None = None,
    cycle: dict | None = None,
    config: dict | None = None,
) -> dict:
    """返回 {enabled, as_of, rows, detail, stats, news, reason}。

    任何数据缺失都降级为空行/空字段, 不抛异常。
    """
    from app.services import watchlist

    cfg = normalize_config(config)
    if not cfg["enabled"]:
        return {"enabled": False, "reason": "自选股复盘未开启", "rows": [], "detail": []}

    entries = watchlist.list_symbols()
    if cfg["group_ids"]:
        wanted = set(cfg["group_ids"])
        entries = [
            e for e in entries
            if wanted & set(e.get("group_ids") or [])
        ]
    symbols = [r["symbol"] for r in entries]
    if not symbols:
        return {"enabled": True, "as_of": None, "rows": [], "detail": [], "news": []}

    try:
        df_e, latest = repo.get_enriched_latest()
    except Exception as e:  # noqa: BLE001
        logger.warning("watchlist digest: enriched unavailable: %s", e)
        return {"enabled": True, "as_of": None, "rows": [], "detail": [], "reason": str(e)}
    if df_e is None or df_e.is_empty():
        return {"enabled": True, "as_of": None, "rows": [], "detail": [], "reason": "enriched 数据未就绪"}

    stock_set = set(df_e["symbol"].to_list())
    stock_symbols = [s for s in symbols if s in stock_set]
    # ETF/指数没有板块归属与板块指数口径, 显式说明而不是静默丢弃
    excluded = [
        {"symbol": s, "reason": "ETF/指数不支持板块相对强度"}
        for s in symbols if s not in stock_set
    ]
    df = df_e.filter(pl.col("symbol").is_in(stock_symbols)) if stock_symbols else pl.DataFrame()
    if df.is_empty():
        return {"enabled": True, "as_of": str(latest) if latest else None,
                "rows": [], "detail": [], "reason": "自选标的不在 enriched 股票缓存中",
                "excluded": excluded}

    # 名称与所属行业 (行业归属取自扩展数据配置, 缺失则为空)
    name_map = repo.get_name_map(stock_symbols)
    try:
        from app.services import preferences

        sector_map = _sector_context_map(repo, stock_symbols, preferences.get_sentiment_cycle_config())
    except Exception as e:  # noqa: BLE001 — 板块信息缺失只影响树干列
        logger.info("watchlist digest: sector context unavailable: %s", e)
        sector_map = {}
    ta_eval_map = _evaluate_method(repo, latest, stock_symbols, cycle, sector_map)
    rows_raw = df.to_dicts()

    # 多周期相对强度: 个体收益 + 对同板块指数/所属板块的超额 + 评级 (5/20/60/120 日)
    from app.services.watchlist_strength import build_strength

    strength = build_strength(repo, latest, stock_symbols, sector_map)
    strength_stocks = strength.get("stocks") or {}

    # 概念热度: 当日概念榜前 10 的名称与领涨股名
    hot_concepts: set[str] = set()
    top_concepts: set[str] = set()
    if overview:
        for it in ((overview.get("concept_rank") or {}).get("leading") or [])[:10]:
            top_concepts.add(str(it.get("name")))
            leader = it.get("leader") or {}
            if leader.get("name"):
                hot_concepts.add(str(leader["name"]))

    cycle_phase = (cycle or {}).get("phase")
    rows: list[dict] = []
    for row in rows_raw:
        symbol = str(row.get("symbol"))
        name = name_map.get(symbol) or str(row.get("name") or symbol)
        st = strength_stocks.get(symbol) or {}
        abs_ret = st.get("abs") or {}
        excess_index = st.get("excess_index") or {}
        excess_sector = st.get("excess_sector") or {}
        ret5 = abs_ret.get("5")
        ret20 = abs_ret.get("20")
        # 5 日相对强弱 = 个股 5 日收益 − 同板块指数 5 日收益 (不再是"指数当日涨跌")
        rs5 = excess_index.get("5")
        sector = sector_map.get(symbol)
        consensus = _consensus_label(row)
        signals = [label for key, label in _SIGNAL_LABELS.items() if row.get(key) is True]
        alert = _alert_score(row, signals, cycle_phase, consensus, ret5,
                             strength_score=st.get("score"), tag=st.get("tag"))
        rows.append({
            "symbol": symbol,
            "name": name,
            "change_pct": _num(row.get("change_pct")),
            "turnover_rate": _num(row.get("turnover_rate")),
            "amount": _num(row.get("amount")),
            "ret5": ret5,
            "ret20": ret20,
            "ret60": abs_ret.get("60"),
            "ret120": abs_ret.get("120"),
            "abs": abs_ret,
            "rs5": rs5,
            "excess_index": excess_index,
            "excess_sector": excess_sector,
            "rps": st.get("rps") or {},
            "strength_rating": st.get("rating") or "—",
            "strength_score": st.get("score"),
            "strength_tag": st.get("tag") or "",
            "board": st.get("board"),
            "index_symbol": st.get("index_symbol"),
            "index_name": st.get("index_name"),
            "sector_members": st.get("sector_members"),
            "trend": _trend_label(row),
            "role": _cycle_role(row, hot_concepts, top_concepts),
            "sector_gate": _sector_gate(sector),
            "sector": None if not sector else sector["name"],
            "sector_phase": None if not sector else sector.get("phase"),
            "consensus": consensus,
            "signals": signals[:4],
            "ta": ta_eval_map.get(symbol),
            "alert_score": alert,
            "_row": row,
        })

    rows.sort(key=lambda r: r["alert_score"], reverse=True)
    total = len(rows)
    kept = rows[: int(cfg["max_rows"])]
    detail_rows = _select_detail(rows, int(cfg["max_detail"]))

    # detail 与 kept 引用同一批 dict: 先补详述字段, 再统一去掉内部 _row
    detail = []
    for item in detail_rows:
        raw = item.pop("_row")
        item["vol_ratio_5d"] = _num(raw.get("vol_ratio_5d"))
        item["macd_hist"] = _num(raw.get("macd_hist"))
        item["rsi_14"] = _num(raw.get("rsi_14"))
        detail.append(item)
    for item in kept:
        item.pop("_row", None)

    news = _fetch_news(detail[: int(cfg["news_top"])]) if cfg["news_top"] else {}

    return {
        "enabled": True,
        "as_of": str(latest) if latest else None,
        "cycle_phase": cycle_phase,
        "rows": kept,
        "detail": detail,
        "total": total,
        "truncated": total - len(kept),
        "news": news,
        "benchmarks": strength.get("index_bench") or {},
        "sector_bench": strength.get("sector_bench") or {},
        "strength_coverage": strength.get("coverage") or {},
        "excluded": excluded,
    }


def _select_detail(rows: list[dict], limit: int) -> list[dict]:
    """按异动分取前 N, 并强制并入 |多周期强弱分| 最高的 N 只。

    长周期极强的票当日可能很平静 (异动分低), 若只按异动分截断, 复盘会漏掉
    最该被分析的相对强度标的。两组合并后按异动分输出。
    """
    by_alert = rows[:limit]
    by_strength = sorted(
        (r for r in rows if r.get("strength_score") is not None),
        key=lambda r: abs(r["strength_score"]), reverse=True,
    )[:limit]
    picked: list[dict] = []
    seen: set[str] = set()
    for item in [*by_alert, *by_strength]:
        if item["symbol"] in seen:
            continue
        seen.add(item["symbol"])
        picked.append(item)
    picked.sort(key=lambda r: r["alert_score"], reverse=True)
    return picked


def _evaluate_method(repo, latest: date | None, symbols: list[str],
                     cycle: dict | None, sector_map: dict[str, dict]) -> dict[str, dict]:
    """按心法对自选股做规则层评估 (brief 场景); 失败返回空字典。"""
    if not latest or not symbols:
        return {}
    try:
        from app.analysis import engine as ta_engine
        from app.analysis import method as ta_method
        from app.services import preferences

        data_dir = getattr(getattr(repo, "store", None), "data_dir", None)
        if data_dir is None:
            return {}
        loaded = ta_method.resolve_for_scenario(
            data_dir, "watchlist_review", preferences.get_ta_method_selection(),
        )
        df = repo.get_enriched_range(
            latest - timedelta(days=200), latest, symbols=symbols,
        )
        if df is None or df.is_empty():
            return {}
        out: dict[str, dict] = {}
        for key, sub in df.sort(["symbol", "date"]).group_by("symbol"):
            sym = str(key[0] if isinstance(key, tuple) else key)
            sector = sector_map.get(sym) or {}
            ev = ta_engine.evaluate(
                loaded, sub, cycle=cycle,
                sector_phase=sector.get("phase"), sector_rs_pct=sector.get("rs_pct"),
            )
            out[sym] = {
                "composite": ev.get("composite"),
                "label": ev.get("rule_label"),
                "confidence": ev.get("confidence"),
                "tree": ev.get("tree_label"),
                "hits": [c["tag"] for c in ev.get("principle_checks") or []
                         if c["result"] == "hit" and c.get("tag")],
            }
        return out
    except Exception as e:  # noqa: BLE001 — 心法评估失败不阻断自选股章节
        logger.warning("watchlist ta evaluation failed: %s", e)
        return {}


def _alert_score(row: dict, signals: list[str], cycle_phase: str | None,
                 consensus: str, ret5: float | None, *,
                 strength_score: float | None = None,
                 tag: str | None = None) -> float:
    pct = abs(_num(row.get("change_pct")) or 0.0) * 100
    score = pct * 2 + len(signals) * 5
    turnover = _num(row.get("turnover_rate")) or 0.0
    score += min(turnover, 20) * 0.5
    if ret5 is not None and abs(ret5) > 0.1:
        score += 10
    if cycle_phase == "退潮" and (_num(row.get("consecutive_limit_ups")) or 0) >= 2:
        score += 15
    if consensus in {"分歧转弱", "一致加速"}:
        score += 8
    # 长周期相对强度极端的标的即使当日平静也值得详述 (逆势强势 / 极弱)
    if tag:
        score += 8
    elif strength_score is not None and abs(strength_score) >= 0.15:
        score += 5
    return round(score, 2)


def _fetch_news(detail: list[dict]) -> dict[str, list[dict]]:
    """对异动靠前的个股检索公告/消息; 未启用或失败时返回空。"""
    cfg = web_search.current_config()
    if not cfg["enabled"] or not cfg["use_in"].get("watchlist", True):
        return {}
    if not web_search.key_status()["configured"]:
        return {}
    out: dict[str, list[dict]] = {}
    for item in detail:
        try:
            hits = web_search.search(
                f"{item['name']} {item['symbol'].split('.')[0]} 公告 OR 消息",
                count=3,
                after=cn_today() - timedelta(days=14),
                purpose="watchlist_review",
            )
        except web_search.WebSearchError as e:
            logger.info("watchlist news skipped for %s: %s", item["symbol"], e)
            break
        if hits:
            out[item["symbol"]] = [
                {"title": h.get("title"), "url": h.get("url"),
                 "summary": h.get("summary") or h.get("snippet")}
                for h in hits
            ]
    return out


def format_table(digest: dict) -> str:
    """总表 Markdown (代码生成, 不让模型重排数字)。"""
    rows = digest.get("rows") or []
    if not rows:
        return ""
    lines = [
        "| 代码 | 名称 | 涨跌 | 5日相对强弱 | 相对强度 | 20日超额(指/板) | 趋势 | 周期角色 | 树干 | 分歧/一致 | 规则标签 | 心法标签 | 今日信号 |",
        "|---|---|---:|---:|---|---:|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rs = _fmt_pp(r.get("rs5"))
        rating = r.get("strength_rating") or "—"
        if r.get("strength_tag"):
            rating = f"{rating}·{r['strength_tag']}"
        excess_20 = f"{_fmt_pp((r.get('excess_index') or {}).get('20'))}/{_fmt_pp((r.get('excess_sector') or {}).get('20'))}"
        signals = "、".join(r.get("signals") or []) or "—"
        ta = r.get("ta") or {}
        ta_tags = "、".join(ta.get("hits") or []) or "—"
        lines.append(
            f"| {r['symbol']} | {r['name']} | {_fmt_pct(r.get('change_pct'))} | {rs} | "
            f"{rating} | {excess_20} | "
            f"{r.get('trend', '—')} | {r.get('role', '—')} | {r.get('sector_gate', '—')} | "
            f"{r.get('consensus', '—')} | {ta.get('label', '—')} | {ta_tags} | {signals} |"
        )
    if digest.get("truncated"):
        lines.append(f"\n> 另有 {digest['truncated']} 只自选未展开（按异动分截断）。")
    return "\n".join(lines)
