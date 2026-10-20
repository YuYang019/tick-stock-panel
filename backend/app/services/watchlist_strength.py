"""自选股多周期相对强度 — 复盘的规则层事实来源。

复盘此前只有一条错误的 5 日相对强弱 (个股 5 日收益 − 指数**当日**涨跌), 且没有
任何板块维度, 导致「短期回调但长周期大幅跑赢」的标的被误判为弱。本模块按
5/20/60/120 个交易日计算:

- 个股绝对收益;
- 对同板块指数 (创业板→创业板指 / 科创板→科创综指 / 主板→沪深300) 的超额;
- 对所属主板块 (等同概念/同行业成分等权) 的超额;
- 全市场 RPS 分位 (仅展示, 不参与评级);
- 透明可审计的强弱评级与「逆势强势 / 相对抗跌」标签。

口径 (金融契约):
- 收益用 enriched 前复权 close 的简单收益; 窗口按该标的自己的交易日序列数
  (停牌缺口不影响窗口定义), 不足 N+1 根为 None, 不外推。
- 超额 = 个股收益 − 基准同窗口收益, 基准两端取个股窗口的实际起止交易日
  (基准缺失该日则取 ≤ 该日的最近一根), 基准缺任一端则为 None。
- 板块收益 = 成员各自全窗口收益的等权平均, 有效成员 < MIN_SECTOR_MEMBERS 为 None。
  成分来自当前快照, 回算历史存在幸存者偏差, 结果带 sector_snapshot 标记。
- 缓存按 (latest, symbols, sector 快照) 300 秒; 数据刷新后 latest 变化自然失效。
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import polars as pl

from app.services.index_const import board_benchmark

logger = logging.getLogger(__name__)

HORIZONS: tuple[int, ...] = (5, 20, 60, 120)
# 评级权重: 5 日只作参考, 中长期为主 (短期噪声大, 容易把强势股的回调误判为弱)
HORIZON_WEIGHTS: dict[int, float] = {5: 0.1, 20: 0.3, 60: 0.3, 120: 0.3}
MIN_SECTOR_MEMBERS = 10
SPAN_CALENDAR_DAYS = 250          # 120 交易日 ≈ 175 自然日, 留足余量
RATING_STRONG_PP = 0.15
RATING_ABOVE_PP = 0.05
RATING_BELOW_PP = -0.05
RATING_WEAK_PP = -0.15
CACHE_TTL_SECONDS = 300

# 全市场 RPS 分位的展示窗口 (与 factors/relative_strength 的 RS_WINDOWS 对齐)
RPS_WINDOWS: tuple[int, ...] = (20, 60, 120)

_cache: dict[str, tuple[float, dict]] = {}


def invalidate_cache() -> None:
    _cache.clear()


# ================================================================
# 收益计算
# ================================================================

def _window_return(closes: list[float | None], n: int) -> float | None:
    """close[-1] / close[-1-n] − 1; 不足 n+1 根或端点为非正数时 None。"""
    if len(closes) < n + 1:
        return None
    last, base = closes[-1], closes[-1 - n]
    if not last or not base or last <= 0 or base <= 0:
        return None
    return last / base - 1


def _close_series(df: pl.DataFrame) -> dict[str, tuple[list[date], list[float | None]]]:
    """[{symbol, date, close}] → {symbol: (dates, closes)}, 按日期升序。"""
    out: dict[str, tuple[list[date], list[float | None]]] = {}
    if df is None or df.is_empty():
        return out
    for key, sub in df.sort(["symbol", "date"]).group_by("symbol", maintain_order=True):
        sym = str(key[0] if isinstance(key, tuple) else key)
        dates = [d for d in sub["date"].to_list()]
        closes = [None if c is None else float(c) for c in sub["close"].to_list()]
        out[sym] = (dates, closes)
    return out


def _benchmark_return(
    bench: tuple[list[date], list[float | None]] | None,
    start: date | None,
    end: date | None,
) -> float | None:
    """基准在 [start, end] 区间的收益; 任一端取 ≤ 该日的最近一根收盘。"""
    if bench is None or start is None or end is None:
        return None
    dates, closes = bench
    if not dates:
        return None

    def close_at(target: date) -> float | None:
        value = None
        for d, c in zip(dates, closes, strict=False):
            if d > target:
                break
            if c is not None and c > 0:
                value = c
        return value

    base = close_at(start)
    last = close_at(end)
    if not base or not last:
        return None
    return last / base - 1


def _index_series(repo, index_symbols: set[str], start: date, end: date) -> dict[str, tuple[list[date], list[float | None]]]:
    """批量读取基准指数收盘序列; 单个指数失败只影响该基准。"""
    out: dict[str, tuple[list[date], list[float | None]]] = {}
    for symbol in sorted(index_symbols):
        try:
            df = repo.get_index_daily(symbol, start, end, columns=["date", "close"])
        except Exception as e:  # noqa: BLE001 — 基准缺失降级为 None, 不阻断复盘
            logger.info("watchlist strength: index %s unavailable: %s", symbol, e)
            continue
        if df is None or df.is_empty():
            continue
        series = _close_series(df.with_columns(pl.lit(symbol).alias("symbol")))
        if symbol in series:
            out[symbol] = series[symbol]
    return out


# ================================================================
# 板块基准
# ================================================================

def _dedup_members(keys: set[str]) -> list[str]:
    """成分键去重: sector_member_map 同时给带后缀与裸代码, 不去重会重复计权。"""
    suffixed = sorted(k for k in keys if "." in k)
    bare = sorted(k for k in keys if "." not in k)
    return suffixed if suffixed else bare


def _sector_returns(
    repo,
    sector_map: dict[str, dict],
    start: date,
    end: date,
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """计算所需板块的等权窗口收益, 返回 ({板块: {ret: {...}, members}}, {板块: 成分代码})。"""
    from app.services.market_overview_builder import sector_member_map

    # {(kind, level): {板块名}} — 同一类别下可能有多只自选分属不同板块,
    # 不能按 kind 单值保存 (否则后一个板块会覆盖前一个)
    wanted: dict[tuple[str, str | None], set[str]] = {}
    for info in sector_map.values():
        name = info.get("name")
        kind = info.get("kind") or "concept"
        if not name:
            continue
        level_key = "2" if kind == "industry" else None
        wanted.setdefault((kind, level_key), set()).add(name)
    if not wanted:
        return {}, {}

    by_sector: dict[str, list[str]] = {}
    member_map_by_kind: dict[tuple[str, str | None], dict[str, set[str]]] = {}
    for key in wanted:
        kind, level_key = key
        level = int(level_key) if level_key else None
        try:
            member_map_by_kind[key] = sector_member_map(repo, kind, level)
        except Exception as e:  # noqa: BLE001
            logger.info("watchlist strength: sector map %s unavailable: %s", kind, e)
            member_map_by_kind[key] = {}
    for (kind, level_key), names in wanted.items():
        full_map = member_map_by_kind.get((kind, level_key), {})
        for name in names:
            members = full_map.get(name)
            if members:
                by_sector[name] = _dedup_members(members)

    if not by_sector:
        return {}, {}

    all_members = sorted({m for members in by_sector.values() for m in members})
    try:
        df = repo.get_enriched_range(start, end, symbols=all_members, columns=["symbol", "date", "close"])
    except Exception as e:  # noqa: BLE001
        logger.info("watchlist strength: sector member history unavailable: %s", e)
        df = None
    if df is None or df.is_empty():
        return {}, by_sector

    series = _close_series(df)
    out: dict[str, dict] = {}
    for name, members in by_sector.items():
        per_window: dict[int, list[float]] = {h: [] for h in HORIZONS}
        valid_members = 0
        for member in members:
            item = series.get(member)
            if item is None:
                continue
            _dates, closes = item
            rets = {h: _window_return(closes, h) for h in HORIZONS}
            if all(v is None for v in rets.values()):
                continue
            valid_members += 1
            for h, value in rets.items():
                if value is not None:
                    per_window[h].append(value)
        if valid_members < MIN_SECTOR_MEMBERS:
            out[name] = {"members": valid_members, "ret": {h: None for h in HORIZONS}}
            continue
        out[name] = {
            "members": valid_members,
            "ret": {
                h: (sum(values) / len(values)) if len(values) >= MIN_SECTOR_MEMBERS else None
                for h, values in per_window.items()
            },
        }
    return out, by_sector


# ================================================================
# 全市场 RPS (仅展示)
# ================================================================

def _rps_map(repo, start: date, end: date, symbols: set[str]) -> dict[str, dict[str, float]]:
    """自选标的的全市场 RPS 分位; 计算失败返回空字典 (不影响其余字段)。"""
    try:
        from app.factors import relative_strength as rs

        all_df = repo.get_enriched_range(
            start, end, columns=["symbol", "date", "close", "volume"],
        )
        if all_df is None or all_df.is_empty():
            return {}
        data_dir = getattr(getattr(repo, "store", None), "data_dir", None)
        ctx = rs.load_context(data_dir, need_industry=False, need_benchmark=False)
        names = [f"rps_{h}" for h in RPS_WINDOWS]
        computed = rs.compute_relative_strength(all_df, names, ctx)
        latest_dates = computed.group_by("symbol").agg(pl.col("date").max().alias("_d"))
        last = computed.join(latest_dates, left_on=["symbol", "date"], right_on=["symbol", "_d"], how="inner")
        keep = [c for c in ["symbol", *names] if c in last.columns]
        out: dict[str, dict[str, float]] = {}
        for row in last.select(keep).to_dicts():
            sym = str(row.get("symbol"))
            if sym not in symbols:
                continue
            out[sym] = {
                str(h): (None if row.get(f"rps_{h}") is None else round(float(row[f"rps_{h}"]), 1))
                for h in RPS_WINDOWS
            }
        return out
    except Exception as e:  # noqa: BLE001 — RPS 只是辅助展示
        logger.info("watchlist strength: rps unavailable: %s", e)
        return {}


# ================================================================
# 评级
# ================================================================

def _weighted_mean(values: dict[int, float | None]) -> float | None:
    """按 HORIZON_WEIGHTS 加权, 缺失窗口重新归一。"""
    acc = 0.0
    total = 0.0
    for h, value in values.items():
        if value is None:
            continue
        weight = HORIZON_WEIGHTS.get(h, 0.0)
        acc += value * weight
        total += weight
    return acc / total if total > 0 else None


def _rating(score: float | None, available_horizons: set[int]) -> str:
    if score is None:
        return "—"
    if score >= RATING_STRONG_PP:
        rating = "极强"
    elif score >= RATING_ABOVE_PP:
        rating = "强"
    elif score > RATING_BELOW_PP:
        rating = "中性"
    elif score > RATING_WEAK_PP:
        rating = "弱"
    else:
        rating = "极弱"
    # 只有 5 日数据时评级置信度低, 上限压在「中性」, 避免短期噪声给出极强/极弱
    if rating in {"极强", "强"} and not (available_horizons & {20, 60, 120}):
        return "中性"
    if rating in {"极弱", "弱"} and not (available_horizons & {20, 60, 120}):
        return "中性"
    return rating


def _tag(abs_ret: dict[int, float | None], excess_index: dict[int, float | None]) -> str:
    a60, e60 = abs_ret.get(60), excess_index.get(60)
    a120, e120 = abs_ret.get(120), excess_index.get(120)
    a20, e20 = abs_ret.get(20), excess_index.get(20)
    if (a60 is not None and e60 is not None and a60 >= 0.15 and e60 >= 0.15) or (
        a120 is not None and e120 is not None and a120 >= 0.20 and e120 >= 0.20
    ):
        return "逆势强势"
    if a20 is not None and e20 is not None and a20 < 0 and e20 >= 0.10:
        return "相对抗跌"
    return ""


# ================================================================
# 对外入口
# ================================================================

def build_strength(
    repo,
    latest: date | None,
    symbols: list[str],
    sector_map: dict[str, dict] | None = None,
) -> dict:
    """自选股多周期相对强度事实包。

    sector_map: {symbol: {"name": 主板块名, "kind": "concept"|"industry", ...}}
    (来自 watchlist_digest._sector_context_map)。缺失时只算对指数与 RPS。
    """
    empty = {
        "span": None,
        "coverage": {"start": None, "end": None, "horizons": {}, "note": "无历史数据"},
        "index_bench": {},
        "sector_bench": {},
        "stocks": {},
    }
    if not latest or not symbols:
        return empty

    sector_map = sector_map or {}
    sector_sig = ",".join(
        sorted(f"{sym}:{(info or {}).get('name') or ''}" for sym, info in sector_map.items())
    )
    cache_key = f"{latest.isoformat()}|{','.join(sorted(symbols))}|{sector_sig}"
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        span = repo.get_enriched_history_span()
    except Exception as e:  # noqa: BLE001
        logger.info("watchlist strength: history span unavailable: %s", e)
        span = None
    if span is None:
        return empty

    end = min(latest, span[1])
    start = max(span[0], end - timedelta(days=SPAN_CALENDAR_DAYS))
    try:
        df = repo.get_enriched_range(start, end, symbols=symbols, columns=["symbol", "date", "close"])
    except Exception as e:  # noqa: BLE001
        logger.info("watchlist strength: stock history unavailable: %s", e)
        df = None
    if df is None or df.is_empty():
        return empty

    series = _close_series(df)
    index_symbols = {board_benchmark(sym)[0] for sym in symbols}
    index_series = _index_series(repo, index_symbols, start, end)
    sector_returns, _members = _sector_returns(repo, sector_map, start, end)
    rps = _rps_map(repo, start, end, set(symbols))

    index_bench: dict[str, dict] = {}
    stocks: dict[str, dict] = {}
    horizons_available: dict[str, bool] = {str(h): False for h in HORIZONS}

    for sym in symbols:
        index_symbol, index_name, matched = board_benchmark(sym)
        bench = index_series.get(index_symbol)
        if bench is not None:
            bench_ret = {
                h: _benchmark_return(bench, *_window_bounds(series.get(sym), h))
                for h in HORIZONS
            }
            if index_symbol not in index_bench:
                index_bench[index_symbol] = {"name": index_name, "ret": bench_ret}
        else:
            bench_ret = {h: None for h in HORIZONS}

        item = series.get(sym)
        closes = item[1] if item else []
        abs_ret = {h: _window_return(closes, h) for h in HORIZONS}
        excess_index = {
            h: (None if abs_ret[h] is None or bench_ret.get(h) is None else abs_ret[h] - bench_ret[h])
            for h in HORIZONS
        }
        sector_info = sector_map.get(sym) or {}
        sector_name = sector_info.get("name")
        sector = sector_returns.get(sector_name or "") or {}
        sector_bench_ret = sector.get("ret") or {}
        excess_sector = {
            h: (None if abs_ret[h] is None or sector_bench_ret.get(h) is None
                else abs_ret[h] - sector_bench_ret[h])
            for h in HORIZONS
        }
        sector_members = sector.get("members")

        available = {h for h in HORIZONS if excess_index.get(h) is not None}
        available_sector = {h for h in HORIZONS if excess_sector.get(h) is not None}
        for h in available:
            horizons_available[str(h)] = True
        score_index = _weighted_mean(excess_index)
        score_sector = _weighted_mean(excess_sector)
        if score_index is not None and score_sector is not None:
            score = 0.6 * score_index + 0.4 * score_sector
        else:
            score = score_index if score_index is not None else score_sector
        rating = _rating(score, available | available_sector)
        tag = _tag(abs_ret, excess_index)

        stocks[sym] = {
            "board": board_of_safe(sym),
            "index_symbol": index_symbol,
            "index_name": index_name,
            "board_matched": matched,
            "abs": {str(h): _r(v) for h, v in abs_ret.items()},
            "excess_index": {str(h): _r(v) for h, v in excess_index.items()},
            "excess_sector": {str(h): _r(v) for h, v in excess_sector.items()},
            "rps": rps.get(sym, {}),
            "sector": sector_name,
            "sector_members": sector_members,
            "rating": rating,
            "score": _r(score),
            "tag": tag,
        }

    note = _coverage_note(horizons_available)
    payload = {
        "span": (start, end),
        "coverage": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "horizons": horizons_available,
            "note": note,
            "sector_snapshot": True,
        },
        "index_bench": {
            k: {"name": v["name"], "ret": {str(h): _r(x) for h, x in v["ret"].items()}}
            for k, v in index_bench.items()
        },
        "sector_bench": {
            k: {
                "members": v.get("members"),
                "ret": {str(h): _r(x) for h, x in (v.get("ret") or {}).items()},
            }
            for k, v in sector_returns.items()
        },
        "stocks": stocks,
    }
    _cache[cache_key] = (now, payload)
    return payload


def _window_bounds(item: tuple[list[date], list[float | None]] | None, n: int) -> tuple[date | None, date | None]:
    """个股 N 日窗口的实际起止交易日 (用于基准同日对齐)。"""
    if not item:
        return None, None
    dates, closes = item
    if len(closes) < n + 1:
        return None, None
    return dates[-1 - n], dates[-1]


def board_of_safe(symbol: str) -> str:
    from app.services.abnormal_moves import board_of

    try:
        return board_of(symbol)
    except Exception:  # noqa: BLE001
        return "主板"


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _coverage_note(horizons: dict[str, bool]) -> str:
    missing = [h for h, ok in horizons.items() if not ok]
    if not missing:
        return ""
    if len(missing) == len(horizons):
        return "无可用历史窗口"
    return f"历史覆盖不足, {'/'.join(missing)} 日窗口不可用"
