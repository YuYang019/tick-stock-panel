"""相对强度 (RPS) 因子族 — 全市场 RPS / 行业相对 / 基准相对。

因子 (分组「相对强度」, 仅股票, 均为 kind="virtual" 但不走 scoring_value_expr 单表达式分发,
由 materialize_relative_strength 在评分物化入口整帧计算):

  rps_{N}            全市场截面分位 (0~100) of N 日对数收益 ln(close / close.shift(N))
  rps_ind_{N}        二级行业内截面分位 (0~100); 行业有效成员 < 5 → null
  industry_rps_{N}   行业等权平均 N 日对数收益在全市场行业间的分位 (0~100), 成员股共享
  excess_{N}         个股 N 日对数收益 − 基准同区间对数收益 (基准 = 中证全指 000985.SH)
  rs_line_new_high_60  RS 线 (close / 基准 close) 今日 ≥ 前 60 日 RS 线最大值 → 1.0, 否则 0.0
  rs_line_slope_20     ln(RS 线) 对 bar 序号的 20 bar OLS 斜率 (每 bar 对数超额趋势)
  N ∈ {20, 60, 120, 250}

口径决策 (金融契约, CONTRIBUTING §3/§5.3):
- 价格: enriched 前复权 close (技术面收益口径); 指数无复权问题。
- 分位: 100 · (升序平均秩 − 1) / (有效数 − 1), 最弱 0 / 最强 100 (O'Neil 口径:"跑赢多少比例");
  有效数 < 下限 (全市场 30, 行业 5) → null, 防止单标的/小样本帧产出假 100。
- 资格 (仅截面排名类 rps*/industry_rps*): close>0、volume>0 (停牌/无量当日不参与)、
  非 ST/退 (按当前维表名称, 非点时 → 结果携带注记)、在股票维表内、上市满 60 个交易日、
  且 N 日收益可计算 (至少 N+1 根 bar)。不合格 → null。excess/rs_line 是个股时序量, 不做资格过滤。
- 基准: 按日期精确 join, 不前向填充; 基准缺日 → 当日及引用该日的窗口为 null。
- 行业 (PIT): ext_hy_ths 为快照表。每次写入快照时同步归档
  ext_data/ext_hy_ths/history/date=YYYY-MM-DD/part.parquet; bar 日期 → 最近一份 date ≤ bar 的归档
  (按 symbol asof backward)。当前快照 part.parquet 以其文件 mtime (北京日期) 视作一份归档
  (mtime 只会晚于真实拉取, 保守)。早于首份归档的 bar 回退到最早一份并标记
  → 检验/回测结果携带「行业归属使用当前快照, 存在未来函数偏差」警告。
- 时序窗口按 bar 数 (实际交易日) 计; 预热 = N+1 / 61 / 21 根。
- 单日快照帧 (每 symbol ≤ 1 行) 无历史: 整族跳过 (缺列 → 下游"不可计算"), 与指标因子同语义。
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.factors import registry as _registry

logger = logging.getLogger(__name__)

GROUP = "相对强度"
BENCHMARK_SYMBOL = "000985.SH"  # 中证全指; 改基准只改此常量 (下游缓存按数据签名失效)
RS_WINDOWS: tuple[int, ...] = (20, 60, 120, 250)
MIN_LISTING_BARS = 60
MIN_MARKET_MEMBERS = 30
MIN_INDUSTRY_MEMBERS = 5
NEW_HIGH_WINDOW = 60
SLOPE_WINDOW = 20
INDUSTRY_CONFIG_ID = "ext_hy_ths"
INDUSTRY_FIELD = "所属同花顺行业"
INDUSTRY_LEVEL = 2
INDUSTRY_COLUMN = "_ind2"  # 帧内临时列 (DSL group_* 的 industry 键亦映射到此)

WARN_INDUSTRY_SNAPSHOT = "行业归属使用当前快照, 存在未来函数偏差"
WARN_ST_NOT_PIT = "RPS 资格的 ST 剔除按当前名称判定 (非点时)"
WARN_SUBSET_UNIVERSE = "指定了标的子集: RPS/行业截面因子需全市场帧, 子集内不计算 (结果为空)"

_TMP_PREFIX = "__rs_"


def _spec(fid: str, label: str, desc: str, warmup: int, *, industry: bool = False) -> _registry.FactorSpec:
    return _registry.FactorSpec(
        id=fid,
        label=label,
        group=GROUP,
        formula_text=desc,
        kind="virtual",
        dependencies=frozenset({"close", "volume"}),
        warmup_bars=warmup,
        asset_types=frozenset({"stock"}),
        incremental_safe=False,
        tags=("relative_strength", "industry_snapshot") if industry else ("relative_strength",),
    )


def _build_specs() -> tuple[_registry.FactorSpec, ...]:
    specs: list[_registry.FactorSpec] = []
    for n in RS_WINDOWS:
        specs.append(_spec(
            f"rps_{n}", f"RPS{n}", f"全市场截面分位(0~100) of {n}日对数收益 ln(close/close[{n}]); "
            f"剔除ST/停牌/上市不满{MIN_LISTING_BARS}日", n + 1,
        ))
    for n in RS_WINDOWS:
        specs.append(_spec(
            f"rps_ind_{n}", f"行业内RPS{n}", f"同花顺二级行业内截面分位(0~100) of {n}日对数收益; "
            f"行业有效成员<{MIN_INDUSTRY_MEMBERS}为空", n + 1, industry=True,
        ))
    for n in RS_WINDOWS:
        specs.append(_spec(
            f"industry_rps_{n}", f"行业RPS{n}", f"所属二级行业等权平均{n}日对数收益的全市场行业分位(0~100)",
            n + 1, industry=True,
        ))
    for n in RS_WINDOWS:
        specs.append(_spec(
            f"excess_{n}", f"{n}日超额收益", f"{n}日对数收益 − 中证全指({BENCHMARK_SYMBOL})同区间对数收益",
            n + 1,
        ))
    specs.append(_spec(
        "rs_line_new_high_60", "RS线60日新高",
        f"RS线=close/中证全指close; 今日RS ≥ 前{NEW_HIGH_WINDOW}日RS最大值为1, 否则0",
        NEW_HIGH_WINDOW + 1,
    ))
    specs.append(_spec(
        "rs_line_slope_20", "RS线20日斜率",
        f"ln(RS线)对bar序号的{SLOPE_WINDOW}日OLS斜率 (每日对数超额趋势)", SLOPE_WINDOW + 1,
    ))
    return tuple(specs)


RS_SPECS: tuple[_registry.FactorSpec, ...] = _build_specs()
RS_FACTOR_IDS: frozenset[str] = frozenset(spec.id for spec in RS_SPECS)
INDUSTRY_FACTOR_IDS: frozenset[str] = frozenset(
    spec.id for spec in RS_SPECS if "industry_snapshot" in spec.tags
)
CROSS_SECTION_FACTOR_IDS: frozenset[str] = frozenset(
    spec.id for spec in RS_SPECS if spec.id.startswith(("rps_", "industry_rps_"))
)


def register_rs_factors() -> None:
    """启动期注册 (registry 模块末尾调用); 已注册同版本则跳过 (幂等)。"""
    for spec in RS_SPECS:
        if _registry.get_factor(spec.id) is None:
            _registry.register_factor(spec)
        _registry._PROTECTED_IDS.add(spec.id)


register_rs_factors()  # 导入即注册 (registry 已在上方完成导入初始化)


def is_rs_factor(name: object) -> bool:
    return str(name) in RS_FACTOR_IDS


def industry_level_expr(raw: pl.Expr, level: int = INDUSTRY_LEVEL) -> pl.Expr:
    """"一级-二级-三级" → 第 level 级; 级数不足取最后一级 (与 market_overview_builder 同口径)。"""
    parts = raw.cast(pl.Utf8).str.strip_chars().str.split("-")
    picked = (
        pl.when(parts.list.len() >= level)
        .then(parts.list.get(level - 1, null_on_oob=True))
        .otherwise(parts.list.last())
    )
    return pl.when(picked.str.len_chars() > 0).then(picked).otherwise(None)


# ───────────────────────────── 数据上下文 ─────────────────────────────


@dataclass(frozen=True)
class RSContext:
    """计算所需外部数据 (纯数据, 便于测试注入)。

    instruments: symbol, name, listing_date(Date)  — None = 维表缺失 (资格降级: 不过滤 ST/上市龄)
    industry_archive: as_of(Date), symbol, industry  — 历史归档 + 当前快照 (as_of = 文件北京日期)
    benchmark: date(Date), bench_close(Float64)
    """

    instruments: pl.DataFrame | None = None
    industry_archive: pl.DataFrame | None = None
    benchmark: pl.DataFrame | None = None


_cache_lock = threading.Lock()
_file_cache: dict[tuple[str, str], tuple[tuple, pl.DataFrame | None]] = {}


def _cached(kind: str, root: Path, signature: tuple, loader) -> pl.DataFrame | None:
    key = (str(root), kind)
    with _cache_lock:
        hit = _file_cache.get(key)
        if hit is not None and hit[0] == signature:
            return hit[1]
    value = loader()  # 不持锁做磁盘读 (CONTRIBUTING §6.2)
    with _cache_lock:
        _file_cache[key] = (signature, value)
    return value


def _stat_sig(paths: Iterable[Path]) -> tuple:
    sig = []
    for path in paths:
        try:
            st = path.stat()
            sig.append((path.name if path.name != "part.parquet" else path.parent.name, st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    return tuple(sig)


def _resolve_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    from app.config import settings

    return Path(settings.data_dir)


def load_instruments(data_dir: Path | None = None) -> pl.DataFrame | None:
    root = _resolve_dir(data_dir)
    path = root / "instruments" / "instruments.parquet"

    def _load() -> pl.DataFrame | None:
        if not path.exists():
            return None
        try:
            raw = pl.read_parquet(path, columns=["symbol", "name", "listing_date"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("RPS: instruments 读取失败, 资格过滤降级: %s", exc)
            return None
        return raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("name").cast(pl.Utf8),
            pl.col("listing_date").cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False).alias("listing_date"),
        ).unique(subset=["symbol"], keep="last")

    return _cached("instruments", root, _stat_sig([path]), _load)


def _industry_history_dir(root: Path) -> Path:
    return root / "ext_data" / INDUSTRY_CONFIG_ID / "history"


def _normalize_industry(df: pl.DataFrame, as_of: date) -> pl.DataFrame:
    if "symbol" not in df.columns or INDUSTRY_FIELD not in df.columns:
        return pl.DataFrame()
    return (
        df.select(
            pl.lit(as_of).cast(pl.Date).alias("as_of"),
            pl.col("symbol").cast(pl.Utf8),
            industry_level_expr(pl.col(INDUSTRY_FIELD)).alias("industry"),
        )
        .filter(pl.col("symbol").is_not_null())
        .unique(subset=["symbol"], keep="last")
    )


def load_industry_archive(data_dir: Path | None = None) -> pl.DataFrame | None:
    """全部历史归档 + 当前快照 → (as_of, symbol, industry); 无任何来源返回 None。"""
    root = _resolve_dir(data_dir)
    hist = _industry_history_dir(root)
    current = root / "ext_data" / INDUSTRY_CONFIG_ID / "part.parquet"
    parts = sorted(hist.glob("date=*/part.parquet")) if hist.exists() else []
    signature = _stat_sig([*parts, current])

    def _load() -> pl.DataFrame | None:
        frames: list[pl.DataFrame] = []
        for part in parts:
            try:
                as_of = date.fromisoformat(part.parent.name.removeprefix("date="))
                frames.append(_normalize_industry(pl.read_parquet(part), as_of))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RPS: 行业归档 %s 读取失败, 跳过: %s", part, exc)
        if current.exists():
            try:
                from app.market_time import CN_TZ

                as_of = datetime.fromtimestamp(current.stat().st_mtime, CN_TZ).date()
                frames.append(_normalize_industry(pl.read_parquet(current), as_of))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RPS: 行业快照读取失败: %s", exc)
        frames = [frame for frame in frames if not frame.is_empty()]
        if not frames:
            return None
        return pl.concat(frames).unique(subset=["as_of", "symbol"], keep="first").sort(["symbol", "as_of"])

    return _cached("industry", root, signature, _load)


def load_benchmark(data_dir: Path | None = None, symbol: str = BENCHMARK_SYMBOL) -> pl.DataFrame | None:
    """指数日K close (kline_index_enriched); 缺失返回 None (excess/rs_line 全 null)。"""
    root = _resolve_dir(data_dir)
    base = root / "kline_index_enriched"
    parts = sorted(base.glob("date=*/part.parquet")) if base.exists() else []
    # 签名: 分区数 + 首尾分区 stat (全量 stat 数千分区过慢; 中间分区重写由 generation 链路覆盖)
    signature = (len(parts), _stat_sig(parts[:1] + parts[-1:]), symbol)

    def _load() -> pl.DataFrame | None:
        if not parts:
            return None
        try:
            from app.parquet import scan_enriched_parquet

            frame = (
                scan_enriched_parquet(str(base / "**" / "*.parquet"))
                .filter(pl.col("symbol") == symbol)
                .select(pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64).alias("bench_close"))
                .collect()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RPS: 基准 %s 读取失败: %s", symbol, exc)
            return None
        frame = frame.filter(pl.col("bench_close") > 0).unique(subset=["date"], keep="last").sort("date")
        return frame if not frame.is_empty() else None

    return _cached(f"benchmark:{symbol}", root, signature, _load)


def load_context(data_dir: Path | None = None, *, need_industry: bool = True, need_benchmark: bool = True) -> RSContext:
    return RSContext(
        instruments=load_instruments(data_dir),
        industry_archive=load_industry_archive(data_dir) if need_industry else None,
        benchmark=load_benchmark(data_dir) if need_benchmark else None,
    )


def clear_caches() -> None:
    with _cache_lock:
        _file_cache.clear()


# ───────────────────────────── 行业 PIT ─────────────────────────────


def earliest_industry_date(archive: pl.DataFrame | None) -> date | None:
    if archive is None or archive.is_empty():
        return None
    return archive["as_of"].min()


def industry_pit_frame(keys: pl.DataFrame, archive: pl.DataFrame | None) -> pl.DataFrame:
    """(symbol, date) → (industry, industry_fallback), 行序与 keys 一致。

    date ≥ 首份归档: asof backward (最近一份 as_of ≤ date 的归属, 按 symbol);
    date < 首份归档: 回退到最早一份归档的归属, industry_fallback=True (未来函数偏差标记)。
    """
    n = keys.height
    if archive is None or archive.is_empty():
        return pl.DataFrame({
            "industry": pl.Series([None] * n, dtype=pl.Utf8),
            "industry_fallback": pl.Series([False] * n, dtype=pl.Boolean),
        })
    first = archive["as_of"].min()
    work = keys.select(pl.col("symbol").cast(pl.Utf8), pl.col("date").cast(pl.Date)).with_row_index("_ri")
    late = work.filter(pl.col("date") >= first).sort("date")
    early = work.filter(pl.col("date") < first)
    pieces: list[pl.DataFrame] = []
    if not late.is_empty():
        pieces.append(
            late.join_asof(
                archive.select("as_of", "symbol", "industry").sort("as_of"),
                left_on="date", right_on="as_of", by="symbol", strategy="backward",
                check_sortedness=False,  # 两侧均已按日期全局排序 (by 分组时 Polars 无法自检)
            ).select("_ri", "industry", pl.lit(False).alias("industry_fallback"))
        )
    if not early.is_empty():
        baseline = archive.filter(pl.col("as_of") == first).select("symbol", "industry")
        pieces.append(
            early.join(baseline, on="symbol", how="left", maintain_order="left")
            .select("_ri", "industry", pl.lit(True).alias("industry_fallback"))
        )
    return pl.concat(pieces).sort("_ri").drop("_ri")


def attach_industry(frame: pl.DataFrame, data_dir: Path | None = None, *, archive: pl.DataFrame | None = None) -> pl.DataFrame:
    """给帧挂二级行业临时列 INDUSTRY_COLUMN (DSL group_* 用); 无行业数据时整列 null。"""
    if INDUSTRY_COLUMN in frame.columns or frame.is_empty():
        return frame
    if archive is None:
        archive = load_industry_archive(data_dir)
    pit = industry_pit_frame(frame.select("symbol", "date"), archive)
    return frame.with_columns(pit["industry"].alias(INDUSTRY_COLUMN))


# ───────────────────────────── 计算 ─────────────────────────────


def _eligibility_expr(work: pl.DataFrame, instruments: pl.DataFrame | None) -> tuple[pl.DataFrame, pl.Expr]:
    """返回 (附加辅助列后的 work, 资格布尔表达式)。work 必须已按 (symbol, date) 排序。"""
    ok = (pl.col("close") > 0).fill_null(False)
    if "volume" in work.columns:
        ok = ok & (pl.col("volume") > 0).fill_null(False)
    if instruments is None or instruments.is_empty():
        # 维表缺失: 无法判断 ST/上市龄 → 退化为帧内 bar 数 ≥ 60 (保守), 不静默放行次新
        bar_no = pl.int_range(pl.len()).over("symbol")
        return work, ok & (bar_no >= MIN_LISTING_BARS)
    # 市场交易日轴 = 帧内日期并集 (全市场帧即交易日历); 序号 0 = 帧首日
    axis = work.select(pl.col("date").unique().sort()).with_row_index(f"{_TMP_PREFIX}dpos")
    first_day = axis["date"][0]
    info = instruments.filter(pl.col("symbol").is_in(work["symbol"].unique().implode()))
    # 上市位置 (交易日序号): 帧内上市 = 首个 ≥ 上市日的交易日序号 (join_asof forward);
    # 早于帧首日 = −(工作日数) 估算 (节假日按交易日计, 至多高估 ~15 日/年, 只影响上市临界附近)
    info = (
        info.sort("listing_date", nulls_last=True)
        .join_asof(axis.rename({"date": f"{_TMP_PREFIX}ad"}), left_on="listing_date",
                   right_on=f"{_TMP_PREFIX}ad", strategy="forward")
        .with_columns(
            pl.when(pl.col("listing_date") < first_day)
            .then(-pl.business_day_count(pl.col("listing_date"), pl.lit(first_day)))
            .otherwise(pl.col(f"{_TMP_PREFIX}dpos").cast(pl.Int64))
            .alias(f"{_TMP_PREFIX}lpos"),
            pl.col("listing_date").is_not_null().alias(f"{_TMP_PREFIX}hasl"),
            pl.col("name").fill_null("").str.to_uppercase().str.contains("ST|退").alias(f"{_TMP_PREFIX}st"),
        )
        .select("symbol", f"{_TMP_PREFIX}lpos", f"{_TMP_PREFIX}hasl", f"{_TMP_PREFIX}st")
    )
    axis_index = axis.with_columns(pl.col(f"{_TMP_PREFIX}dpos").cast(pl.Int64))
    work = (
        work.join(info, on="symbol", how="left", maintain_order="left")
        .join(axis_index, on="date", how="left", maintain_order="left")
    )
    aged = (pl.col(f"{_TMP_PREFIX}dpos") - pl.col(f"{_TMP_PREFIX}lpos")) >= MIN_LISTING_BARS
    # 无上市日 / 不在股票维表 (ETF/指数/退市) → 不合格 (保守, 设计文档 §4.3)
    listed_ok = (pl.col(f"{_TMP_PREFIX}hasl") & aged).fill_null(False)
    not_st = (~pl.col(f"{_TMP_PREFIX}st")).fill_null(False)
    return work, ok & listed_ok & not_st


MARKET_COVERAGE_MIN = 0.8


def _covers_market(frame: pl.DataFrame, instruments: pl.DataFrame | None) -> bool:
    """帧是否覆盖全市场: 帧内标的数 ≥ 80% × 帧末日前已上市的股票维表标的数。

    维表缺失时无法判断, 放行 (仅靠截面最少 30 只的下限兜底)。
    """
    if instruments is None or instruments.is_empty():
        return True
    last = frame["date"].cast(pl.Date).max()
    listed = instruments.filter(pl.col("listing_date").is_not_null() & (pl.col("listing_date") <= last)).height
    if listed == 0:
        return True
    return frame["symbol"].n_unique() >= MARKET_COVERAGE_MIN * listed


def _percentile(value: pl.Expr, partition: list[str], min_count: int) -> pl.Expr:
    count = value.count().over(partition)
    rank = value.rank(method="average").over(partition)
    pct = pl.when(count > 1).then((rank - 1.0) / (count - 1.0) * 100.0).otherwise(None)
    return pl.when(value.is_not_null() & (count >= min_count)).then(pct).otherwise(None)



def _window_of(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def compute_relative_strength(frame: pl.DataFrame, names: Collection[str], ctx: RSContext) -> pl.DataFrame:
    """纯计算入口: 把请求且帧上缺失的 RS 因子列补到帧上 (行序不变)。"""
    wanted = [str(n) for n in dict.fromkeys(names) if is_rs_factor(n) and str(n) not in frame.columns]
    if not wanted or frame.is_empty():
        return frame
    if not {"symbol", "date", "close"}.issubset(frame.columns):
        logger.warning("RS factors skipped: frame lacks symbol/date/close")
        return frame
    if frame.height <= frame["symbol"].n_unique():
        logger.warning("RS factors skipped on single-day snapshot (need history): %s", wanted)
        return frame

    base_cols = ["symbol", "date", "close", *(["volume"] if "volume" in frame.columns else [])]
    work = (
        frame.select(base_cols)
        .with_columns(pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64))
        .with_row_index(f"{_TMP_PREFIX}ri")
        .sort(["symbol", "date"])
    )
    windows = sorted({_window_of(n) for n in wanted if not n.startswith("rs_line_")})
    log_close = pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None)
    work = work.with_columns(log_close.alias(f"{_TMP_PREFIX}lc"))
    lc = pl.col(f"{_TMP_PREFIX}lc")
    work = work.with_columns([
        (lc - lc.shift(n).over("symbol")).alias(f"{_TMP_PREFIX}ret{n}") for n in windows
    ])

    needs_cross = any(n in CROSS_SECTION_FACTOR_IDS or n.startswith("rps_ind_") for n in wanted)
    if needs_cross and not _covers_market(frame, ctx.instruments):
        # 单标的 K 线 / 按 symbol 分批的流水线帧 / 子集帧: 截面分位会变成"批内排名", 必须 fail-closed
        logger.warning("RS cross-sectional factors skipped: frame does not cover the market universe")
        wanted_cross = {n for n in wanted if n in CROSS_SECTION_FACTOR_IDS or n.startswith("rps_ind_")}
        frame = frame.with_columns([pl.lit(None, dtype=pl.Float64).alias(n) for n in sorted(wanted_cross)])
        wanted = [n for n in wanted if n not in wanted_cross]
        needs_cross = False
        if not wanted:
            return frame
    if needs_cross:
        work, eligible = _eligibility_expr(work, ctx.instruments)
        work = work.with_columns(eligible.alias(f"{_TMP_PREFIX}ok"))
    needs_industry = any(n in INDUSTRY_FACTOR_IDS for n in wanted)
    if needs_industry:
        pit = industry_pit_frame(work.select("symbol", "date"), ctx.industry_archive)
        work = work.with_columns(pit["industry"].alias(f"{_TMP_PREFIX}ind"))
    ok = pl.col(f"{_TMP_PREFIX}ok") if needs_cross else pl.lit(True)
    ind = pl.col(f"{_TMP_PREFIX}ind")

    outputs: list[pl.Expr] = []
    for name in wanted:
        if name.startswith("rps_ind_"):
            ret = pl.col(f"{_TMP_PREFIX}ret{_window_of(name)}")
            valid = pl.when(ok & ret.is_finite() & ind.is_not_null()).then(ret).otherwise(None)
            outputs.append(_percentile(valid, ["date", f"{_TMP_PREFIX}ind"], MIN_INDUSTRY_MEMBERS).alias(name))
        elif name.startswith("rps_"):
            ret = pl.col(f"{_TMP_PREFIX}ret{_window_of(name)}")
            valid = pl.when(ok & ret.is_finite()).then(ret).otherwise(None)
            outputs.append(_percentile(valid, ["date"], MIN_MARKET_MEMBERS).alias(name))
    if outputs:
        work = work.with_columns(outputs)

    industry_names = [n for n in wanted if n.startswith("industry_rps_")]
    if industry_names:
        work = _attach_industry_rps(work, industry_names, ok)

    bench_names = [n for n in wanted if n.startswith(("excess_", "rs_line_"))]
    if bench_names:
        work = _attach_benchmark_factors(work, bench_names, ctx.benchmark)

    work = work.with_columns([pl.lit(None, dtype=pl.Float64).alias(n) for n in wanted if n not in work.columns])
    result = work.sort(f"{_TMP_PREFIX}ri").select([pl.col(n).cast(pl.Float64) for n in wanted])
    return frame.hstack(result.get_columns())


def _attach_industry_rps(work: pl.DataFrame, names: list[str], ok: pl.Expr) -> pl.DataFrame:
    ind_col = f"{_TMP_PREFIX}ind"
    for name in names:
        n = _window_of(name)
        ret = pl.col(f"{_TMP_PREFIX}ret{n}")
        member = work.select(
            "date", ind_col,
            pl.when(ok & ret.is_finite() & pl.col(ind_col).is_not_null()).then(ret).otherwise(None).alias("_v"),
        ).filter(pl.col("_v").is_not_null())
        industry = (
            member.group_by(["date", ind_col])
            .agg(pl.col("_v").mean().alias("_m"), pl.len().alias("_n"))
            .filter(pl.col("_n") >= MIN_INDUSTRY_MEMBERS)
            # 行业数下限同小组纪律: 当日有效行业 < 5 时分位无意义
            .with_columns(_percentile(pl.col("_m"), ["date"], MIN_INDUSTRY_MEMBERS).alias(name))
            .select("date", ind_col, name)
        )
        work = work.join(industry, on=["date", ind_col], how="left", maintain_order="left")
    return work


def _attach_benchmark_factors(work: pl.DataFrame, names: list[str], benchmark: pl.DataFrame | None) -> pl.DataFrame:
    if benchmark is None or benchmark.is_empty():
        logger.warning("RS benchmark %s unavailable: %s → null", BENCHMARK_SYMBOL, names)
        return work
    bench = benchmark.select(pl.col("date").cast(pl.Date), pl.col("bench_close").cast(pl.Float64))
    work = work.join(bench, on="date", how="left", maintain_order="left")  # 精确日期 join, 不前向填充
    blc = pl.when(pl.col("bench_close") > 0).then(pl.col("bench_close").log()).otherwise(None)
    work = work.with_columns(blc.alias(f"{_TMP_PREFIX}blc"))
    lc = pl.col(f"{_TMP_PREFIX}lc")
    blc_col = pl.col(f"{_TMP_PREFIX}blc")
    # ln(RS) = ln(close) − ln(bench); 基准缺日 → null, 不跨缺口填充
    lrs = lc - blc_col
    exprs: list[pl.Expr] = []
    for name in names:
        if name.startswith("excess_"):
            n = _window_of(name)
            # 基准取同一对日期 (个股 t 与 t−N 根 bar 的日期), 个股停牌缺口两端对齐
            bench_ret = blc_col - blc_col.shift(n).over("symbol")
            exprs.append((pl.col(f"{_TMP_PREFIX}ret{n}") - bench_ret).alias(name))
        elif name == "rs_line_new_high_60":
            prev_max = lrs.shift(1).rolling_max(NEW_HIGH_WINDOW, min_samples=NEW_HIGH_WINDOW).over("symbol")
            exprs.append(
                pl.when(lrs.is_not_null() & prev_max.is_not_null())
                .then((lrs >= prev_max).cast(pl.Float64))
                .otherwise(None)
                .alias(name)
            )
        elif name == "rs_line_slope_20":
            exprs.append(_ols_slope(lrs, SLOPE_WINDOW).alias(name))
    return work.with_columns(exprs) if exprs else work


def _ols_slope(y: pl.Expr, window: int) -> pl.Expr:
    """y 对 bar 序号 t 的滚动 OLS 斜率: (E[ty] − E[t]E[y]) / Var(t); 窗口内任一 null → null。

    t 用 symbol 内局部序号; 等距整数窗口 Var(t) = (w²−1)/12 为常数。
    """
    t = pl.int_range(pl.len(), dtype=pl.Int64).cast(pl.Float64)
    mean_ty = (t * y).rolling_mean(window, min_samples=window)
    mean_t = t.rolling_mean(window, min_samples=window)
    mean_y = y.rolling_mean(window, min_samples=window)
    var_t = (window * window - 1) / 12.0
    return ((mean_ty - mean_t * mean_y) / var_t).over("symbol")


def _expand_names(names: Collection[str]) -> tuple[set[str], bool]:
    """请求名 → (需物化的 RS 因子集合, 是否需要行业列 for DSL group_*)。

    展开 composite 成员与 custom (DSL) 公式引用的 RS 因子, 使组合/自定义因子可直接引用 rps_*。
    """
    rs: set[str] = set()
    group_needed = False
    for raw in names:
        name = str(raw)
        if is_rs_factor(name):
            rs.add(name)
            continue
        spec = _registry.get_factor(name)
        if spec is None:
            continue
        if spec.kind == "composite":
            rs.update(member for member, _ in spec.components if is_rs_factor(member))
        elif spec.kind == "custom":
            from app.factors.dsl import compile_formula_cached

            compiled = compile_formula_cached(spec.formula_text)
            rs.update(ref for ref in compiled.referenced_factors if is_rs_factor(ref))
            group_needed = group_needed or bool(compiled.group_keys)
    return rs, group_needed


def materialize_relative_strength(
    frame: pl.DataFrame,
    names: Collection[str],
    *,
    data_dir: Path | None = None,
    ctx: RSContext | None = None,
) -> pl.DataFrame:
    """评分物化入口挂钩 (strategy/scoring.materialize_scoring_columns 调用)。"""
    rs_names, _ = _expand_names(names)
    rs_names -= set(frame.columns)
    if not rs_names or frame.is_empty():
        return frame
    if ctx is None:
        ctx = load_context(
            data_dir,
            need_industry=bool(rs_names & INDUSTRY_FACTOR_IDS),
            need_benchmark=any(n.startswith(("excess_", "rs_line_")) for n in rs_names),
        )
    return compute_relative_strength(frame, sorted(rs_names), ctx)


# ───────────────────────────── 结果注记 ─────────────────────────────


def result_warnings(
    names: Collection[str],
    start: date,
    *,
    symbols: Collection[str] | None = None,
    data_dir: Path | None = None,
    archive: pl.DataFrame | None = None,
) -> list[str]:
    """检验/回测结果的降级注记 (结果字段 warnings)。

    - 行业类因子 (含 DSL group_* 自定义因子) 且 start 早于首份行业归档 → 行业快照偏差;
    - 截面类 RPS → ST 非点时; 指定标的子集 → 子集分位非全市场。
    """
    rs_names, group_needed = _expand_names(names)
    warnings: list[str] = []
    if rs_names & INDUSTRY_FACTOR_IDS or group_needed:
        if archive is None:
            archive = load_industry_archive(data_dir)
        first = earliest_industry_date(archive)
        if first is None:
            warnings.append("无行业归属数据 (ext_hy_ths 未拉取), 行业相对因子为空")
        elif start < first:
            warnings.append(f"{WARN_INDUSTRY_SNAPSHOT} ({start} ~ {first - timedelta(days=1)} 回退到 {first} 的归属)")
    cross = rs_names & (CROSS_SECTION_FACTOR_IDS | INDUSTRY_FACTOR_IDS)
    if cross:
        warnings.append(WARN_ST_NOT_PIT)
        if symbols:
            warnings.append(WARN_SUBSET_UNIVERSE)
    return warnings



# ───────────────────────────── 归档写入 ─────────────────────────────


def archive_industry_snapshot(df: pl.DataFrame, data_dir: Path, as_of: date) -> int:
    """把本次拉取/上传的行业归属写入 history/date=as_of (同日覆盖, 幂等)。

    只存 (symbol, 原始行业串): 解析口径变化时历史归档可重算。失败只告警, 不影响快照写入。
    """
    if "symbol" not in df.columns or INDUSTRY_FIELD not in df.columns or df.is_empty():
        return 0
    out = (
        df.select(pl.col("symbol").cast(pl.Utf8), pl.col(INDUSTRY_FIELD).cast(pl.Utf8))
        .filter(pl.col("symbol").is_not_null() & pl.col(INDUSTRY_FIELD).is_not_null())
        .unique(subset=["symbol"], keep="last")
    )
    if out.is_empty():
        return 0
    from app.services.fs_utils import atomic_write_parquet

    target = _industry_history_dir(Path(data_dir)) / f"date={as_of.isoformat()}" / "part.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(out, target)
    logger.info("行业归属归档: %s (%d 行)", target, out.height)
    return out.height


# ───────────────────────────── 矩阵回测适配 ─────────────────────────────


def matrix_values(market, name: str) -> np.ndarray:
    """matrix_native 回测评分适配: MarketDataMatrix (T×A) → 长表 → 同一计算 → T×A float32。

    只取 close 有效的 (日期, 标的) 格点组成长表 (与 panel 路径"无 bar 即无行"一致),
    因此 shift(N) 按实际 bar 数计, 两条路径逐位同口径。
    """
    close = np.asarray(market.close)
    rows, assets = close.shape
    out = np.full((rows, assets), np.nan, dtype=np.float32)
    valid = np.isfinite(close)
    if not valid.any():
        return out
    if any(len(str(label)) > 10 for label in market.timestamp_labels[:1]):
        logger.warning("RS factor %s skipped on intraday matrix (daily-only factor)", name)
        return out
    t_idx, a_idx = np.nonzero(valid)
    dates = pl.Series(list(market.timestamp_labels), dtype=pl.Utf8).str.slice(0, 10).str.to_date()
    volume = np.asarray(market.volume, dtype=np.float64)[valid]
    frame = pl.DataFrame({
        "symbol": np.asarray(market.symbols, dtype=object)[a_idx].astype(str),
        "date": dates.gather(t_idx),
        "close": close[valid].astype(np.float64),
        "volume": np.nan_to_num(volume, nan=0.0),
    })
    result = materialize_relative_strength(frame, [name])
    if name in result.columns:
        out[t_idx, a_idx] = result[name].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float32)
    return out
