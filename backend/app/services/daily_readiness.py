"""盘后数据就绪判定 — 目标交易日的日K / enriched / 指数是否已真实落盘。

背景: 部分数据源 (如 Tushare) 当日日K在收盘后 1~数小时才发布。盘后管道
请求 [昨天 ~ 今天] 时上游只返回昨天, 旧逻辑把"请求跨度"当成新增天数,
复盘再按"最新有数据日"回退到昨天, 结果生成了前一天的报告。

本模块只读磁盘分区元数据 (parquet footer, 不解压数据页), 供盘后补拉与
复盘门控共用。判定口径:
- 日K: 目标日分区存在, 非盘中快照, 行数不低于前一分区的 DAILY_MIN_RATIO;
- enriched: 目标日分区行数不低于目标日日K的 ENRICHED_MIN_RATIO
  (停牌/新股预热不足的标的本就可能没有指标, 不要求严格相等);
- 指数: 开启指数拉取且本地有指数历史时, 目标日分区行数不低于前一分区的比例。
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path

from app.market_time import CN_TZ

logger = logging.getLogger(__name__)

DAILY_MIN_RATIO = 0.9
ENRICHED_MIN_RATIO = 0.95
INDEX_MIN_RATIO = 0.9
# 实时侧写入的当日行: quote_ts 早于 15:00 视为盘中快照 (与 data_integrity 的
# 历史日 15:30 定版线不同: 当日 15:00~15:30 的收盘定版行可作为当日数据)。
_TODAY_SNAPSHOT_CUTOFF = dt_time(15, 0)


@dataclass
class Readiness:
    target_date: str
    ready: bool
    daily_latest: str | None = None
    enriched_latest: str | None = None
    index_latest: str | None = None
    daily_rows: int = 0
    enriched_rows: int = 0
    index_rows: int | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _partition_dates(root: Path) -> list[date]:
    out: list[date] = []
    if not root.exists():
        return out
    for part in root.glob("date=*"):
        try:
            out.append(date.fromisoformat(part.name.removeprefix("date=")))
        except ValueError:
            continue
    out.sort()
    return out


def _rows_and_quote_ts(part_dir: Path) -> tuple[int, int | None]:
    """分区总行数与 max(quote_ts) (毫秒); 只读 parquet 元数据。"""
    import pyarrow.parquet as pq

    rows = 0
    q_max: int | None = None
    for path in sorted(part_dir.glob("*.parquet")):
        try:
            meta = pq.read_metadata(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("readiness metadata skipped %s: %s", path, e)
            continue
        rows += meta.num_rows
        names = [meta.schema.column(i).name for i in range(meta.num_columns)]
        if "quote_ts" not in names:
            continue
        idx = names.index("quote_ts")
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(idx).statistics
            if stats is not None and stats.has_min_max and stats.max is not None:
                v = int(stats.max)
                q_max = v if q_max is None else max(q_max, v)
    return rows, q_max


def _is_intraday_snapshot(day: date, part_dir: Path, q_max: int | None) -> bool:
    if q_max is None:
        return False
    try:
        ts = datetime.fromtimestamp(q_max / 1000, tz=CN_TZ)
    except (OverflowError, OSError, ValueError):
        return False
    if ts.date() > day:
        return False
    if ts.date() == day and ts.time() >= _TODAY_SNAPSHOT_CUTOFF:
        return False
    # 复用完整性检测的停牌残留判别: batch 权威行为主时不算快照
    from app.services.data_integrity import _partition_is_snapshot
    return _partition_is_snapshot(day, part_dir, q_max)


def _prev_rows(root: Path, dates: list[date], target: date) -> int | None:
    prev = [d for d in dates if d < target]
    if not prev:
        return None
    rows, _ = _rows_and_quote_ts(root / f"date={prev[-1].isoformat()}")
    return rows


def check_ready(data_dir: Path, target: date, *, check_index: bool | None = None) -> Readiness:
    """判定 target 当日数据是否已完整落盘。只读, 不抛异常。"""
    from app.tickflow.repository import enriched_dirname

    data_dir = Path(data_dir)
    if check_index is None:
        from app.services import preferences
        check_index = preferences.get_pipeline_pull_index()

    daily_root = data_dir / "kline_daily"
    enriched_root = data_dir / enriched_dirname("stock")
    index_root = data_dir / "kline_index_daily"

    daily_dates = _partition_dates(daily_root)
    enriched_dates = _partition_dates(enriched_root)
    index_dates = _partition_dates(index_root)

    res = Readiness(
        target_date=target.isoformat(),
        ready=False,
        daily_latest=daily_dates[-1].isoformat() if daily_dates else None,
        enriched_latest=enriched_dates[-1].isoformat() if enriched_dates else None,
        index_latest=index_dates[-1].isoformat() if index_dates else None,
    )
    try:
        if target not in daily_dates:
            res.reasons.append(f"日K 缺少 {target} (最新 {res.daily_latest or '无'})")
        else:
            part = daily_root / f"date={target.isoformat()}"
            res.daily_rows, q_max = _rows_and_quote_ts(part)
            prev = _prev_rows(daily_root, daily_dates, target)
            if _is_intraday_snapshot(target, part, q_max):
                res.reasons.append(f"日K {target} 仍是盘中快照")
            elif res.daily_rows <= 0:
                res.reasons.append(f"日K {target} 为空")
            elif prev and res.daily_rows < prev * DAILY_MIN_RATIO:
                res.reasons.append(f"日K {target} 仅 {res.daily_rows} 行 (前一交易日 {prev} 行)")

        if target not in enriched_dates:
            res.reasons.append(f"指标缺少 {target} (最新 {res.enriched_latest or '无'})")
        else:
            res.enriched_rows, _ = _rows_and_quote_ts(enriched_root / f"date={target.isoformat()}")
            if res.daily_rows and res.enriched_rows < res.daily_rows * ENRICHED_MIN_RATIO:
                res.reasons.append(
                    f"指标 {target} 仅 {res.enriched_rows} 行, 日K {res.daily_rows} 行"
                )

        # 指数: 用户开启拉取且本地已有指数历史才校验 (从未拉过指数的部署不卡)
        if check_index and index_dates:
            if target not in index_dates:
                res.reasons.append(f"指数缺少 {target} (最新 {res.index_latest})")
            else:
                res.index_rows, _ = _rows_and_quote_ts(index_root / f"date={target.isoformat()}")
                prev = _prev_rows(index_root, index_dates, target)
                if prev and res.index_rows < prev * INDEX_MIN_RATIO:
                    res.reasons.append(f"指数 {target} 仅 {res.index_rows} 行 (前一交易日 {prev} 行)")
    except Exception as e:  # noqa: BLE001
        res.reasons.append(f"就绪检查异常: {e}")

    res.ready = not res.reasons
    return res


# 上游探针样本: 沪深主板高流动性标的, 任一返回目标日即认为上游已发布
_PROBE_SYMBOLS = ("600000.SH", "000001.SZ", "600519.SH")


def probe_upstream_daily(capset, target: date) -> bool | None:
    """少量样本探测上游是否已有 target 日K, 不落盘。

    True=已发布, False=确认尚未发布, None=无法探测 (调用方按需直接跑完整管道)。
    TickFlow 实时行情路径 (quote.pool) 每次都写当日, 无需探测 → None。
    """
    import polars as pl

    from app.services import preferences
    from app.tickflow.capabilities import Cap

    start = datetime.combine(target, dt_time.min)
    end = datetime.combine(target, dt_time.max)
    try:
        provider_name = preferences.get_daily_data_provider()
        df = None
        if provider_name != "tickflow":
            from app.data_providers import custom as custom_sources
            if custom_sources.provider_has_dataset(provider_name, "daily"):
                provider = custom_sources.get_provider(provider_name)
                df = provider.get_daily(list(_PROBE_SYMBOLS), start_time=start, end_time=end)
        if df is None:
            if capset is not None and capset.has(Cap.QUOTE_POOL):
                return None
            if capset is None or not capset.has(Cap.KLINE_DAILY_BATCH):
                return None
            from app.services import kline_sync
            df = kline_sync.sync_daily_batch(list(_PROBE_SYMBOLS), start_time=start, end_time=end)
        if df is None or df.is_empty() or "date" not in df.columns:
            return False
        dates = df.select(pl.col("date").cast(pl.Date, strict=False)).to_series()
        return bool((dates == target).any())
    except Exception as e:  # noqa: BLE001
        logger.info("upstream daily probe failed (按无法探测处理): %s", e)
        return None
