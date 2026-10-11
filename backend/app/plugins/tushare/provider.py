"""Tushare Pro 内置数据源 provider。

方法签名对齐 custom.GenericHTTPProvider(service 分流点按这套签名调用),
注入 custom loader 注册表后, 各 service 无需改动即可路由到本 provider。

实现数据集(按 token 实际权限动态裁剪, 见下文"权限探测"):
  - daily        股票 daily / ETF fund_daily / 指数 index_daily, 原始价;
                 全市场窗口按 trade_date 逐日拉(1 请求覆盖全市场), 少量标的按 ts_code 拉
  - adj_factor   股票 adj_factor / ETF fund_adj; 累积后复权因子 → 单事件比值
  - minute       股票 stk_mins(1min/5min/...), 深历史
  - financial    income / balancesheet / cashflow / fina_indicator(合并报表), shares 恒空
  - instruments  stock_basic(L/P) + daily_basic 最新股本
realtime 未实现(实测 rt_k 无权限 40203), 不声明 → 自动回退 TickFlow。

单位与口径 (CONTRIBUTING §3.1, 实测 2026-10):
  - daily/fund_daily/index_daily: vol 已是手, 直接透传; amount 千元 → ×1000 元。
  - stk_mins: vol 为股 → floor(/100) 手; amount 已是元; trade_time 为北京墙钟 naive。
  - daily_basic: total_share/float_share 万股 → ×10000 股(instruments 维表口径为股)。
  - fina_indicator 百分比指标(roe/毛利率等)为百分点数值(30.5 = 30.5%), 与 fuyao 一致
    原样透传 — 项目 metrics 表与前端财务页/回测因子均按百分点消费, 不做 /100。
  - adj_factor 为累积后复权因子(上市首日=1, 只增不减); 本项目 ex_factor 为单事件比值
    (pre/post, 非累积), 累积链由 pipeline._apply_adj_factor 构建 →
    ex_factor(t) = adj(t) / adj(t_prev), 仅输出比值≠1 的事件日。

权限探测:
  - __init__ 不发请求; 只读本地缓存 data/cache/tushare/capabilities.json
    (token 指纹 sha256 前 8 位 + 时间, TTL 24h, token 变更即失效)。
  - 缓存缺失/过期时, 首次读 config.datasets 触发后台线程对每个数据集代表接口试拉 1 行
    (短超时), 不阻塞 loader 启动扫描与请求线程。
  - 仅 TusharePermissionError 判"无权限"并裁剪声明; 网络/限频等错误判"未知"并保留声明
    (避免网络抖动让能力永久消失), 稍后重探。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from app.data_providers.normalizer import DAILY_COLS, normalize_daily
from app.plugins.tushare import client as tushare_client
from app.plugins.tushare.client import (
    TushareAuthError,
    TushareClient,
    TushareError,
    TusharePermissionError,
)

logger = logging.getLogger(__name__)

# 声明全集(运行时按探测结果裁剪)。realtime 未实现不进全集, 见模块 docstring。
_DATASETS = ("daily", "adj_factor", "minute", "financial")
# 数据集 → 代表接口 + 1 行试拉参数(只看权限, 不看数据内容)
_PROBES: dict[str, tuple[str, dict]] = {
    "daily": ("daily", {"ts_code": "000001.SZ", "limit": 1}),
    "adj_factor": ("adj_factor", {"ts_code": "000001.SZ", "limit": 1}),
    "minute": ("stk_mins", {"ts_code": "000001.SZ", "freq": "1min", "limit": 1}),
    "financial": ("income", {"ts_code": "000001.SZ", "limit": 1}),
}

API_KEY_ENV = "TUSHARE_TOKEN"
SECRETS_FIELD = "tushare_api_key"  # UI 配置的 Key 存 secrets.json, 优先级高于 .env

_CAPS_FILE = "capabilities.json"
_CAPS_TTL_S = 24 * 3600
_CAPS_RETRY_S = 600  # 有"未知"结果时 10 分钟后再探, 不在每次读 datasets 时狂打接口
_PROBE_TIMEOUT_S = 8.0  # 探测专用短超时: 启动后台线程里也不长时间挂起

_ADJ_LOOKBACK_DAYS = 30  # 单事件比值需要窗口起点前一个交易日的累积因子(容忍长期停牌)
_ADJ_EPS = 1e-9  # 累积因子未变化时相邻值逐位相等, 比值恰为 1; eps 只吸收浮点噪声
_SYMBOL_BATCH = 50  # 单标的模式每批产出的标的数(iter_daily 有界批次)
_FINANCIAL_HISTORY_PERIODS = 8  # 财务首装历史: 最近 8 期(约 2 年)
_FINANCIAL_LOOKBACK_DAYS = {True: 400, False: 800}  # latest_only → 公告日回看窗口
_MINUTE_DEFAULT_DAYS = 5

_DAILY_API = {"stock": "daily", "etf": "fund_daily", "index": "index_daily"}
_ADJ_API = {"stock": "adj_factor", "etf": "fund_adj"}  # 指数无复权因子
_MINUTE_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min"}


def get_api_key() -> str:
    from app import secrets_store

    return secrets_store.get_env_backed_secret(SECRETS_FIELD, API_KEY_ENV)


def availability() -> tuple[bool, str]:
    """loader 启动自检: token 已配置(secrets.json 或 .env)才注册为可切换数据源。不抛异常、不发请求。"""
    if get_api_key():
        return True, "ok"
    return False, f"缺少 API Key(可在下方输入框直接填写,或配置环境变量 {API_KEY_ENV})"


def probe_api_key(api_key: str) -> tuple[bool, str]:
    """用候选 token 实探一次 trade_cal(先探后存)。不落盘, 返回信息不含 token。"""
    client = None
    try:
        client = tushare_client.TushareClient(token=api_key, timeout=10.0)
        client.query("trade_cal", {"exchange": "SSE", "limit": 1})
        return True, "ok"
    except TushareError as e:
        return False, f"Key 无效或网络失败: {e}"
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()


def _fingerprint(token: str) -> str:
    """token 指纹(sha256 前 8 位): 缓存按 token 隔离, 文件里不存 token 本身。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8] if token else ""


def _cache_dir() -> Path:
    from app.config import settings

    d = settings.data_dir / "cache" / "tushare"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _weekdays(start_d: date, end_d: date) -> list[date]:
    """窗口内工作日。逐日全市场拉取用: 节假日请求返回空, 代价远小于额外依赖交易日历。"""
    out = []
    d = start_d
    while d <= end_d:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _as_naive(dt: datetime) -> datetime:
    """带时区 → 北京墙钟 naive; naive 视为已是北京墙钟(项目统一口径)。"""
    if dt.tzinfo is None:
        return dt
    from zoneinfo import ZoneInfo

    return dt.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)


def _parse_ymd(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=False)


def _iso_ymd(value) -> str | None:
    """Tushare YYYYMMDD → ISO 日期字符串(项目财务表 period_end/announce_date 存储口径)。"""
    if not value:
        return None
    text = str(value)
    try:
        return datetime.strptime(text[:8], "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class _TushareConfig:
    """轻量 config shim, 让 custom loader 的 provider_has_dataset 能识别本 provider。

    datasets 为属性: 读取时顺带检查探测是否需要刷新(后台线程, 不阻塞调用方)。
    """

    name = "tushare"
    display_name = "tushare"
    path = None
    builtin = True

    def __init__(self, owner: TushareProvider) -> None:
        self._owner = owner

    @property
    def datasets(self) -> dict:
        self._owner._maybe_schedule_probe()
        return self._owner._datasets


class TushareProvider:
    """Tushare 数据源。数据集声明按 token 权限动态裁剪(config.datasets)。"""

    name = "tushare"
    builtin = True

    def __init__(self, client: TushareClient | None = None, *, auto_probe: bool = True) -> None:
        """client 可注入(测试); auto_probe=False 时只用缓存, 不自动起后台探测。不发任何请求。"""
        self._client = client
        self._auto_probe = auto_probe
        self._lock = threading.Lock()
        self._probing = False
        self._fp = _fingerprint(get_api_key())
        self._status: dict[str, str] = dict.fromkeys(_DATASETS, "unknown")
        self._next_probe_at = 0.0  # 0 = 尽快探测
        self._datasets: dict = dict.fromkeys(_DATASETS)
        cached = self._load_caps()
        if cached is not None:
            statuses, probed_at = cached
            self._apply_status(statuses)
            complete = all(v in ("ok", "denied") for v in statuses.values())
            self._next_probe_at = probed_at + (_CAPS_TTL_S if complete else _CAPS_RETRY_S)
        self.config = _TushareConfig(self)

    def close(self) -> None:  # loader.load_all 重建注册表时会对每个 provider 调 close
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None

    def _get_client(self) -> TushareClient:
        if self._client is None:
            self._client = tushare_client.TushareClient(token=get_api_key())
        return self._client

    # ---- 权限探测 ----
    def _caps_path(self) -> Path:
        return _cache_dir() / _CAPS_FILE

    def _load_caps(self) -> tuple[dict[str, str], float] | None:
        """读缓存; 指纹不符(换 token)/ 过期 / 损坏 → None。"""
        try:
            raw = json.loads(self._caps_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get("fingerprint") != self._fp:
            return None
        try:
            probed_at = float(raw.get("probed_at") or 0)
        except (TypeError, ValueError):
            return None
        if time.time() - probed_at >= _CAPS_TTL_S:
            return None
        statuses = raw.get("datasets")
        if not isinstance(statuses, dict):
            return None
        return {k: str(statuses.get(k, "unknown")) for k in _DATASETS}, probed_at

    def _save_caps(self, statuses: dict[str, str], probed_at: float) -> None:
        from app.services.fs_utils import atomic_write_text

        payload = {"fingerprint": self._fp, "probed_at": probed_at, "datasets": statuses}
        try:
            atomic_write_text(self._caps_path(), json.dumps(payload, ensure_ascii=False, indent=2))
        except OSError as e:  # 缓存写失败不致命, 下次启动再探
            logger.warning("Tushare 能力缓存写入失败: %s", e)

    def _apply_status(self, statuses: dict[str, str]) -> None:
        """整体替换 datasets dict(读侧无锁也只会看到旧/新两种完整状态)。"""
        self._status = dict(statuses)
        self._datasets = {k: None for k in _DATASETS if statuses.get(k) != "denied"}

    def _maybe_schedule_probe(self) -> None:
        if not self._auto_probe or not self._fp:
            return
        # 测试进程(无 conftest 隔离, 开发机可能设有真实 TUSHARE_TOKEN)里其他测试会经
        # loader 注册本 provider 并读 datasets → 未注入假 client 时禁止后台探测, 保证不触网
        if self._client is None and os.environ.get("PYTEST_CURRENT_TEST"):
            return
        with self._lock:
            if self._probing or time.time() < self._next_probe_at:
                return
            self._probing = True
        threading.Thread(target=self._probe_worker, name="tushare-caps-probe", daemon=True).start()

    def _probe_worker(self) -> None:
        try:
            self.refresh_capabilities()
        except Exception as e:  # noqa: BLE001 — 后台线程兜底, 不影响主流程
            logger.warning("Tushare 权限探测失败: %s", e)
        finally:
            with self._lock:
                self._probing = False

    def refresh_capabilities(self) -> dict[str, str]:
        """同步探测每个数据集的代表接口, 更新声明并写缓存。返回 {dataset: ok|denied|unknown}。

        合并规则: 本轮"未知"不覆盖上轮确定结论(网络抖动不改变已知权限)。
        """
        client = self._client
        own = client is None
        if own:
            client = tushare_client.TushareClient(token=get_api_key(), timeout=_PROBE_TIMEOUT_S)
        statuses: dict[str, str] = {}
        try:
            for ds, (api, params) in _PROBES.items():
                try:
                    client.query(api, params)
                    statuses[ds] = "ok"
                except TusharePermissionError:
                    statuses[ds] = "denied"
                except TushareError as e:
                    logger.info("Tushare 探测 %s(%s) 结果未知, 保留声明: %s", ds, api, e)
                    statuses[ds] = "unknown"
        finally:
            if own:
                with contextlib.suppress(Exception):
                    client.close()
        merged = {
            k: (self._status.get(k, "unknown") if v == "unknown" else v) for k, v in statuses.items()
        }
        now = time.time()
        self._apply_status(merged)
        complete = all(v in ("ok", "denied") for v in merged.values())
        self._next_probe_at = now + (_CAPS_TTL_S if complete else _CAPS_RETRY_S)
        self._save_caps(merged, now)
        denied = sorted(k for k, v in merged.items() if v == "denied")
        if denied:
            logger.info("Tushare token 无以下数据集权限, 已回退 TickFlow: %s", ", ".join(denied))
        return merged

    # ---- daily ----
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """日K → 内部契约(原始价, volume 手, amount 元)。见 iter_daily。"""
        chunks = [
            df
            for df in self.iter_daily(
                symbols, start_time=start_time, end_time=end_time,
                asset_type=asset_type, on_chunk_done=on_chunk_done,
            )
            if not df.is_empty()
        ]
        return pl.concat(chunks, how="diagonal_relaxed") if chunks else pl.DataFrame()

    def iter_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """分批产出日K。取数模式按请求数择优:

        - 逐日(trade_date): 1 请求覆盖全市场, 工作日数 < 标的数时更省(全市场同步);
          某日失败直接抛出 — 静默跳过会造成全市场缺一天, 由 kline_sync 丢弃 staging;
        - 单标的(ts_code + 区间, offset 分页): 少量标的/长窗口; 单标的失败软跳过。
        index_daily 不支持按日期取全市场, 恒走单标的。
        """
        api = _DAILY_API.get(asset_type)
        if not symbols or api is None:
            return
        end_d = (end_time or datetime.now()).date()
        start_d = (start_time.date() if start_time else end_d - timedelta(days=365))
        if start_d > end_d:
            return
        symset = set(symbols)
        days = _weekdays(start_d, end_d)
        if asset_type != "index" and len(days) < len(symbols):
            for i, d in enumerate(days):
                raw = self._get_client().query_all(api, {"trade_date": _ymd(d)})
                if on_chunk_done:
                    on_chunk_done(i + 1, len(days))
                df = self._daily_frame(raw, symset)
                if not df.is_empty():
                    yield df
            return

        batches = [symbols[i:i + _SYMBOL_BATCH] for i in range(0, len(symbols), _SYMBOL_BATCH)]
        for bi, batch in enumerate(batches):
            frames = []
            for sym in batch:
                try:
                    frames.append(self._get_client().query_all(
                        api, {"ts_code": sym, "start_date": _ymd(start_d), "end_date": _ymd(end_d)}
                    ))
                except (TusharePermissionError, TushareAuthError):
                    raise  # 权限/token 问题对所有标的都成立, 逐只重试只会白耗配额
                except TushareError as e:
                    logger.warning("Tushare 日K拉取失败 %s [%s ~ %s]: %s", sym, start_d, end_d, e)
            raw = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
            if on_chunk_done:
                on_chunk_done(bi + 1, len(batches))
            df = self._daily_frame(raw, symset)
            if not df.is_empty():
                yield df

    def _daily_frame(self, raw: pl.DataFrame, symset: set[str]) -> pl.DataFrame:
        """daily/fund_daily/index_daily 原始帧 → 内部日K(amount 千元→元, vol 已是手)。"""
        need = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
        if raw.is_empty() or not need <= set(raw.columns):
            if not raw.is_empty():
                logger.warning("Tushare 日K响应缺少字段 %s, 疑似接口结构变化", sorted(need - set(raw.columns)))
            return pl.DataFrame()
        df = raw.filter(pl.col("ts_code").is_in(sorted(symset))).select(
            pl.col("ts_code").cast(pl.String).alias("symbol"),
            _parse_ymd("trade_date").alias("date"),
            *(pl.col(c).cast(pl.Float64, strict=False) for c in ("open", "high", "low", "close")),
            pl.col("vol").cast(pl.Float64, strict=False).alias("volume"),
            (pl.col("amount").cast(pl.Float64, strict=False) * 1000.0).alias("amount"),
        )
        out = normalize_daily(df, source=self.name)
        if out.is_empty():
            return out
        cols = [c for c in DAILY_COLS if c in out.columns]
        return out.select(cols).unique(subset=["symbol", "date"], keep="last").sort(["symbol", "date"])

    # ---- adj_factor ----
    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """除权因子 → 内部契约(symbol/trade_date/ex_factor, 单事件比值非累积)。

        Tushare 给逐日累积后复权因子 adj(t); 事件日比值 = adj(t)/adj(前一交易日)(>1 表示
        除权使价格下移, 与 pipeline 的 pre/post 口径一致)。窗口起点向前多取 30 天,
        保证起点当天的事件也有"前一日"可比。
        """
        schema = {"symbol": pl.String, "trade_date": pl.Date, "ex_factor": pl.Float64}
        api = _ADJ_API.get(asset_type)
        if not symbols or api is None:
            return pl.DataFrame(schema=schema)
        end_d = (end_time or datetime.now()).date()
        start_d = start_time.date() if start_time else None
        fetch_start = start_d - timedelta(days=_ADJ_LOOKBACK_DAYS) if start_d else None
        symset = set(symbols)

        frames: list[pl.DataFrame] = []
        days = _weekdays(fetch_start, end_d) if fetch_start else []
        if fetch_start and len(days) < len(symbols):
            # 逐日全市场: 某日缺失会把跨缺口的比值记到错误日期 → 失败直接抛出(fail-closed)
            for i, d in enumerate(days):
                frames.append(self._get_client().query_all(api, {"trade_date": _ymd(d)}))
                if on_chunk_done:
                    on_chunk_done(i + 1, len(days))
        else:
            for i, sym in enumerate(symbols):
                params = {"ts_code": sym, "end_date": _ymd(end_d)}
                if fetch_start:
                    params["start_date"] = _ymd(fetch_start)
                try:
                    frames.append(self._get_client().query_all(api, params))
                except (TusharePermissionError, TushareAuthError):
                    raise
                except TushareError as e:
                    # 单标的失败: 整只跳过(不输出半段序列推出的伪事件)
                    logger.warning("Tushare 复权因子拉取失败 %s: %s", sym, e)
                if on_chunk_done:
                    on_chunk_done(i + 1, len(symbols))
        raw = pl.concat([f for f in frames if not f.is_empty()], how="diagonal_relaxed") if any(
            not f.is_empty() for f in frames
        ) else pl.DataFrame()
        events = cumulative_to_events(raw, symset)
        if start_d is not None and not events.is_empty():
            events = events.filter(pl.col("trade_date") >= start_d)
        return events if not events.is_empty() else pl.DataFrame(schema=schema)

    # ---- minute ----
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
        freq: str = "1m",
    ) -> pl.DataFrame:
        """分钟K → [symbol, datetime(北京墙钟 naive), OHLC, volume(手), amount(元)]。

        stk_mins 只覆盖股票: ETF/指数抛错 → kline_sync 自动回退 TickFlow(而非返回空帧
        被当成"该标的无分钟数据")。单标的失败同样抛出, 交给调用方整体回退。
        """
        if asset_type != "stock":
            raise TushareError(f"Tushare stk_mins 不覆盖 {asset_type} 分钟K")
        tf = _MINUTE_FREQ.get(freq)
        if tf is None:
            raise TushareError(f"Tushare 分钟K不支持周期 {freq}")
        end_dt = _as_naive(end_time) if end_time else datetime.now()
        start_dt = _as_naive(start_time) if start_time else end_dt - timedelta(days=_MINUTE_DEFAULT_DAYS)
        frames = []
        for i, sym in enumerate(symbols or []):
            raw = self._get_client().query_all("stk_mins", {
                "ts_code": sym,
                "freq": tf,
                "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            })
            frames.append(_minute_frame(raw))
            if on_chunk_done:
                on_chunk_done(i + 1, len(symbols))
        frames = [f for f in frames if not f.is_empty()]
        if not frames:
            return pl.DataFrame(schema=_MINUTE_SCHEMA)
        return pl.concat(frames, how="vertical_relaxed").sort(["symbol", "datetime"])

    # ---- financial ----
    def get_financials(
        self,
        table: str,
        symbols: list[str],
        latest_only: bool = True,
    ) -> pl.DataFrame:
        """财务数据 → canonical 列(symbol/period_end/announce_date/指标)。

        - 只取合并报表(report_type=1), 同一报告期多次修订取最新版(update_flag 高者);
        - latest_only=True 取每股最近 1 期, 否则最近 8 期(首装历史);
        - shares: 无可靠全量股本历史接口, 恒返回空(已有存量靠合并写入保留)。
        单股失败软跳过; 权限/token 错误对全体成立, 直接抛出交给 financial_sync 记录。
        """
        if table == "shares":
            logger.info("Tushare 插件未接入股本表, shares 跳过 (已有数据保留)")
            return pl.DataFrame()
        spec = _FINANCIAL_TABLES.get(table)
        if spec is None:
            return pl.DataFrame()
        api, field_map = spec
        start = date.today() - timedelta(days=_FINANCIAL_LOOKBACK_DAYS[bool(latest_only)])
        rows: list[dict] = []
        for sym in symbols:
            try:
                raw = self._get_client().query_all(api, {"ts_code": sym, "start_date": _ymd(start)})
            except (TusharePermissionError, TushareAuthError):
                raise
            except TushareError as e:
                logger.warning("Tushare 财务 %s %s 失败: %s", api, sym, e)
                continue
            rows.extend(_financial_rows(sym, raw, field_map, latest_only))
        # 稀疏列(如仅银行有值的字段)前 100 行可能全空 → 全量扫描推断 (#458)
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()

    # ---- instruments ----
    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        """标的维表: tickflow Instrument 形状(symbol/name/code/exchange/region/type + ext),
        供 instrument_sync._flatten_instruments 复用。

        覆盖上市(L)与暂停上市(P); 退市(D)不进维表 — 维表驱动全市场同步与选股,
        退市股会让每轮同步白打数百个请求。股本取最近交易日 daily_basic(万股 → 股)。
        失败软返回 [](instrument_sync 将回退 TickFlow)。
        """
        if asset_type != "stock":
            return []
        client = self._get_client()
        fields = "ts_code,symbol,name,exchange,list_date,list_status"
        try:
            basics = [client.query("stock_basic", {"list_status": s}, fields) for s in ("L", "P")]
        except TushareError as e:
            logger.warning("Tushare stock_basic 拉取失败: %s", e)
            return []
        basic = pl.concat([b for b in basics if not b.is_empty()], how="diagonal_relaxed") if any(
            not b.is_empty() for b in basics
        ) else pl.DataFrame()
        if basic.is_empty() or "ts_code" not in basic.columns:
            logger.warning("Tushare stock_basic 返回为空或缺少 ts_code, 疑似接口结构变化")
            return []
        shares = self._latest_shares(client)
        out = []
        for r in basic.iter_rows(named=True):
            sym = r.get("ts_code")
            if not sym:
                continue
            code, _, suffix = str(sym).partition(".")
            total, flt = shares.get(sym, (None, None))
            out.append({
                "symbol": sym,
                "name": r.get("name"),
                "code": code,
                "exchange": suffix or None,
                "region": "CN",
                "type": "stock",
                "ext": {
                    "listing_date": _iso_ymd(r.get("list_date")),
                    "total_shares": total,
                    "float_shares": flt,
                    "tick_size": 0.01,
                    "limit_up": None,  # 涨跌停价由 enriched 管道按规则计算, 不伪造
                    "limit_down": None,
                },
            })
        return out

    def _latest_shares(self, client: TushareClient) -> dict[str, tuple[float | None, float | None]]:
        """最近交易日 daily_basic 股本(万股 → 股)。盘后才发布当日数据, 故从今天往前回找。

        失败只影响股本列(换手率降级), 不致命。
        """
        today = date.today()
        try:
            cal = client.query("trade_cal", {
                "exchange": "SSE", "is_open": "1",
                "start_date": _ymd(today - timedelta(days=20)), "end_date": _ymd(today),
            }, "cal_date")
            days = sorted((str(d) for d in cal["cal_date"].to_list() if d), reverse=True) if (
                "cal_date" in cal.columns
            ) else []
            for d in days[:5]:
                df = client.query_all("daily_basic", {"trade_date": d}, "ts_code,total_share,float_share")
                if df.is_empty() or "ts_code" not in df.columns:
                    continue
                return {
                    r["ts_code"]: (
                        (t * 10_000) if (t := _to_float(r.get("total_share"))) is not None else None,
                        (f * 10_000) if (f := _to_float(r.get("float_share"))) is not None else None,
                    )
                    for r in df.iter_rows(named=True)
                }
        except TushareError as e:
            logger.warning("Tushare daily_basic 股本拉取失败, 维表股本留空: %s", e)
        return {}

    def trading_days(self) -> set:
        """近一年交易日集合 (与 fuyao 同协议, 供交易日探针)。失败抛 TushareError。"""
        today = date.today()
        cal = self._get_client().query("trade_cal", {
            "exchange": "SSE", "is_open": "1",
            "start_date": _ymd(today - timedelta(days=366)), "end_date": _ymd(today),
        }, "cal_date")
        if "cal_date" not in cal.columns:
            return set()
        return {
            d for d in (datetime.strptime(str(x), "%Y%m%d").date() for x in cal["cal_date"] if x)
        }

    # ---- 测试(设置页试拉) ----
    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        if dataset not in self.config.datasets:
            return {
                "provider": self.name, "dataset": dataset, "rows": 0,
                "error": f"Tushare 插件未接入或 token 无 {dataset} 权限(自动回退 TickFlow)",
            }
        syms = [s for s in (symbols or [])][:3] or ["000001.SZ"]
        now = datetime.now()
        try:
            if dataset == "daily":
                df = self.get_daily(syms, now - timedelta(days=30), now)
            elif dataset == "adj_factor":
                df = self.get_adj_factors(syms, now - timedelta(days=365), now)
            elif dataset == "minute":
                df = self.get_minute(syms[:1], now - timedelta(days=3), now)
            else:
                df = self.get_financials("metrics", syms[:1], latest_only=True)
        except TushareError as e:
            return {"provider": self.name, "dataset": dataset, "rows": 0, "error": str(e)}
        head = df.head(5).to_dicts()
        for row in head:  # date/datetime → ISO 字符串, 保证 JSON 可序列化
            for k, v in list(row.items()):
                if isinstance(v, (date, datetime)):
                    row[k] = v.isoformat()
        return {
            "provider": self.name, "dataset": dataset, "rows": df.height,
            "columns": df.columns, "preview": head,
        }


# ================================================================
# 纯函数转换(模块级, 便于单测)
# ================================================================

def cumulative_to_events(raw: pl.DataFrame, symset: set[str] | None = None) -> pl.DataFrame:
    """Tushare 逐日累积因子 → 单事件比值 [symbol, trade_date, ex_factor]。

    ex_factor(t) = adj(t) / adj(上一条记录)。累积因子在非事件日逐位不变, 比值恰为 1 →
    只保留 |比值-1| > eps 的行。每个标的的首条记录没有"上一条", 不产生事件
    (上市首日或拉取窗口起点, 后者由调用方多取 30 天回看覆盖)。
    """
    schema = {"symbol": pl.String, "trade_date": pl.Date, "ex_factor": pl.Float64}
    if raw.is_empty() or not {"ts_code", "trade_date", "adj_factor"} <= set(raw.columns):
        return pl.DataFrame(schema=schema)
    df = raw.select(
        pl.col("ts_code").cast(pl.String).alias("symbol"),
        _parse_ymd("trade_date").alias("trade_date"),
        pl.col("adj_factor").cast(pl.Float64, strict=False).alias("adj"),
    ).drop_nulls()
    if symset is not None:
        df = df.filter(pl.col("symbol").is_in(sorted(symset)))
    df = (
        df.filter(pl.col("adj") > 0)
        .unique(subset=["symbol", "trade_date"], keep="last")
        .sort(["symbol", "trade_date"])
        .with_columns((pl.col("adj") / pl.col("adj").shift(1).over("symbol")).alias("ex_factor"))
        .filter(pl.col("ex_factor").is_not_null() & ((pl.col("ex_factor") - 1.0).abs() > _ADJ_EPS))
    )
    return df.select("symbol", "trade_date", "ex_factor").cast(schema)


_MINUTE_SCHEMA = {
    "symbol": pl.String, "datetime": pl.Datetime("us"),
    "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
    "volume": pl.Float64, "amount": pl.Float64,
}


def _minute_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """stk_mins 原始帧 → canonical 8 列。vol 股 → floor(/100) 手; amount 已是元。

    显式 schema: 避免稀疏 amount 列被推断为 Null 后 append 失败 (#458 同类问题)。
    stk_mins 默认按时间倒序返回 → 统一升序。
    """
    need = {"ts_code", "trade_time", "open", "high", "low", "close", "vol"}
    if raw.is_empty() or not need <= set(raw.columns):
        if not raw.is_empty():
            logger.warning("Tushare 分钟K响应缺少字段 %s, 疑似接口结构变化", sorted(need - set(raw.columns)))
        return pl.DataFrame(schema=_MINUTE_SCHEMA)
    amount = (
        pl.col("amount").cast(pl.Float64, strict=False)
        if "amount" in raw.columns else pl.lit(None, dtype=pl.Float64)
    )
    return (
        raw.select(
            pl.col("ts_code").cast(pl.String).alias("symbol"),
            pl.col("trade_time").cast(pl.String)
            .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False).alias("datetime"),
            *(pl.col(c).cast(pl.Float64, strict=False) for c in ("open", "high", "low", "close")),
            (pl.col("vol").cast(pl.Float64, strict=False) / 100.0).floor().alias("volume"),
            amount.alias("amount"),
        )
        .drop_nulls(subset=["symbol", "datetime"])
        .unique(subset=["symbol", "datetime"], keep="last")
        .sort(["symbol", "datetime"])
        .cast(_MINUTE_SCHEMA)
    )


# 字段映射: Tushare 原始字段 → 项目 canonical 列名(TickFlow 口径, 前端财务页与回测
# FUNDAMENTAL_FACTORS 按此消费; 目标列与 fuyao 映射一致)。映射之外的数值字段以原名透传。
_INCOME_FIELD_MAP = {
    "total_revenue": "revenue",
    "oper_cost": "operating_cost",
    "sell_exp": "selling_expense",
    "admin_exp": "admin_expense",
    "rd_exp": "rd_expense",
    "fin_exp": "financial_expense",
    "operate_profit": "operating_profit",
    "non_oper_income": "non_operating_income",
    "non_oper_exp": "non_operating_expense",
    "total_profit": "total_profit",
    "income_tax": "income_tax",
    "n_income": "net_income",
    "n_income_attr_p": "net_income_attributable",
    "basic_eps": "basic_eps",
    "diluted_eps": "diluted_eps",
}
_BALANCE_FIELD_MAP = {
    "total_assets": "total_assets",
    "total_cur_assets": "total_current_assets",
    "total_nca": "total_non_current_assets",
    "money_cap": "cash_and_equivalents",
    "accounts_receiv": "accounts_receivable",
    "inventories": "inventory",
    "fix_assets": "fixed_assets",
    "intan_assets": "intangible_assets",
    "goodwill": "goodwill",
    "total_liab": "total_liabilities",
    "total_cur_liab": "total_current_liabilities",
    "total_ncl": "total_non_current_liabilities",
    "st_borr": "short_term_borrowing",
    "lt_borr": "long_term_borrowing",
    "acct_payable": "accounts_payable",
    "total_hldr_eqy_inc_min_int": "total_equity",
    "total_hldr_eqy_exc_min_int": "equity_attributable",
    "undistr_porfit": "retained_earnings",  # Tushare 原字段即此拼写
    "minority_int": "minority_interest",
}
_CASHFLOW_FIELD_MAP = {
    "n_cashflow_act": "net_operating_cash_flow",
    "n_cashflow_inv_act": "net_investing_cash_flow",
    "n_cash_flows_fnc_act": "net_financing_cash_flow",
    "c_pay_acq_const_fiolta": "capex",
    "n_incr_cash_cash_equ": "net_cash_change",
}
# fina_indicator 百分比指标为百分点数值, 与 fuyao 一致原样透传(见模块 docstring)
_METRICS_FIELD_MAP = {
    "eps": "eps_basic",
    "dt_eps": "eps_diluted",
    "bps": "bps",
    "ocfps": "ocfps",
    "roe_waa": "roe",  # 加权平均 ROE, 与 fuyao index_weighted_avg_roe 同口径
    "roe_dt": "roe_diluted",  # 近似: Tushare 无"稀释 ROE", 取扣非 ROE
    "roa": "roa",
    "grossprofit_margin": "gross_margin",
    "netprofit_margin": "net_margin",
    "debt_to_assets": "debt_to_asset_ratio",
    "or_yoy": "revenue_yoy",
    "netprofit_yoy": "net_income_yoy",
    "ocf_to_or": "operating_cash_to_revenue",
    "inv_turn": "inventory_turnover",
    "profit_dedt": "net_income_deducted",
}
# 项目财务表名 → (Tushare 接口, 字段映射)
_FINANCIAL_TABLES = {
    "income": ("income", _INCOME_FIELD_MAP),
    "balance_sheet": ("balancesheet", _BALANCE_FIELD_MAP),
    "cash_flow": ("cashflow", _CASHFLOW_FIELD_MAP),
    "metrics": ("fina_indicator", _METRICS_FIELD_MAP),
}
_FIN_META_COLS = {
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "end_type", "update_flag",
}


def _financial_rows(
    symbol: str, raw: pl.DataFrame, field_map: dict[str, str], latest_only: bool
) -> list[dict]:
    """单股原始财务帧 → canonical 行。只取合并报表; 同报告期多版本取最新修订。

    announce_date 取 ann_date(公告日), 缺失回退 f_ann_date: 回测按公告日 +1 生效,
    用较晚的公告日更保守, 不会提前泄露未公告数据(CONTRIBUTING §3.4)。
    """
    if raw.is_empty() or "end_date" not in raw.columns:
        return []
    df = raw
    if "report_type" in df.columns:
        df = df.filter(pl.col("report_type").cast(pl.String) == "1")
    sort_cols = ["end_date"] + [c for c in ("update_flag", "ann_date") if c in df.columns]
    df = (
        df.with_columns(pl.col(c).cast(pl.String) for c in sort_cols)
        .sort(sort_cols, descending=True, nulls_last=True)
        .unique(subset=["end_date"], keep="first", maintain_order=True)
        .head(1 if latest_only else _FINANCIAL_HISTORY_PERIODS)
    )
    canonical = set(field_map.values())
    out = []
    for r in df.iter_rows(named=True):
        row: dict = {
            "symbol": symbol,
            "period_end": _iso_ymd(r.get("end_date")),
            "announce_date": _iso_ymd(r.get("ann_date") or r.get("f_ann_date")),
        }
        for src, dst in field_map.items():
            row[dst] = _to_float(r.get(src))
        # 映射之外的数值字段原名透传为扩展列; 与 canonical 同名的原字段(如 roe)不得覆盖映射结果
        for src, value in r.items():
            if (
                src in field_map or src in _FIN_META_COLS or src in canonical or src in row
                or not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                continue
            row[src] = value
        out.append(row)
    return out
