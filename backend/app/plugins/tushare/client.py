"""Tushare Pro HTTP 客户端。

职责: 认证、统一信封解包、错误分类、按接口限频、offset 分页。不知道 provider / services 层。
不引入 tushare SDK(其依赖 pandas 且封装了全局 token), 直接 POST 官方 HTTP 网关:
  请求 {api_name, token, params, fields} → 响应 {code, msg, data: {fields, items}}。

安全: token 只放在请求体里, 不进 URL; 所有异常/日志消息统一经 _scrub 擦除 token,
即使上游 msg 回显了 token 也不会外泄到日志或设置页。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable

import httpx
import polars as pl

logger = logging.getLogger(__name__)

BASE_URL = "http://api.tushare.pro"

_DEFAULT_RPM = 200  # 默认每接口每分钟 200 次(2000 积分档常见上限), TUSHARE_RPM 覆盖
_RATE_RETRY_BACKOFF_S = (3.0, 10.0, 30.0)  # 限频退避重试(最多 3 次)
_DEFAULT_PAGE_SIZE = 2000  # 未知单次上限的接口取保守页大小, 避免服务端截断被误判为末页
_MAX_PAGES = 200  # 分页页数上限: 防 has_more 异常/服务端忽略 offset 导致死循环

# 实测/官方文档给出的单次返回上限。页大小 ≤ 上限时"短页即末页"才成立。
PAGE_LIMITS = {
    "daily": 6000,
    "stk_mins": 8000,
}

# 错误码(官方): 40203 无接口权限; 40101/-2001 token 无效。限频与权限共用 40203,
# 只能靠 msg 区分 → 先判限频再判权限, 否则限频会被误判为"无权限"而永久裁剪数据集。
_PERMISSION_CODES = {40203}
_AUTH_CODES = {40101, -2001}
_RATE_MARKERS = ("每分钟最多访问", "每天最多访问", "每小时最多访问", "访问频率")
_AUTH_MARKERS = ("token不对", "token 不对", "token无效", "token 无效", "请确认token")


class TushareError(Exception):
    """Tushare 接口错误(网络失败 / HTTP 非 200 / 信封 code != 0)。"""


class TusharePermissionError(TushareError):
    """无接口权限(code=40203 且非限频)。唯一会让 provider 裁剪数据集的错误。"""


class TushareAuthError(TushareError):
    """token 无效。"""


class TushareRateLimitError(TushareError):
    """限频(重试耗尽后抛出)。不代表无权限。"""


def classify_error(code, msg: str) -> TushareError:
    """信封错误 → 异常类型。限频优先于权限判定(两者可能同为 40203)。"""
    text = str(msg or "")
    try:
        icode = int(code)
    except (TypeError, ValueError):
        icode = None
    detail = f"Tushare 接口错误 code={code}: {text}"
    if any(m in text for m in _RATE_MARKERS):
        return TushareRateLimitError(detail)
    if icode in _PERMISSION_CODES:
        return TusharePermissionError(detail)
    if icode in _AUTH_CODES or any(m in text for m in _AUTH_MARKERS):
        return TushareAuthError(detail)
    return TushareError(detail)


def _rpm_from_env() -> int:
    try:
        v = int(os.environ.get("TUSHARE_RPM", "") or _DEFAULT_RPM)
    except ValueError:
        return _DEFAULT_RPM
    return v if v > 0 else _DEFAULT_RPM


class RateLimiter:
    """按 api_name 的 60s 滑动窗口限频(进程内共享, 线程安全)。

    Tushare 限频按"接口 × 每分钟"计, 多个 client 实例(探测 / 同步)共享同一窗口,
    所以默认用模块级单例。clock/sleep 可注入供测试。
    """

    def __init__(
        self,
        rpm: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = rpm or _rpm_from_env()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def acquire(self, api_name: str) -> None:
        while True:
            with self._lock:
                now = self._clock()
                q = self._hits.setdefault(api_name, deque())
                while q and now - q[0] >= 60.0:
                    q.popleft()
                if len(q) < self.rpm:
                    q.append(now)
                    return
                wait = 60.0 - (now - q[0])
            self._sleep(max(wait, 0.01))


_SHARED_LIMITER = RateLimiter()


class TushareClient:
    """Tushare Pro HTTP 客户端 (线程安全: httpx.Client 可并发复用, 限频器带锁)。"""

    def __init__(
        self,
        token: str,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token:
            raise TushareError("未配置 TUSHARE_TOKEN")
        self._token = token
        self._base_url = base_url
        self._limiter = limiter or _SHARED_LIMITER
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _scrub(self, text: str) -> str:
        return str(text).replace(self._token, "***") if self._token else str(text)

    # ---- 内部 ----
    def _post(self, api_name: str, params: dict, fields: str) -> dict:
        """单次请求 + 信封解包。返回 data 容器 {fields, items, has_more?}。"""
        self._limiter.acquire(api_name)
        body = {"api_name": api_name, "token": self._token, "params": params, "fields": fields}
        try:
            resp = self._http.post(self._base_url, json=body)
        except httpx.HTTPError as e:
            raise TushareError(self._scrub(f"网络请求失败({api_name}): {e}")) from None
        if resp.status_code != 200:
            raise TushareError(f"HTTP {resp.status_code}: {api_name}")
        try:
            payload = resp.json()
        except ValueError:
            raise TushareError(f"响应不是 JSON: {api_name}") from None
        code = payload.get("code")
        if code not in (0, "0", None):
            err = classify_error(code, self._scrub(payload.get("msg") or ""))
            raise type(err)(f"{err} ({api_name})")
        data = payload.get("data")
        return data if isinstance(data, dict) else {}

    def _post_with_retry(self, api_name: str, params: dict, fields: str) -> dict:
        """限频错误退避重试(最多 3 次); 其余错误直接抛出。"""
        for attempt in range(len(_RATE_RETRY_BACKOFF_S) + 1):
            try:
                return self._post(api_name, params, fields)
            except TushareRateLimitError as e:
                if attempt >= len(_RATE_RETRY_BACKOFF_S):
                    raise
                wait = _RATE_RETRY_BACKOFF_S[attempt]
                logger.warning("Tushare %s 触发限频, %.0fs 后重试: %s", api_name, wait, e)
                self._sleep(wait)
        raise TushareRateLimitError(f"限频重试耗尽 ({api_name})")  # pragma: no cover

    @staticmethod
    def _frame(data: dict) -> pl.DataFrame:
        """{fields, items} → DataFrame。空 items 时保留列名(全 Null 列), 便于下游判空。"""
        fields = list(data.get("fields") or [])
        items = data.get("items") or []
        if not fields:
            return pl.DataFrame()
        if not items:
            return pl.DataFrame(schema=dict.fromkeys(fields, pl.Null))
        # strict=False: 同列 int/float 混排(如 vol 偶发整数)时向上转型而非报错
        return pl.DataFrame(items, schema=fields, orient="row", strict=False, infer_schema_length=None)

    # ---- 公共 ----
    def query(self, api_name: str, params: dict | None = None, fields: str = "") -> pl.DataFrame:
        """单次查询(不分页)。"""
        return self._frame(self._post_with_retry(api_name, dict(params or {}), fields))

    def query_all(
        self,
        api_name: str,
        params: dict | None = None,
        fields: str = "",
        page_size: int | None = None,
        max_pages: int = _MAX_PAGES,
    ) -> pl.DataFrame:
        """offset 分页拉全。终止条件: has_more=False / 空页 / 短页(页大小 ≤ 已知上限时可靠)。

        达到页数上限仍未结束 → 抛 TushareError(fail-closed, 不返回半截数据冒充全量)。
        """
        size = page_size or min(PAGE_LIMITS.get(api_name, _DEFAULT_PAGE_SIZE), _DEFAULT_PAGE_SIZE * 3)
        frames: list[pl.DataFrame] = []
        offset = 0
        for _ in range(max_pages):
            data = self._post_with_retry(
                api_name, {**(params or {}), "limit": size, "offset": offset}, fields
            )
            df = self._frame(data)
            if df.height:
                frames.append(df)
            has_more = data.get("has_more")
            if df.height == 0 or has_more is False or (has_more is None and df.height < size):
                break
            offset += df.height
        else:
            raise TushareError(f"分页超过上限 {max_pages} 页 ({api_name})")
        if not frames:
            return self._frame(data)
        return pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
