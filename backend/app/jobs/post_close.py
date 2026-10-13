"""盘后数据补拉与复盘门控。

问题: 部分数据源当日日K在收盘后数小时才发布。盘后管道 (默认 15:35) 拿不到
当天数据时, 复盘按"最新有数据日"回退, 生成了前一天的报告。

本模块在不改动用户同步/复盘时间的前提下:
- 盘后管道跑完后用 daily_readiness 判定当日是否就绪; 未就绪则每
  RETRY_INTERVAL_MIN 分钟补拉一次, 最晚 DEADLINE (北京时间 22:00);
- 补拉前先用少量样本探测上游, 确认已发布才跑完整管道 (避免重复重活);
- 复盘到点时数据未齐 → 进入等待; 之后任一管道 (定时/补拉/手动) 就绪即生成;
- 截止仍缺数据 → 记录失败, 经复盘 SSE 通道提示; 跨日自动停止旧日重试。

状态只放内存: 重启后由 recover_on_start 依据报告归档与磁盘数据重新推导。
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dt_time

from app.market_time import CN_TZ, cn_now

logger = logging.getLogger(__name__)

RETRY_JOB_ID = "post_close_retry"
DEADLINE_JOB_ID = "post_close_deadline"
REVIEW_CATCHUP_JOB_ID = "scheduled_review_catchup"
RECOVER_JOB_ID = "post_close_recover"
RETRY_INTERVAL_MIN = 30
DEADLINE = dt_time(22, 0)
REPORT_SOURCE = "scheduled"


@dataclass
class _DayState:
    day: date
    ready: bool = False
    review_waiting: bool = False
    final_failed: bool = False
    attempts: int = 0
    last_reasons: tuple[str, ...] = ()


_lock = threading.Lock()
_state: _DayState | None = None
_review_running = False
_scheduler = None
_repo = None
_pipeline_fn = None


def configure(scheduler, repo, pipeline_fn) -> None:
    """start_scheduler 注入调度器、仓库与盘后管道闭包 (补拉复用同一闭包)。"""
    global _scheduler, _repo, _pipeline_fn
    _scheduler, _repo, _pipeline_fn = scheduler, repo, pipeline_fn


def _today() -> date:
    return cn_now().date()


def _deadline_dt(day: date) -> datetime:
    return datetime.combine(day, DEADLINE, tzinfo=CN_TZ)


def _day_state(day: date) -> _DayState:
    """当日状态; 跨日自动换新 (旧日等待/重试一律作废)。调用方持锁。"""
    global _state
    if _state is None or _state.day != day:
        _state = _DayState(day=day)
    return _state


def _push(evt: dict) -> None:
    try:
        from app.jobs.daily_pipeline import _get_app_state
        app_state = _get_app_state()
        qs = getattr(app_state, "quote_service", None) if app_state else None
        if qs:
            qs.push_review_event(json.dumps(evt, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass


def _push_status(level: str, message: str) -> None:
    """复盘 SSE 通道上的状态提示 (前端 toast), 不改变复盘生成状态。"""
    _push({"type": "status", "level": level, "message": message})


def _remove_job(job_id: str) -> None:
    if _scheduler is None:
        return
    with contextlib.suppress(Exception):
        _scheduler.remove_job(job_id)


def _data_dir():
    return _repo.store.data_dir if _repo is not None else None


def check_ready(day: date):
    from app.services.daily_readiness import check_ready as _check
    return _check(_data_dir(), day)


def _is_trading_day(day: date) -> bool | None:
    """交易日判定: 周末 False; 工作日走探针, 未知按 None (照工作日尝试)。"""
    if day.weekday() >= 5:
        return False
    try:
        from app.services.trading_day import is_trading_day
        now = cn_now()
        if now.date() != day:
            return None
        return is_trading_day(now)
    except Exception:  # noqa: BLE001
        return None


def auto_review_archived(day: date) -> bool:
    """当日定时复盘是否已归档 (只认定时来源; 用户手动报告不挡自动复盘)。"""
    try:
        from app.services import market_recap_reports
        iso = day.isoformat()
        return any(
            r.get("as_of") == iso and r.get("source") == REPORT_SOURCE
            for r in market_recap_reports.list_reports()
        )
    except Exception:  # noqa: BLE001
        return False


def _review_due(now: datetime) -> bool:
    from app.services import preferences
    sched = preferences.get_review_schedule()
    if not sched["enabled"]:
        return False
    due = now.replace(hour=sched["hour"], minute=sched["minute"], second=0, microsecond=0)
    return now >= due


# ----------------------------------------------------------------
# 管道完成钩子 (定时 / 补拉 / 手动 共用)
# ----------------------------------------------------------------

def after_pipeline(source: str) -> bool:
    """任一盘后管道结束后调用: 判定当日就绪, 驱动补拉或触发等待中的复盘。

    返回当日是否就绪。source: scheduled | retry | manual。
    """
    day = _today()
    if _data_dir() is None:
        return False
    try:
        readiness = check_ready(day)
    except Exception as e:  # noqa: BLE001
        logger.warning("post_close readiness check failed: %s", e)
        return False

    with _lock:
        st = _day_state(day)
        st.last_reasons = tuple(readiness.reasons)
        st.ready = readiness.ready
        if source in ("scheduled", "retry"):
            st.attempts += 1

    if readiness.ready:
        logger.info("post_close: %s 数据就绪 (source=%s)", day, source)
        _remove_job(RETRY_JOB_ID)
        _remove_job(DEADLINE_JOB_ID)
        maybe_trigger_review(day)
        return True

    logger.warning("post_close: %s 数据未就绪 (source=%s): %s",
                   day, source, "; ".join(readiness.reasons))
    if source in ("scheduled", "retry"):
        schedule_retry(day)
    return False


def schedule_retry(day: date, delay_min: int = RETRY_INTERVAL_MIN) -> None:
    """安排下一次补拉 (≤ delay_min 后, 不晚于截止)。"""
    from app.services import preferences

    if _scheduler is None or day != _today():
        return
    with _lock:
        st = _day_state(day)
        if st.ready or st.final_failed:
            return

    now = cn_now()
    deadline = _deadline_dt(day)
    if now >= deadline:
        _final_fail(day)
        return

    if not preferences.get_pipeline_pull_a_share():
        # 关闭 A 股日K拉取: 不擅自补拉, 只在截止时收尾等待中的复盘
        logger.info("post_close: A 股日K拉取已关闭, 不自动补拉 %s", day)
        _ensure_deadline_job(day)
        return

    run_at = min(now + timedelta(minutes=delay_min), deadline)
    from apscheduler.triggers.date import DateTrigger
    _scheduler.add_job(
        _retry_tick,
        args=[day],
        trigger=DateTrigger(run_date=run_at, timezone="Asia/Shanghai"),
        id=RETRY_JOB_ID,
        misfire_grace_time=600,
        replace_existing=True,
    )
    logger.info("post_close: 已安排 %s 补拉 @ %s", day, run_at.strftime("%H:%M"))


def _ensure_deadline_job(day: date) -> None:
    if _scheduler is None:
        return
    from apscheduler.triggers.date import DateTrigger
    _scheduler.add_job(
        _deadline_tick,
        args=[day],
        trigger=DateTrigger(run_date=_deadline_dt(day), timezone="Asia/Shanghai"),
        id=DEADLINE_JOB_ID,
        misfire_grace_time=600,
        replace_existing=True,
    )


def _retry_tick(day: date) -> None:
    """补拉一次: 已就绪→收尾; 先探测上游, 确认已发布才跑完整管道。"""
    if day != _today():
        logger.info("post_close: 跨日, 停止 %s 的补拉", day)
        return
    if _is_trading_day(day) is False:
        logger.info("post_close: %s 休市, 停止补拉", day)
        return
    try:
        if check_ready(day).ready:
            after_pipeline("manual")  # 期间已被手动同步补齐, 只收尾不计次
            return
    except Exception:  # noqa: BLE001
        pass

    from app.jobs import daily_pipeline
    from app.services.daily_readiness import probe_upstream_daily

    app_state = daily_pipeline._get_app_state()
    capset = getattr(app_state, "capabilities", None) if app_state else None
    probe = probe_upstream_daily(capset, day)
    final_attempt = cn_now() >= _deadline_dt(day) - timedelta(minutes=1)
    if probe is False:
        logger.info("post_close: 上游暂无 %s 日K%s", day, ", 已到截止" if final_attempt else "")
        with _lock:
            _day_state(day).attempts += 1
        if final_attempt:
            _final_fail(day)
        else:
            schedule_retry(day)
        return

    if _pipeline_fn is None:
        return
    daily_pipeline._scheduled_pipeline_task(_pipeline_fn, source="retry")


def _deadline_tick(day: date) -> None:
    if day != _today():
        return
    try:
        if check_ready(day).ready:
            after_pipeline("manual")
            return
    except Exception:  # noqa: BLE001
        pass
    _final_fail(day)


def _final_fail(day: date) -> None:
    from app.services import preferences

    with _lock:
        st = _day_state(day)
        if st.final_failed or st.ready:
            return
        st.final_failed = True
        reasons = "; ".join(st.last_reasons) or "上游未返回当日数据"
        waiting = st.review_waiting
    _remove_job(RETRY_JOB_ID)
    _remove_job(DEADLINE_JOB_ID)

    trading = _is_trading_day(day)
    if trading is None:
        msg = f"{day} 截至 {DEADLINE:%H:%M} 仍无当日数据(交易日未知, 可能休市): {reasons}"
    elif not preferences.get_pipeline_pull_a_share():
        msg = f"{day} 当日数据未更新: A 股日K拉取已关闭, 未自动补拉"
    else:
        msg = f"{day} 截至 {DEADLINE:%H:%M} 当日数据仍未就绪, 停止补拉: {reasons}"
    logger.warning("post_close: %s", msg)
    if waiting and not auto_review_archived(day):
        _push({"type": "status", "level": "error", "message": f"今日复盘未生成 — {msg}"})


# ----------------------------------------------------------------
# 复盘门控
# ----------------------------------------------------------------

def mark_review_waiting(day: date, reasons: list[str]) -> None:
    """复盘到点但数据未齐: 进入等待, 并确保有补拉/截止任务能收尾。"""
    from app.services import preferences

    with _lock:
        st = _day_state(day)
        first = not st.review_waiting
        st.review_waiting = True
        st.last_reasons = tuple(reasons)
        failed = st.final_failed
    if failed:
        return
    if first:
        hint = "完成补拉后自动生成" if preferences.get_pipeline_pull_a_share() else \
            "A 股日K拉取已关闭, 请手动同步"
        _push_status("info", f"{day} 当日数据尚未就绪, 复盘等待中({hint})")
    if _scheduler is not None and _scheduler.get_job(RETRY_JOB_ID) is None:
        if cn_now() >= _deadline_dt(day):
            _final_fail(day)
        elif preferences.get_pipeline_pull_a_share():
            schedule_retry(day)
        else:
            _ensure_deadline_job(day)


def cancel_review_waiting() -> None:
    """关闭定时复盘时调用: 取消等待中的补生成 (补拉数据不受影响)。"""
    with _lock:
        if _state is not None:
            _state.review_waiting = False
    _remove_job(REVIEW_CATCHUP_JOB_ID)


def maybe_trigger_review(day: date) -> None:
    """数据就绪后: 若复盘已到点、未归档, 立即补生成一次。"""
    if _scheduler is None or _repo is None or day != _today():
        return
    if not _review_due(cn_now()) or auto_review_archived(day):
        return
    from apscheduler.triggers.date import DateTrigger

    from app.jobs.daily_pipeline import _run_scheduled_review
    _scheduler.add_job(
        _run_scheduled_review,
        args=[_repo],
        kwargs={"target": day},
        trigger=DateTrigger(run_date=cn_now(), timezone="Asia/Shanghai"),
        id=REVIEW_CATCHUP_JOB_ID,
        misfire_grace_time=3600,
        replace_existing=True,
    )
    logger.info("post_close: %s 数据就绪, 触发复盘补生成", day)


def try_begin_review() -> bool:
    global _review_running
    with _lock:
        if _review_running:
            return False
        _review_running = True
        return True


def end_review(day: date, archived: bool) -> None:
    global _review_running
    with _lock:
        _review_running = False
        if archived:
            _day_state(day).review_waiting = False


# ----------------------------------------------------------------
# 重启恢复
# ----------------------------------------------------------------

def schedule_recover(delay_s: int = 60) -> None:
    if _scheduler is None:
        return
    from apscheduler.triggers.date import DateTrigger
    _scheduler.add_job(
        recover_on_start,
        trigger=DateTrigger(run_date=cn_now() + timedelta(seconds=delay_s),
                            timezone="Asia/Shanghai"),
        id=RECOVER_JOB_ID,
        misfire_grace_time=600,
        replace_existing=True,
    )


def recover_on_start() -> None:
    """重启后: 盘后窗口内且当日未就绪 → 恢复补拉/等待; 已就绪且复盘欠账 → 补生成。"""
    from app.services import preferences

    now = cn_now()
    day = now.date()
    sched = preferences.get_pipeline_schedule()
    start = now.replace(hour=sched["hour"], minute=sched["minute"], second=0, microsecond=0)
    if now < start or now >= _deadline_dt(day) or _data_dir() is None:
        return
    if _is_trading_day(day) is False:
        return
    readiness = check_ready(day)
    if readiness.ready:
        with _lock:
            _day_state(day).ready = True
        maybe_trigger_review(day)
        return
    logger.info("post_close: 重启恢复, %s 数据未就绪: %s", day, "; ".join(readiness.reasons))
    with _lock:
        _day_state(day).last_reasons = tuple(readiness.reasons)
    # 重启后尽快补拉一次, 不等满一个间隔
    schedule_retry(day, delay_min=1)
    if _review_due(now) and not auto_review_archived(day):
        mark_review_waiting(day, readiness.reasons)
