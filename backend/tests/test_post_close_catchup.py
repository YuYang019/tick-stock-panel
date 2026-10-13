"""盘后补拉与复盘日期门控。

复现: 上游当日日K未发布时, 管道请求 [昨天 ~ 今天] 只拿到昨天, 旧逻辑报
"覆盖 1 天", 复盘回退到最新有数据日 → 生成前一天报告。
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from app.jobs import daily_pipeline, post_close
from app.market_time import CN_TZ
from app.services import daily_readiness, preferences

TODAY = date(2026, 10, 13)
YDAY = date(2026, 10, 12)


# ── 磁盘夹具 ─────────────────────────────────────────────────────────

def _write(data_dir: Path, table: str, day: date, rows: int, quote_ts: int | None = None) -> None:
    part = data_dir / table / f"date={day.isoformat()}"
    part.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "symbol": [f"{i:06d}.SZ" for i in range(rows)],
        "volume": [100.0] * rows,
        "quote_ts": pl.Series([quote_ts] * rows, dtype=pl.Int64),
    })
    df.write_parquet(part / "part.parquet")


def _full_day(data_dir: Path, day: date, rows: int = 100, index_rows: int = 10) -> None:
    _write(data_dir, "kline_daily", day, rows)
    _write(data_dir, "kline_daily_enriched", day, rows)
    _write(data_dir, "kline_index_daily", day, index_rows)


def _ms(day: date, hh: int, mm: int) -> int:
    return int(datetime(day.year, day.month, day.day, hh, mm, tzinfo=CN_TZ).timestamp() * 1000)


# ── 就绪判定 ─────────────────────────────────────────────────────────

def test_only_yesterday_is_not_ready(tmp_path):
    _full_day(tmp_path, YDAY)
    r = daily_readiness.check_ready(tmp_path, TODAY, check_index=True)
    assert not r.ready
    assert r.daily_latest == YDAY.isoformat()


def test_full_today_is_ready(tmp_path):
    _full_day(tmp_path, YDAY)
    _full_day(tmp_path, TODAY)
    assert daily_readiness.check_ready(tmp_path, TODAY, check_index=True).ready


def test_partial_today_is_not_ready(tmp_path):
    _full_day(tmp_path, YDAY)
    _full_day(tmp_path, TODAY, rows=20)
    r = daily_readiness.check_ready(tmp_path, TODAY, check_index=True)
    assert not r.ready and any("仅 20 行" in x for x in r.reasons)


def test_intraday_snapshot_is_not_ready(tmp_path):
    _full_day(tmp_path, YDAY)
    _write(tmp_path, "kline_daily", TODAY, 100, quote_ts=_ms(TODAY, 11, 30))
    _write(tmp_path, "kline_daily_enriched", TODAY, 100)
    _write(tmp_path, "kline_index_daily", TODAY, 10)
    r = daily_readiness.check_ready(tmp_path, TODAY, check_index=True)
    assert not r.ready and any("快照" in x for x in r.reasons)


def test_close_quotes_after_15_count_as_today(tmp_path):
    _full_day(tmp_path, YDAY)
    _write(tmp_path, "kline_daily", TODAY, 100, quote_ts=_ms(TODAY, 15, 1))
    _write(tmp_path, "kline_daily_enriched", TODAY, 100)
    _write(tmp_path, "kline_index_daily", TODAY, 10)
    assert daily_readiness.check_ready(tmp_path, TODAY, check_index=True).ready


def test_enriched_lagging_is_not_ready(tmp_path):
    _full_day(tmp_path, YDAY)
    _write(tmp_path, "kline_daily", TODAY, 100)
    _write(tmp_path, "kline_index_daily", TODAY, 10)
    r = daily_readiness.check_ready(tmp_path, TODAY, check_index=True)
    assert not r.ready and any("指标缺少" in x for x in r.reasons)
    _write(tmp_path, "kline_daily_enriched", TODAY, 80)
    assert not daily_readiness.check_ready(tmp_path, TODAY, check_index=True).ready


def test_index_required_only_when_enabled(tmp_path):
    _full_day(tmp_path, YDAY)
    _write(tmp_path, "kline_daily", TODAY, 100)
    _write(tmp_path, "kline_daily_enriched", TODAY, 100)
    assert not daily_readiness.check_ready(tmp_path, TODAY, check_index=True).ready
    assert daily_readiness.check_ready(tmp_path, TODAY, check_index=False).ready


# ── 管道真实落盘报告 ─────────────────────────────────────────────────

class _StopError(Exception):
    pass


def test_run_now_reports_actual_dates_not_request_span(monkeypatch, tmp_path):
    """请求 [昨天 ~ 今天] 但上游只返回昨天: 新增 0 天, 并提示上游暂无今天。"""
    from app.services import data_integrity, instrument_sync, kline_sync

    _write(tmp_path, "kline_daily", YDAY, 10)
    monkeypatch.setattr(daily_pipeline, "cn_today", lambda: TODAY)
    monkeypatch.setattr(instrument_sync, "sync_instruments", lambda data_dir: 0)
    monkeypatch.setattr(daily_pipeline, "_resolve_universe", lambda capset, repo=None: ["000001.SZ"])
    monkeypatch.setattr(daily_pipeline, "_invalidate", lambda table=None: None)
    monkeypatch.setattr(data_integrity, "scan_recent_integrity", lambda *a, **k: [])
    monkeypatch.setattr(daily_pipeline._prefs, "get_pipeline_pull_a_share", lambda: True)
    monkeypatch.setattr(kline_sync, "sync_and_persist_daily_batch",
                        lambda *a, **k: _write(tmp_path, "kline_daily", YDAY, 10) or 10)
    msgs: list[str] = []

    def progress(stage, pct, msg, **_k):
        if stage == "sync_daily" and pct == 45:
            msgs.append(msg)
            raise _StopError()

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), latest_daily_date=lambda: YDAY)
    with pytest.raises(_StopError):
        daily_pipeline.run_now(repo, SimpleNamespace(has=lambda k: False), on_progress=progress)
    assert msgs and "新增 0 天" in msgs[0] and f"上游暂无 {TODAY}" in msgs[0]


# ── 调度编排 ─────────────────────────────────────────────────────────

class FakeScheduler:
    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def add_job(self, func, trigger=None, args=None, kwargs=None, id=None, **_k):
        self.jobs[id] = {"func": func, "trigger": trigger, "args": args or [], "kwargs": kwargs or {}}

    def remove_job(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        del self.jobs[job_id]

    def get_job(self, job_id):
        return self.jobs.get(job_id)


@pytest.fixture
def env(monkeypatch, tmp_path):
    prefs_path = tmp_path / "preferences.json"
    monkeypatch.setattr(preferences, "_path", lambda: prefs_path)
    preferences._invalidate_cache()
    preferences.set_review_schedule(True, 16, 0)

    now = {"t": datetime(2026, 10, 13, 15, 40, tzinfo=CN_TZ)}
    monkeypatch.setattr(post_close, "cn_now", lambda: now["t"])
    monkeypatch.setattr(daily_pipeline, "cn_today", lambda: now["t"].date())
    monkeypatch.setattr(post_close, "_is_trading_day", lambda d: True)
    pushed: list[dict] = []
    monkeypatch.setattr(post_close, "_push", lambda evt: pushed.append(evt))
    reports: list[dict] = []
    monkeypatch.setattr("app.services.market_recap_reports.list_reports", lambda: list(reports))
    monkeypatch.setattr("app.services.market_recap_reports.save_report",
                        lambda d: reports.append(d) or d)

    sched = FakeScheduler()
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    post_close.configure(sched, repo, lambda on_progress=None: {})
    monkeypatch.setattr(post_close, "_state", None)
    monkeypatch.setattr(post_close, "_review_running", False)
    _full_day(tmp_path, YDAY)
    yield SimpleNamespace(now=now, sched=sched, repo=repo, data=tmp_path,
                          pushed=pushed, reports=reports)
    post_close.configure(None, None, None)
    preferences._invalidate_cache()


def test_not_ready_schedules_retry_and_no_mining(env, monkeypatch):
    mining: list = []
    monkeypatch.setattr(daily_pipeline, "_run_tracked", lambda *a: True)
    monkeypatch.setattr("app.services.mining_schedule.run_weekly_mining",
                        lambda s: mining.append(s))
    daily_pipeline._scheduled_pipeline_task(lambda: None)
    job = env.sched.jobs[post_close.RETRY_JOB_ID]
    assert job["trigger"].run_date.strftime("%H:%M") == "16:10"
    assert mining == []


def test_ready_triggers_mining_and_clears_retry(env, monkeypatch):
    mining: list = []
    _full_day(env.data, TODAY)
    env.sched.jobs[post_close.RETRY_JOB_ID] = {}
    monkeypatch.setattr(daily_pipeline, "_run_tracked", lambda *a: True)
    monkeypatch.setattr("app.services.mining_schedule.run_weekly_mining",
                        lambda s: mining.append(s) or {})
    daily_pipeline._scheduled_pipeline_task(lambda: None)
    assert post_close.RETRY_JOB_ID not in env.sched.jobs
    assert len(mining) == 1


def test_retry_capped_at_deadline(env):
    env.now["t"] = datetime(2026, 10, 13, 21, 50, tzinfo=CN_TZ)
    post_close.schedule_retry(TODAY)
    assert env.sched.jobs[post_close.RETRY_JOB_ID]["trigger"].run_date.strftime("%H:%M") == "22:00"


def test_retry_probe_absent_skips_full_pipeline(env, monkeypatch):
    calls: list = []
    monkeypatch.setattr(daily_readiness, "probe_upstream_daily", lambda capset, d: False)
    monkeypatch.setattr(daily_pipeline, "_scheduled_pipeline_task", lambda *a, **k: calls.append(a))
    post_close._retry_tick(TODAY)
    assert calls == []
    assert post_close.RETRY_JOB_ID in env.sched.jobs


def test_retry_probe_present_runs_pipeline(env, monkeypatch):
    calls: list = []
    monkeypatch.setattr(daily_readiness, "probe_upstream_daily", lambda capset, d: True)
    monkeypatch.setattr(daily_pipeline, "_scheduled_pipeline_task",
                        lambda fn, source="scheduled": calls.append(source))
    post_close._retry_tick(TODAY)
    assert calls == ["retry"]


def test_retry_stops_across_day(env, monkeypatch):
    env.now["t"] = datetime(2026, 10, 14, 9, 0, tzinfo=CN_TZ)
    monkeypatch.setattr(daily_readiness, "probe_upstream_daily",
                        lambda *a: pytest.fail("跨日不应再探测"))
    post_close._retry_tick(TODAY)
    assert env.sched.jobs == {}


def test_retry_stops_on_holiday(env, monkeypatch):
    monkeypatch.setattr(post_close, "_is_trading_day", lambda d: False)
    monkeypatch.setattr(daily_readiness, "probe_upstream_daily",
                        lambda *a: pytest.fail("休市不应探测"))
    post_close._retry_tick(TODAY)


def test_deadline_final_failure_notifies_waiting_review(env, monkeypatch):
    post_close.mark_review_waiting(TODAY, ["日K 缺少"])
    env.now["t"] = datetime(2026, 10, 13, 22, 0, tzinfo=CN_TZ)
    monkeypatch.setattr(daily_readiness, "probe_upstream_daily", lambda capset, d: False)
    post_close._retry_tick(TODAY)
    errors = [e for e in env.pushed if e.get("level") == "error"]
    assert len(errors) == 1 and "今日复盘未生成" in errors[0]["message"]
    # 截止后不再安排补拉, 重复截止也只提示一次
    assert post_close.RETRY_JOB_ID not in env.sched.jobs
    post_close._final_fail(TODAY)
    assert len([e for e in env.pushed if e.get("level") == "error"]) == 1


def test_unknown_trading_day_reports_possible_holiday(env, monkeypatch):
    monkeypatch.setattr(post_close, "_is_trading_day", lambda d: None)
    post_close.mark_review_waiting(TODAY, ["日K 缺少"])
    post_close._final_fail(TODAY)
    assert "可能休市" in env.pushed[-1]["message"]


def test_pull_disabled_does_not_retry(env, monkeypatch):
    monkeypatch.setattr(preferences, "get_pipeline_pull_a_share", lambda: False)
    post_close.schedule_retry(TODAY)
    assert post_close.RETRY_JOB_ID not in env.sched.jobs
    assert post_close.DEADLINE_JOB_ID in env.sched.jobs
    env.now["t"] = datetime(2026, 10, 13, 22, 0, tzinfo=CN_TZ)
    post_close.mark_review_waiting(TODAY, ["日K 缺少"])
    assert "拉取已关闭" in env.pushed[-1]["message"]


# ── 复盘门控 ─────────────────────────────────────────────────────────

def _patch_review(monkeypatch, env, as_of: str | None = None):
    streamed: list = []

    async def fake_stream(repo, qs, ds, as_of=None):
        streamed.append(as_of)
        return "正文", {"as_of": (as_of_override or as_of.isoformat()), "emotion_label": ""}

    as_of_override = as_of
    monkeypatch.setattr("app.secrets_store.get_ai_key", lambda: "sk-test")
    monkeypatch.setattr(daily_pipeline, "_stream_review_with_retry", fake_stream)
    monkeypatch.setattr(daily_pipeline, "_maybe_push_review", lambda c, m: None)
    return streamed


def test_review_waits_when_not_ready_then_generates_once(env, monkeypatch):
    streamed = _patch_review(monkeypatch, env)
    env.now["t"] = datetime(2026, 10, 13, 16, 0, tzinfo=CN_TZ)
    asyncio.run(daily_pipeline._run_scheduled_review(env.repo))
    assert streamed == [] and env.reports == []
    assert any(e.get("level") == "info" for e in env.pushed)
    assert post_close.RETRY_JOB_ID in env.sched.jobs

    # 补拉就绪 → 触发补生成, 显式传今天
    env.now["t"] = datetime(2026, 10, 13, 17, 30, tzinfo=CN_TZ)
    _full_day(env.data, TODAY)
    assert post_close.after_pipeline("retry") is True
    job = env.sched.jobs[post_close.REVIEW_CATCHUP_JOB_ID]
    asyncio.run(job["func"](*job["args"], **job["kwargs"]))
    assert streamed == [TODAY]
    assert [r["as_of"] for r in env.reports] == [TODAY.isoformat()]

    # 重复触发 (cron 补跑 / 手动同步) 不再归档
    asyncio.run(daily_pipeline._run_scheduled_review(env.repo))
    post_close.maybe_trigger_review(TODAY)
    assert len(env.reports) == 1


def test_review_discards_stale_as_of(env, monkeypatch):
    _patch_review(monkeypatch, env, as_of=YDAY.isoformat())
    _full_day(env.data, TODAY)
    env.now["t"] = datetime(2026, 10, 13, 16, 0, tzinfo=CN_TZ)
    asyncio.run(daily_pipeline._run_scheduled_review(env.repo))
    assert env.reports == []


def test_manual_report_does_not_block_auto_review(env, monkeypatch):
    streamed = _patch_review(monkeypatch, env)
    _full_day(env.data, TODAY)
    env.reports.append({"as_of": TODAY.isoformat(), "content": "手动"})
    env.now["t"] = datetime(2026, 10, 13, 16, 0, tzinfo=CN_TZ)
    asyncio.run(daily_pipeline._run_scheduled_review(env.repo))
    assert streamed == [TODAY]


def test_ready_before_review_time_does_not_generate_early(env):
    _full_day(env.data, TODAY)
    env.now["t"] = datetime(2026, 10, 13, 15, 40, tzinfo=CN_TZ)
    post_close.after_pipeline("scheduled")
    assert post_close.REVIEW_CATCHUP_JOB_ID not in env.sched.jobs


def test_manual_sync_completes_waiting_review(env, monkeypatch):
    _patch_review(monkeypatch, env)
    env.now["t"] = datetime(2026, 10, 13, 16, 0, tzinfo=CN_TZ)
    asyncio.run(daily_pipeline._run_scheduled_review(env.repo))
    _full_day(env.data, TODAY)
    from app.api import pipeline as pipeline_api
    pipeline_api._after_manual_pipeline()
    assert post_close.REVIEW_CATCHUP_JOB_ID in env.sched.jobs
    assert post_close.RETRY_JOB_ID not in env.sched.jobs


def test_disabled_review_cancels_waiting(env, monkeypatch):
    _patch_review(monkeypatch, env)
    env.now["t"] = datetime(2026, 10, 13, 16, 0, tzinfo=CN_TZ)
    asyncio.run(daily_pipeline._run_scheduled_review(env.repo))
    preferences.set_review_schedule(False, 16, 0)
    post_close.cancel_review_waiting()
    _full_day(env.data, TODAY)
    post_close.after_pipeline("retry")
    assert post_close.REVIEW_CATCHUP_JOB_ID not in env.sched.jobs


def test_recover_on_start_resumes_waiting(env):
    env.now["t"] = datetime(2026, 10, 13, 18, 0, tzinfo=CN_TZ)
    post_close.recover_on_start()
    run_at = env.sched.jobs[post_close.RETRY_JOB_ID]["trigger"].run_date
    assert run_at.strftime("%H:%M") == "18:01"
    assert post_close._state.review_waiting


def test_recover_on_start_generates_owed_review(env):
    _full_day(env.data, TODAY)
    env.now["t"] = datetime(2026, 10, 13, 18, 0, tzinfo=CN_TZ)
    post_close.recover_on_start()
    assert post_close.REVIEW_CATCHUP_JOB_ID in env.sched.jobs


def test_recover_outside_window_noop(env):
    env.now["t"] = datetime(2026, 10, 13, 23, 0, tzinfo=CN_TZ)
    post_close.recover_on_start()
    assert env.sched.jobs == {}
