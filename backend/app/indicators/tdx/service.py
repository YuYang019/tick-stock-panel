"""指标序列计算服务 (API 层保持薄, 计算在此)。

数据口径与 /api/kline/daily 一致: 同一读取 (repo.get_daily_asset) + 同一实时
蜡烛合并 (api/kline.daily_rows_with_live, 由调用方以 merge_live 注入)。
起点向前扩 warmup 对应的自然日数先算, 再裁回请求区间 → dates 与日K rows 一一对齐。
"""
from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta

import polars as pl

from app.indicators.tdx.compiler import CompiledIndicator, IndicatorOutput, evaluate, stick_columns

MAX_SERIES_IDS = 10
_FRAME_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def warmup_calendar_days(warmup_bars: int) -> int:
    """交易日 → 自然日 (A 股约 1.48 自然日/交易日, 留长假余量)。"""
    return int(max(warmup_bars, 1) * 1.6) + 30


def _to_date_str(value) -> str:
    return str(value)[:10]


def rows_to_frame(symbol: str, rows: list[dict]) -> pl.DataFrame:
    """日K rows (含实时行, date 可能为 str) → 计算帧 (symbol, date[str], OHLCVA Float64)。"""
    schema = {"symbol": pl.Utf8, "date": pl.Utf8, **{c: pl.Float64 for c in _FRAME_COLUMNS}}
    if not rows:
        return pl.DataFrame(schema=schema)
    records = []
    for row in rows:
        record = {"symbol": symbol, "date": _to_date_str(row.get("date"))}
        for column in _FRAME_COLUMNS:
            raw = row.get(column)
            try:
                record[column] = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                record[column] = None
        records.append(record)
    return pl.DataFrame(records, schema=schema).unique(subset=["date"], keep="last").sort("date")


def load_frame(
    repo,
    symbol: str,
    start: date,
    end: date,
    warmup_bars: int,
    merge_live: Callable[[str, str, pl.DataFrame], list[dict]],
    live_fallback: Callable[[str, int], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    asset_type = repo.resolve_asset_type(symbol)
    extended_start = start - timedelta(days=warmup_calendar_days(warmup_bars))
    df = repo.get_daily_asset(asset_type, symbol, extended_start, end)
    if (df is None or df.is_empty()) and live_fallback is not None:
        # 本地无该标的: 与 /api/kline/daily 同口径实时拉取 (Free/未同步), 否则图上有 K 线却无指标线
        span_days = (end - extended_start).days
        try:
            df = live_fallback(symbol, int(span_days / 1.4) + 30)
        except Exception:  # noqa: BLE001 — 拉取失败按无数据处理, 与日K 502 分开
            df = pl.DataFrame()
        if not df.is_empty() and "date" in df.columns:
            df = df.filter(
                (pl.col("date").cast(pl.Utf8).str.slice(0, 10) >= extended_start.isoformat())
                & (pl.col("date").cast(pl.Utf8).str.slice(0, 10) <= end.isoformat())
            )
    rows = merge_live(symbol, asset_type, df if df is not None else pl.DataFrame())
    return rows_to_frame(symbol, rows)


def _clean(values: list) -> list:
    out = []
    for value in values:
        if value is None or isinstance(value, bool):
            out.append(value)
        elif isinstance(value, float) and not math.isfinite(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _output_payload(output: IndicatorOutput, frame: pl.DataFrame) -> dict:
    payload = {
        "key": output.key, "name": output.name, "kind": output.kind, "color": output.color,
        "thick": output.thick, "dashed": output.dashed, "visible": output.visible,
        "values": None, "stick": None,
    }
    if output.stick is not None:
        cond_col, p1_col, p2_col = stick_columns(output)
        cond = frame[cond_col].to_list() if cond_col in frame.columns else []
        payload["stick"] = {
            "p1": _clean(frame[p1_col].to_list()) if p1_col in frame.columns else [],
            "p2": _clean(frame[p2_col].to_list()) if p2_col in frame.columns else [],
            "cond": [bool(v) if v is not None and math.isfinite(v) else False for v in cond],
            "width": output.stick[3],
            "empty": output.stick[4],
        }
    else:
        payload["values"] = _clean(frame[output.column].to_list()) if output.column in frame.columns else []
    return payload


def compute_outputs(compiled: CompiledIndicator, frame: pl.DataFrame, start: date) -> list[dict]:
    """在含预热段的帧上求值, 裁回 date >= start; 返回各输出 payload (与 date_strings 对齐)。"""
    evaluated = evaluate(compiled, frame)
    cropped = evaluated.filter(pl.col("date") >= start.isoformat())
    return [_output_payload(output, cropped) for output in compiled.outputs]


def date_strings(frame: pl.DataFrame, start: date) -> list[str]:
    if frame.is_empty():
        return []
    return frame.filter(pl.col("date") >= start.isoformat())["date"].to_list()
