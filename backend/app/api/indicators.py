"""通达信自定义指标 API — CRUD / 校验 / 函数清单 / 序列 / 预览。

只做胶水: 参数校验 → store/service → 注册表与缓存失效。计算在 indicators/tdx/service.py。
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.indicators.tdx import catalog, service, store
from app.indicators.tdx import factors as indicator_factors
from app.indicators.tdx.compiler import MAX_FORMULA_LENGTH, CompiledIndicator, compile_indicator
from app.market_time import cn_today

router = APIRouter(prefix="/api/indicators", tags=["indicators"])

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
_DEFAULT_SPAN_DAYS = 365


def _data_dir(request: Request) -> Path:
    root = getattr(getattr(getattr(request.app.state, "repo", None), "store", None), "data_dir", None)
    if root is None:
        raise HTTPException(status_code=500, detail="数据目录不可用")
    return Path(root)


def _compiled_payload(compiled: CompiledIndicator) -> dict:
    return {
        "ok": compiled.ok,
        "errors": [error.to_dict() for error in compiled.errors],
        "warnings": [warning.to_dict() for warning in compiled.warnings],
        "outputs": [_output_meta(output) for output in compiled.outputs],
        "warmup_bars": compiled.warmup_bars if compiled.ok else 0,
    }


def _output_meta(output) -> dict:
    return {
        "name": output.name, "key": output.key, "kind": output.kind, "color": output.color,
        "thick": output.thick, "dashed": output.dashed, "visible": output.visible,
    }


def _list_item(definition: dict) -> dict:
    compiled = compile_indicator(str(definition.get("formula", "")))
    return {
        "id": definition.get("id"),
        "name": definition.get("name"),
        "pane": definition.get("pane", "main"),
        "builtin": bool(definition.get("builtin", False)),
        "description": definition.get("description", ""),
        "formula": definition.get("formula", ""),
        "outputs": [_output_meta(output) for output in compiled.outputs],
        "warnings": [warning.to_dict() for warning in compiled.warnings],
        "errors": [error.to_dict() for error in compiled.errors],
        "created_at": definition.get("created_at"),
        "updated_at": definition.get("updated_at"),
    }


def _invalidate(request: Request) -> None:
    """指标增删改会改变 ind_* 因子集合与取值: 失效信号表达式与计算缓存 (同信号保存路径)。"""
    from app.indicators.pipeline import invalidate_custom_signals
    from app.strategy import custom_signals

    custom_signals.invalidate_intraday_cache()
    invalidate_custom_signals()
    try:
        from app.services import strategy_cache

        strategy_cache.clear_cache(_data_dir(request))
    except Exception:  # noqa: BLE001 — 缓存清理失败不阻断保存
        pass
    repo = request.app.state.repo
    if hasattr(repo, "clear_cache"):
        repo.clear_cache()


# ── 列表 / 函数清单 / 校验 ──────────────────────────────────


@router.get("")
def list_indicators(request: Request) -> list[dict]:
    return [_list_item(definition) for definition in store.load_all(_data_dir(request))]


@router.get("/functions")
def list_functions() -> dict:
    return {
        "functions": catalog.FUNCTION_DOCS,
        "variables": catalog.VARIABLE_DOCS,
        "modifiers": catalog.MODIFIER_DOCS,
    }


class ValidateRequest(BaseModel):
    formula: str = Field(..., max_length=MAX_FORMULA_LENGTH)


@router.post("/validate")
def validate_formula(req: ValidateRequest) -> dict:
    return _compiled_payload(compile_indicator(req.formula))


# ── CRUD ──────────────────────────────────────────────────


class IndicatorCreateRequest(BaseModel):
    id: str | None = Field(default=None, max_length=40)
    name: str = Field(..., min_length=1, max_length=32)
    formula: str = Field(..., min_length=1, max_length=MAX_FORMULA_LENGTH)
    pane: str = Field(default="main", pattern="^(main|sub)$")
    description: str = Field(default="", max_length=500)


class IndicatorUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=32)
    formula: str = Field(..., min_length=1, max_length=MAX_FORMULA_LENGTH)
    pane: str = Field(default="main", pattern="^(main|sub)$")
    description: str = Field(default="", max_length=500)


def _slug(name: str) -> str:
    base = re.sub(r"_+", "_", "".join(ch if ch.isascii() and ch.isalnum() else "_" for ch in name.lower())).strip("_")
    return (base or "ind")[:32]


def _unique_id(data_dir: Path, base: str) -> str:
    candidate, suffix = base, 2
    while store.exists(data_dir, candidate):
        candidate, suffix = f"{base[:36]}_{suffix}", suffix + 1
    return candidate


def _compile_or_400(formula: str) -> CompiledIndicator:
    compiled = compile_indicator(formula)
    if not compiled.ok:
        raise HTTPException(status_code=400, detail={"message": "公式无效", **_compiled_payload(compiled)})
    return compiled


def _save_and_register(request: Request, definition: dict) -> list[str]:
    data_dir = _data_dir(request)
    try:
        store.validate(definition)
        factor_ids = indicator_factors.register_indicator(definition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save_one(data_dir, definition)
    _invalidate(request)
    return factor_ids


@router.post("")
def create_indicator(req: IndicatorCreateRequest, request: Request) -> dict:
    data_dir = _data_dir(request)
    _compile_or_400(req.formula)
    if req.id and req.id.strip():
        indicator_id = req.id.strip()
        if not store.ID_RE.match(indicator_id):
            raise HTTPException(status_code=400, detail="id 仅允许小写字母数字下划线, 1-40 字符")
        if store.exists(data_dir, indicator_id):
            raise HTTPException(status_code=409, detail=f"指标 id 已存在: {indicator_id}")
    else:
        indicator_id = _unique_id(data_dir, _slug(req.name))
    stamp = store.now()
    definition = {
        "id": indicator_id, "name": req.name.strip(), "formula": req.formula, "pane": req.pane,
        "description": req.description, "builtin": False, "created_at": stamp, "updated_at": stamp,
    }
    factor_ids = _save_and_register(request, definition)
    return {"ok": True, "id": indicator_id, "factor_ids": factor_ids, "indicator": _list_item(definition)}


@router.put("/{indicator_id}")
def update_indicator(indicator_id: str, req: IndicatorUpdateRequest, request: Request) -> dict:
    data_dir = _data_dir(request)
    target = store.load_one(data_dir, indicator_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"指标不存在: {indicator_id}")
    if target.get("builtin"):
        raise HTTPException(status_code=400, detail="预设指标不可修改, 请另存为新指标")
    _compile_or_400(req.formula)
    removed = sorted(indicator_factors.factor_ids_for(target) - indicator_factors.factor_ids_for(
        {**target, "formula": req.formula, "name": req.name}
    ))
    references = _find_references(data_dir, indicator_id, only=set(removed)) if removed else []
    if references:
        raise HTTPException(status_code=409, detail={
            "message": "修改会移除仍被引用的输出线, 拒绝保存", "removed_factors": removed, "references": references,
        })
    target.update({
        "name": req.name.strip(), "formula": req.formula, "pane": req.pane,
        "description": req.description, "updated_at": store.now(),
    })
    factor_ids = _save_and_register(request, target)
    return {"ok": True, "id": indicator_id, "factor_ids": factor_ids, "indicator": _list_item(target)}


def _find_references(data_dir: Path, indicator_id: str, only: set[str] | None = None) -> list[str]:
    """扫描自定义信号 / 策略 / 自定义因子中对 ind_{id}_* 的引用 (删除前 fail-closed)。"""
    prefix = f"{indicator_factors.FACTOR_PREFIX}{indicator_id}_"
    pattern = re.compile(re.escape(prefix) + r"[a-z0-9_]+")

    def hit(text: str) -> bool:
        found = set(pattern.findall(text))
        if only is None:
            return bool(found)
        # 键可能是另一键的前缀 (o_x 与 o_x_2): 按完整 token 比较
        return bool(found & only)

    references: list[str] = []
    for subdir, label in (
        (data_dir / "user_data" / "custom_signals", "custom_signals"),
        (data_dir / "user_data" / "custom_factors", "custom_factors"),
        (data_dir / "strategies", "strategies"),
    ):
        if not subdir.is_dir():
            continue
        for file in sorted(subdir.rglob("*")):
            if not file.is_file() or file.suffix not in {".json", ".py"}:
                continue
            try:
                if hit(file.read_text(encoding="utf-8")):
                    references.append(f"{label}/{file.relative_to(subdir).as_posix()}")
            except (OSError, UnicodeDecodeError):
                continue
    return references


@router.delete("/{indicator_id}")
def delete_indicator(indicator_id: str, request: Request) -> dict:
    data_dir = _data_dir(request)
    target = store.load_one(data_dir, indicator_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"指标不存在: {indicator_id}")
    if target.get("builtin"):
        raise HTTPException(status_code=400, detail="预设指标不可删除")
    references = _find_references(data_dir, indicator_id)
    if references:
        raise HTTPException(status_code=409, detail={"message": "该指标仍被引用, 拒绝删除", "references": references})
    store.delete_one(data_dir, indicator_id)
    removed = indicator_factors.unregister_indicator(indicator_id)
    _invalidate(request)
    return {"ok": True, "id": indicator_id, "removed_factors": removed}


# ── 序列 / 预览 ───────────────────────────────────────────


def _parse_range(start_date: str | None, end_date: str | None) -> tuple[date, date]:
    try:
        end = date.fromisoformat(end_date) if end_date else cn_today()
        start = date.fromisoformat(start_date) if start_date else end - timedelta(days=_DEFAULT_SPAN_DAYS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="日期格式应为 YYYY-MM-DD") from exc
    if start > end:
        raise HTTPException(status_code=400, detail="start_date 不能晚于 end_date")
    return start, end


def _check_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=400, detail="symbol 非法")
    return symbol


def _merge_live(request: Request):
    from app.api.kline import daily_rows_with_live

    return lambda symbol, asset_type, df: daily_rows_with_live(request, symbol, asset_type, df)


def _live_fallback(request: Request):
    from app.api.kline import live_enriched_frame

    return lambda symbol, count: live_enriched_frame(request, symbol, count)


def _series_for(request: Request, compiled_by_id: dict[str, CompiledIndicator], symbol: str, start: date, end: date):
    repo = request.app.state.repo
    warmup = max((c.warmup_bars for c in compiled_by_id.values()), default=1)
    frame = service.load_frame(
        repo, symbol, start, end, warmup, _merge_live(request), _live_fallback(request),
    )
    dates: list[str] = service.date_strings(frame, start)
    indicators: dict[str, dict] = {}
    for indicator_id, compiled in compiled_by_id.items():
        indicators[indicator_id] = {"outputs": service.compute_outputs(compiled, frame, start)}
    return dates, indicators


@router.get("/series")
def get_series(
    request: Request,
    symbol: str = Query(..., max_length=20),
    ids: str = Query(..., max_length=500, description="逗号分隔的指标 id, 最多 10 个"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
) -> dict:
    symbol = _check_symbol(symbol)
    id_list = list(dict.fromkeys(part.strip() for part in ids.split(",") if part.strip()))
    if not id_list:
        raise HTTPException(status_code=400, detail="ids 不能为空")
    if len(id_list) > service.MAX_SERIES_IDS:
        raise HTTPException(status_code=400, detail=f"单次最多 {service.MAX_SERIES_IDS} 个指标")
    start, end = _parse_range(start_date, end_date)
    data_dir = _data_dir(request)
    errors: dict[str, str] = {}
    compiled_by_id: dict[str, CompiledIndicator] = {}
    for indicator_id in id_list:
        definition = store.load_one(data_dir, indicator_id) if store.ID_RE.match(indicator_id) else None
        if definition is None:
            errors[indicator_id] = "指标不存在"
            continue
        compiled = compile_indicator(str(definition.get("formula", "")))
        if not compiled.ok:
            errors[indicator_id] = compiled.errors[0].message
            continue
        compiled_by_id[indicator_id] = compiled
    dates, indicators = _series_for(request, compiled_by_id, symbol, start, end) if compiled_by_id else ([], {})
    return {"symbol": symbol, "dates": dates, "indicators": indicators, "errors": errors}


class PreviewRequest(BaseModel):
    formula: str = Field(..., min_length=1, max_length=MAX_FORMULA_LENGTH)
    symbol: str = Field(..., min_length=1, max_length=20)
    start_date: str | None = None
    end_date: str | None = None


@router.post("/preview")
def preview(req: PreviewRequest, request: Request) -> dict:
    symbol = _check_symbol(req.symbol)
    compiled = _compile_or_400(req.formula)
    start, end = _parse_range(req.start_date, req.end_date)
    dates, indicators = _series_for(request, {"preview": compiled}, symbol, start, end)
    return {
        "symbol": symbol,
        "dates": dates,
        "outputs": indicators["preview"]["outputs"],
        "warnings": [warning.to_dict() for warning in compiled.warnings],
        "warmup_bars": compiled.warmup_bars,
    }
