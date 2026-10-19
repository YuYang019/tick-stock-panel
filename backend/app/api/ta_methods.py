"""技术面心法 API —— 心法库 CRUD、特征目录、预览与场景选择。

心法文件: data/user_data/ta_methods/{id}.json; 出厂心法只读。
预览接口只跑规则层, 不调用 AI (「试跑 AI」由前端走个股分析流)。
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/ta-methods", tags=["ta-methods"])


def _data_dir(request: Request):
    repo = getattr(request.app.state, "repo", None)
    return getattr(getattr(repo, "store", None), "data_dir", None)


@router.get("")
def list_methods(request: Request) -> dict:
    """出厂 + 用户心法列表 + 当前选择。"""
    from app.analysis import method as ta_method
    from app.services import preferences

    data_dir = _data_dir(request)
    if data_dir is None:
        return {"methods": [], "selection": preferences.get_ta_method_selection()}
    return {
        "methods": ta_method.list_methods(data_dir),
        "selection": preferences.get_ta_method_selection(),
    }


@router.get("/features")
def list_features() -> dict:
    """已注册特征目录 (供心法编辑器渲染参数表单)。"""
    from app.analysis.features import list_features as _list

    return {"features": _list()}


class MethodIn(BaseModel):
    method: dict


class CopyIn(BaseModel):
    source_id: str
    new_id: str | None = None   # 留空 = 自动生成未占用的副本 id
    name: str | None = None


class SelectionIn(BaseModel):
    active: str | None = None
    per_scenario: dict = {}


class PreviewIn(BaseModel):
    symbol: str
    method_id: str | None = None
    as_of: date | None = None


@router.put("/selection")
def update_selection(req: SelectionIn) -> dict:
    """保存心法选择: 全局 active + 按场景覆盖。"""
    from app.services import preferences

    return preferences.set_ta_method_selection(req.model_dump())


@router.get("/{method_id}")
def get_method(request: Request, method_id: str) -> dict:
    from app.analysis import method as ta_method

    data_dir = _data_dir(request)
    if data_dir is None:
        raise HTTPException(status_code=503, detail="数据目录未就绪")
    loaded = ta_method.load_method(method_id, data_dir)
    if loaded is None:
        raise HTTPException(status_code=404, detail="心法不存在")
    return {"method": loaded, "factory": method_id == ta_method.FACTORY_ID}


@router.put("/{method_id}")
def save_method(request: Request, method_id: str, req: MethodIn) -> dict:
    from app.analysis import method as ta_method

    data_dir = _data_dir(request)
    if data_dir is None:
        raise HTTPException(status_code=503, detail="数据目录未就绪")
    payload = dict(req.method)
    payload["id"] = method_id
    try:
        saved = ta_method.save_method(data_dir, payload)
    except ta_method.MethodValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"method": saved}


@router.delete("/{method_id}")
def delete_method(request: Request, method_id: str) -> dict:
    from app.analysis import method as ta_method

    data_dir = _data_dir(request)
    if data_dir is None:
        raise HTTPException(status_code=503, detail="数据目录未就绪")
    try:
        ta_method.delete_method(data_dir, method_id)
    except ta_method.MethodValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True}


@router.post("/copy")
def copy_method(request: Request, req: CopyIn) -> dict:
    from app.analysis import method as ta_method

    data_dir = _data_dir(request)
    if data_dir is None:
        raise HTTPException(status_code=503, detail="数据目录未就绪")
    try:
        new_id = req.new_id or ta_method.suggest_copy_id(data_dir, req.source_id)
        saved = ta_method.copy_method(data_dir, req.source_id, new_id, req.name)
    except ta_method.MethodConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ta_method.MethodValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"method": saved}


@router.post("/preview")
def preview(request: Request, req: PreviewIn) -> dict:
    """规则层预览: 三层定位 + 特征事实 + 原则核验 + 拼装后的提示词 (不调用 AI)。"""
    from app.analysis import engine as ta_engine
    from app.analysis import method as ta_method
    from app.services import preferences

    repo = getattr(request.app.state, "repo", None)
    data_dir = _data_dir(request)
    if repo is None or data_dir is None:
        raise HTTPException(status_code=503, detail="数据仓库未就绪")

    if req.method_id:
        loaded = ta_method.load_method(req.method_id, data_dir)
        if loaded is None:
            raise HTTPException(status_code=404, detail="心法不存在")
    else:
        loaded = ta_method.resolve_for_scenario(
            data_dir, "stock_analysis", preferences.get_ta_method_selection(),
        )

    from app.services.stock_analyzer import _load_financials, _load_kline, _ta_context

    df = _load_kline(repo, req.symbol)
    if df.is_empty():
        raise HTTPException(status_code=404, detail=f"{req.symbol} 暂无日 K 数据")
    fins = _load_financials(data_dir, req.symbol)
    ctx = _ta_context(repo, req.symbol, df, fins)
    evaluation = ta_engine.evaluate(
        loaded, df,
        cycle=ctx.get("cycle"), style=ctx.get("style"),
        sector_phase=ctx.get("sector_phase"), sector_rs_pct=ctx.get("sector_rs_pct"),
        benchmark_ret20=ctx.get("benchmark_ret20"), financials=fins,
    )
    system, user = ta_engine.build_ta_prompt(loaded, evaluation, "stock_analysis")
    return {
        "method_id": loaded["id"],
        "evaluation": evaluation,
        "prompt": {"system": system, "user": user},
    }
