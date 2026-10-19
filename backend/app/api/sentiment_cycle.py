"""情绪周期与市场风格 API — 时序查询 + 口径配置。

计算在 app.services.sentiment_cycle / market_style (纯函数), API 层只做装配。
"""
from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/sentiment-cycle", tags=["sentiment-cycle"])


def _repo(request: Request):
    return getattr(request.app.state, "repo", None)


@router.get("")
def get_sentiment_cycle(
    request: Request,
    as_of: Annotated[date | None, Query()] = None,
) -> dict:
    """当前（或指定日期）情绪周期: 阶段、判定依据、近 20 日轨迹、当日指标。"""
    from app.services import market_style, preferences, sentiment_cycle

    repo = _repo(request)
    if repo is None:
        return {"available": False, "reason": "数据仓库未就绪"}
    style = market_style.get_style(repo, as_of, preferences.get_market_style_config())
    payload = sentiment_cycle.get_cycle(
        repo, as_of, preferences.get_sentiment_cycle_config(),
        style=(style or {}).get("style"),
    )
    if payload is None:
        return {"available": False, "reason": "enriched 历史数据不足, 请先同步日 K"}
    return {"available": True, **payload}


@router.get("/style")
def get_market_style(
    request: Request,
    as_of: Annotated[date | None, Query()] = None,
) -> dict:
    """当前市场风格: 大/小市值组收益、价差与风格判定。"""
    from app.services import market_style, preferences

    repo = _repo(request)
    if repo is None:
        return {"available": False, "reason": "数据仓库未就绪"}
    payload = market_style.get_style(repo, as_of, preferences.get_market_style_config())
    if payload is None:
        return {"available": False, "reason": "enriched 历史数据不足, 请先同步日 K"}
    return {"available": True, **payload}


@router.get("/sectors")
def get_sector_cycles(
    request: Request,
    kind: Annotated[str, Query(pattern="^(concept|industry)$")] = "concept",
    as_of: Annotated[date | None, Query()] = None,
) -> dict:
    """板块周期: 当日涨幅靠前的板块各自所处阶段 (按板块涨幅排名顺序)。"""
    from app.services import preferences, sentiment_cycle

    repo = _repo(request)
    if repo is None:
        return {"available": False, "reason": "数据仓库未就绪", "sectors": []}
    cycles = sentiment_cycle.get_sector_cycles(
        repo, as_of, preferences.get_sentiment_cycle_config(), kind=kind,
    )
    if not cycles:
        return {
            "available": False,
            "reason": "未配置概念/行业归属的扩展数据, 或历史数据不足",
            "sectors": [],
        }
    return {
        "available": True,
        "kind": kind,
        "sectors": [{"name": name, **state} for name, state in cycles.items()],
    }


class SentimentCycleConfigIn(BaseModel):
    config: dict | None = None


@router.get("/config")
def get_sentiment_cycle_config() -> dict:
    """情绪周期口径: 当前配置 + 出厂默认 + 阶段顺序与合法转移。"""
    from app.services import preferences, sentiment_cycle

    raw = preferences.get_sentiment_cycle_config()
    payload = sentiment_cycle.public_config(raw)
    payload["customized"] = raw is not None
    return payload


@router.put("/config")
def update_sentiment_cycle_config(req: SentimentCycleConfigIn) -> dict:
    """保存情绪周期口径; config 传 null/{} = 恢复出厂。"""
    from app.services import preferences, sentiment_cycle

    saved = preferences.set_sentiment_cycle_config(req.config)
    payload = sentiment_cycle.public_config(saved)
    payload["customized"] = saved is not None
    return payload
