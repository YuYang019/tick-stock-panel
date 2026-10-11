"""指标 API: CRUD、builtin 保护、validate、series 对齐 kline 日期与 warmup 裁剪、ids 上限、空数据。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import indicators as indicators_api
from app.api import kline as kline_api
from app.api import signals as signals_api
from app.factors.registry import get_factor
from app.indicators.tdx import factors as indicator_factors
from app.indicators.tdx import store

TODAY = date(2026, 10, 12)
SYMBOL = "600000.SH"


def _history(n: int = 400) -> pl.DataFrame:
    days = [TODAY - timedelta(days=1 + i) for i in range(n)][::-1]
    days = [d for d in days if d.weekday() < 5]
    close = [10.0 + 0.01 * i for i in range(len(days))]
    return pl.DataFrame({
        "symbol": [SYMBOL] * len(days), "date": days, "open": close, "high": close, "low": close,
        "close": close, "volume": [1000.0] * len(days), "amount": [1e6] * len(days),
    })


class _Repo:
    def __init__(self, data_dir: Path, history: pl.DataFrame) -> None:
        self.store = SimpleNamespace(data_dir=data_dir)
        self.history = history
        self.calls: list[tuple[date, date]] = []

    def resolve_asset_type(self, symbol):
        return "stock"

    def get_daily_asset(self, asset_type, symbol, start, end, columns=None):
        self.calls.append((start, end))
        frame = self.history.filter((pl.col("symbol") == symbol) & (pl.col("date") >= start) & (pl.col("date") <= end))
        return frame

    def get_instruments_asset(self, asset_type):
        return pl.DataFrame()

    def get_instruments(self):
        return pl.DataFrame()


class _Quotes:
    def __init__(self, close: float | None) -> None:
        self.close = close

    def get_enriched_today(self):
        if self.close is None:
            return pl.DataFrame(), None
        return pl.DataFrame({"symbol": [SYMBOL], "open": [self.close], "high": [self.close], "low": [self.close],
                             "close": [self.close], "volume": [10.0], "amount": [100.0], "change_pct": [0.0]}), TODAY


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_api, "cn_today", lambda: TODAY)
    monkeypatch.setattr(indicators_api, "cn_today", lambda: TODAY)
    monkeypatch.setattr(kline_api, "_get_stock_info", lambda repo, symbol: {"name": "浦发银行"})
    monkeypatch.setattr(kline_api, "_attach_ext", lambda resp, *a, **k: resp)
    # 本地无数据时的实时拉取回退默认不出网; 需要时由用例覆盖
    monkeypatch.setattr(kline_api, "live_enriched_frame", lambda request, symbol, count: pl.DataFrame())
    app = FastAPI()
    app.include_router(indicators_api.router)
    app.include_router(kline_api.router)
    app.include_router(signals_api.router)
    repo = _Repo(tmp_path, _history())
    app.state.repo = repo
    app.state.quote_service = _Quotes(close=99.0)
    store.seed_if_empty(tmp_path)
    indicator_factors.load_into_registry(tmp_path)
    yield TestClient(app), repo, tmp_path
    for definition in store.load_all(tmp_path):
        indicator_factors.unregister_indicator(definition["id"])


def test_seed_listed_and_registered(env) -> None:
    client, _repo, _ = env
    items = client.get("/api/indicators").json()
    assert [i["id"] for i in items] == ["zhixing"]
    zhixing = items[0]
    assert zhixing["builtin"] is True and zhixing["pane"] == "main"
    assert [o["name"] for o in zhixing["outputs"]] == ["知行短期趋势线", "知行多空线"]
    for output in zhixing["outputs"]:
        spec = get_factor(f"ind_zhixing_{output['key']}")
        assert spec is not None and spec.kind == "indicator" and spec.group == "自定义指标"
        assert spec.label == f"知行.{output['name']}"


def test_seed_idempotent_after_user_deletes(tmp_path) -> None:
    assert store.seed_if_empty(tmp_path) == 1
    (tmp_path / "user_data" / "indicators" / "zhixing.json").unlink()
    assert store.seed_if_empty(tmp_path) == 0


def test_functions_and_validate(env) -> None:
    client, _, _ = env
    functions = client.get("/api/indicators/functions").json()
    assert {"MA", "EMA", "SMA", "CROSS"} <= {f["name"] for f in functions["functions"]}
    assert functions["variables"] and functions["modifiers"]
    ok = client.post("/api/indicators/validate", json={"formula": "A:MA(C,5),COLORRED;"}).json()
    assert ok["ok"] is True and ok["warmup_bars"] == 5
    assert ok["outputs"][0] == {"name": "A", "key": "a", "kind": "line", "color": "#FF0000",
                                "thick": 1, "dashed": False, "visible": True}
    bad = client.post("/api/indicators/validate", json={"formula": "A:MA(C,5;"}).json()
    assert bad["ok"] is False and bad["errors"][0]["line"] == 1 and bad["errors"][0]["code"] == "T002"


def test_crud_and_builtin_protection(env) -> None:
    client, _, data_dir = env
    created = client.post("/api/indicators", json={"name": "My MACD", "formula": "DIF:EMA(C,12)-EMA(C,26);DEA:EMA(DIF,9);", "pane": "sub"})
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["id"] == "my_macd"
    assert sorted(body["factor_ids"]) == ["ind_my_macd_dea", "ind_my_macd_dif"]
    # 同名再建 → slug 加后缀
    again = client.post("/api/indicators", json={"name": "My MACD", "formula": "X:C;", "pane": "sub"}).json()
    assert again["id"] == "my_macd_2"
    # 显式 id 冲突
    assert client.post("/api/indicators", json={"id": "my_macd", "name": "x", "formula": "X:C;"}).status_code == 409
    # 编译失败拒绝保存
    assert client.post("/api/indicators", json={"name": "bad", "formula": "X:FOO;"}).status_code == 400
    assert not (data_dir / "user_data" / "indicators" / "bad.json").exists()
    # 更新: 输出改名 → 旧因子注销, 新因子注册
    updated = client.put("/api/indicators/my_macd", json={"name": "My MACD", "formula": "DIF:EMA(C,12)-EMA(C,26);", "pane": "sub"})
    assert updated.status_code == 200
    assert get_factor("ind_my_macd_dea") is None and get_factor("ind_my_macd_dif") is not None
    # builtin
    assert client.put("/api/indicators/zhixing", json={"name": "x", "formula": "A:C;"}).status_code == 400
    assert client.delete("/api/indicators/zhixing").status_code == 400
    # 删除
    assert client.delete("/api/indicators/my_macd").json()["removed_factors"] == ["ind_my_macd_dif"]
    assert get_factor("ind_my_macd_dif") is None
    assert client.delete("/api/indicators/my_macd").status_code == 404


def test_path_traversal_rejected(env) -> None:
    client, _, _ = env
    assert client.post("/api/indicators", json={"id": "../evil", "name": "x", "formula": "A:C;"}).status_code in (400, 422)
    assert client.delete("/api/indicators/..%2Fevil").status_code in (404, 400)
    with pytest.raises(ValueError):
        store._path(Path("/tmp"), "../x")


def test_delete_referenced_by_signal_is_409(env) -> None:
    client, _, data_dir = env
    client.post("/api/indicators", json={"id": "ma2", "name": "双均线", "formula": "FAST:MA(C,5);SLOW:MA(C,10);"})
    signal_dir = data_dir / "user_data" / "custom_signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    (signal_dir / "golden.json").write_text(
        '{"id":"golden","conditions":[{"left":"ind_ma2_fast","op":">","right":"field:ind_ma2_slow"}]}', encoding="utf-8",
    )
    response = client.delete("/api/indicators/ma2")
    assert response.status_code == 409
    assert response.json()["detail"]["references"] == ["custom_signals/golden.json"]
    assert get_factor("ind_ma2_fast") is not None
    # 编辑移除被引用输出线同样拒绝
    blocked = client.put("/api/indicators/ma2", json={"name": "双均线", "formula": "FAST:MA(C,5);"})
    assert blocked.status_code == 409


def test_series_aligns_with_kline_daily_and_crops_warmup(env) -> None:
    client, repo, _ = env
    start, end = "2026-06-01", "2026-10-12"
    daily = client.get("/api/kline/daily", params={"symbol": SYMBOL, "start_date": start, "end_date": end}).json()
    series = client.get("/api/indicators/series", params={"symbol": SYMBOL, "ids": "zhixing", "start_date": start, "end_date": end}).json()
    kline_dates = [str(row["date"])[:10] for row in daily["rows"]]
    assert series["dates"] == kline_dates
    assert series["dates"][-1] == TODAY.isoformat()  # 含当日 live candle
    assert series["errors"] == {}
    outputs = series["indicators"]["zhixing"]["outputs"]
    assert all(len(o["values"]) == len(kline_dates) for o in outputs)
    # 预热向前扩展: 读取起点早于请求 start, 首个返回值已非空 (114 日均线)
    assert repo.calls[-1][0] < date.fromisoformat(start) - timedelta(days=114)
    assert all(o["values"][0] is not None for o in outputs)
    # live 行参与计算: 末值受 99 元实时价影响
    short = outputs[0]["values"]
    assert short[-1] > short[-2] + 1


def test_series_ids_limit_unknown_and_empty(env) -> None:
    client, repo, _ = env
    ids = ",".join(f"x{i}" for i in range(11))
    assert client.get("/api/indicators/series", params={"symbol": SYMBOL, "ids": ids}).status_code == 400
    unknown = client.get("/api/indicators/series", params={"symbol": SYMBOL, "ids": "nope,../x"}).json()
    assert unknown["indicators"] == {} and set(unknown["errors"]) == {"nope", "../x"}
    repo.history = repo.history.head(0)
    client.app.state.quote_service = _Quotes(close=None)
    empty = client.get("/api/indicators/series", params={"symbol": SYMBOL, "ids": "zhixing"}).json()
    assert empty["dates"] == []
    assert all(o["values"] == [] for o in empty["indicators"]["zhixing"]["outputs"])


def test_series_live_fallback_matches_kline_when_no_local_data(env, monkeypatch) -> None:
    """本地无该标的日K: 指标与 /api/kline/daily 走同一实时拉取, 日期仍一一对齐。"""
    client, repo, _ = env
    fetched = repo.history
    repo.history = repo.history.head(0)
    client.app.state.quote_service = _Quotes(close=None)
    monkeypatch.setattr(kline_api, "live_enriched_frame", lambda request, symbol, count: fetched)
    start = str(fetched["date"].cast(pl.Utf8).str.slice(0, 10).to_list()[-40])
    kline = client.get("/api/kline/daily", params={"symbol": SYMBOL, "start_date": start, "days": 2000}).json()
    series = client.get(
        "/api/indicators/series", params={"symbol": SYMBOL, "ids": "zhixing", "start_date": start},
    ).json()
    assert series["dates"]
    assert series["dates"] == [str(r["date"])[:10] for r in kline["rows"]][-len(series["dates"]):]
    assert any(v is not None for v in series["indicators"]["zhixing"]["outputs"][0]["values"])


def test_preview_unsaved_formula(env) -> None:
    client, _, _ = env
    response = client.post("/api/indicators/preview", json={
        "formula": "A:C/0;STICKLINE(C>O,O,C,4,0);", "symbol": SYMBOL, "start_date": "2026-09-01",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    line, stick = body["outputs"]
    assert len(line["values"]) == len(body["dates"]) and all(v is None for v in line["values"])
    assert stick["kind"] == "stick" and stick["values"] is None
    assert len(stick["stick"]["cond"]) == len(body["dates"]) and stick["stick"]["width"] == 4.0
    assert client.post("/api/indicators/preview", json={"formula": "A:FOO;", "symbol": SYMBOL}).status_code == 400


def test_signal_options_include_indicator_group_and_factor_list_excludes(env) -> None:
    client, _, _ = env
    options = client.get("/api/custom-signals/options").json()
    group = next(g for g in options["groups"] if g["key"] == "indicator")
    assert group["label"] == "自定义指标"
    labels = {f["label"] for f in group["fields"]}
    assert {"知行.知行短期趋势线", "知行.知行多空线"} == labels
    assert set(options["indicatorFields"]) == {f["key"] for f in group["fields"]}

    from app.api.factors import router as factors_router

    app = FastAPI()
    app.include_router(factors_router)
    factor_ids = {f["id"] for f in TestClient(app).get("/api/factors").json()["factors"]}
    assert not any(fid.startswith("ind_") for fid in factor_ids)
