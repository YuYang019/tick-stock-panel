"""心法体系: 特征、schema 校验、三层评估、钩子与提示词拼装。"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import polars as pl
import pytest

from app.analysis import engine, features, method


def _frame(n: int = 80, *, rising: bool = True, base: float = 10.0) -> pl.DataFrame:
    """上升/下降趋势 + 周期波动 (波动用于产生分型高低点)。"""
    rows = []
    start = date(2026, 5, 1)
    price = base
    for i in range(n):
        wave = 1 + 0.03 * math.sin(i / 4)   # 每 4 日一个摆动
        price *= (1.006 if rising else 0.994)
        mid = price * wave
        if rising:
            ma5, ma10, ma20, ma60 = (price * k for k in (0.995, 0.99, 0.985, 0.98))
        else:
            ma5, ma10, ma20, ma60 = (price * k for k in (1.005, 1.01, 1.015, 1.02))
        rows.append({
            "date": start + timedelta(days=i),
            "open": mid * 0.995, "high": mid * 1.01, "low": mid * 0.99,
            "close": mid, "volume": 1e6 * (1 + 0.1 * (i % 3)),
            "change_pct": 0.006 if rising else -0.006,
            "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
            "macd_dif": 0.2 if rising else -0.2, "macd_dea": 0.1 if rising else -0.1,
            "macd_hist": 0.1 if rising else -0.1,
            "high_60d": price * (1.02 if not rising else 1.0),
            "low_60d": price * 0.8,
            "turnover_rate": 3.0, "amplitude": 0.02, "vol_ratio_5d": 1.2,
            "consecutive_limit_ups": 0, "atr_14": price * 0.03,
            "rsi_14": 60.0,
        })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


# ---------- 特征 ----------

def test_swing_structure_rising():
    out = features.run_feature("swing_structure", _frame())
    assert out.score is not None and out.score > 0
    assert "上升" in out.state


def test_swing_structure_falling():
    out = features.run_feature("swing_structure", _frame(rising=False))
    assert out.score is not None and out.score < 0
    assert "下降" in out.state


def test_ma_alignment_states():
    assert "多头" in features.run_feature("ma_alignment", _frame()).state
    assert "空头" in features.run_feature("ma_alignment", _frame(rising=False)).state


def test_consensus_states():
    df = _frame(30).with_columns([
        pl.lit(1).alias("consecutive_limit_ups"),
        pl.lit(1.5).alias("turnover_rate"),
    ])
    assert features.run_feature("consensus_state", df).state == "一致加速"

    df = _frame(30).with_columns([
        pl.lit(12.0).alias("turnover_rate"),
        pl.lit(0.09).alias("amplitude"),
        pl.lit(-0.03).alias("change_pct"),
        pl.lit(9.0).alias("close"), pl.lit(10.0).alias("high"), pl.lit(9.0).alias("low"),
    ])
    assert features.run_feature("consensus_state", df).state == "分歧转弱"


def test_missing_columns_returns_uncomputable():
    df = pl.DataFrame({"date": [date(2026, 1, 1)], "close": [10.0]})
    out = features.run_feature("ma_alignment", df)
    assert out.score is None
    assert out.state == "数据不足"


def test_unknown_feature_is_reported():
    out = features.run_feature("nope", _frame())
    assert out.score is None
    assert "未注册" in out.state


def test_fundamental_floor_uses_canonical_fields():
    """字段名必须是标准化口径: roe / net_income_yoy (metrics), net_income_attributable (income)。"""
    out = features.run_feature("fundamental_floor", _frame(), {
        "financials": {
            "metrics": [{"roe": 15, "net_income_yoy": 20, "revenue_yoy": 12}],
            "income": [{"net_income_attributable": 1e8}],
        },
    })
    assert out.score > 0
    assert out.facts["net_profit_yoy"] == 20

    # 亏损 + 净利大幅下滑 → 薄弱
    out = features.run_feature("fundamental_floor", _frame(), {
        "financials": {
            "metrics": [{"roe": 1, "net_income_yoy": -60}],
            "income": [{"net_income_attributable": -1e7}],
        },
    })
    assert out.state == "基本面薄弱"
    assert any("净利润为负" in e for e in out.evidence)


# ---------- schema ----------

def test_factory_method_valid():
    m = method.factory_method()
    assert m["id"] == method.FACTORY_ID
    assert len(m["modules"]) == 8
    assert len(m["maxims"]) == 7


def test_validate_rejects_unknown_feature():
    with pytest.raises(method.MethodValidationError):
        method.validate({
            "id": "x1", "modules": [
                {"id": "m", "features": [{"id": "not_registered"}]},
            ],
        })


def test_validate_rejects_unknown_hook_source():
    with pytest.raises(method.MethodValidationError):
        method.validate({
            "id": "x1",
            "modules": [{"id": "m", "features": ["ma_alignment"]}],
            "maxims": [{"id": "a", "hooks": [{"source": "shell:rm", "when": {"eq": 1}}]}],
        })


def test_validate_rejects_bad_weight_and_duplicate_modules():
    with pytest.raises(method.MethodValidationError):
        method.validate({"id": "x1", "modules": [{"id": "m", "weight": -1}]})
    with pytest.raises(method.MethodValidationError):
        method.validate({"id": "x1", "modules": [{"id": "m"}, {"id": "m"}]})


def test_save_load_copy_delete_roundtrip(tmp_path):
    clone = method.copy_method(tmp_path, method.FACTORY_ID, "my-v1", "我的心法")
    assert clone["id"] == "my-v1" and clone["based_on"] == method.FACTORY_ID
    loaded = method.load_method("my-v1", tmp_path)
    assert loaded["name"] == "我的心法"
    clone["name"] = "改名"
    saved = method.save_method(tmp_path, clone)
    assert saved["version"] == 2
    assert (tmp_path / "user_data" / "ta_methods" / ".history" / "my-v1.1.json").exists()
    method.delete_method(tmp_path, "my-v1")
    assert method.load_method("my-v1", tmp_path) is None


def test_factory_cannot_be_overwritten(tmp_path):
    with pytest.raises(method.MethodValidationError):
        method.save_method(tmp_path, method.factory_method())


def test_resolve_for_scenario_falls_back(tmp_path):
    resolved = method.resolve_for_scenario(
        tmp_path, "stock_analysis", {"active": "missing", "per_scenario": {}},
    )
    assert resolved["id"] == method.FACTORY_ID


# ---------- 引擎 ----------

def test_evaluate_produces_three_layers():
    m = method.factory_method()
    ev = engine.evaluate(
        m, _frame(),
        cycle={"phase": "发酵", "extreme": ""},
        style={"style": "题材主导", "note": "存量博弈"},
        sector_phase="发酵", sector_rs_pct=0.8,
    )
    assert ev["market_gate"]["passed"] is True
    assert ev["sector_gate"]["passed"] is True
    assert ev["composite"] > 0
    assert ev["rule_label"] in {"企稳", "反弹"}
    assert set(ev["effective_weights"]) == {
        "trend", "momentum", "volume_price", "key_levels", "phase", "structure", "logic",
    }


def test_phase_override_changes_weights():
    m = method.factory_method()
    kw = dict(sector_phase="发酵", sector_rs_pct=0.8)
    up = engine.evaluate(m, _frame(), cycle={"phase": "发酵"}, **kw)
    down = engine.evaluate(m, _frame(), cycle={"phase": "退潮"}, **kw)
    assert up["effective_weights"]["momentum"] > down["effective_weights"]["momentum"]
    assert down["effective_weights"]["key_levels"] > up["effective_weights"]["key_levels"]


def test_market_gate_lowers_confidence():
    m = method.factory_method()
    kw = dict(sector_phase="发酵", sector_rs_pct=0.8)
    normal = engine.evaluate(m, _frame(), cycle={"phase": "发酵"}, **kw)
    weak = engine.evaluate(m, _frame(), cycle={"phase": "退潮", "extreme": "亏钱"}, **kw)
    assert weak["market_gate"]["passed"] is False
    assert weak["confidence"] < normal["confidence"]


def test_hooks_produce_hits_and_tags():
    m = method.factory_method()
    df = _frame().with_columns([
        pl.lit(1).alias("consecutive_limit_ups"),
        pl.lit(1.0).alias("turnover_rate"),
    ])
    ev = engine.evaluate(
        m, df, cycle={"phase": "发酵", "extreme": "赚钱"},
        style={"style": "题材主导"}, sector_phase="发酵", sector_rs_pct=0.8,
    )
    checks = {c["id"]: c for c in ev["principle_checks"]}
    assert checks["divergence_consensus"]["result"] == "hit"
    assert checks["divergence_consensus"]["tag"] == "透支风险"
    assert checks["extreme_reversal"]["result"] == "hit"
    assert checks["everything_cycles"]["result"] == "n/a"


def test_logic_quadrant():
    m = method.factory_method()
    ev = engine.evaluate(
        m, _frame(),
        financials={"metrics": [{"roe": 18, "net_income_yoy": 30}],
                    "income": [{"net_income_attributable": 1e8}]},
        concept_heat=0.9, recent_limit_ups=3,
        sector_phase="发酵", sector_rs_pct=0.8, cycle={"phase": "发酵"},
    )
    assert "爆发前提" in (ev["logic_quality"]["quadrant"] or "")


def test_build_ta_prompt_contains_layers_and_hides_disabled():
    m = method.factory_method()
    for mod in m["modules"]:
        if mod["id"] == "phase":
            mod["enabled"] = False
    ev = engine.evaluate(m, _frame(), cycle={"phase": "发酵"}, sector_phase="发酵", sector_rs_pct=0.8)
    system, user = engine.build_ta_prompt(m, ev, "stock_analysis")
    assert "投资心法" in system
    assert "三层分析框架" in system
    assert "底层逻辑三问" in system
    assert "市场阶段" not in system   # 禁用模块的原则不出现在提示词
    assert "三层定位" in user
    assert "原则核验" in user


def test_build_ta_prompt_truncates_principles():
    m = method.factory_method()
    m["modules"][1]["principles"] = "原则" * 3000
    ev = engine.evaluate(m, _frame())
    system, _ = engine.build_ta_prompt(m, ev, "watchlist_review")
    assert len(system) < 20000


def test_market_review_scenario_excludes_stock_level_maxims():
    m = method.factory_method()
    ev = engine.evaluate(m, _frame(), cycle={"phase": "发酵"})
    system, _ = engine.build_ta_prompt(m, ev, "market_review")
    assert "买在分歧" not in system
    assert "万物皆周期" in system


# ---------- API ----------

def test_api_crud_and_preview(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import ta_methods as api
    from app.services import preferences

    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr(preferences, "save", lambda upd: None)

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)

    listed = client.get("/api/ta-methods").json()
    assert listed["methods"] == []   # 数据目录未就绪时降级

    features = client.get("/api/ta-methods/features").json()
    assert {f["id"] for f in features["features"]} >= {"swing_structure", "consensus_state"}


def test_api_requires_data_dir(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import ta_methods as api

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    assert client.get("/api/ta-methods/fusion-v1").status_code == 503


def test_api_with_data_dir(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import ta_methods as api
    from app.services import preferences

    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr(preferences, "save", lambda upd: None)

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app = FastAPI()
    app.state.repo = repo
    app.include_router(api.router)
    client = TestClient(app)

    listed = client.get("/api/ta-methods").json()
    assert listed["methods"][0]["id"] == method.FACTORY_ID
    assert listed["methods"][0]["factory"] is True

    factory = client.get("/api/ta-methods/fusion-v1").json()
    assert factory["factory"] is True

    copied = client.post("/api/ta-methods/copy", json={
        "source_id": method.FACTORY_ID, "new_id": "mine", "name": "我的心法",
    })
    assert copied.status_code == 200
    assert (tmp_path / "user_data" / "ta_methods" / "mine.json").exists()

    # 出厂心法不可覆盖
    assert client.put("/api/ta-methods/fusion-v1", json={"method": factory["method"]}).status_code == 400

    # 非法特征 → 400
    bad = dict(factory["method"])
    bad["id"] = "mine"
    bad["modules"] = [{"id": "m", "features": [{"id": "nope"}]}]
    assert client.put("/api/ta-methods/mine", json={"method": bad}).status_code == 400

    assert client.delete("/api/ta-methods/mine").status_code == 200
    assert client.get("/api/ta-methods/mine").status_code == 404

    selection = client.put("/api/ta-methods/selection", json={
        "active": "mine", "per_scenario": {"watchlist_review": "fusion-v1", "bogus": "x"},
    }).json()
    assert selection["per_scenario"] == {"watchlist_review": "fusion-v1"}


def test_api_preview_without_repo():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import ta_methods as api

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    res = client.post("/api/ta-methods/preview", json={"symbol": "600000.SH"})
    assert res.status_code == 503



# ---------- 回归: review 修复项 ----------

def test_copy_does_not_overwrite_existing(tmp_path):
    method.copy_method(tmp_path, method.FACTORY_ID, "mine-v1", "第一份")
    edited = method.load_method("mine-v1", tmp_path)
    edited["name"] = "已编辑"
    method.save_method(tmp_path, edited)
    with pytest.raises(method.MethodConflictError):
        method.copy_method(tmp_path, method.FACTORY_ID, "mine-v1", "第二份")
    assert method.load_method("mine-v1", tmp_path)["name"] == "已编辑"


def test_suggest_copy_id_skips_taken(tmp_path):
    first = method.suggest_copy_id(tmp_path, method.FACTORY_ID)
    method.copy_method(tmp_path, method.FACTORY_ID, first)
    second = method.suggest_copy_id(tmp_path, method.FACTORY_ID)
    assert first != second
    assert second.startswith(f"{method.FACTORY_ID}-copy")


def test_load_method_rejects_path_traversal(tmp_path):
    outside = tmp_path / "user_data" / "evil.json"
    outside.parent.mkdir(parents=True)
    outside.write_text(json.dumps(method.factory_method() | {"id": "evil"}))
    assert method.load_method("../evil", tmp_path) is None


def test_principles_prompt_has_no_empty_stock_sections():
    """大盘复盘只注入原则段, 不出现空维度表/规则评分 0 等噪音。"""
    text = engine.build_principles_prompt(method.factory_method(), "market_review")
    assert "投资心法" in text
    assert "万物皆周期" in text
    assert "买在分歧" not in text
    assert "技术面维度" not in text
    assert "规则评分" not in text


def test_api_copy_conflict_returns_409(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import ta_methods as api

    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(api.router)
    client = TestClient(app)
    body = {"source_id": method.FACTORY_ID, "new_id": "dup"}
    assert client.post("/api/ta-methods/copy", json=body).status_code == 200
    assert client.post("/api/ta-methods/copy", json=body).status_code == 409
    # 不指定 new_id 时自动生成不冲突的 id
    auto = client.post("/api/ta-methods/copy", json={"source_id": method.FACTORY_ID})
    assert auto.status_code == 200
    assert auto.json()["method"]["id"] != "dup"


def test_sector_context_uses_membership_not_leader(monkeypatch):
    """板块强弱分位来自归属映射 + 全量排名, 阶段来自板块周期; 宽基标签不参与。"""
    from app.services import market_overview_builder as mob
    from app.services import sentiment_cycle, stock_analyzer

    class _Repo:
        def get_enriched_latest(self):
            return pl.DataFrame({"symbol": ["600001.SH"], "name": ["普通成分股"],
                                 "change_pct": [0.01], "amount": [1e8]}), date(2026, 10, 19)

    # 板块0 最强 ... 板块10 最弱; 「融资融券」为宽基 (成分数超上限), 排第一也要被剔除
    ranked = [{"name": "融资融券", "count": 3800}] + [{"name": f"板块{i}", "count": 50} for i in range(11)]
    monkeypatch.setattr(mob, "sector_memberships",
                        lambda repo, sym, kind, level=None: ["融资融券", "板块2"] if kind == "concept" else [])
    monkeypatch.setattr(mob, "dimension_rank_all", lambda rows, repo, kind, level=None: ranked)
    monkeypatch.setattr("app.services.market_mainline._resolve_filter_config",
                        lambda cfg=None: {"min_members": 4, "max_members": 600, "blacklist": set()})
    calls = []

    def fake_cycles(repo, as_of, cfg, *, sectors=None, kind="concept"):
        calls.append((tuple(sectors or ()), kind))
        return {"板块2": {"phase": "发酵"}}

    monkeypatch.setattr(sentiment_cycle, "get_sector_cycles", fake_cycles)
    phase, pct = stock_analyzer._sector_context(_Repo(), "600001.SH")
    assert phase == "发酵"
    assert pct == pytest.approx(0.8)
    assert calls == [(("板块2",), "concept")]   # 只算主板块, 不算宽基

    monkeypatch.setattr(mob, "sector_memberships", lambda *a, **k: [])
    assert stock_analyzer._sector_context(_Repo(), "600001.SH") == (None, None)


def test_tradable_sector_filters_style_tags(monkeypatch):
    from app.services import sentiment_cycle

    monkeypatch.setattr("app.services.market_mainline._resolve_filter_config",
                        lambda cfg=None: {"min_members": 4, "max_members": 600, "blacklist": {"屏蔽"}})
    small = {f"{i:06d}" for i in range(50)} | {f"{i:06d}.SZ" for i in range(50)}
    huge = {f"{i:06d}" for i in range(3000)}
    assert sentiment_cycle.tradable_sector("固态电池", small)
    assert not sentiment_cycle.tradable_sector("融资融券", huge)
    assert not sentiment_cycle.tradable_sector("屏蔽", small)
    assert not sentiment_cycle.tradable_sector("微型", {"000001", "000002"})


def test_stock_report_structure_lists_new_sections():
    text = engine.stock_report_structure(method.factory_method())
    for key in ("树干校验", "逻辑质量", "底层逻辑三问", "谁在买?"):
        assert key in text
    assert "不要输出「原则核验」表" in text


def test_principle_checks_table_from_code():
    ev = engine.evaluate(
        method.factory_method(), _frame(),
        cycle={"phase": "发酵", "extreme": "赚钱"}, style={"style": "题材主导"},
    )
    table = engine.principle_checks_table(ev)
    assert "### 原则核验(规则层)" in table
    assert "| 极致赚钱后是极致亏钱 | 市场层 | 命中 | 透支风险 |" in table
    assert "| 万物皆周期 | 总纲 | 不适用 |" in table
    assert engine.principle_checks_table({}) == ""


async def test_stock_stream_appends_check_table_and_sources(monkeypatch, tmp_path):
    """原则核验表与资料来源在模型正文之后由代码追加。"""
    import json as _json

    from app.services import stock_analyzer

    df = _frame()
    monkeypatch.setattr(stock_analyzer, "_load_kline", lambda repo, sym: df)
    monkeypatch.setattr(stock_analyzer, "_load_financials", lambda d, s: {})
    ev = engine.evaluate(method.factory_method(), df, cycle={"phase": "发酵", "extreme": "赚钱"})
    monkeypatch.setattr(stock_analyzer, "_evaluate_ta", lambda *a: (ev, "SYS", "USR"))

    async def fake_news(repo, symbol):
        return [{"title": "公告", "url": "https://e.com/n", "summary": "s"}], ""

    monkeypatch.setattr(stock_analyzer, "_load_stock_news", fake_news)
    captured = {}

    async def fake_stream(messages, **kw):
        captured["system"] = messages[0]["content"]
        yield "模型正文"

    monkeypatch.setattr("app.services.ai_provider.stream_ai_text", fake_stream)

    class _Repo:
        def resolve_asset_type(self, s):
            return "stock"

    events = [_json.loads(e) async for e in stock_analyzer.analyze_stock_stream(_Repo(), tmp_path, "600001.SH")]
    deltas = [e["content"] for e in events if e["type"] == "delta"]
    assert deltas[0] == "模型正文"
    assert "原则核验(规则层)" in deltas[-1]
    assert "https://e.com/n" in deltas[-1]
    assert events[-1]["type"] == "done"
    # 章节结构在基础模板之后 (越靠后优先级越高)
    assert captured["system"].index("本次报告章节") > captured["system"].index("SYS")


def test_relative_strength_feature_uses_sixty_day_on_conflict():
    """20 日跑输但 60 日大幅跑赢时, 以较长窗口定 state, 不再判成弱。"""
    df = _frame(80)
    out = features.run_feature("relative_strength", df, {
        "benchmark_ret20": 0.15,      # 个股 20 日约 +7.6%, 跑输 7.4pp
        "benchmark_ret60": -0.30,     # 60 日跑赢很多
        "benchmark_name": "创业板指",
    })
    assert out.facts["relative_20d"] < 0
    assert out.facts["relative_60d"] > 0
    assert out.state == "强于创业板指"
    assert out.score > 0
    assert out.facts["benchmark"] == "创业板指"


def test_relative_strength_reports_missing_benchmark_name_default():
    out = features.run_feature("relative_strength", _frame(80), {})
    assert out.state == "绝对动量"
    assert out.facts["relative_20d"] is None
