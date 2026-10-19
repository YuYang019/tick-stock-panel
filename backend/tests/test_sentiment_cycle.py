"""情绪周期: 指标口径、状态机转移与防抖、配置合并。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.services import sentiment_cycle as sc


def _days(n: int) -> list[date]:
    start = date(2026, 8, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _frame(rows: list[dict]) -> pl.DataFrame:
    """rows: {symbol,date,close,high,low,limit_up,broken,pct,boards} → enriched 形状。"""
    out = []
    for r in rows:
        close = r["close"]
        out.append({
            "symbol": r["symbol"],
            "date": r["date"],
            "name": r.get("name", "测试股"),
            "change_pct": r.get("pct", 0.0),
            "raw_close": close,
            "raw_high": r.get("high", close),
            "raw_low": r.get("low", close),
            "turnover_rate": r.get("turnover", 3.0),
            "consecutive_limit_ups": r.get("boards", 0),
            "limit_up_price": r.get("limit_up", close * 1.1),
            "limit_down_price": r.get("limit_down", close * 0.9),
        })
    return pl.DataFrame(out)


def _metrics_from(rows: list[dict], **kw) -> pl.DataFrame:
    return sc.compute_daily_metrics(_frame(rows), **kw)


# ---------- 指标 ----------

def test_limit_counts_and_premium():
    d1, d2 = _days(2)
    rows = [
        # A 昨日涨停, 今日 +3% (溢价)
        {"symbol": "A", "date": d1, "close": 11.0, "limit_up": 11.0, "pct": 0.10, "boards": 1},
        {"symbol": "A", "date": d2, "close": 11.33, "pct": 0.03, "boards": 0},
        # B 昨日涨停, 今日 -6% (大面)
        {"symbol": "B", "date": d1, "close": 11.0, "limit_up": 11.0, "pct": 0.10, "boards": 1},
        {"symbol": "B", "date": d2, "close": 10.34, "pct": -0.06, "boards": 0},
        # C 昨日炸板, 今日 +1%
        {"symbol": "C", "date": d1, "close": 10.5, "high": 11.0, "limit_up": 11.0, "pct": 0.05},
        {"symbol": "C", "date": d2, "close": 10.6, "pct": 0.01},
    ]
    m = _metrics_from(rows).to_dicts()
    day2 = m[-1]
    assert day2["limit_up"] == 0
    assert day2["broken"] == 0
    assert day2["premium_lu"] == pytest.approx((0.03 - 0.06) / 2)
    assert day2["premium_broken"] == pytest.approx(0.01)
    assert day2["big_loss"] == 1
    day1 = m[0]
    assert day1["limit_up"] == 2
    assert day1["broken"] == 1
    assert day1["seal_rate"] == pytest.approx(2 / 3 * 100)


def test_one_word_board_excluded_from_promotion_denominator():
    d1, d2 = _days(2)
    rows = [
        # 一字板: 开=收=涨停, 次日继续涨停 → 不计入 1进2 分母
        {"symbol": "A", "date": d1, "close": 11.0, "high": 11.0, "low": 11.0,
         "limit_up": 11.0, "pct": 0.10, "boards": 1},
        {"symbol": "A", "date": d2, "close": 12.1, "limit_up": 12.1, "pct": 0.10, "boards": 2},
        # 换手板: 次日未涨停 → 计入分母, 未晋级
        {"symbol": "B", "date": d1, "close": 11.0, "high": 11.0, "low": 10.2,
         "limit_up": 11.0, "pct": 0.10, "boards": 1},
        {"symbol": "B", "date": d2, "close": 11.1, "pct": 0.01, "boards": 0},
    ]
    m = _metrics_from(rows).to_dicts()
    assert m[-1]["promo_1_2"] == pytest.approx(0.0)
    assert m[-1]["first_board"] == 0


def test_promotion_rate_computed():
    d1, d2 = _days(2)
    rows = [
        {"symbol": "A", "date": d1, "close": 11.0, "high": 11.0, "low": 10.2,
         "limit_up": 11.0, "pct": 0.10, "boards": 1},
        {"symbol": "A", "date": d2, "close": 12.1, "limit_up": 12.1, "pct": 0.10, "boards": 2},
        {"symbol": "B", "date": d1, "close": 11.0, "high": 11.0, "low": 10.2,
         "limit_up": 11.0, "pct": 0.10, "boards": 1},
        {"symbol": "B", "date": d2, "close": 11.1, "pct": 0.01, "boards": 0},
    ]
    m = _metrics_from(rows).to_dicts()
    assert m[-1]["promo_1_2"] == pytest.approx(50.0)


def test_height_broken_and_high_nuke():
    d1, d2 = _days(2)
    rows = [
        {"symbol": "A", "date": d1, "close": 13.31, "limit_up": 13.31, "pct": 0.10, "boards": 3},
        {"symbol": "A", "date": d2, "close": 12.24, "pct": -0.08, "boards": 0},
    ]
    m = _metrics_from(rows).to_dicts()
    assert m[-1]["height_broken"] == 1
    assert m[-1]["high_nuke"] == 1


def test_filters_st_new_and_bj():
    d1 = _days(1)[0]
    rows = [
        {"symbol": "A", "date": d1, "close": 11.0, "limit_up": 11.0, "pct": 0.1, "boards": 1},
        {"symbol": "B", "date": d1, "close": 11.0, "limit_up": 11.0, "pct": 0.1,
         "boards": 1, "name": "ST测试"},
        {"symbol": "C.BJ", "date": d1, "close": 11.0, "limit_up": 11.0, "pct": 0.1, "boards": 1},
    ]
    m = _metrics_from(rows).to_dicts()
    assert m[0]["limit_up"] == 1
    m_bj = _metrics_from(rows, include_bj=True).to_dicts()
    assert m_bj[0]["limit_up"] == 2


# ---------- 状态机 ----------

def _metrics_rows(spec: list[dict]) -> pl.DataFrame:
    """直接构造每日指标表 (跳过个股聚合), 便于精确控制状态机输入。"""
    base = {
        "limit_up": 50, "limit_down": 5, "broken": 10, "first_board": 30,
        "max_height": 5, "height_broken": 0, "seal_rate": 80.0,
        "seal_rate_drop": 0.0, "promo_1_2": 30.0, "promo_2_3": 30.0,
        "promo_3p": 20.0, "premium_lu": 0.02, "premium_multi": 0.03,
        "premium_broken": -0.01, "big_loss": 3, "high_nuke": 0, "up_ratio": 0.55,
        "avg_pct": 0.003,
    }
    out = []
    for i, over in enumerate(spec):
        row = {**base, **over, "date": _days(len(spec))[i]}
        out.append(row)
    return pl.DataFrame(out).with_columns(pl.col("date").cast(pl.Date))


ICE_ROW = {"limit_up": 5, "max_height": 2, "premium_lu": -0.02, "big_loss": 12,
           "seal_rate": 40.0, "up_ratio": 0.25, "first_board": 10}


def test_ice_point_to_repair_with_confirmation():
    spec = [ICE_ROW] * 6 + [
        # 冰点: 涨停极少、高度低、溢价负、大面多
        dict(ICE_ROW),
        # 修复第 1 天: 溢价转正、首板增加、大面下降
        {"limit_up": 25, "max_height": 3, "premium_lu": 0.01, "big_loss": 5,
         "seal_rate": 65.0, "up_ratio": 0.5, "first_board": 45},
        # 修复第 2 天: 连续满足 → 切换
        {"limit_up": 30, "max_height": 3, "premium_lu": 0.02, "big_loss": 3,
         "seal_rate": 70.0, "up_ratio": 0.55, "first_board": 50},
    ]
    states = sc.classify(_metrics_rows(spec))
    phases = [s.phase for s in states]
    assert phases[6] == "冰点"
    # 第 1 天仅成为候选, 不切换
    assert states[7].phase == "冰点"
    assert states[7].pending == "修复"
    assert states[8].phase == "修复"


def test_single_day_noise_does_not_switch():
    spec = [{}] * 8
    spec[6] = {"limit_up": 5, "max_height": 2, "premium_lu": -0.02, "big_loss": 12,
               "seal_rate": 40.0, "up_ratio": 0.25}
    spec[7] = {"limit_up": 20, "premium_lu": 0.01, "big_loss": 4, "seal_rate": 65.0}
    states = sc.classify(_metrics_rows(spec))
    assert states[-1].phase == states[-2].phase  # 1 天不足以切换


def test_illegal_transition_rejected():
    """冰点不能直接进入发酵。"""
    spec = [ICE_ROW] * 3
    # 发酵条件全部满足, 但冰点 → 发酵不在转移表
    spec += [{"limit_up": 80, "max_height": 8, "seal_rate": 85.0,
              "promo_1_2": 40.0, "premium_multi": 0.05, "premium_lu": 0.03}] * 3
    states = sc.classify(_metrics_rows(spec))
    assert "发酵" not in [s.phase for s in states[3:]]


def test_config_override_changes_confirm_days():
    spec = [ICE_ROW] * 6 + [
        dict(ICE_ROW),
        {"limit_up": 25, "max_height": 3, "premium_lu": 0.01, "big_loss": 5,
         "seal_rate": 65.0, "up_ratio": 0.5, "first_board": 45},
    ]
    cfg = {"confirm_days": 1}
    states = sc.classify(_metrics_rows(spec), cfg)
    assert states[-1].phase == "修复"


def test_merge_config_clamps_and_ignores_unknown():
    cfg = sc.merge_config({
        "lookback": 1, "confirm_days": 99, "big_loss_pct": "-5",
        "phases": {"不存在": {"conditions": []}, "冰点": {"conditions": [
            {"key": "limit_up", "cmp": "bogus", "value": 1, "weight": 2},
        ]}},
    })
    assert cfg["lookback"] == 2
    assert cfg["confirm_days"] == 99
    assert cfg["big_loss_pct"] == -5.0
    assert "不存在" not in cfg["phases"]
    cond = cfg["phases"]["冰点"]["conditions"][0]
    assert cond["cmp"] == "pct_lte"  # 非法运算符保留出厂值
    assert cond["weight"] == 2


def test_public_metrics_percent_units():
    spec = [{"premium_lu": 0.035, "premium_multi": -0.012}]
    states = sc.classify(_metrics_rows(spec))
    assert states[0].metrics["premium_lu"] == pytest.approx(3.5)
    assert states[0].metrics["premium_multi"] == pytest.approx(-1.2)


def test_recap_block_renders_phase():
    states = sc.classify(_metrics_rows([{}] * 5))
    payload = {**states[-1].to_dict(), "history": []}
    text = sc.recap_block(payload)
    assert "当前阶段" in text
    assert states[-1].phase in text
    assert sc.recap_block(None) == ""


# ---------- API ----------

class _StubRepo:
    def __init__(self, df, latest):
        self._df = df
        self._latest = latest

    def get_enriched_latest(self):
        return self._df, self._latest

    def get_enriched_history(self, target, lookback):
        return self._df

    def get_enriched_history_span(self):
        return (self._df["date"].min(), self._df["date"].max())


def _stub_df(days: int = 20) -> pl.DataFrame:
    rows = []
    for d in _days(days):
        for i in range(6):
            rows.append({
                "symbol": f"S{i}", "date": d, "name": "测试股",
                "change_pct": 0.02 if i < 2 else -0.01,
                "raw_close": 11.0 if i < 2 else 10.0,
                "raw_high": 11.0 if i < 2 else 10.0,
                "raw_low": 11.0 if i < 2 else 10.0,
                "turnover_rate": 3.0,
                "consecutive_limit_ups": 1 if i < 2 else 0,
                "limit_up_price": 11.0,
                "limit_down_price": 9.0,
                "close": 11.0 if i < 2 else 10.0,
                "float_shares": 1e9 if i == 0 else (1e7 if i >= 4 else 5e8),
                "ma20": 10.5 if i < 2 else 10.0,
            })
    return pl.DataFrame(rows).cast({"date": pl.Date})


def test_api_returns_cycle_payload(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import sentiment_cycle as api
    from app.services import preferences

    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr(preferences, "_path", lambda: tmp_path / "preferences.json")
    sc.invalidate_cache()
    df = _stub_df()
    app = FastAPI()
    app.state.repo = _StubRepo(df, df["date"].max())
    app.include_router(api.router)
    client = TestClient(app)

    payload = client.get("/api/sentiment-cycle").json()
    assert payload["available"] is True
    assert payload["phase"] in sc.PHASES
    assert len(payload["history"]) <= 20
    assert "metrics" in payload

    style = client.get("/api/sentiment-cycle/style").json()
    assert style["available"] is True
    assert style["style"] in {"题材主导", "机构主导", "普涨", "普跌", "切换中", "均衡"}


def test_api_config_roundtrip(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import sentiment_cycle as api
    from app.services import preferences

    # 恢复出厂分支会直接写盘 (不走 save): 必须把 _path 指到临时目录,
    # 否则会覆盖真实的 data/user_data/preferences.json
    store: dict = {}
    monkeypatch.setattr(preferences, "_path", lambda: tmp_path / "preferences.json")
    monkeypatch.setattr(preferences, "load", lambda: dict(store))
    monkeypatch.setattr(preferences, "save", lambda upd: store.update(upd))

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)

    payload = client.get("/api/sentiment-cycle/config").json()
    assert payload["customized"] is False
    assert payload["config"]["confirm_days"] == 2

    payload = client.put("/api/sentiment-cycle/config", json={"config": {"confirm_days": 3}}).json()
    assert payload["customized"] is True
    assert payload["config"]["confirm_days"] == 3

    payload = client.put("/api/sentiment-cycle/config", json={"config": None}).json()
    assert payload["customized"] is False


# ---------- 回归: review 修复项 ----------

def _monotonic_metrics(n: int = 71) -> pl.DataFrame:
    df = pl.DataFrame({"date": _days(n)}).with_columns(pl.col("date").cast(pl.Date))
    for k in sc._METRIC_KEYS:
        df = df.with_columns(pl.Series(k, [float(i) for i in range(n)]))
    return df


def test_rolling_percentile_has_no_lookahead():
    """改写未来数据不得改变历史日期的分位 (CONTRIBUTING §5.3)。"""
    base = _monotonic_metrics()
    altered = base.with_columns(
        pl.when(pl.int_range(pl.len()) > 40).then(-100.0)
        .otherwise(pl.col("limit_up")).alias("limit_up")
    )
    a = sc._add_derived(base, 60)["_pct_limit_up"].to_list()
    b = sc._add_derived(altered, 60)["_pct_limit_up"].to_list()
    assert a[:41] == b[:41]


def test_rolling_percentile_not_saturated():
    """旧实现窗口 71 行除以 60, 单调序列有 15 行被截到 >=0.95。"""
    pcts = sc._add_derived(_monotonic_metrics(), 60)["_pct_limit_up"].to_list()
    # 单调递增: 每天都是窗口新高 → 分位恒为 1.0, 但早期样本不足为 None
    assert pcts[: sc.MIN_HISTORY_DAYS - 1] == [None] * (sc.MIN_HISTORY_DAYS - 1)
    assert all(p == pytest.approx(1.0) for p in pcts[sc.MIN_HISTORY_DAYS:])

    import random
    random.seed(7)
    noisy = _monotonic_metrics().with_columns(
        pl.Series("limit_up", [random.random() for _ in range(71)])
    )
    vals = [p for p in sc._add_derived(noisy, 60)["_pct_limit_up"].to_list() if p is not None]
    assert sum(p >= 0.95 for p in vals) <= 6   # 随机序列的极值应很少


def test_height_broken_requires_prior_multi_board():
    """前一日无连板股时不能判定断板 (旧实现恒为 1)。"""
    rows = []
    for d in _days(2):
        for sym in ("A", "B"):
            rows.append({"symbol": sym, "date": d, "close": 10.0, "high": 10.2,
                         "low": 9.8, "pct": -0.01, "boards": 0})
    m = _metrics_from(rows).to_dicts()
    assert [r["height_broken"] for r in m] == [0, 0]

    # 昨日 3 板, 今日未封 → 断板
    d1, d2 = _days(2)
    rows = [
        {"symbol": "A", "date": d1, "close": 13.31, "limit_up": 13.31, "pct": 0.10, "boards": 3},
        {"symbol": "A", "date": d2, "close": 13.0, "pct": -0.02, "boards": 0},
    ]
    assert _metrics_from(rows).to_dicts()[-1]["height_broken"] == 1


def test_extreme_flag_rare_on_noise():
    import random
    random.seed(3)
    spec = [
        {"limit_up": random.randint(20, 80), "max_height": random.randint(2, 8),
         "premium_lu": random.uniform(-0.02, 0.04), "seal_rate": random.uniform(55, 85),
         "big_loss": random.randint(0, 15), "limit_down": random.randint(0, 20),
         "high_nuke": random.randint(0, 5)}
        for _ in range(70)
    ]
    states = sc.classify(_metrics_rows(spec))
    flagged = sum(1 for s in states if s.extreme)
    assert flagged <= 14, f"极致预警过多: {flagged}/70"


# ---------- 板块周期 / 风格联动 ----------

def test_missing_limit_columns_returns_empty():
    """instruments 未加载时 enriched 缓存无连板列: 必须返回空而不是抛错或近似。"""
    df = pl.DataFrame({
        "symbol": ["A"], "date": [_days(1)[0]], "change_pct": [0.01],
        "raw_close": [10.0], "raw_high": [10.0], "raw_low": [10.0],
    })
    assert sc.compute_daily_metrics(df).is_empty()


def test_avg_pct_metric_and_percent_units():
    rows = [
        {"symbol": "A", "date": _days(1)[0], "close": 10.0, "pct": 0.02},
        {"symbol": "B", "date": _days(1)[0], "close": 10.0, "pct": -0.01},
    ]
    m = _metrics_from(rows).to_dicts()[0]
    assert m["avg_pct"] == pytest.approx(0.005)
    states = sc.classify(_metrics_rows([{"avg_pct": 0.012}]))
    assert states[0].metrics["avg_pct"] == pytest.approx(1.2)


def test_style_linkage_lowers_retreat_threshold():
    """机构主导时退潮门槛下调: 原本差一点才达标的退潮在最新一日成为候选。"""
    cfg = {"confirm_days": 1, "score_threshold": 0.6, "style_retreat_discount": 0.15}
    # 先进入分歧
    spec = [{}] * 12 + [{
        "max_height": 5, "seal_rate_drop": 15.0, "premium_lu": -0.01, "broken": 40,
    }] * 3
    # 最后一天: 退潮四个条件只满足前两个 (1.2 + 1.0) / 3.8 ≈ 0.58 < 0.6
    spec.append({"height_broken": 1, "premium_multi": -0.01, "big_loss": 3, "high_nuke": 0})
    neutral = sc.classify(_metrics_rows(spec), cfg)
    linked = sc.classify(_metrics_rows(spec), cfg, style="机构主导")
    assert neutral[-1].phase != "退潮"
    assert linked[-1].phase == "退潮"
    # 联动只作用于最新一日, 不追溯历史
    assert [s.phase for s in neutral[:-1]] == [s.phase for s in linked[:-1]]


def test_sector_rules_are_separate_and_mergeable():
    cfg = sc.merge_config({"sector_phases": {"冰点": {"conditions": [
        {"key": "up_ratio", "value": 0.3},
    ]}}})
    sector_cond = {c["key"]: c for c in cfg["sector_phases"]["冰点"]["conditions"]}
    assert sector_cond["up_ratio"]["value"] == 0.3
    # 全市场规则不受影响
    market_keys = {c["key"] for c in cfg["phases"]["冰点"]["conditions"]}
    assert "up_ratio" not in market_keys


def _sector_frame(n_days: int, members: list[str], hot_last_day: bool) -> pl.DataFrame:
    rows = []
    for di, d in enumerate(_days(n_days)):
        last = di == n_days - 1
        for i, sym in enumerate(members):
            pct = 0.0 + 0.002 * ((di + i) % 5 - 2)
            boards = 0
            close = 10.0
            if last and hot_last_day:
                pct = 0.06
                if i < len(members) // 3:
                    pct, boards, close = 0.10, 1, 11.0
            rows.append({
                "symbol": sym, "date": d, "name": "成分股", "change_pct": pct,
                "raw_close": close, "raw_high": close, "raw_low": close * 0.97,
                "turnover_rate": 3.0, "consecutive_limit_ups": boards,
                "limit_up_price": 11.0, "limit_down_price": 9.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_sector_states_skip_small_sectors_and_use_sector_rules():
    members = [f"{i:06d}.SZ" for i in range(15)]
    small = [f"{i:06d}.SH" for i in range(5)]
    df = pl.concat([_sector_frame(40, members, True), _sector_frame(40, small, True)])
    cfg = sc.merge_config({"confirm_days": 1})
    out = sc._sector_states(df, {"热门": set(members), "小板块": set(small)}, ["热门", "小板块"], cfg)
    assert "小板块" not in out          # 成分股少于 SECTOR_MIN_MEMBERS 不计算
    hot = out["热门"]
    assert hot["members"] == 15
    assert hot["limit_up"] == 5
    assert hot["phase"] in {"修复", "发酵", "高潮"}   # 全面大涨日不应被判为冰点/退潮


def test_sector_states_match_bare_codes():
    members = [f"{i:06d}.SZ" for i in range(12)]
    df = _sector_frame(30, members, False)
    bare_keys = {m.split(".")[0] for m in members}
    out = sc._sector_states(df, {"裸代码板块": bare_keys}, ["裸代码板块"], sc.merge_config(None))
    assert out["裸代码板块"]["members"] == 12


def test_get_sector_cycles_without_memberships(monkeypatch):
    from app.services import market_overview_builder as mob

    sc.invalidate_cache()
    monkeypatch.setattr(mob, "sector_member_map", lambda repo, kind, level=None: {})
    assert sc.get_sector_cycles(object()) == {}


def test_sector_recap_block_format():
    text = sc.sector_recap_block({
        "固态电池": {"phase": "发酵", "pending": "高潮", "days_in_phase": 3, "members": 40,
                  "limit_up": 8, "max_height": 4, "avg_pct": 3.21},
    })
    assert "固态电池: 发酵" in text
    assert "候选 高潮" in text
    assert "平均 +3.21%" in text
    assert sc.sector_recap_block({}) == ""


def test_api_sector_endpoint_degrades(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import sentiment_cycle as api
    from app.services import preferences

    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr(sc, "get_sector_cycles", lambda *a, **k: {})
    app = FastAPI()
    app.state.repo = object()
    app.include_router(api.router)
    client = TestClient(app)
    payload = client.get("/api/sentiment-cycle/sectors").json()
    assert payload["available"] is False and payload["sectors"] == []
    assert client.get("/api/sentiment-cycle/sectors?kind=bogus").status_code == 422
