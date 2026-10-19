"""TushareProvider 契约与单位标准化测试。

不依赖真实网络与 token: 注入假 client(按 api_name 返回预置帧), 验证单位口径
(amount 千元→元、分钟 vol 股→手)、日期解析、资产类型路由、累积复权因子→单事件比值
(经 pipeline._apply_adj_factor 回放: 最新价不变 + 除权日前后连续)、财务列映射与
合并报表过滤、权限裁剪(仅 40203 裁剪, 网络错误保留)与缓存(TTL / token 指纹)、
client 错误分类 / 限频重试 / 分页终止 / token 不外泄、availability / probe / loader 注册。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime

import polars as pl
import pytest

from app.indicators.pipeline import _apply_adj_factor
from app.plugins.tushare import client as tc
from app.plugins.tushare import provider as tp
from app.plugins.tushare.provider import TushareProvider

FAKE_TOKEN = "fake-token-for-tests-0123456789"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """所有测试: 缓存目录指向 tmp, token 固定为假值, 不读真实 secrets/.env。"""
    monkeypatch.setattr(tp, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(tp, "get_api_key", lambda: FAKE_TOKEN)


class _FakeClient:
    """按 api_name 分派: handler(params) → DataFrame 或抛异常。记录 (api, params)。"""

    def __init__(self, handlers: dict | None = None):
        self.handlers = handlers or {}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def _run(self, api, params):
        self.calls.append((api, dict(params or {})))
        h = self.handlers.get(api)
        if h is None:
            return pl.DataFrame()
        out = h(dict(params or {})) if callable(h) else h
        if isinstance(out, Exception):
            raise out
        return out

    def query(self, api, params=None, fields=""):
        return self._run(api, params)

    def query_all(self, api, params=None, fields="", page_size=None, max_pages=200):
        return self._run(api, params)

    def close(self):
        self.closed = True


def _provider(handlers=None) -> tuple[TushareProvider, _FakeClient]:
    fake = _FakeClient(handlers)
    return TushareProvider(client=fake, auto_probe=False), fake


def _daily_rows(code: str, days: list[str], close: float = 10.0) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": [code] * len(days),
        "trade_date": days,
        "open": [close] * len(days), "high": [close + 1] * len(days),
        "low": [close - 1] * len(days), "close": [close] * len(days),
        "pre_close": [close] * len(days), "change": [0.0] * len(days),
        "pct_chg": [-0.8737] * len(days),
        "vol": [1234.0] * len(days),  # 手
        "amount": [5678.9] * len(days),  # 千元
    })


# ---- 1. daily ----


def test_daily_units_amount_thousand_to_yuan_and_date_parsed():
    provider, fake = _provider({"daily": _daily_rows("600519.SH", ["20261009", "20261010"])})
    df = provider.get_daily(["600519.SH"], datetime(2026, 9, 1), datetime(2026, 10, 10))
    assert df.columns[:8] == ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    assert df["date"].dtype == pl.Date
    assert df["date"].to_list() == [date(2026, 10, 9), date(2026, 10, 10)]
    assert df["volume"].to_list() == [1234.0, 1234.0]  # 已是手, 不再 /100
    assert df["amount"][0] == pytest.approx(5_678_900.0)  # 千元 × 1000
    # 单标的长窗口 → 按 ts_code + 区间拉
    api, params = fake.calls[0]
    assert api == "daily" and params["ts_code"] == "600519.SH"
    assert params["start_date"] == "20260901" and params["end_date"] == "20261010"


@pytest.mark.parametrize(("asset", "api"), [("etf", "fund_daily"), ("index", "index_daily")])
def test_daily_routes_etf_and_index_to_own_api(asset, api):
    provider, fake = _provider({api: _daily_rows("510300.SH", ["20261010"], close=4.0)})
    df = provider.get_daily(["510300.SH"], datetime(2026, 9, 1), datetime(2026, 10, 10), asset_type=asset)
    assert df.height == 1 and df["amount"][0] == pytest.approx(5_678_900.0)
    assert {c[0] for c in fake.calls} == {api}


def test_daily_whole_market_pulls_by_trade_date_and_filters_symbols():
    """标的数 > 工作日数 → 逐日全市场拉取, 每日 1 请求; 只保留请求的标的。"""
    syms = [f"{i:06d}.SZ" for i in range(1, 11)]

    def handler(params):
        return _daily_rows(syms[0], [params["trade_date"]]).vstack(
            _daily_rows("999999.SZ", [params["trade_date"]])
        )

    provider, fake = _provider({"daily": handler})
    progress = []
    df = provider.get_daily(
        syms, datetime(2026, 10, 5), datetime(2026, 10, 9),
        on_chunk_done=lambda c, t: progress.append((c, t)),
    )
    assert [p["trade_date"] for _, p in fake.calls] == [
        "20261005", "20261006", "20261007", "20261008", "20261009"
    ]
    assert set(df["symbol"].to_list()) == {syms[0]}
    assert progress[-1] == (5, 5)


def test_daily_index_never_uses_trade_date_mode():
    syms = [f"00000{i}.SH" for i in range(1, 9)]
    provider, fake = _provider({"index_daily": pl.DataFrame()})
    provider.get_daily(syms, datetime(2026, 10, 8), datetime(2026, 10, 9), asset_type="index")
    assert all("ts_code" in p and "trade_date" not in p for _, p in fake.calls)


def test_daily_per_symbol_soft_fail_but_permission_raises():
    def handler(params):
        if params["ts_code"] == "000002.SZ":
            return tc.TushareError("网络请求失败")
        return _daily_rows(params["ts_code"], ["20261009"])

    provider, _ = _provider({"daily": handler})
    df = provider.get_daily(["000001.SZ", "000002.SZ"], datetime(2026, 1, 1), datetime(2026, 10, 9))
    assert df["symbol"].to_list() == ["000001.SZ"]

    provider, _ = _provider({"daily": tc.TusharePermissionError("code=40203")})
    with pytest.raises(tc.TusharePermissionError):
        provider.get_daily(["000001.SZ"], datetime(2026, 1, 1), datetime(2026, 10, 9))


def test_daily_schema_change_returns_empty_not_fake_data():
    provider, _ = _provider({"daily": pl.DataFrame({"ts_code": ["000001.SZ"], "foo": [1]})})
    assert provider.get_daily(["000001.SZ"], datetime(2026, 1, 1), datetime(2026, 10, 9)).is_empty()


def test_daily_empty_symbols_or_unknown_asset():
    provider, fake = _provider()
    assert provider.get_daily([], None, None).is_empty()
    assert provider.get_daily(["x"], None, None, asset_type="bond").is_empty()
    assert fake.calls == []


# ---- 2. adj_factor: 累积 → 单事件比值 ----


def _adj_rows(code: str, pairs: list[tuple[str, float]]) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": [code] * len(pairs),
        "trade_date": [d for d, _ in pairs],
        "adj_factor": [a for _, a in pairs],
    })


def test_adj_no_event_returns_empty_schema():
    provider, _ = _provider({"adj_factor": _adj_rows("600519.SH", [
        ("20261005", 8.0), ("20261006", 8.0), ("20261007", 8.0),
    ])})
    df = provider.get_adj_factors(["600519.SH"], None, datetime(2026, 10, 7))
    assert df.is_empty()
    assert df.schema == {"symbol": pl.String, "trade_date": pl.Date, "ex_factor": pl.Float64}


def test_adj_event_ratio_round_trips_through_pipeline():
    """核心: 输出喂给 pipeline._apply_adj_factor 后最新价不变、除权日前后连续。

    样例: 除权前收盘 10.00, 每股分红 0.5 → 除权参考价 9.50。Tushare 后复权因子
    在除权日由 a 升为 a × 10/9.5; 本插件输出 ex_factor = 10/9.5 (>1)。
    """
    a0 = 1.2345
    a1 = a0 * 10.0 / 9.5
    provider, _ = _provider({"adj_factor": _adj_rows("600519.SH", [
        ("20261005", a0), ("20261006", a0), ("20261007", a1), ("20261008", a1),
    ])})
    factors = provider.get_adj_factors(["600519.SH"], None, datetime(2026, 10, 8))
    assert factors["trade_date"].to_list() == [date(2026, 10, 7)]
    assert factors["ex_factor"][0] == pytest.approx(10.0 / 9.5)

    raw = pl.DataFrame({
        "symbol": ["600519.SH"] * 4,
        "date": [date(2026, 10, d) for d in (5, 6, 7, 8)],
        "open": [10.0, 10.0, 9.5, 9.6], "high": [10.0, 10.0, 9.5, 9.6],
        "low": [10.0, 10.0, 9.5, 9.6], "close": [10.0, 10.0, 9.5, 9.6],
    })
    adj = _apply_adj_factor(raw, factors).sort("date")
    closes = adj["close"].to_list()
    assert closes[-1] == pytest.approx(9.6)  # 前复权: 最新价不变
    assert closes[1] == pytest.approx(9.5)  # 除权前一日 10.00 → 9.50, 与除权日参考价连续
    assert closes[2] == pytest.approx(9.5)
    assert adj.filter(pl.col("date") == date(2026, 10, 7))["ex_rights"][0] is True


def test_adj_multiple_symbols_and_events_window_filtered():
    rows = _adj_rows("000001.SZ", [
        ("20260901", 1.0), ("20260915", 1.1), ("20261001", 1.1), ("20261008", 1.21),
    ]).vstack(_adj_rows("000002.SZ", [("20261001", 2.0), ("20261008", 3.0)]))
    provider, fake = _provider({"adj_factor": rows})
    df = provider.get_adj_factors(
        ["000001.SZ", "000002.SZ"], datetime(2026, 10, 1), datetime(2026, 10, 9)
    )
    # 2026-09-15 事件在窗口外 → 过滤; 窗口起点前 30 天回看保证 10-01 起的事件有前值可比
    assert df.sort(["symbol", "trade_date"]).rows() == [
        ("000001.SZ", date(2026, 10, 8), pytest.approx(1.1)),
        ("000002.SZ", date(2026, 10, 8), pytest.approx(1.5)),
    ]
    assert fake.calls[0][1]["start_date"] == "20260901"  # 10-01 − 30 天


def test_adj_first_row_is_never_an_event():
    """上市首日/窗口首条无前值, 不产生事件(不能把首个累积值当比值)。"""
    out = tp.cumulative_to_events(_adj_rows("x.SZ", [("20261008", 3.7)]))
    assert out.is_empty()


def test_adj_etf_uses_fund_adj_and_index_returns_empty():
    provider, fake = _provider({"fund_adj": _adj_rows("510300.SH", [
        ("20261005", 1.0), ("20261006", 1.02),
    ])})
    df = provider.get_adj_factors(["510300.SH"], None, datetime(2026, 10, 6), asset_type="etf")
    assert df["ex_factor"][0] == pytest.approx(1.02)
    assert fake.calls[0][0] == "fund_adj"
    assert provider.get_adj_factors(["000001.SH"], None, None, asset_type="index").is_empty()


def test_adj_by_date_mode_failure_raises_fail_closed():
    """逐日全市场模式缺一天会把跨缺口比值记到错误日期 → 抛出而非静默。"""
    syms = [f"{i:06d}.SZ" for i in range(1, 200)]
    provider, _ = _provider({"adj_factor": tc.TushareError("网络请求失败")})
    with pytest.raises(tc.TushareError):
        provider.get_adj_factors(syms, datetime(2026, 10, 1), datetime(2026, 10, 9))


# ---- 3. minute ----


def _mins_rows(code: str) -> pl.DataFrame:
    # stk_mins 默认时间倒序
    return pl.DataFrame({
        "ts_code": [code, code],
        "trade_time": ["2026-10-09 09:32:00", "2026-10-09 09:31:00"],
        "close": [10.1, 10.0], "open": [10.0, 9.9], "high": [10.2, 10.0], "low": [9.9, 9.8],
        "vol": [42600.0, 199.0],  # 股
        "amount": [430260.0, 1990.0],  # 元
    })


def test_minute_volume_shares_to_lots_and_naive_beijing_datetime():
    provider, fake = _provider({"stk_mins": lambda p: _mins_rows(p["ts_code"])})
    df = provider.get_minute(["600519.SH"], datetime(2026, 10, 9, 9, 30), datetime(2026, 10, 9, 15, 0))
    assert df.columns == ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
    assert df["datetime"].dtype == pl.Datetime("us") and df["datetime"].dtype.time_zone is None
    assert df["datetime"].to_list() == [datetime(2026, 10, 9, 9, 31), datetime(2026, 10, 9, 9, 32)]
    assert df["volume"].to_list() == [1.0, 426.0]  # floor(股/100)
    assert df["amount"].to_list() == [1990.0, 430260.0]  # 元, 透传
    params = fake.calls[0][1]
    assert params["freq"] == "1min"
    assert params["start_date"] == "2026-10-09 09:30:00" and params["end_date"] == "2026-10-09 15:00:00"


def test_minute_non_stock_raises_for_tickflow_fallback():
    provider, _ = _provider()
    with pytest.raises(tc.TushareError):
        provider.get_minute(["510300.SH"], None, None, asset_type="etf")
    with pytest.raises(tc.TushareError):
        provider.get_minute(["600519.SH"], None, None, freq="1d")


def test_minute_empty_returns_canonical_schema():
    provider, _ = _provider({"stk_mins": pl.DataFrame()})
    df = provider.get_minute(["600519.SH"], None, None)
    assert df.is_empty() and "datetime" in df.columns


# ---- 4. financial ----


def _income_rows() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": ["600519.SH"] * 4,
        "ann_date": ["20261020", "20261021", "20260801", "20260801"],
        "f_ann_date": ["20261020", "20261021", "20260801", "20260801"],
        "end_date": ["20260930", "20260930", "20260630", "20260630"],
        "report_type": ["1", "1", "1", "2"],  # 2 = 单季合并, 应被过滤
        "update_flag": ["0", "1", "1", "1"],
        "total_revenue": [100.0, 101.0, 60.0, 999.0],
        "n_income_attr_p": [50.0, 51.0, 30.0, 999.0],
        "basic_eps": [1.5, 1.51, 0.9, 9.9],
        "ebit": [70.0, 71.0, 40.0, 999.0],  # 映射之外 → 原名透传
    })


def test_financial_income_mapping_consolidated_latest_revision():
    provider, fake = _provider({"income": _income_rows()})
    df = provider.get_financials("income", ["600519.SH"], latest_only=False)
    assert df["period_end"].to_list() == ["2026-09-30", "2026-06-30"]
    q3 = df.row(0, named=True)
    assert q3["revenue"] == 101.0  # 同报告期取最新修订 (update_flag=1)
    assert q3["net_income_attributable"] == 51.0
    assert q3["announce_date"] == "2026-10-21"
    assert q3["ebit"] == 71.0
    assert 999.0 not in df["revenue"].to_list()  # 非合并报表 (report_type=2) 不混入
    assert fake.calls[0][0] == "income"


def test_financial_latest_only_single_period():
    provider, _ = _provider({"income": _income_rows()})
    df = provider.get_financials("income", ["600519.SH"], latest_only=True)
    assert df.height == 1 and df["period_end"][0] == "2026-09-30"


def test_financial_metrics_percent_points_and_roe_not_overwritten():
    raw = pl.DataFrame({
        "ts_code": ["600519.SH"], "ann_date": ["20261021"], "end_date": ["20260930"],
        "eps": [1.5], "bps": [180.0], "roe": [25.0], "roe_waa": [26.5],
        "grossprofit_margin": [91.2], "debt_to_assets": [18.3], "or_yoy": [12.0],
    })
    provider, fake = _provider({"fina_indicator": raw})
    row = provider.get_financials("metrics", ["600519.SH"]).row(0, named=True)
    assert fake.calls[0][0] == "fina_indicator"
    assert row["roe"] == 26.5  # 加权 ROE; 原始 roe 列不得覆盖 canonical roe
    assert row["gross_margin"] == 91.2  # 百分点数值原样透传(与 fuyao / 前端口径一致)
    assert row["debt_to_asset_ratio"] == 18.3
    assert row["eps_basic"] == 1.5 and row["bps"] == 180.0
    assert row["roa"] is None  # 缺失字段置 None, 不伪造


@pytest.mark.parametrize(("table", "api"), [("balance_sheet", "balancesheet"), ("cash_flow", "cashflow")])
def test_financial_balance_and_cashflow_routing(table, api):
    raw = pl.DataFrame({
        "ts_code": ["000001.SZ"], "ann_date": ["20261021"], "end_date": ["20260930"],
        "report_type": ["1"], "total_assets": [1e9], "n_cashflow_act": [2e8],
    })
    provider, fake = _provider({api: raw})
    row = provider.get_financials(table, ["000001.SZ"]).row(0, named=True)
    assert fake.calls[0][0] == api
    if table == "balance_sheet":
        assert row["total_assets"] == 1e9
    else:
        assert row["net_operating_cash_flow"] == 2e8


def test_financial_shares_empty_and_soft_fail_per_symbol():
    provider, fake = _provider({"income": lambda p: (
        tc.TushareError("网络") if p["ts_code"] == "BAD.SZ" else _income_rows()
    )})
    assert provider.get_financials("shares", ["600519.SH"]).is_empty()
    assert fake.calls == []
    df = provider.get_financials("income", ["BAD.SZ", "600519.SH"])
    assert df["symbol"].to_list() == ["600519.SH"]


# ---- instruments ----


def test_instruments_shape_and_share_units():
    basic = pl.DataFrame({
        "ts_code": ["600519.SH", "430047.BJ"], "symbol": ["600519", "430047"],
        "name": ["贵州茅台", "诺思兰德"], "exchange": ["SSE", "BSE"],
        "list_date": ["20010827", "20200727"], "list_status": ["L", "L"],
    })
    provider, fake = _provider({
        "stock_basic": lambda p: basic if p["list_status"] == "L" else pl.DataFrame(),
        "trade_cal": pl.DataFrame({"cal_date": ["20261009", "20261008"]}),
        "daily_basic": pl.DataFrame({
            "ts_code": ["600519.SH"], "total_share": [125619.78], "float_share": [125619.78],
        }),
    })
    rows = provider.get_instruments("stock")
    assert [r["symbol"] for r in rows] == ["600519.SH", "430047.BJ"]
    mt = rows[0]
    assert mt["code"] == "600519" and mt["exchange"] == "SH" and mt["type"] == "stock"
    assert mt["ext"]["listing_date"] == "2001-08-27"
    assert mt["ext"]["total_shares"] == pytest.approx(1_256_197_800.0)  # 万股 → 股
    assert rows[1]["ext"]["float_shares"] is None  # 无股本行 → None, 不伪造
    assert {p["list_status"] for a, p in fake.calls if a == "stock_basic"} == {"L", "P"}
    assert [p.get("trade_date") for a, p in fake.calls if a == "daily_basic"] == ["20261009"]

    from app.services.instrument_sync import _flatten_instruments

    flat = _flatten_instruments(rows)[0]
    assert flat["listing_date"] == "2001-08-27" and flat["total_shares"] == pytest.approx(1_256_197_800.0)
    assert provider.get_instruments("etf") == []


def test_instruments_failure_soft_returns_empty():
    provider, _ = _provider({"stock_basic": tc.TushareError("网络")})
    assert provider.get_instruments() == []


# ---- 5. 权限裁剪与能力缓存 ----


def _probe_handlers(**over):
    base = {"daily": pl.DataFrame(), "adj_factor": pl.DataFrame(),
            "stk_mins": pl.DataFrame(), "income": pl.DataFrame()}
    base.update(over)
    return base


def test_init_sends_no_request_and_declares_full_set():
    provider, fake = _provider(_probe_handlers())
    assert set(provider.config.datasets) == {"daily", "adj_factor", "minute", "financial"}
    assert "realtime" not in provider.config.datasets  # rt_k 未实现, 不声明
    assert fake.calls == []


def test_permission_error_trims_dataset_network_error_keeps_it(tmp_path):
    provider, fake = _provider(_probe_handlers(
        stk_mins=tc.TusharePermissionError("code=40203 没有接口访问权限"),
        income=tc.TushareError("网络请求失败: timeout"),
    ))
    statuses = provider.refresh_capabilities()
    assert statuses == {"daily": "ok", "adj_factor": "ok", "minute": "denied", "financial": "unknown"}
    assert "minute" not in provider.config.datasets
    assert "financial" in provider.config.datasets  # 网络错误 ≠ 无权限
    assert [a for a, _ in fake.calls] == ["daily", "adj_factor", "stk_mins", "income"]

    cached = json.loads((tmp_path / "capabilities.json").read_text(encoding="utf-8"))
    assert cached["fingerprint"] == tp._fingerprint(FAKE_TOKEN)
    assert FAKE_TOKEN not in json.dumps(cached)  # 缓存里只有指纹, 无 token


def test_rate_limit_during_probe_is_not_permission_denied():
    provider, _ = _provider(_probe_handlers(daily=tc.classify_error(40203, "抱歉，您每分钟最多访问该接口200次")))
    assert provider.refresh_capabilities()["daily"] == "unknown"
    assert "daily" in provider.config.datasets


def test_unknown_does_not_override_previous_definitive_result():
    provider, fake = _provider(_probe_handlers(stk_mins=tc.TusharePermissionError("40203")))
    provider.refresh_capabilities()
    fake.handlers["stk_mins"] = tc.TushareError("网络抖动")
    assert provider.refresh_capabilities()["minute"] == "denied"
    assert "minute" not in provider.config.datasets


def test_new_provider_loads_trimmed_cache_without_requests():
    provider, _ = _provider(_probe_handlers(stk_mins=tc.TusharePermissionError("40203")))
    provider.refresh_capabilities()
    fresh, fake = _provider(_probe_handlers())
    assert "minute" not in fresh.config.datasets
    assert fake.calls == []


def test_cache_invalidated_by_ttl_and_token_change(monkeypatch, tmp_path):
    provider, _ = _provider(_probe_handlers(stk_mins=tc.TusharePermissionError("40203")))
    provider.refresh_capabilities()

    # token 变更 → 指纹不符 → 缓存失效, 回到全集声明
    monkeypatch.setattr(tp, "get_api_key", lambda: "another-fake-token")
    assert "minute" in _provider()[0].config.datasets

    # TTL 过期 → 失效
    monkeypatch.setattr(tp, "get_api_key", lambda: FAKE_TOKEN)
    path = tmp_path / "capabilities.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["probed_at"] = time.time() - tp._CAPS_TTL_S - 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert "minute" in _provider()[0].config.datasets

    # 损坏 → 失效且不抛
    path.write_text("{not json", encoding="utf-8")
    assert "minute" in _provider()[0].config.datasets


def test_datasets_access_schedules_background_probe_once(monkeypatch):
    """首次读 datasets 起后台探测(不阻塞), 完成后裁剪生效; 有效缓存期内不重复探测。"""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)  # 注入了假 client, 允许探测
    fake = _FakeClient(_probe_handlers(stk_mins=tc.TusharePermissionError("40203")))
    provider = TushareProvider(client=fake, auto_probe=True)
    assert fake.calls == []
    _ = provider.config.datasets
    for _ in range(200):
        if "minute" not in provider._datasets and not provider._probing:
            break
        time.sleep(0.01)
    assert "minute" not in provider.config.datasets
    n = len(fake.calls)
    _ = provider.config.datasets
    time.sleep(0.05)
    assert len(fake.calls) == n == 4


def test_auto_probe_disabled_without_injected_client_under_pytest(monkeypatch):
    """测试进程里未注入 client 的 provider(loader 注册的实例)不得后台触网。"""
    started = []
    monkeypatch.setattr(tp.threading, "Thread", lambda *a, **k: started.append(1))
    provider = TushareProvider()
    _ = provider.config.datasets
    assert started == []


def test_test_dataset_preview_and_undeclared():
    provider, _ = _provider({"daily": _daily_rows("000001.SZ", ["20261009"])})
    out = provider.test_dataset("daily", ["000001.SZ"])
    assert out["rows"] == 1 and out["preview"][0]["date"] == "2026-10-09"
    assert "error" in provider.test_dataset("realtime")


# ---- 6. client ----


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Http:
    """按顺序返回预置响应, 记录请求体。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def post(self, url, json=None):
        self.bodies.append(json)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    def close(self):
        pass


def _ok(fields, items, **extra):
    return _Resp({"code": 0, "msg": "", "data": {"fields": fields, "items": items, **extra}})


def _client(monkeypatch, responses, sleeps=None):
    http = _Http(responses)
    monkeypatch.setattr(tc.httpx, "Client", lambda **kw: http)
    limiter = tc.RateLimiter(rpm=10_000, sleep=lambda s: None)
    sleep = (lambda s: sleeps.append(s)) if sleeps is not None else (lambda s: None)
    return tc.TushareClient(token=FAKE_TOKEN, limiter=limiter, sleep=sleep), http


def test_client_unpacks_envelope_and_sends_body(monkeypatch):
    client, http = _client(monkeypatch, [_ok(["ts_code", "close"], [["600519.SH", 1480.0]])])
    df = client.query("daily", {"ts_code": "600519.SH"}, "ts_code,close")
    assert df.rows() == [("600519.SH", 1480.0)]
    assert http.bodies[0] == {
        "api_name": "daily", "token": FAKE_TOKEN,
        "params": {"ts_code": "600519.SH"}, "fields": "ts_code,close",
    }


def test_client_empty_items_keeps_columns(monkeypatch):
    client, _ = _client(monkeypatch, [_ok(["ts_code", "close"], [])])
    df = client.query("daily")
    assert df.is_empty() and df.columns == ["ts_code", "close"]


@pytest.mark.parametrize(("code", "msg", "exc"), [
    (40203, "抱歉，您没有接口访问权限", tc.TusharePermissionError),
    (40101, "token不对，请确认", tc.TushareAuthError),
    (-2001, "您的token不对，请确认。", tc.TushareAuthError),
    (40203, "抱歉，您每分钟最多访问该接口200次", tc.TushareRateLimitError),
    (50000, "系统内部错误", tc.TushareError),
])
def test_classify_error(code, msg, exc):
    err = tc.classify_error(code, msg)
    assert type(err) is exc


def test_client_permission_error_raised(monkeypatch):
    client, _ = _client(monkeypatch, [_Resp({"code": 40203, "msg": "没有接口访问权限", "data": None})])
    with pytest.raises(tc.TusharePermissionError):
        client.query("rt_k")


def test_client_rate_limit_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    limited = _Resp({"code": 40203, "msg": "抱歉，您每分钟最多访问该接口200次", "data": None})
    client, http = _client(monkeypatch, [limited, limited, _ok(["a"], [[1]])], sleeps)
    assert client.query("daily").rows() == [(1,)]
    assert len(http.bodies) == 3 and sleeps == [3.0, 10.0]


def test_client_rate_limit_gives_up_after_three_retries(monkeypatch):
    sleeps: list[float] = []
    limited = _Resp({"code": 40203, "msg": "每分钟最多访问该接口", "data": None})
    client, http = _client(monkeypatch, [limited], sleeps)
    with pytest.raises(tc.TushareRateLimitError):
        client.query("daily")
    assert len(http.bodies) == 4 and len(sleeps) == 3


def test_client_pagination_merges_and_stops_on_short_page(monkeypatch):
    client, http = _client(monkeypatch, [
        _ok(["a"], [[1], [2]]), _ok(["a"], [[3], [4]]), _ok(["a"], [[5]]),
    ])
    df = client.query_all("x", {"k": 1}, page_size=2)
    assert df["a"].to_list() == [1, 2, 3, 4, 5]
    assert [b["params"]["offset"] for b in http.bodies] == [0, 2, 4]
    assert all(b["params"]["limit"] == 2 and b["params"]["k"] == 1 for b in http.bodies)


def test_client_pagination_respects_has_more_and_empty_page(monkeypatch):
    client, http = _client(monkeypatch, [_ok(["a"], [[1]], has_more=True), _ok(["a"], [])])
    assert client.query_all("x", page_size=1)["a"].to_list() == [1]
    assert len(http.bodies) == 2
    client, http = _client(monkeypatch, [_ok(["a"], [[1], [2]], has_more=False)])
    assert client.query_all("x", page_size=2).height == 2 and len(http.bodies) == 1


def test_client_pagination_page_cap_fails_closed(monkeypatch):
    client, http = _client(monkeypatch, [_ok(["a"], [[1]], has_more=True)])
    with pytest.raises(tc.TushareError, match="分页超过上限"):
        client.query_all("x", page_size=1, max_pages=3)
    assert len(http.bodies) == 3


def test_client_errors_never_leak_token(monkeypatch):
    import httpx

    client, _ = _client(monkeypatch, [httpx.ConnectError(f"boom token={FAKE_TOKEN}")])
    with pytest.raises(tc.TushareError) as ei:
        client.query("daily")
    assert FAKE_TOKEN not in str(ei.value)

    echo = _Resp({"code": 40101, "msg": f"token {FAKE_TOKEN} 不对", "data": None})
    client, _ = _client(monkeypatch, [echo])
    with pytest.raises(tc.TushareError) as ei:
        client.query("daily")
    assert FAKE_TOKEN not in str(ei.value)


def test_client_http_and_json_errors(monkeypatch):
    # 502 属瞬时故障: 重试耗尽后仍失败才抛出
    client, http = _client(monkeypatch, [_Resp({}, status=502)] * 4)
    with pytest.raises(tc.TushareError, match="HTTP 502"):
        client.query("daily")
    assert len(http.bodies) == 4
    client, _ = _client(monkeypatch, [_Resp(ValueError("bad"))])
    with pytest.raises(tc.TushareError, match="JSON"):
        client.query("daily")


def test_client_retries_transient_5xx(monkeypatch):
    """上游偶发 502 不应让整段同步失败 (2026-10-19 全市场日K中断的根因)。"""
    sleeps: list[float] = []
    client, http = _client(monkeypatch, [
        _Resp({}, status=502),
        _Resp({}, status=503),
        _ok(["ts_code", "close"], [["600519.SH", 1480.0]]),
    ], sleeps=sleeps)
    df = client.query("daily", {"trade_date": "20261019"})
    assert df.rows() == [("600519.SH", 1480.0)]
    assert len(http.bodies) == 3
    assert sleeps == [tc._TRANSIENT_RETRY_BACKOFF_S[0], tc._TRANSIENT_RETRY_BACKOFF_S[1]]


def test_client_does_not_retry_permission_or_auth(monkeypatch):
    for payload in ({"code": 40203, "msg": "没有接口访问权限", "data": None},
                    {"code": 40101, "msg": "token 不对", "data": None}):
        client, http = _client(monkeypatch, [_Resp(payload)] * 5)
        with pytest.raises(tc.TushareError):
            client.query("daily")
        assert len(http.bodies) == 1, "权限/token 错误不得重试"


def test_transient_retry_does_not_consume_rate_budget(monkeypatch):
    """限频与瞬时故障各自计数: 限频 1 次 + 502 3 次, 两类预算不互相占用。"""
    sleeps: list[float] = []
    client, http = _client(monkeypatch, [
        _Resp({"code": 40203, "msg": "抱歉，您每分钟最多访问该接口200次", "data": None}),
        _Resp({}, status=502),
        _Resp({}, status=502),
        _Resp({}, status=502),
        _ok(["ts_code"], [["600519.SH"]]),
    ], sleeps=sleeps)
    assert client.query("daily").height == 1
    assert len(http.bodies) == 5


def test_client_requires_token():
    with pytest.raises(tc.TushareError):
        tc.TushareClient(token="")


def test_rate_limiter_sliding_window_per_api():
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    lim = tc.RateLimiter(rpm=2, clock=lambda: now[0], sleep=sleep)
    lim.acquire("daily")
    lim.acquire("daily")
    lim.acquire("income")  # 不同接口独立计数
    assert sleeps == []
    lim.acquire("daily")  # 第 3 次须等到窗口滑出
    assert sleeps and sum(sleeps) == pytest.approx(60.0)


def test_rpm_env_override(monkeypatch):
    monkeypatch.setenv("TUSHARE_RPM", "50")
    assert tc.RateLimiter().rpm == 50
    monkeypatch.setenv("TUSHARE_RPM", "abc")
    assert tc.RateLimiter().rpm == 200


# ---- 7. availability / probe_api_key / loader ----


def test_get_api_key_secrets_store_then_env(monkeypatch):
    from app import secrets_store

    monkeypatch.undo()  # 撤掉 autouse 的 get_api_key 替身, 验证真实解析
    monkeypatch.setenv(tp.API_KEY_ENV, "env-token")
    monkeypatch.setattr(secrets_store, "load", lambda: {tp.SECRETS_FIELD: "ui-token"})
    assert tp.get_api_key() == "ui-token"
    monkeypatch.setattr(secrets_store, "load", lambda: {})
    assert tp.get_api_key() == "env-token"


def test_availability_two_states(monkeypatch):
    assert tp.availability() == (True, "ok")
    monkeypatch.setattr(tp, "get_api_key", lambda: "")
    ok, reason = tp.availability()
    assert ok is False and tp.API_KEY_ENV in reason


def test_probe_api_key_ok_and_invalid(monkeypatch):
    fake = _FakeClient({"trade_cal": pl.DataFrame({"cal_date": ["20261009"]})})
    monkeypatch.setattr(tp, "tushare_client", type("M", (), {"TushareClient": lambda **kw: fake}))
    assert tp.probe_api_key("candidate") == (True, "ok")
    assert fake.calls[0][0] == "trade_cal" and fake.closed

    bad = _FakeClient({"trade_cal": tc.TushareAuthError("Tushare 接口错误 code=-2001: token不对")})
    monkeypatch.setattr(tp, "tushare_client", type("M", (), {"TushareClient": lambda **kw: bad}))
    ok, reason = tp.probe_api_key("bad-candidate")
    assert ok is False and "bad-candidate" not in reason


def test_manifest_and_loader_registration(monkeypatch):
    from app.data_providers.custom import loader

    manifest = loader.plugin_manifest("tushare")
    assert manifest is not None
    assert manifest["entry"] == "app.plugins.tushare.provider:TushareProvider"
    assert manifest["runtime"] == "none" and manifest["api_key_env"] == tp.API_KEY_ENV
    assert set(manifest["datasets"]) == {"daily", "adj_factor", "minute", "financial"}

    monkeypatch.setattr(loader, "_PROVIDERS", {})
    monkeypatch.setattr(loader, "_PLUGIN_STATUS", {})
    monkeypatch.setattr(loader, "_plugin_key_masked", lambda n, e: "")
    monkeypatch.setattr(tp, "availability", lambda: (True, "ok"))
    loader._register_one_plugin(manifest)
    provider = loader._PROVIDERS["tushare"]
    assert isinstance(provider, TushareProvider) and provider.builtin is True
    assert loader.provider_has_dataset("tushare", "daily")
    assert not loader.provider_has_dataset("tushare", "realtime")

    # 运行时裁剪对 provider_has_dataset 与能力矩阵(list_plugins)同时生效
    provider._apply_status({"daily": "ok", "adj_factor": "ok", "minute": "denied", "financial": "ok"})
    assert not loader.provider_has_dataset("tushare", "minute")
    status = next(p for p in loader.list_plugins() if p["name"] == "tushare")
    assert "minute" not in status["datasets"] and "daily" in status["datasets"]


def test_loader_unavailable_without_token(monkeypatch):
    from app.data_providers.custom import loader

    monkeypatch.setattr(loader, "_PROVIDERS", {})
    monkeypatch.setattr(loader, "_PLUGIN_STATUS", {})
    monkeypatch.setattr(loader, "_plugin_key_masked", lambda n, e: "")
    monkeypatch.setattr(tp, "get_api_key", lambda: "")
    loader._register_one_plugin(loader.plugin_manifest("tushare"))
    assert "tushare" not in loader._PROVIDERS
    status = loader.list_plugins()[0]
    assert status["available"] is False and status["datasets"]  # 未注册按 manifest 展示


def test_loader_probe_plugin_key_dispatch(monkeypatch):
    from app.data_providers.custom import loader

    monkeypatch.setattr(tp, "probe_api_key", lambda key: (key == "good", "x"))
    assert loader.probe_plugin_key("tushare", "good")[0] is True
    assert loader.probe_plugin_key("tushare", "bad")[0] is False
