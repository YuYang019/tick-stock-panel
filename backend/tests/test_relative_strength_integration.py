"""相对强度因子: DSL group_* 算子 + 因子检验/回测接入 (预热、注记)。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.factors import relative_strength as rs
from app.factors.dsl import FACTOR_COLUMN, compile_formula

DAY0 = date(2025, 1, 1)


def _frame(n_symbols: int = 40, n_days: int = 90) -> pl.DataFrame:
    rows = []
    for i in range(n_symbols):
        for t in range(n_days):
            rows.append({
                "symbol": f"S{i:03d}", "date": DAY0 + timedelta(days=t),
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.0 * (1 + 0.001 * (i + 1)) ** t + (0.3 if (i + t) % 5 == 0 else 0.0),
                "volume": 1000.0, "amount": 1e6, "turnover_rate": 1.0,
            })
    return pl.DataFrame(rows)


def _archive(n_symbols: int = 40) -> pl.DataFrame:
    # S000-S029 → A, S030-S035 → B, S036-S039 → C (< 5 → null)
    return pl.DataFrame({
        "as_of": [DAY0] * n_symbols,
        "symbol": [f"S{i:03d}" for i in range(n_symbols)],
        "industry": ["A" if i < 30 else ("B" if i < 36 else "C") for i in range(n_symbols)],
    })


@pytest.fixture
def fake_industry(monkeypatch):
    archive = _archive()
    monkeypatch.setattr(rs, "load_industry_archive", lambda *a, **k: archive)
    return archive


# ───────────────────────────── DSL ─────────────────────────────


def test_dsl_group_ops_parse_and_metadata():
    compiled = compile_formula("group_rank(close, industry)")
    assert compiled.ok and compiled.cross_sectional
    assert compiled.group_keys == frozenset({"industry"})
    assert compiled.dependencies == frozenset({"close"})  # industry 不是数据列依赖
    nested = compile_formula("group_zscore(ts_delta(close, 5), industry) + rank(rps_20)")
    assert nested.ok and nested.warmup_bars >= 21
    assert "rps_20" in nested.referenced_factors


@pytest.mark.parametrize("formula, code", [
    ("group_rank(close, sector)", "E003"),       # 非法分组键
    ("group_rank(close, 3)", "E003"),            # 分组键必须是标识符
    ("group_mean(close)", "E003"),               # 参数数量
    ("ts_mean(group_rank(close, industry), 5)", "E009"),  # 截面嵌在时序窗口内
    ("group_rank(industry, industry)", "E001"),  # industry 不是数据列
])
def test_dsl_group_ops_errors(formula, code):
    compiled = compile_formula(formula)
    assert not compiled.ok
    assert code in {error.code for error in compiled.errors}


def test_dsl_group_rank_compute_and_small_group_null(fake_industry):
    frame = _frame()
    compiled = compile_formula("group_rank(close, industry)")
    out = compiled.frame_transform(frame)
    assert rs.INDUSTRY_COLUMN not in out.columns  # 临时分组列用后即删
    last = out.filter(pl.col("date") == out["date"].max())
    value = dict(zip(last["symbol"].to_list(), last[FACTOR_COLUMN].to_list(), strict=True))
    a_close = last.filter(pl.col("symbol").is_in([f"S{i:03d}" for i in range(30)]))
    top = a_close.sort("close")["symbol"][-1]
    assert value[top] == pytest.approx(1.0)  # rank / count, 与 rank() 同 (0,1] 口径
    assert value["S037"] is None  # C 组 4 只 < 5


def test_dsl_group_mean_and_zscore(fake_industry):
    frame = _frame()
    out_mean = compile_formula("group_mean(close, industry)").frame_transform(frame)
    out_z = compile_formula("group_zscore(close, industry)").frame_transform(frame)
    last_day = frame["date"].max()
    b = frame.filter((pl.col("date") == last_day) & pl.col("symbol").is_in([f"S{i:03d}" for i in range(30, 36)]))
    b_mean = b["close"].mean()
    got = out_mean.filter((pl.col("date") == last_day) & (pl.col("symbol") == "S030"))[FACTOR_COLUMN].item()
    assert got == pytest.approx(b_mean)
    z = out_z.filter((pl.col("date") == last_day) & pl.col("symbol").is_in(b["symbol"].implode()))[FACTOR_COLUMN]
    assert z.mean() == pytest.approx(0.0, abs=1e-9)


def test_dsl_group_without_industry_data_is_null(monkeypatch):
    monkeypatch.setattr(rs, "load_industry_archive", lambda *a, **k: None)
    out = compile_formula("group_rank(close, industry)").frame_transform(_frame())
    assert out[FACTOR_COLUMN].null_count() == out.height


# ───────────────────────────── 因子检验接入 ─────────────────────────────


class _Engine:
    def __init__(self, panel: pl.DataFrame) -> None:
        self.panel = panel
        self.calls: list[dict] = []

    def load_panel(self, symbols, start, end, columns, asset_type):
        self.calls.append({"start": start, "end": end, "columns": columns})
        return self.panel.filter((pl.col("date") >= start) & (pl.col("date") <= end)).select(
            [c for c in columns if c in self.panel.columns]
        )


def test_factor_backtest_warmup_and_warnings(monkeypatch, fake_industry):
    from app.backtest.factor import FactorBacktestService, FactorConfig

    instruments = pl.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(40)], "name": ["正常"] * 40,
        "listing_date": [date(2020, 1, 1)] * 40,
    })
    ctx = rs.RSContext(instruments=instruments, industry_archive=fake_industry)
    monkeypatch.setattr(rs, "load_context", lambda *a, **k: ctx)
    monkeypatch.setattr(rs, "load_industry_archive", lambda *a, **k: _archive().with_columns(
        pl.lit(date(2025, 3, 1)).alias("as_of")))  # 首份归档晚于检验起点 → 快照偏差注记
    engine = _Engine(_frame())
    config = FactorConfig(
        factor_name="rps_ind_20", symbols=None, start=date(2025, 2, 1), end=date(2025, 3, 31),
        n_groups=3, rebalance="daily",
    )
    result = FactorBacktestService(engine).run(config)
    assert result.error is None, result.error
    assert result.n_dates > 0
    # 预热按注册表 warmup_bars (21 → ≥120 自然日下限); rps_250 → 251 bar ≈ 416 自然日
    assert engine.calls[0]["start"] <= config.start - timedelta(days=120)
    assert FactorBacktestService._warmup_calendar_days(["rps_250"]) >= 251 * 1.6
    assert any(rs.WARN_INDUSTRY_SNAPSHOT in w for w in result.warnings)


def test_factor_batch_item_carries_warnings(monkeypatch, fake_industry):
    from app.backtest.factor import FactorBacktestService, FactorBatchConfig

    monkeypatch.setattr(rs, "load_context", lambda *a, **k: rs.RSContext(industry_archive=fake_industry))
    engine = _Engine(_frame())
    result = FactorBacktestService(engine).run_batch(FactorBatchConfig(
        factor_names=["rps_20", "momentum_5d"], symbols=None,
        start=date(2025, 2, 15), end=date(2025, 3, 31), n_groups=3, rebalance="daily",
    ))
    by_name = {item.factor_name: item for item in result.results}
    assert by_name["rps_20"].error is None
    assert rs.WARN_ST_NOT_PIT in by_name["rps_20"].warnings
    assert by_name["momentum_5d"].warnings == []


def test_history_window_batches_keep_cross_section_rps_for_signals(monkeypatch, tmp_path):
    """按 symbol 分批的历史窗口: 信号引用的 rps_* 须在全市场帧上排名, 不能被批内守卫置空。"""
    from app.indicators import pipeline
    from app.strategy import custom_signals

    sig = {"id": "rps_strong", "name": "强势", "kind": "entry",
           "conditions": [{"left": "rps_20", "op": ">=", "right": "90"}]}
    custom_signals.validate(sig)
    exprs = custom_signals.build_expressions([sig])
    monkeypatch.setattr(pipeline, "_get_custom_signal_exprs", lambda: exprs)
    frame = _frame()
    monkeypatch.setattr(rs, "load_instruments", lambda *a, **k: pl.DataFrame({
        "symbol": frame["symbol"].unique().to_list(),
        "name": [f"N{i}" for i in range(frame["symbol"].n_unique())],
        "listing_date": [DAY0 - timedelta(days=400)] * frame["symbol"].n_unique(),
    }))

    batched = pipeline.compute_enriched_history_window(frame, tmp_path, sym_batch=10)
    whole = pipeline.compute_enriched_history_window(frame, tmp_path, sym_batch=1000)
    column = custom_signals.column_name("rps_strong")
    assert batched["rps_20"].is_not_null().sum() > 0
    joined = batched.select("symbol", "date", "rps_20", column).join(
        whole.select("symbol", "date", pl.col("rps_20").alias("w"), pl.col(column).alias("wc")),
        on=["symbol", "date"],
    )
    assert joined.filter(pl.col("rps_20").ne_missing(pl.col("w"))).is_empty()
    assert joined.filter(pl.col(column).ne_missing(pl.col("wc"))).is_empty()
    assert joined[column].fill_null(False).any()
