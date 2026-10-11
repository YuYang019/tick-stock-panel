"""相对强度 (RPS) 因子族: 资格、分位、行业 PIT、基准相对、无未来函数。"""
from __future__ import annotations

import math
import os
from datetime import date, timedelta

import polars as pl
import pytest

from app.factors import relative_strength as rs

N_SYMBOLS = 40
N_DAYS = 90
DAY0 = date(2025, 1, 1)


def _dates(n: int = N_DAYS) -> list[date]:
    return [DAY0 + timedelta(days=i) for i in range(n)]


def _frame(n_symbols: int = N_SYMBOLS, n_days: int = N_DAYS, drift=None) -> pl.DataFrame:
    """symbol i 日收益恒为 drift(i) → N 日对数收益 = N·ln(1+drift), 强弱顺序 = i 升序。"""
    drift = drift or (lambda i: 0.001 * (i + 1))
    rows = []
    for i in range(n_symbols):
        for t, d in enumerate(_dates(n_days)):
            rows.append({"symbol": f"S{i:03d}", "date": d, "close": 10.0 * (1 + drift(i)) ** t, "volume": 1000.0})
    return pl.DataFrame(rows)


def _instruments(n_symbols: int = N_SYMBOLS, **overrides) -> pl.DataFrame:
    rows = []
    for i in range(n_symbols):
        sym = f"S{i:03d}"
        name, listing = overrides.get(sym, (f"股票{i}", date(2020, 1, 1)))
        rows.append({"symbol": sym, "name": name, "listing_date": listing})
    return pl.DataFrame(rows, schema={"symbol": pl.Utf8, "name": pl.Utf8, "listing_date": pl.Date})


def _ctx(**kw) -> rs.RSContext:
    kw.setdefault("instruments", _instruments())
    return rs.RSContext(**kw)


def _at(out: pl.DataFrame, sym: str, d: date, col: str):
    return out.filter((pl.col("symbol") == sym) & (pl.col("date") == d))[col].item()


def test_registry_entries():
    from app.factors.registry import get_factor, unregister_factor

    for n in rs.RS_WINDOWS:
        for prefix in ("rps_", "rps_ind_", "industry_rps_", "excess_"):
            spec = get_factor(f"{prefix}{n}")
            assert spec is not None and spec.group == "相对强度"
            assert spec.warmup_bars == n + 1
            assert spec.asset_types == frozenset({"stock"})
    assert get_factor("rs_line_new_high_60").warmup_bars == 61
    assert get_factor("rs_line_slope_20").warmup_bars == 21
    with pytest.raises(ValueError, match="内置因子不可注销"):
        unregister_factor("rps_20")


def test_scoring_warmup_honors_rs():
    from app.strategy.scoring import scoring_warmup_bars

    assert scoring_warmup_bars({"rps_250": 1.0, "momentum_20d": 1.0}) == 251
    assert scoring_warmup_bars({"rs_line_new_high_60": 1.0}) == 61


def test_percentile_correctness():
    out = rs.compute_relative_strength(_frame(), ["rps_20"], _ctx())
    last = _dates()[-1]
    # 40 只严格单调: 最弱 0, 最强 100, 第 k 只 = 100·k/39
    for i in (0, 10, 39):
        assert _at(out, f"S{i:03d}", last, "rps_20") == pytest.approx(100.0 * i / (N_SYMBOLS - 1))
    # 前 20 根 bar 无 20 日收益 → null; 第 21 根 (t=20) 起有值
    assert _at(out, "S005", _dates()[19], "rps_20") is None
    assert _at(out, "S005", _dates()[20], "rps_20") is not None
    assert out.height == _frame().height  # 行序/行数不变


def test_row_order_preserved_on_shuffled_input():
    frame = _frame().sample(fraction=1.0, shuffle=True, seed=7)
    out = rs.compute_relative_strength(frame, ["rps_20", "excess_20"], _ctx(benchmark=_bench()))
    assert out.select("symbol", "date").equals(frame.select("symbol", "date"))
    ref = rs.compute_relative_strength(_frame(), ["rps_20"], _ctx())
    joined = out.join(ref.select("symbol", "date", pl.col("rps_20").alias("ref")), on=["symbol", "date"])
    assert joined.filter(pl.col("rps_20").ne_missing(pl.col("ref"))).is_empty()


def test_eligibility_st_new_listing_suspended():
    last = _dates()[-1]
    inst = _instruments(**{
        "S039": ("*ST强势", date(2020, 1, 1)),        # ST → null
        "S038": ("次新", last - timedelta(days=30)),   # 帧内上市 < 60 交易日 → null
        "S037": ("无上市日", None),                     # 上市日缺失 → 保守剔除
    })
    frame = _frame().with_columns(
        pl.when((pl.col("symbol") == "S036") & (pl.col("date") == last)).then(0.0)
        .otherwise(pl.col("volume")).alias("volume")  # 停牌/无量当日 → null
    )
    out = rs.compute_relative_strength(frame, ["rps_20"], _ctx(instruments=inst))
    for sym in ("S039", "S038", "S037", "S036"):
        assert _at(out, sym, last, "rps_20") is None
    # 剩余 36 只重新排名: S035 成为最强 = 100
    assert _at(out, "S035", last, "rps_20") == pytest.approx(100.0)
    assert _at(out, "S000", last, "rps_20") == pytest.approx(0.0)


def test_listing_age_counts_trading_bars():
    # 上市日 = 帧内第 10 根 bar → 第 70 根 bar 起满 60 个交易日
    days = _dates()
    inst = _instruments(**{"S039": ("新股", days[10])})
    out = rs.compute_relative_strength(_frame(), ["rps_20"], _ctx(instruments=inst))
    assert _at(out, "S039", days[69], "rps_20") is None
    assert _at(out, "S039", days[70], "rps_20") is not None


def test_single_symbol_frame_is_null_not_fake_100():
    frame = _frame().filter(pl.col("symbol") == "S010")
    out = rs.compute_relative_strength(frame, ["rps_20", "rps_ind_20", "industry_rps_20"], _ctx())
    for col in ("rps_20", "rps_ind_20", "industry_rps_20"):
        assert out[col].null_count() == out.height


def test_subset_frame_below_market_coverage_is_null():
    # 维表 100 只, 帧只有 40 只 (< 80%) → 截面因子 fail-closed
    out = rs.compute_relative_strength(_frame(), ["rps_20"], _ctx(instruments=_instruments(100)))
    assert out["rps_20"].null_count() == out.height


def test_single_day_snapshot_skipped():
    frame = _frame().filter(pl.col("date") == _dates()[-1])
    out = rs.compute_relative_strength(frame, ["rps_20"], _ctx())
    assert "rps_20" not in out.columns  # 缺列 → 下游"不可计算"


def test_industry_level_parsing():
    frame = pl.DataFrame({"v": ["电子-半导体-半导体设备", "银行-银行-股份制银行", "单级", "", None]})
    got = frame.select(rs.industry_level_expr(pl.col("v")).alias("x"))["x"].to_list()
    assert got == ["半导体", "银行", "单级", None, None]


# ───────────────────────────── 行业 ─────────────────────────────


def _archive(rows: list[tuple[date, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"as_of": a, "symbol": s, "industry": i} for a, s, i in rows],
        schema={"as_of": pl.Date, "symbol": pl.Utf8, "industry": pl.Utf8},
    )


def _two_industries(as_of: date = DAY0) -> pl.DataFrame:
    # S000-S029 → A (30 只), S030-S035 → B (6 只), S036-S039 → C (4 只, < 5 → null)
    rows = []
    for i in range(N_SYMBOLS):
        ind = "A" if i < 30 else ("B" if i < 36 else "C")
        rows.append((as_of, f"S{i:03d}", ind))
    return _archive(rows)


def test_group_rank_within_industry_and_small_group_null():
    out = rs.compute_relative_strength(_frame(), ["rps_ind_20"], _ctx(industry_archive=_two_industries()))
    last = _dates()[-1]
    assert _at(out, "S029", last, "rps_ind_20") == pytest.approx(100.0)  # A 组最强
    assert _at(out, "S000", last, "rps_ind_20") == pytest.approx(0.0)
    assert _at(out, "S030", last, "rps_ind_20") == pytest.approx(0.0)    # B 组最弱
    assert _at(out, "S035", last, "rps_ind_20") == pytest.approx(100.0)
    assert _at(out, "S037", last, "rps_ind_20") is None                  # C 组 4 只 < 5


def test_industry_rps_market_percentile_of_industry_mean():
    # 6 个行业各 5~10 只, 行业均值强弱 = 行业序号
    rows = [(DAY0, f"S{i:03d}", f"I{i // 7}") for i in range(N_SYMBOLS)]  # I0..I5, I5 只有 5 只
    out = rs.compute_relative_strength(_frame(), ["industry_rps_20"], _ctx(industry_archive=_archive(rows)))
    last = _dates()[-1]
    vals = {f"I{k}": _at(out, f"S{k * 7:03d}", last, "industry_rps_20") for k in range(6)}
    assert vals["I0"] == pytest.approx(0.0)
    assert vals["I5"] == pytest.approx(100.0)
    assert vals["I2"] == pytest.approx(40.0)
    # 同行业成员共享同值
    assert _at(out, "S014", last, "industry_rps_20") == _at(out, "S020", last, "industry_rps_20")


def test_industry_pit_lookup_and_fallback_flag():
    d = _dates()
    archive = _archive([
        (d[30], "S000", "A"), (d[30], "S001", "A"),
        (d[60], "S000", "B"),  # S000 在 d60 起改归 B
    ])
    keys = pl.DataFrame({"symbol": ["S000", "S000", "S000", "S000", "S001", "S002"],
                         "date": [d[10], d[30], d[59], d[60], d[89], d[89]]})
    pit = rs.industry_pit_frame(keys, archive)
    assert pit["industry"].to_list() == ["A", "A", "A", "B", "A", None]
    assert pit["industry_fallback"].to_list() == [True, False, False, False, False, False]
    assert rs.earliest_industry_date(archive) == d[30]


def test_industry_pit_no_archive():
    keys = pl.DataFrame({"symbol": ["S000"], "date": [DAY0]})
    pit = rs.industry_pit_frame(keys, None)
    assert pit["industry"].to_list() == [None]


def test_result_warnings_snapshot_bias(tmp_path):
    archive = _two_industries(as_of=date(2026, 10, 11))
    warns = rs.result_warnings(["rps_ind_20"], date(2026, 1, 1), archive=archive)
    assert any(rs.WARN_INDUSTRY_SNAPSHOT in w for w in warns)
    assert rs.WARN_ST_NOT_PIT in warns
    # 起点不早于首份归档 → 无快照偏差注记
    assert not any(rs.WARN_INDUSTRY_SNAPSHOT in w for w in
                   rs.result_warnings(["rps_ind_20"], date(2026, 10, 12), archive=archive))
    # 非行业因子 → 无行业注记; 非 RS 因子 → 无任何注记
    assert not any(rs.WARN_INDUSTRY_SNAPSHOT in w for w in rs.result_warnings(["rps_20"], date(2020, 1, 1), archive=archive))
    assert rs.result_warnings(["momentum_20d"], date(2020, 1, 1), archive=archive) == []
    assert rs.WARN_SUBSET_UNIVERSE in rs.result_warnings(["rps_20"], date(2026, 1, 1), symbols=["S000"], archive=archive)


def test_archive_write_and_load_roundtrip(tmp_path):
    raw = pl.DataFrame({"symbol": ["600000.SH", "000001.SZ"], rs.INDUSTRY_FIELD: ["银行-银行-股份制银行", "银行-银行-股份制银行"]})
    assert rs.archive_industry_snapshot(raw, tmp_path, date(2026, 10, 1)) == 2
    raw2 = raw.with_columns(pl.lit("电子-半导体-芯片").alias(rs.INDUSTRY_FIELD))
    rs.archive_industry_snapshot(raw2, tmp_path, date(2026, 10, 5))
    rs.clear_caches()
    archive = rs.load_industry_archive(tmp_path)
    assert archive is not None
    assert sorted(archive["as_of"].unique().to_list()) == [date(2026, 10, 1), date(2026, 10, 5)]
    keys = pl.DataFrame({"symbol": ["600000.SH", "600000.SH", "600000.SH"],
                         "date": [date(2026, 9, 1), date(2026, 10, 3), date(2026, 10, 6)]})
    pit = rs.industry_pit_frame(keys, archive)
    assert pit["industry"].to_list() == ["银行", "银行", "半导体"]
    assert pit["industry_fallback"].to_list() == [True, False, False]


def test_write_ext_parquet_archives_industry_snapshot(tmp_path):
    from app.services.ext_data import ExtConfig, ExtField, write_ext_parquet

    config = ExtConfig(
        id=rs.INDUSTRY_CONFIG_ID, label="扩展行业", mode="snapshot",
        fields=[ExtField(name="symbol", dtype="string", label="代码"),
                ExtField(name=rs.INDUSTRY_FIELD, dtype="string", label="行业")],
    )
    df = pl.DataFrame({"symbol": ["600000.SH"], rs.INDUSTRY_FIELD: ["银行-银行-股份制银行"]})
    write_ext_parquet(df, config, tmp_path, snapshot_date=date(2026, 10, 2), keep_strategy_cache=True)
    part = tmp_path / "ext_data" / rs.INDUSTRY_CONFIG_ID / "history" / "date=2026-10-02" / "part.parquet"
    assert part.exists()
    assert pl.read_parquet(part)[rs.INDUSTRY_FIELD].to_list() == ["银行-银行-股份制银行"]


def test_current_snapshot_counts_as_archive_by_mtime(tmp_path):
    cfg = tmp_path / "ext_data" / rs.INDUSTRY_CONFIG_ID
    cfg.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600000.SH"], rs.INDUSTRY_FIELD: ["银行-银行-股份制银行"]}).write_parquet(cfg / "part.parquet")
    ts = 1_790_000_000  # 2026-09-21 北京时间
    os.utime(cfg / "part.parquet", (ts, ts))
    rs.clear_caches()
    archive = rs.load_industry_archive(tmp_path)
    assert archive["as_of"].to_list() == [date(2026, 9, 21)]


# ───────────────────────────── 基准 ─────────────────────────────


def _bench(n_days: int = N_DAYS, daily: float = 0.002, drop: tuple[int, ...] = ()) -> pl.DataFrame:
    rows = [{"date": d, "bench_close": 1000.0 * (1 + daily) ** t} for t, d in enumerate(_dates(n_days)) if t not in drop]
    return pl.DataFrame(rows)


def test_excess_vs_benchmark():
    out = rs.compute_relative_strength(_frame(), ["excess_20"], _ctx(benchmark=_bench()))
    last = _dates()[-1]
    expected = 20 * math.log(1.011) - 20 * math.log(1.002)  # S010 drift = 0.011
    assert _at(out, "S010", last, "excess_20") == pytest.approx(expected, rel=1e-9)


def test_excess_missing_benchmark_date_is_null_no_ffill():
    d = _dates()
    out = rs.compute_relative_strength(_frame(), ["excess_20"], _ctx(benchmark=_bench(drop=(60,))))
    assert _at(out, "S010", d[60], "excess_20") is None   # 基准缺当日
    assert _at(out, "S010", d[80], "excess_20") is None   # 窗口起点 (t−20 = 60) 缺
    assert _at(out, "S010", d[81], "excess_20") is not None


def test_benchmark_unavailable_all_null():
    out = rs.compute_relative_strength(_frame(), ["excess_20", "rs_line_slope_20"], _ctx(benchmark=None))
    assert out["excess_20"].null_count() == out.height
    assert out["rs_line_slope_20"].null_count() == out.height


def test_rs_line_slope_exact():
    out = rs.compute_relative_strength(_frame(), ["rs_line_slope_20"], _ctx(benchmark=_bench()))
    # ln RS 线性: 斜率 = ln(1.011) − ln(1.002)
    assert _at(out, "S010", _dates()[-1], "rs_line_slope_20") == pytest.approx(math.log(1.011 / 1.002), rel=1e-9)
    assert _at(out, "S010", _dates()[18], "rs_line_slope_20") is None
    assert _at(out, "S010", _dates()[19], "rs_line_slope_20") is not None


def test_rs_line_new_high_values():
    # S000 drift 0.001 < 基准 0.002 → RS 线单调下降 → 永不新高; S039 drift 0.04 → 持续新高
    out = rs.compute_relative_strength(_frame(), ["rs_line_new_high_60"], _ctx(benchmark=_bench()))
    last = _dates()[-1]
    assert _at(out, "S000", last, "rs_line_new_high_60") == 0.0
    assert _at(out, "S039", last, "rs_line_new_high_60") == 1.0
    assert _at(out, "S039", _dates()[59], "rs_line_new_high_60") is None  # 前 60 日不满窗


@pytest.mark.parametrize("factor", ["rs_line_new_high_60", "rs_line_slope_20", "excess_60", "rps_60", "rps_ind_20"])
def test_no_look_ahead_future_shuffle(factor):
    """篡改 cutoff 之后的未来价格 (个股与基准), cutoff 及之前的因子值必须逐位不变。"""
    import random

    cutoff = _dates()[70]
    base = _frame()
    rng = random.Random(42)
    noise = pl.Series([rng.uniform(0.5, 2.0) for _ in range(base.height)])
    tampered = base.with_columns(
        pl.when(pl.col("date") > cutoff).then(pl.col("close") * noise).otherwise(pl.col("close")).alias("close")
    )
    bench = _bench()
    bench_t = bench.with_columns(
        pl.when(pl.col("date") > cutoff).then(pl.col("bench_close") * 3.0).otherwise(pl.col("bench_close"))
    )
    ctx_a = _ctx(benchmark=bench, industry_archive=_two_industries())
    ctx_b = _ctx(benchmark=bench_t, industry_archive=_two_industries())
    a = rs.compute_relative_strength(base, [factor], ctx_a).filter(pl.col("date") <= cutoff)
    b = rs.compute_relative_strength(tampered, [factor], ctx_b).filter(pl.col("date") <= cutoff)
    assert a[factor].is_not_null().sum() > 0
    assert a[factor].equals(b[factor])
    # 反证: cutoff 之后确实变了 (测试有效性)
    a_all = rs.compute_relative_strength(base, [factor], ctx_a).filter(pl.col("date") > cutoff)
    b_all = rs.compute_relative_strength(tampered, [factor], ctx_b).filter(pl.col("date") > cutoff)
    assert not a_all[factor].equals(b_all[factor])


# ───────────────────────────── 物化入口 ─────────────────────────────


def test_materialize_scoring_columns_routes_rs(monkeypatch):
    from app.strategy import scoring

    ctx = _ctx(benchmark=_bench())
    monkeypatch.setattr(rs, "load_context", lambda *a, **k: ctx)
    out = scoring.materialize_scoring_columns(_frame(), ["rps_20", "excess_20"])
    assert {"rps_20", "excess_20"}.issubset(out.columns)
    assert out["rps_20"].is_not_null().sum() > 0
    # 已物化后 scoring_value_expr 直接列引用
    assert scoring.scoring_value_expr(out.columns, "rps_20") is not None
    # 未物化帧: 不伪造 (返回 None → 不可计算)
    assert scoring.scoring_value_expr(_frame().columns, "rps_20") is None


def test_custom_signals_history_path_materializes_rs(monkeypatch):
    from app.strategy import custom_signals

    monkeypatch.setattr(rs, "load_context", lambda *a, **k: _ctx())
    assert "rps_60" in custom_signals.allowed_fields()
    exprs = {"csg_rps_strong": pl.col("rps_60") >= 90}
    out = custom_signals.materialize_factor_columns(_frame(), exprs)
    assert "rps_60" in out.columns
    assert out.filter(pl.col("date") == _dates()[-1])["rps_60"].is_not_null().all()


def test_rps_20_delta_5_is_time_diff_of_cross_sectional_rps20():
    frame = _frame()
    ctx = _ctx()
    both = rs.compute_relative_strength(frame, ["rps_20", "rps_20_delta_5"], ctx)
    only = rs.compute_relative_strength(frame, ["rps_20_delta_5"], ctx)
    assert "rps_20" not in only.columns  # 中间列不外泄
    expected = (
        both.sort(["symbol", "date"])
        .with_columns((pl.col("rps_20") - pl.col("rps_20").shift(5).over("symbol")).alias("_e"))
        .sort(["symbol", "date"])
    )
    got = only.sort(["symbol", "date"])["rps_20_delta_5"].to_list()
    assert got == expected["_e"].to_list()
    assert any(v is not None for v in got)


def test_rps_20_delta_5_registered_with_warmup():
    from app.factors.registry import get_factor

    spec = get_factor("rps_20_delta_5")
    assert spec is not None and spec.group == "相对强度" and spec.warmup_bars == 26
    assert "rps_20_delta_5" in rs.CROSS_SECTION_FACTOR_IDS
