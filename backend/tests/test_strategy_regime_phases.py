"""情绪周期择时 (REGIME_PHASES / regime_filter.phases)。

- 回测 mask: 按 T-1 阶段白名单过滤开仓日, 与 states/min_score 取交集
- 策略加载: REGIME_PHASES 只接受 market_phase 已定义的阶段 (fail-closed)
- 选股: T-1 阶段不在白名单 → 清空候选并给出原因; 卖出信号不受影响
- 回测默认: 未显式传 regime_filter 时以策略 REGIME_PHASES 为默认, 显式 {} 关闭
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from app.backtest.regime_alignment import build_regime_filter_mask, regime_filter_active
from app.services import regime_builder
from app.strategy.engine import StrategyDataContext, StrategyEngine

LABELS = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")
REGIME = {
    "2026-01-05": {"state": "strong", "score": 80, "phase": "rally"},
    "2026-01-06": {"state": "strong", "score": 75, "phase": "ebb"},
    "2026-01-07": {"state": "weak", "score": 20, "phase": "ignite"},
    "2026-01-08": {"state": "range", "score": 50, "phase": "repair"},
}


def test_phase_mask_uses_previous_day_phase():
    mask = build_regime_filter_mask(LABELS, {"phases": ["rally", "ignite"]}, REGIME)
    # 首日无 T-1 → 放行; 01-06 看 01-05(rally) 放行; 01-07 看 01-06(ebb) 拦; 01-08 看 01-07(ignite) 放行
    assert mask.tolist() == [True, True, False, True]


def test_phase_mask_intersects_with_states():
    mask = build_regime_filter_mask(LABELS, {"phases": ["rally", "ignite"], "states": ["strong"]}, REGIME)
    # 01-08 的 T-1 (01-07) 阶段 ignite 但 state=weak → 拦
    assert mask.tolist() == [True, True, False, False]


def test_phase_missing_on_previous_day_blocks_when_phases_required():
    regime = {k: {key: v for key, v in val.items() if key != "phase"} for k, val in REGIME.items()}
    mask = build_regime_filter_mask(LABELS, {"phases": ["rally"]}, regime)
    assert mask.tolist() == [True, False, False, False]


def test_regime_filter_active():
    assert not regime_filter_active(None)
    assert not regime_filter_active({})
    assert regime_filter_active({"phases": ["rally"]})
    assert regime_filter_active({"min_score": 0})


def _strategy_code(phases: str) -> str:
    return f'''import polars as pl
META = {{
    "id": "custom_phase_gate", "name": "t", "asset_types": ["stock"], "timeframes": ["1d"],
    "scoring": {{"close": 1.0}},
}}
EXECUTION_BACKEND = "polars_expr"
EXIT_SIGNALS = ["signal_exit_x"]
REGIME_PHASES = {phases}
def filter(df, params):
    return pl.lit(True)
'''


def _engine(tmp_path, phases: str) -> StrategyEngine:
    custom = tmp_path / "strategies" / "custom"
    custom.mkdir(parents=True)
    (custom / "custom_phase_gate.py").write_text(_strategy_code(phases), encoding="utf-8")
    return StrategyEngine(strategy_dirs=[custom])


def _write_regime(tmp_path, rows: dict[str, str]) -> None:
    path = regime_builder.regime_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "date": [date.fromisoformat(d) for d in rows],
        "state": ["range"] * len(rows),
        "score": [50.0] * len(rows),
        "phase": list(rows.values()),
    }).write_parquet(path)


def _context(as_of: date) -> StrategyDataContext:
    return StrategyDataContext(
        asset_type="stock", timeframe="1d", as_of=as_of,
        current=pl.DataFrame({
            "symbol": ["000001.SZ", "600000.SH"],
            "date": [as_of, as_of],
            "close": [10.0, 20.0],
            "signal_exit_x": [True, False],
        }),
    )


def test_strategy_loading_rejects_unknown_phase(tmp_path):
    engine = _engine(tmp_path, '["rally", "moon"]')
    assert not engine.has("custom_phase_gate")
    assert any("REGIME_PHASES" in e["error"] for e in engine.load_errors())


@pytest.mark.parametrize("prev_phase, blocked", [("ebb", True), ("rally", False)])
def test_screener_blocks_entries_by_previous_day_phase(tmp_path, prev_phase, blocked):
    engine = _engine(tmp_path, '["rally", "ignite"]')
    assert engine.get("custom_phase_gate").regime_phases == ("rally", "ignite")
    # as_of 当天阶段是 rally, 但开仓只看 T-1 (01-08)
    _write_regime(tmp_path, {"2026-01-08": prev_phase, "2026-01-09": "rally"})

    result = engine.run("custom_phase_gate", _context(date(2026, 1, 9)),
                        overrides={"basic_filter": {"enabled": False}})
    if blocked:
        assert result.rows == [] and result.total == 0
        assert result.regime_blocked and "退潮" in result.regime_blocked
        # 卖出信号照常返回: 持仓离场不受情绪阶段影响
        assert [h["symbol"] for h in result.exit_signal_hits] == ["000001.SZ"]
    else:
        assert result.total == 2 and result.regime_blocked is None


def test_no_regime_phases_means_no_gate(tmp_path):
    engine = _engine(tmp_path, "[]")
    _write_regime(tmp_path, {"2026-01-08": "ebb"})
    result = engine.run("custom_phase_gate", _context(date(2026, 1, 9)),
                        overrides={"basic_filter": {"enabled": False}})
    assert result.total == 2 and result.regime_blocked is None
