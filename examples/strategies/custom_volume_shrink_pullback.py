"""放量缩量回踩 — 短期强势且强于大盘的股票, 放量拉升后缩量洗盘至 KDJ 超卖的埋伏点"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
)
from app.backtest.matrix import (
    valid_shift as shift,
)

META = {
    "id": "custom_volume_shrink_pullback",
    "name": "放量缩量回踩",
    "description": "前置: 近期 RPS20 强势且跑赢中证全指; 近 N 日有放量阳线, 今日缩量回调 5%~15%、KDJ J 值超卖且守住 MA60",
    "tags": ["埋伏", "缩量", "KDJ超卖", "相对强度", "情绪择时"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 0.5e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "strong_days", "label": "强势回看天数", "type": "int", "default": 10, "min": 3, "max": 30, "step": 1},
        {"id": "rps20_peak_min", "label": "回看期内 RPS20 最高值下限", "type": "float", "default": 80.0, "min": 50.0, "max": 99.0, "step": 5.0},
        {"id": "excess_min", "label": "20日超额收益下限 (对中证全指)", "type": "float", "default": 0.0, "min": -0.2, "max": 0.5, "step": 0.01},
        {"id": "rs_slope_min", "label": "RS线20日斜率下限", "type": "float", "default": 0.0, "min": -0.01, "max": 0.02, "step": 0.001},
        {"id": "surge_ratio", "label": "放量倍数 (量比)", "type": "float", "default": 2.0, "min": 1.3, "max": 6.0, "step": 0.1},
        {"id": "surge_change_pct", "label": "放量日最低涨幅%", "type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5},
        {"id": "surge_lookback", "label": "放量回看天数", "type": "int", "default": 10, "min": 3, "max": 30, "step": 1},
        {"id": "shrink_ratio", "label": "缩量上限 (量比)", "type": "float", "default": 0.6, "min": 0.2, "max": 1.0, "step": 0.05},
        {"id": "j_max", "label": "KDJ J 值上限 (超卖)", "type": "float", "default": -10.0, "min": -40.0, "max": 20.0, "step": 1.0},
        {"id": "j_recent_days", "label": "J 超卖允许回看天数", "type": "int", "default": 2, "min": 1, "max": 5, "step": 1},
        {"id": "pullback_min_pct", "label": "自放量后最高收盘回调下限%", "type": "float", "default": 5.0, "min": 0.0, "max": 30.0, "step": 1.0},
        {"id": "pullback_max_pct", "label": "回调上限% (超过视为出货)", "type": "float", "default": 15.0, "min": 5.0, "max": 40.0, "step": 1.0},
    ],
    "scoring": {"rps_20": 0.3, "excess_20": 0.3, "rs_line_slope_20": 0.4},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_volume_shrink", "signal_kdj_oversold"]
EXIT_SIGNALS = ["signal_heavy_volume_drop", "signal_ma60_breakdown", "signal_kdj_overbought"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 10
REGIME_PHASES = ["ignite", "rally", "climax", "repair"]  # 退潮/冰点不埋伏

RULES = """
前置条件 (必须同时满足):
1. 短期强势: 近 10 个交易日内 RPS20 最高值 ≥ 80 (看回调前是否强过, 而非回调当日)
2. 强于大盘: 20 日超额收益 (对中证全指) > 0, 且 RS 线 20 日斜率 > 0
触发条件:
3. 前期放量: 近 10 日 (不含今日) 出现过量比 ≥ 2 且涨幅 ≥ 3% 的放量阳线
4. 缩量回调: 今日量比 ≤ 0.6, 收盘较放量后最高收盘回调 5%~15%
5. 超卖: KDJ J ≤ -10 (今日或近 2 日内)
6. 支撑: 收盘 ≥ MA60 且 MA60 上行
离场: 放量收阴 (量比 ≥ 2 且跌幅 ≥ 3%) / 跌破 MA60 / J ≥ 80 反弹兑现; 止损 -6%, 最长 10 日
情绪择时: 前一交易日为退潮/冰点时不开仓
"""


def _rolling_any(cond: np.ndarray, days: int, start: int = 0) -> np.ndarray:
    """近 days 根 (从 shift=start 起) 内任一为真。"""
    out = np.zeros(cond.shape, dtype=bool)
    for k in range(start, start + days):
        out |= np.nan_to_num(shift(cond.astype(np.float32), k)).astype(bool) if k else cond.astype(bool)
    return out


def _rolling_max(values: np.ndarray, days: int, start: int = 0) -> np.ndarray:
    out = np.full(values.shape, -np.inf, dtype=np.float64)
    for k in range(start, start + days):
        v = shift(values, k) if k else values
        out = np.fmax(out, np.where(np.isfinite(v), v, -np.inf))
    return np.where(np.isfinite(out), out, np.nan)


class VolumeShrinkPullbackStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        lookback = max(int(params.get("surge_lookback", 10)), int(params.get("strong_days", 10)))
        return 61 + lookback

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        change = matrix_feature(market, "change_pct")
        vol_ratio = matrix_feature(market, "vol_ratio_5d")
        j = matrix_feature(market, "kdj_j")
        ma60 = matrix_feature(market, "ma60")
        rps20 = matrix_feature(market, "rps_20")
        excess20 = matrix_feature(market, "excess_20")
        rs_slope = matrix_feature(market, "rs_line_slope_20")

        # 前置: 短期强势 (回看期内 RPS20 峰值) + 强于大盘
        strong = _rolling_max(rps20, int(params.get("strong_days", 10))) >= float(params.get("rps20_peak_min", 80.0))
        beat_market = (excess20 > float(params.get("excess_min", 0.0))) & (
            rs_slope > float(params.get("rs_slope_min", 0.0))
        )

        # 前期放量阳线 (不含今日)
        lookback = int(params.get("surge_lookback", 10))
        surge = (vol_ratio >= float(params.get("surge_ratio", 2.0))) & (
            change >= float(params.get("surge_change_pct", 3.0)) / 100.0
        )
        had_surge = _rolling_any(surge, lookback, start=1)
        # 自放量以来的回调: 相对回看期内最高收盘
        peak = _rolling_max(close, lookback, start=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pullback = (peak - close) / peak * 100.0

        shrink = vol_ratio <= float(params.get("shrink_ratio", 0.6))
        oversold = _rolling_any(j <= float(params.get("j_max", -10.0)), int(params.get("j_recent_days", 2)))
        support = (close >= ma60) & (ma60 > shift(ma60, 5))
        entry = (
            strong & beat_market & had_surge & shrink & oversold & support
            & (pullback >= float(params.get("pullback_min_pct", 5.0)))
            & (pullback <= float(params.get("pullback_max_pct", 15.0)))
        )

        heavy_drop = (vol_ratio >= 2.0) & (change <= -0.03)
        lose_ma60 = close < ma60
        overbought = j >= 80.0
        exit_ = heavy_drop | lose_ma60 | overbought
        code = np.where(heavy_drop, 0, np.where(lose_ma60, 1, np.where(overbought, 2, -1)))
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=code.astype(np.int16),
            entry_signal_ids=("signal_volume_shrink",),
            exit_signal_ids=("signal_heavy_volume_drop", "signal_ma60_breakdown", "signal_kdj_overbought"),
        )


MATRIX_STRATEGY = VolumeShrinkPullbackStrategy()
