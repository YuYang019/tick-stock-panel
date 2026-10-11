"""均线多头发散 — MA5>MA10>MA20>MA60 多头排列, 且短中均线间距正在拉开 (趋势加速)"""

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
    "id": "custom_ma_bull_divergence",
    "name": "均线多头发散",
    "description": "MA5>MA10>MA20>MA60 多头排列, MA5 与 MA20 间距较 N 日前扩大且 MA20 上行, 趋势加速段",
    "tags": ["均线", "多头", "趋势"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 1e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "spread_lookback", "label": "发散对比天数", "type": "int", "default": 5, "min": 2, "max": 20, "step": 1},
        {"id": "min_spread_pct", "label": "MA5-MA20 最小间距%", "type": "float", "default": 2.0, "min": 0.0, "max": 15.0, "step": 0.5},
        {"id": "max_bias_pct", "label": "收盘偏离MA20上限% (防追高)", "type": "float", "default": 15.0, "min": 3.0, "max": 40.0, "step": 1.0},
    ],
    "scoring": {"rps_ind_60": 0.4, "momentum_20d": 0.3, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_ma_bull_divergence"]
EXIT_SIGNALS = ["signal_ma_dead_5_10", "signal_ma20_breakdown"]
STOP_LOSS = -0.07
MAX_HOLD_DAYS = 20

RULES = """
1. 均线多头排列: MA5 > MA10 > MA20 > MA60
2. 发散: (MA5-MA20)/MA20 大于阈值, 且比 N 日前更大; MA20 较 N 日前抬升
3. 防追高: 收盘偏离 MA20 不超过上限
4. 离场: MA5 下穿 MA10 或收盘跌破 MA20
"""


class MABullDivergenceStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close"})

    def required_warmup_bars(self, params: dict) -> int:
        return 60 + int(params.get("spread_lookback", 5))

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        n = max(1, int(params.get("spread_lookback", 5)))
        ma5 = matrix_feature(market, "ma5")
        ma10 = matrix_feature(market, "ma10")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        close = market.close
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = (ma5 - ma20) / ma20 * 100.0
            bias = (close - ma20) / ma20 * 100.0
        aligned = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
        diverging = (spread >= float(params.get("min_spread_pct", 2.0))) & (spread > shift(spread, n))
        rising = ma20 > shift(ma20, n)
        entry = aligned & diverging & rising & (bias <= float(params.get("max_bias_pct", 15.0)))

        dead_5_10 = (ma5 < ma10) & (shift(ma5, 1) >= shift(ma10, 1))
        lose_ma20 = close < ma20
        exit_ = dead_5_10 | lose_ma20
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(dead_5_10, 0, np.where(lose_ma20, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_ma_bull_divergence",),
            exit_signal_ids=("signal_ma_dead_5_10", "signal_ma20_breakdown"),
        )


MATRIX_STRATEGY = MABullDivergenceStrategy()
