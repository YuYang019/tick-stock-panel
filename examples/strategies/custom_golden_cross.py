"""金叉共振 — MA5 上穿 MA20 与 MACD 金叉在近 3 日内共振, 且在 MA60 之上"""

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
    "id": "custom_golden_cross",
    "name": "金叉共振",
    "description": "今日 MA5 上穿 MA20, 近 N 日内 MACD 金叉 (可选零轴上), 收盘在 MA60 之上",
    "tags": ["均线", "金叉", "MACD"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 0.5e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "macd_window", "label": "MACD 金叉回看天数", "type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
        {"id": "above_zero", "label": "要求 MACD 零轴上 (DIF>0)", "type": "bool", "default": False},
        {"id": "require_ma60", "label": "要求收盘在 MA60 之上", "type": "bool", "default": True},
    ],
    "scoring": {"vol_ratio_5d": 0.4, "rps_ind_60": 0.3, "momentum_20d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_ma_golden_5_20", "signal_macd_golden"]
EXIT_SIGNALS = ["signal_ma_dead_5_20", "signal_macd_dead"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 20

RULES = """
1. 均线金叉: 今日 MA5 上穿 MA20 (昨日 MA5 ≤ MA20)
2. MACD 共振: 近 N 日内 DIF 上穿 DEA; 可选要求 DIF > 0 (零轴上金叉更强)
3. 趋势过滤: 收盘 > MA60
4. 离场: MA5 下穿 MA20 或 MACD 死叉
"""


class GoldenCrossStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 80

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        ma5 = matrix_feature(market, "ma5")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        dif = matrix_feature(market, "macd_dif")
        dea = matrix_feature(market, "macd_dea")

        ma_cross = (ma5 > ma20) & (shift(ma5, 1) <= shift(ma20, 1))
        macd_cross = (dif > dea) & (shift(dif, 1) <= shift(dea, 1))
        recent = macd_cross.copy()
        for k in range(1, max(1, int(params.get("macd_window", 3)))):
            recent |= shift(macd_cross, k).astype(bool)

        entry = ma_cross & recent & (dif > dea)
        if params.get("above_zero", False):
            entry &= dif > 0
        if params.get("require_ma60", True):
            entry &= close > ma60

        ma_dead = (ma5 < ma20) & (shift(ma5, 1) >= shift(ma20, 1))
        macd_dead = (dif < dea) & (shift(dif, 1) >= shift(dea, 1))
        exit_ = ma_dead | macd_dead
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(ma_dead, 0, np.where(macd_dead, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_ma_golden_5_20", "signal_macd_golden"),
            exit_signal_ids=("signal_ma_dead_5_20", "signal_macd_dead"),
        )


MATRIX_STRATEGY = GoldenCrossStrategy()
