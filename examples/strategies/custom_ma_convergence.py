"""均线粘合突破 — MA5/10/20/60 粘合收敛后放量阳线向上脱离 (变盘启动点)"""

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
    "id": "custom_ma_convergence",
    "name": "均线粘合突破",
    "description": "MA5/10/20/60 四线极差收窄并持续 N 日, 今日放量阳线收于四线之上",
    "tags": ["均线", "粘合", "突破"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 0.5e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "spread_pct_max", "label": "粘合带宽上限% (四线极差/MA60)", "type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5},
        {"id": "squeeze_days", "label": "粘合持续天数", "type": "int", "default": 5, "min": 3, "max": 20, "step": 1},
        {"id": "min_change_pct", "label": "突破日最低涨幅%", "type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5},
        {"id": "min_vol_ratio", "label": "突破日最低量比", "type": "float", "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1},
    ],
    "scoring": {"vol_ratio_5d": 0.4, "change_pct": 0.3, "rps_ind_20": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_ma_convergence_breakout"]
EXIT_SIGNALS = ["signal_ma10_breakdown"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15

RULES = """
1. 粘合: 昨日起连续 N 日, MA5/10/20/60 四线极差 / MA60 ≤ 带宽上限
2. 突破: 今日收盘站上四线最高者, 涨幅 ≥ 阈值, 量比 ≥ 阈值
3. 离场: 收盘跌破 MA10
"""


class MAConvergenceStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        return 60 + int(params.get("squeeze_days", 5))

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        ma5 = matrix_feature(market, "ma5")
        ma10 = matrix_feature(market, "ma10")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        top = np.maximum(np.maximum(ma5, ma10), np.maximum(ma20, ma60))
        bot = np.minimum(np.minimum(ma5, ma10), np.minimum(ma20, ma60))
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = (top - bot) / ma60 * 100.0
        tight = spread <= float(params.get("spread_pct_max", 3.0))
        days = max(2, int(params.get("squeeze_days", 5)))
        squeeze = shift(tight, 1).astype(bool)
        for k in range(2, days + 1):
            squeeze &= shift(tight, k).astype(bool)

        entry = (
            squeeze
            & (close > top)
            & (matrix_feature(market, "change_pct") >= float(params.get("min_change_pct", 3.0)) / 100.0)
            & (matrix_feature(market, "vol_ratio_5d") >= float(params.get("min_vol_ratio", 1.5)))
        )
        exit_ = close < ma10
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_ma_convergence_breakout",),
            exit_signal_ids=("signal_ma10_breakdown",),
        )


MATRIX_STRATEGY = MAConvergenceStrategy()
