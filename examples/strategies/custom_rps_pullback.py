"""强势回踩 — 中期强势股 (RPS120 高) 短期回调到 MA20 附近企稳 (强中选弱)"""

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
    "id": "custom_rps_pullback",
    "name": "强势回踩",
    "description": "RPS120 高位的中期强势股, 短期 RPS20 回落、回踩 MA20 不破且当日收阳, 趋势中的低吸点",
    "tags": ["相对强度", "回调", "低吸"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 1e8, "exclude_st": True, "exclude_new_days": 120,
    },
    "params": [
        {"id": "rps_long_min", "label": "RPS120 下限 (中期强势)", "type": "float", "default": 80.0, "min": 50.0, "max": 99.0, "step": 5.0},
        {"id": "rps_short_max", "label": "RPS20 上限 (短期回调)", "type": "float", "default": 50.0, "min": 10.0, "max": 90.0, "step": 5.0},
        {"id": "touch_pct", "label": "最低价距 MA20 容差%", "type": "float", "default": 2.0, "min": 0.0, "max": 6.0, "step": 0.5},
    ],
    "scoring": {"rps_120": 0.5, "industry_rps_60": 0.3, "rps_ind_120": 0.2},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_rps_pullback"]
EXIT_SIGNALS = ["signal_ma60_breakdown", "signal_rps_rebound_done"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15

RULES = """
1. 中期强势: RPS120 ≥ 下限, MA20 > MA60 (中期趋势向上)
2. 短期回调: RPS20 ≤ 上限 (近 20 日相对走弱, 因子检验显示短期 RPS 在 A 股偏反转)
3. 回踩企稳: 当日最低价触及 MA20 (± 容差), 收盘 ≥ MA20 且收阳
4. 离场: 收盘跌破 MA60, 或 RPS20 回升至 90 以上 (短期反弹兑现)
"""


class RPSPullbackStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"open", "close", "low"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 130

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        low = market.low
        open_ = market.open
        rps120 = matrix_feature(market, "rps_120")
        rps20 = matrix_feature(market, "rps_20")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        tol = float(params.get("touch_pct", 2.0)) / 100.0
        entry = (
            (rps120 >= float(params.get("rps_long_min", 80.0)))
            & (rps20 <= float(params.get("rps_short_max", 50.0)))
            & (ma20 > ma60)
            & (low <= ma20 * (1.0 + tol))
            & (close >= ma20)
            & (close > open_)
        )
        lose_ma60 = close < ma60
        done = (rps20 >= 90.0) & (shift(rps20, 1) < 90.0)
        exit_ = lose_ma60 | done
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(lose_ma60, 0, np.where(done, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_rps_pullback",),
            exit_signal_ids=("signal_ma60_breakdown", "signal_rps_rebound_done"),
        )


MATRIX_STRATEGY = RPSPullbackStrategy()
