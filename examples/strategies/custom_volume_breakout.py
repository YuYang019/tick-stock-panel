"""放量突破 — 放量阳线突破 60 日新高, 收盘接近最高价 (实体饱满)"""

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
    "id": "custom_volume_breakout",
    "name": "放量突破",
    "description": "收盘创 60 日新高 (突破前 60 日最高收盘), 量比 ≥ 阈值, 涨幅达标且收在当日高位",
    "tags": ["突破", "放量", "新高"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 1e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "min_vol_ratio", "label": "最低量比 (对5日均量)", "type": "float", "default": 2.0, "min": 1.2, "max": 6.0, "step": 0.1},
        {"id": "min_change_pct", "label": "最低涨幅%", "type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5},
        {"id": "min_close_position", "label": "收盘位置下限 (0=最低 1=最高)", "type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "max_change_pct", "label": "最高涨幅% (排除一字/过热)", "type": "float", "default": 9.5, "min": 3.0, "max": 20.0, "step": 0.5},
    ],
    "scoring": {"vol_ratio_5d": 0.4, "rps_ind_20": 0.3, "change_pct": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_n_day_high", "signal_volume_surge"]
EXIT_SIGNALS = ["signal_ma10_breakdown", "signal_breakout_fail"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15

RULES = """
1. 新高: 今日收盘 > 前 60 个交易日的最高收盘价 (不含今日)
2. 放量: 量比 (对前5日均量) ≥ 阈值
3. 质量: 涨幅在 [下限, 上限] 内, 收盘位于当日振幅高位 (实体饱满)
4. 离场: 收盘跌破 MA10, 或跌回突破前高之下 (假突破)
"""


class VolumeBreakoutStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "high", "low", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 70

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        prior_high = shift(matrix_feature(market, "high_60d"), 1)  # 前 60 日最高收盘 (不含今日)
        change = matrix_feature(market, "change_pct")
        vol_ratio = matrix_feature(market, "vol_ratio_5d")
        position = matrix_feature(market, "close_position")
        ma10 = matrix_feature(market, "ma10")

        entry = (
            (close > prior_high)
            & (vol_ratio >= float(params.get("min_vol_ratio", 2.0)))
            & (change >= float(params.get("min_change_pct", 3.0)) / 100.0)
            & (change <= float(params.get("max_change_pct", 9.5)) / 100.0)
            & (position >= float(params.get("min_close_position", 0.7)))
        )
        lose_ma10 = close < ma10
        fail = close < prior_high * 0.97
        exit_ = lose_ma10 | fail
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(lose_ma10, 0, np.where(fail, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_n_day_high", "signal_volume_surge"),
            exit_signal_ids=("signal_ma10_breakdown", "signal_breakout_fail"),
        )


MATRIX_STRATEGY = VolumeBreakoutStrategy()
