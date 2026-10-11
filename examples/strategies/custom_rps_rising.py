"""RPS 新锐 — 短期强度刚跃升、中期还没走远, 板块共振 + 情绪择时 (抓启动段, 避开鱼尾)"""

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
    "id": "custom_rps_rising",
    "name": "RPS 新锐",
    "description": "RPS20 近 5 日从中游跃升到前列, RPS60 未过热、距 60 日低点涨幅有限; 行业同步走强; 仅在情绪启动/主升/高潮期开仓",
    "tags": ["相对强度", "RPS", "启动", "情绪择时"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 1e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "rps20_min", "label": "今日 RPS20 下限", "type": "float", "default": 85.0, "min": 60.0, "max": 99.0, "step": 1.0},
        {"id": "rps20_prev_max", "label": "N 日前 RPS20 上限 (刚变强)", "type": "float", "default": 60.0, "min": 20.0, "max": 90.0, "step": 5.0},
        {"id": "jump_days", "label": "跃升回看天数 N", "type": "int", "default": 5, "min": 2, "max": 15, "step": 1},
        {"id": "rps60_max", "label": "RPS60 上限 (未走远)", "type": "float", "default": 85.0, "min": 50.0, "max": 100.0, "step": 5.0},
        {"id": "from_low_max_pct", "label": "距 60 日低点最大涨幅%", "type": "float", "default": 40.0, "min": 10.0, "max": 100.0, "step": 5.0},
        {"id": "industry_rps_min", "label": "行业 RPS20 下限 (板块共振)", "type": "float", "default": 70.0, "min": 0.0, "max": 99.0, "step": 5.0},
        {"id": "min_vol_ratio", "label": "最低量比", "type": "float", "default": 1.3, "min": 0.8, "max": 4.0, "step": 0.1},
    ],
    "scoring": {"rps_20_delta_5": 0.4, "industry_rps_20": 0.3, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_rps_rising"]
EXIT_SIGNALS = ["signal_ma10_breakdown", "signal_rps_fade"]
STOP_LOSS = -0.07
MAX_HOLD_DAYS = 15
# 情绪周期择时: 退潮/冰点/修复期不开仓 (选股端按 T-1 阶段拦截, 回测默认同口径)
REGIME_PHASES = ["ignite", "rally", "climax"]

RULES = """
1. 强度跃升: 今日 RPS20 ≥ 下限, 且 N 日前 RPS20 ≤ 上限 (刚从中游冲上来, 而非一直在高位)
2. 未走远: RPS60 ≤ 上限, 收盘距 60 日最低收盘涨幅 ≤ 上限 (避开鱼尾)
3. 板块共振: 所属行业 RPS20 ≥ 下限 (板块行情而非个股孤狼)
4. 量能与趋势: 量比 ≥ 下限, 收盘 > MA20 且 MA20 上行
5. 离场: 收盘跌破 MA10, 或 RPS20 回落到 60 以下 (强度衰退); 止损 -7%, 最长 15 日
6. 情绪择时: 仅在前一交易日情绪阶段为 启动/主升/高潮 时开仓
"""


class RPSRisingStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        return 61 + int(params.get("jump_days", 5))

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        n = max(1, int(params.get("jump_days", 5)))
        close = market.close
        rps20 = matrix_feature(market, "rps_20")
        rps60 = matrix_feature(market, "rps_60")
        ind20 = matrix_feature(market, "industry_rps_20")
        ma10 = matrix_feature(market, "ma10")
        ma20 = matrix_feature(market, "ma20")
        from_low = matrix_feature(market, "distance_from_low_60d")  # close / low_60d − 1
        entry = (
            (rps20 >= float(params.get("rps20_min", 85.0)))
            & (shift(rps20, n) <= float(params.get("rps20_prev_max", 60.0)))
            & (rps60 <= float(params.get("rps60_max", 85.0)))
            & (from_low <= float(params.get("from_low_max_pct", 40.0)) / 100.0)
            & (ind20 >= float(params.get("industry_rps_min", 70.0)))
            & (matrix_feature(market, "vol_ratio_5d") >= float(params.get("min_vol_ratio", 1.3)))
            & (close > ma20) & (ma20 > shift(ma20, 5))
        )
        lose_ma10 = close < ma10
        fade = rps20 < 60.0
        exit_ = lose_ma10 | fade
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(lose_ma10, 0, np.where(fade, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_rps_rising",),
            exit_signal_ids=("signal_ma10_breakdown", "signal_rps_fade"),
        )


MATRIX_STRATEGY = RPSRisingStrategy()
