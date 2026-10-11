"""RPS 强度 — 欧奈尔式相对强度: 中期 RPS 领先 + 行业内领先 + 趋势确认"""

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
    "id": "custom_rps_strength",
    "name": "RPS 强度",
    "description": "RPS120 与行业内 RPS60 双高, 站上 MA60/MA120 且 MA60 上行; 跌破 MA20 或 RPS120 回落离场",
    "tags": ["相对强度", "RPS", "趋势"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 1e8, "exclude_st": True, "exclude_new_days": 120,
    },
    "params": [
        {"id": "rps_min", "label": "RPS120 下限", "type": "float", "default": 90.0, "min": 50.0, "max": 99.0, "step": 1.0},
        {"id": "rps_ind_min", "label": "行业内 RPS60 下限", "type": "float", "default": 80.0, "min": 0.0, "max": 99.0, "step": 5.0},
        {"id": "rps_exit", "label": "RPS120 离场线", "type": "float", "default": 70.0, "min": 30.0, "max": 95.0, "step": 5.0},
        {"id": "max_bias_pct", "label": "收盘偏离MA20上限%", "type": "float", "default": 20.0, "min": 5.0, "max": 50.0, "step": 1.0},
    ],
    "scoring": {"rps_120": 0.4, "rps_ind_60": 0.3, "excess_60": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_rps_strength"]
EXIT_SIGNALS = ["signal_ma20_breakdown", "signal_rps_fade"]
STOP_LOSS = -0.08
MAX_HOLD_DAYS = 30

RULES = """
1. 相对强度: RPS120 ≥ 下限 (全市场前 10%), 行业内 RPS60 ≥ 下限
2. 趋势确认: 收盘 > MA60 且 > MA120, MA60 较 10 日前抬升
3. 防追高: 收盘偏离 MA20 不超过上限
4. 离场: 收盘跌破 MA20, 或 RPS120 跌破离场线
注意: 因子检验显示 A 股近两年短期 RPS (20/60) 偏反转, 本策略以 120 日中期强度为主,
建议回测后再调整阈值; 行业内 RPS 在 2026-10-11 之前使用当前行业快照 (存在未来函数偏差)。
"""


class RPSStrengthStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 130

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        rps120 = matrix_feature(market, "rps_120")
        rps_ind60 = matrix_feature(market, "rps_ind_60")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        ma120 = matrix_feature(market, "ma120")
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = (close - ma20) / ma20 * 100.0
        entry = (
            (rps120 >= float(params.get("rps_min", 90.0)))
            & (rps_ind60 >= float(params.get("rps_ind_min", 80.0)))
            & (close > ma60) & (close > ma120)
            & (ma60 > shift(ma60, 10))
            & (bias <= float(params.get("max_bias_pct", 20.0)))
        )
        lose_ma20 = close < ma20
        fade = rps120 < float(params.get("rps_exit", 70.0))
        exit_ = lose_ma20 | fade
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(lose_ma20, 0, np.where(fade, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_rps_strength",),
            exit_signal_ids=("signal_ma20_breakdown", "signal_rps_fade"),
        )


MATRIX_STRATEGY = RPSStrengthStrategy()
