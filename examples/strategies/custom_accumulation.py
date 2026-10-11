"""主力建仓形态 — 走势稳步抬升、红多绿少、量能红长绿短、波动不高 (样本: 300502 新易盛 2025-04~07)"""

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
    "id": "custom_accumulation",
    "name": "主力建仓",
    "description": "近 N 日收盘对数价线性抬升 (R² 高、斜率为正), 上涨日占多数且上涨日均量明显大于下跌日, "
                   "无大阴线、日波动不高、回撤小; 区间涨幅未过热",
    "tags": ["形态", "建仓", "量价", "趋势"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3, "amount_min": 1e8, "exclude_st": True, "exclude_new_days": 60,
    },
    "params": [
        {"id": "window", "label": "形态窗口 N (交易日)", "type": "int", "default": 20, "min": 10, "max": 60, "step": 5},
        {"id": "r2_min", "label": "走势平滑度 R² 下限", "type": "float", "default": 0.75, "min": 0.4, "max": 0.95, "step": 0.05},
        {"id": "slope_min_pct", "label": "日均抬升% 下限 (对数斜率)", "type": "float", "default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1},
        {"id": "up_ratio_min", "label": "上涨日占比下限 (红多绿少)", "type": "float", "default": 0.55, "min": 0.4, "max": 0.8, "step": 0.05},
        {"id": "vol_ud_min", "label": "上涨日均量 / 下跌日均量 下限 (红长绿短)", "type": "float", "default": 1.2, "min": 1.0, "max": 3.0, "step": 0.05},
        {"id": "worst_day_pct", "label": "窗口内最大单日跌幅% 上限 (无大阴线)", "type": "float", "default": -5.0, "min": -10.0, "max": -2.0, "step": 0.5},
        {"id": "daily_std_max_pct", "label": "日涨跌幅标准差% 上限 (波动不高)", "type": "float", "default": 3.0, "min": 1.0, "max": 6.0, "step": 0.25},
        {"id": "drawdown_max_pct", "label": "距窗口最高收盘回撤% 上限", "type": "float", "default": 8.0, "min": 2.0, "max": 20.0, "step": 1.0},
        {"id": "gain_max_pct", "label": "窗口涨幅% 上限 (未过热)", "type": "float", "default": 45.0, "min": 10.0, "max": 100.0, "step": 5.0},
        {"id": "limit_up_max", "label": "窗口内涨停次数上限 (非游资脉冲)", "type": "int", "default": 2, "min": 0, "max": 5, "step": 1},
    ],
    "scoring": {"rps_ind_20": 0.4, "excess_20": 0.3, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_accumulation"]
EXIT_SIGNALS = ["signal_ma20_breakdown", "signal_heavy_volume_drop"]
# 不设固定止损: 回测显示 -8% 止损在形态内正常波动中被反复打掉 (样本内/外均拖累),
# 风控交给形态破坏 (跌破 MA20, 次日开盘离场) 与放量大阴
STOP_LOSS = None
MAX_HOLD_DAYS = 20

RULES = """
样本: 300502 新易盛 2025-04-16 ~ 2025-07-14, 60 日涨 145%; 实测特征:
  log 价格线性 R² 0.96、日均抬升 1.5%; 上涨日 60%、上涨日均量 = 下跌日 1.40 倍;
  最大单日跌 -4.4%、区间最大回撤 -5.7%; 涨停仅 2 次 (非游资连板)
可计算定义 (近 N=20 日, 不含未来数据):
1. 走势逐步抬升: ln(收盘) 对时间的线性回归 R² ≥ 0.75, 日均斜率 ≥ 0.5%
2. 红多绿少: 上涨日占比 ≥ 55%
3. 量能红长绿短: 上涨日平均成交量 / 下跌日平均成交量 ≥ 1.2
4. 波动不高: 最大单日跌幅 > -5%, 日涨跌幅标准差 ≤ 3%, 距窗口高点回撤 ≤ 8%
5. 不追高: 窗口涨幅 ≤ 45%, 涨停次数 ≤ 2, 收盘 > MA20
离场: 收盘跌破 MA20 (形态破坏) 或放量大阴 (量比 ≥ 2 且跌幅 ≥ 4%); 无固定止损, 最长 20 日
说明: 形态检出约在启动后第 4~6 周 (样本股首次触发 2025-05-22, 其后仍涨约 55%),
不是买在起涨点, 而是确认"有组织资金在持续吸筹"后跟随。
"""


def _rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    """沿时间轴 (axis 0) 的 n 根滚动和; 窗口不满或含 NaN 为 NaN。"""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] < n:
        return out
    c = np.cumsum(np.nan_to_num(x, nan=0.0), axis=0)
    bad = np.cumsum(~np.isfinite(x), axis=0)
    total = c[n - 1:].copy()
    total[1:] -= c[:-n]
    nbad = bad[n - 1:].copy()
    nbad[1:] -= bad[:-n]
    total[nbad > 0] = np.nan
    out[n - 1:] = total
    return out


def _rolling_extreme(x: np.ndarray, n: int, fn) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for k in range(n):
        v = shift(x, k) if k else x
        out = v if k == 0 else fn(out, v)
    return out


class AccumulationStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        return int(params.get("window", 20)) + 21

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        n = max(5, int(params.get("window", 20)))
        close = market.close.astype(np.float64)
        volume = np.asarray(market.volume, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            lc = np.where(close > 0, np.log(close), np.nan)
            chg = close / shift(close, 1) - 1.0

        # 滚动线性回归 (t = 窗口内 0..n-1): slope = cov(t,y)/var(t), R² = corr²
        t = np.arange(close.shape[0], dtype=np.float64)[:, None] * np.ones((1, close.shape[1]))
        t = np.where(np.isfinite(lc), t, np.nan)
        sy, st = _rolling_sum(lc, n), _rolling_sum(t, n)
        syy, stt, sty = _rolling_sum(lc * lc, n), _rolling_sum(t * t, n), _rolling_sum(t * lc, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sty / n - (st / n) * (sy / n)
            var_t = stt / n - (st / n) ** 2
            var_y = syy / n - (sy / n) ** 2
            slope = cov / var_t
            r2 = cov * cov / (var_t * var_y)

        up = np.where(np.isfinite(chg), (chg > 0).astype(np.float64), np.nan)
        dn = np.where(np.isfinite(chg), (chg < 0).astype(np.float64), np.nan)
        n_up, n_dn = _rolling_sum(up, n), _rolling_sum(dn, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            up_vol = _rolling_sum(np.nan_to_num(up) * volume, n) / n_up
            dn_vol = _rolling_sum(np.nan_to_num(dn) * volume, n) / n_dn
            vol_ud = up_vol / dn_vol
            mean_c = _rolling_sum(chg, n) / n
            std_c = np.sqrt(np.maximum(_rolling_sum(chg * chg, n) / n - mean_c ** 2, 0.0))
        worst = _rolling_extreme(chg, n, np.fmin)
        peak = _rolling_extreme(close, n, np.fmax)
        limit_ups = _rolling_sum(np.where(np.isfinite(chg), (chg >= 0.095).astype(np.float64), np.nan), n)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = close / shift(close, n) - 1.0
            drawdown = (peak - close) / peak
        ma20 = matrix_feature(market, "ma20")

        entry = (
            (r2 >= float(params.get("r2_min", 0.75)))
            & (slope >= float(params.get("slope_min_pct", 0.5)) / 100.0)
            & (n_up / n >= float(params.get("up_ratio_min", 0.55)))
            & (vol_ud >= float(params.get("vol_ud_min", 1.2)))
            & (worst > float(params.get("worst_day_pct", -5.0)) / 100.0)
            & (std_c <= float(params.get("daily_std_max_pct", 3.0)) / 100.0)
            & (drawdown <= float(params.get("drawdown_max_pct", 8.0)) / 100.0)
            & (gain <= float(params.get("gain_max_pct", 45.0)) / 100.0)
            & (limit_ups <= int(params.get("limit_up_max", 2)))
            & (close > ma20)
        )
        lose_ma20 = close < ma20
        heavy_drop = (matrix_feature(market, "vol_ratio_5d") >= 2.0) & (chg <= -0.04)
        exit_ = lose_ma20 | heavy_drop
        return make_signal_matrix(
            market.shape,
            entry=np.nan_to_num(entry).astype(np.uint8),
            exit=np.nan_to_num(exit_).astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(lose_ma20, 0, np.where(heavy_drop, 1, -1)).astype(np.int16),
            entry_signal_ids=("signal_accumulation",),
            exit_signal_ids=("signal_ma20_breakdown", "signal_heavy_volume_drop"),
        )


MATRIX_STRATEGY = AccumulationStrategy()
