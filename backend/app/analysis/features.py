"""规则层特征注册表 —— 每个特征一个纯函数。

特征输入是单只标的的日线帧 (enriched 列, 按日期升序), 输出 FeatureResult:
- score: -1..1 的方向性评分 (None 表示数据不足, 不可计算);
- state: 短标签, 直接进入提示词与看板;
- facts: 给模型看的数值字典;
- evidence: 一句话依据列表。

本模块不依赖回测引擎。特征若要进入回测, 需经因子注册表注册为可表达式化的
因子 (virtual/custom); 当前特征含多阶段逻辑, 暂以规则层评估输出, 后续可按
DSL 重写为 custom 因子再做 IC/分层检验。
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    score: float | None
    state: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": None if self.score is None else round(self.score, 3),
            "state": self.state,
            "facts": self.facts,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    label: str
    module: str
    fn: Callable[[pl.DataFrame, dict], FeatureResult]
    params: dict[str, Any] = field(default_factory=dict)
    factor: bool = False


_REGISTRY: dict[str, FeatureSpec] = {}


def register_feature(spec: FeatureSpec) -> None:
    _REGISTRY[spec.id] = spec


def get_feature(feature_id: str) -> FeatureSpec | None:
    return _REGISTRY.get(feature_id)


def list_features() -> list[dict]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "module": spec.module,
            "params": spec.params,
            "factor": spec.factor,
        }
        for spec in _REGISTRY.values()
    ]


def run_feature(feature_id: str, df: pl.DataFrame, params: dict | None = None) -> FeatureResult:
    spec = _REGISTRY.get(feature_id)
    if spec is None:
        return FeatureResult(None, state="未注册", evidence=[f"特征 {feature_id} 未注册"])
    merged = {**spec.params, **(params or {})}
    try:
        return spec.fn(df, merged)
    except Exception as e:  # noqa: BLE001 — 单特征失败不应让整份分析失败
        logger.warning("feature %s failed: %s", feature_id, e)
        return FeatureResult(None, state="计算失败", evidence=[str(e)[:80]])


# ================================================================
# 工具
# ================================================================

def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _col(df: pl.DataFrame, name: str) -> list[float | None]:
    if name not in df.columns:
        return []
    return [_num(v) for v in df[name].to_list()]


def _last(values: list[float | None]) -> float | None:
    for v in reversed(values):
        if v is not None:
            return v
    return None


def _clamp(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


# ================================================================
# 趋势结构 (道氏 + 趋势跟踪)
# ================================================================

def _pivots(highs: list[float | None], lows: list[float | None], lookback: int) -> tuple[list[float], list[float]]:
    """分型高低点: 窗口内最高/最低即为枢轴。"""
    ph: list[float] = []
    pl_: list[float] = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        window_h = [h for h in highs[i - lookback:i + lookback + 1] if h is not None]
        window_l = [lo for lo in lows[i - lookback:i + lookback + 1] if lo is not None]
        if len(window_h) < 2 * lookback + 1 or len(window_l) < 2 * lookback + 1:
            continue
        h = highs[i]
        lo = lows[i]
        if h is not None and h >= max(window_h):
            ph.append(h)
        if lo is not None and lo <= min(window_l):
            pl_.append(lo)
    return ph, pl_


def _swing_structure(df: pl.DataFrame, params: dict) -> FeatureResult:
    lookback = int(params.get("swing_lookback", 5))
    highs = _col(df, "high")
    lows = _col(df, "low")
    ph, plw = _pivots(highs, lows, lookback)
    if len(ph) < 2 or len(plw) < 2:
        return FeatureResult(None, state="数据不足")
    hh = ph[-1] > ph[-2]
    hl = plw[-1] > plw[-2]
    if hh and hl:
        state, score = "HH-HL 上升结构", 0.8
    elif not hh and not hl:
        state, score = "LH-LL 下降结构", -0.8
    elif hh and not hl:
        state, score = "高点抬高/低点降低 扩张", 0.2
    else:
        state, score = "高点降低/低点抬高 收敛", 0.0
    return FeatureResult(score, state, {
        "pivot_highs": [round(x, 2) for x in ph[-3:]],
        "pivot_lows": [round(x, 2) for x in plw[-3:]],
    }, [f"最近两个高点 {ph[-2]:.2f}→{ph[-1]:.2f}", f"最近两个低点 {plw[-2]:.2f}→{plw[-1]:.2f}"])


def _ma_alignment(df: pl.DataFrame, params: dict) -> FeatureResult:
    ma5, ma10, ma20, ma60 = (_last(_col(df, k)) for k in ("ma5", "ma10", "ma20", "ma60"))
    close = _last(_col(df, "close"))
    if None in (ma5, ma10, ma20, ma60, close):
        return FeatureResult(None, state="数据不足")
    ma20_series = [v for v in _col(df, "ma20") if v is not None]
    slope = None
    if len(ma20_series) >= 6 and ma20_series[-6]:
        slope = ma20_series[-1] / ma20_series[-6] - 1
    if ma5 > ma10 > ma20 > ma60:
        state, score = "多头排列", 0.8
    elif ma5 < ma10 < ma20 < ma60:
        state, score = "空头排列", -0.8
    else:
        state, score = "均线缠绕", 0.0
    if slope is not None:
        score += _clamp(slope * 10, -0.2, 0.2)
        state += "·MA20上行" if slope > 0.005 else ("·MA20下行" if slope < -0.005 else "")
    return FeatureResult(_clamp(score), state, {
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "ma20_slope_5d": None if slope is None else round(slope * 100, 2),
    }, [f"MA5/10/20/60 = {ma5:.2f}/{ma10:.2f}/{ma20:.2f}/{ma60:.2f}"])


# ================================================================
# 动量
# ================================================================

def _relative_strength(df: pl.DataFrame, params: dict) -> FeatureResult:
    close = _col(df, "close")
    if len(close) < 21 or not close[-1] or not close[-21]:
        return FeatureResult(None, state="数据不足")
    ret20 = close[-1] / close[-21] - 1
    benchmark = params.get("benchmark_ret20")
    rs = None if benchmark is None else ret20 - float(benchmark)
    if rs is None:
        state = "绝对动量"
        score = _clamp(ret20 * 5)
    else:
        state = "强于大盘" if rs > 0 else "弱于大盘"
        score = _clamp(rs * 10)
    return FeatureResult(score, state, {
        "ret20": round(ret20 * 100, 2),
        "relative_20d": None if rs is None else round(rs * 100, 2),
    }, [f"20 日收益 {ret20 * 100:+.2f}%"] + ([f"相对基准 {rs * 100:+.2f}%"] if rs is not None else []))


def _macd_momentum(df: pl.DataFrame, params: dict) -> FeatureResult:
    hist = _col(df, "macd_hist")
    dif = _col(df, "macd_dif")
    hist = [h for h in hist if h is not None]
    if len(hist) < 5:
        return FeatureResult(None, state="数据不足")
    slope = hist[-1] - hist[-4]
    above_zero = dif and dif[-1] is not None and dif[-1] > 0
    if slope > 0 and above_zero:
        state, score = "红柱扩张", 0.7
    elif slope > 0:
        state, score = "零轴下金叉动能", 0.3
    elif slope < 0 and not above_zero:
        state, score = "绿柱扩张", -0.7
    else:
        state, score = "红柱收敛", -0.3
    return FeatureResult(score, state, {
        "macd_hist": hist[-1], "hist_slope_3d": round(slope, 4),
    }, [f"MACD 柱 {hist[-1]:.4f}, 3 日变化 {slope:+.4f}"])


def _new_high_proximity(df: pl.DataFrame, params: dict) -> FeatureResult:
    close = _last(_col(df, "close"))
    high60 = _last(_col(df, "high_60d"))
    if close is None or not high60:
        return FeatureResult(None, state="数据不足")
    dist = close / high60 - 1
    if dist >= 0:
        state, score = "创 60 日新高", 0.9
    elif dist > -0.03:
        state, score = "逼近 60 日高点", 0.5
    elif dist > -0.1:
        state, score = "距高点 10% 内", 0.1
    else:
        state, score = "远离 60 日高点", -0.3
    return FeatureResult(score, state, {"distance_to_high_60d": round(dist * 100, 2)},
                         [f"距 60 日高点 {dist * 100:+.2f}%"])


# ================================================================
# 量价
# ================================================================

def _vp_confirmation(df: pl.DataFrame, params: dict) -> FeatureResult:
    pct = _col(df, "change_pct")
    vol = _col(df, "volume")
    if len(pct) < 20 or len(vol) < 20:
        return FeatureResult(None, state="数据不足")
    up_vol = [v for p, v in zip(pct[-20:], vol[-20:], strict=False) if p is not None and v is not None and p > 0]
    down_vol = [v for p, v in zip(pct[-20:], vol[-20:], strict=False) if p is not None and v is not None and p < 0]
    if not up_vol or not down_vol:
        return FeatureResult(None, state="数据不足")
    ratio = (sum(up_vol) / len(up_vol)) / (sum(down_vol) / len(down_vol))
    latest_pct = pct[-1]
    latest_vol = vol[-1]
    avg_vol = sum(v for v in vol[-6:-1] if v is not None) / max(1, len([v for v in vol[-6:-1] if v is not None]))
    vol_ratio = latest_vol / avg_vol if latest_vol and avg_vol else None
    if ratio > 1.2:
        state, score = "上涨放量·量价配合", 0.6
    elif ratio < 0.8:
        state, score = "上涨缩量·量价背离", -0.5
    else:
        state, score = "量价中性", 0.0
    if latest_pct is not None and vol_ratio is not None and latest_pct > 0.03 and vol_ratio < 0.7:
        state, score = "缩量上涨·动能不足", -0.4
    return FeatureResult(score, state, {
        "up_down_volume_ratio": round(ratio, 2),
        "latest_vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
    }, [f"20 日上涨/下跌均量比 {ratio:.2f}", f"当日量比 {vol_ratio:.2f}" if vol_ratio else ""])


def _effort_result(df: pl.DataFrame, params: dict) -> FeatureResult:
    """威科夫「努力与结果」: 放量但价格推进很小 = 努力无结果。"""
    vol_ratio = _last(_col(df, "vol_ratio_5d"))
    pct = _last(_col(df, "change_pct"))
    if vol_ratio is None or pct is None:
        return FeatureResult(None, state="数据不足")
    if vol_ratio >= 1.8 and abs(pct) < 0.01:
        state, score = "放量滞涨·努力无结果", -0.5
    elif vol_ratio >= 1.8 and pct > 0.02:
        state, score = "放量推进", 0.5
    elif vol_ratio <= 0.7 and pct < 0:
        state, score = "缩量回调", 0.2
    else:
        state, score = "量价常态", 0.0
    return FeatureResult(score, state, {
        "vol_ratio_5d": vol_ratio, "change_pct": pct,
    }, [f"量比 {vol_ratio:.2f}, 涨跌 {pct * 100:+.2f}%"])


def _consensus_state(df: pl.DataFrame, params: dict) -> FeatureResult:
    """分歧 / 一致状态 (心法⑥)。"""
    turnover = _last(_col(df, "turnover_rate"))
    amplitude = _last(_col(df, "amplitude")) or 0.0
    boards = _last(_col(df, "consecutive_limit_ups")) or 0
    high = _last(_col(df, "high"))
    low = _last(_col(df, "low"))
    close = _last(_col(df, "close"))
    close_pos = None
    if high is not None and low is not None and close is not None and high > low:
        close_pos = (close - low) / (high - low)
    if boards > 0 and turnover is not None and turnover < 3:
        state, score = "一致加速", -0.3
    elif turnover is not None and turnover >= 8 and amplitude >= 0.07:
        if close_pos is not None and close_pos >= 0.5:
            state, score = "分歧换手", 0.2
        else:
            state, score = "分歧转弱", -0.6
    else:
        state, score = "中性", 0.0
    return FeatureResult(score, state, {
        "turnover_rate": turnover, "amplitude": amplitude,
        "close_position": None if close_pos is None else round(close_pos, 2),
    }, [f"换手 {turnover:.2f}%, 振幅 {amplitude * 100:.2f}%" if turnover is not None else "换手缺失"])


# ================================================================
# 关键位置 + 价格行为
# ================================================================

def _level_proximity(df: pl.DataFrame, params: dict) -> FeatureResult:
    from app.indicators.levels import compute_levels

    close = _last(_col(df, "close"))
    atr = _last(_col(df, "atr_14"))
    if close is None or not atr:
        return FeatureResult(None, state="数据不足")
    levels = compute_levels(df)
    supports = [p["value"] for p in levels.get("sr", []) if p["value"] < close]
    resistances = [p["value"] for p in levels.get("sr", []) if p["value"] > close]
    support = max(supports) if supports else None
    resistance = min(resistances) if resistances else None
    to_support = None if support is None else (close - support) / atr
    to_resistance = None if resistance is None else (resistance - close) / atr
    if to_support is not None and to_support <= 1.0 and (to_resistance is None or to_resistance > 2):
        state, score = "贴近支撑", 0.4
    elif to_resistance is not None and to_resistance <= 1.0:
        state, score = "贴近压力", -0.3
    else:
        state, score = "位于区间中部", 0.0
    return FeatureResult(score, state, {
        "support": support, "resistance": resistance,
        "support_atr": None if to_support is None else round(to_support, 2),
        "resistance_atr": None if to_resistance is None else round(to_resistance, 2),
    }, [f"最近支撑 {support}" if support else "无支撑位", f"最近压力 {resistance}" if resistance else "无压力位"])


def _breakout_quality(df: pl.DataFrame, params: dict) -> FeatureResult:
    close = _last(_col(df, "close"))
    high60 = _last(_col(df, "high_60d"))
    vol_ratio = _last(_col(df, "vol_ratio_5d"))
    if close is None or not high60:
        return FeatureResult(None, state="数据不足")
    if close <= high60:
        return FeatureResult(0.0, "未突破", {"distance": round((close / high60 - 1) * 100, 2)})
    strong = vol_ratio is not None and vol_ratio >= 1.3
    state, score = ("放量突破", 0.7) if strong else ("缩量突破·待确认", -0.1)
    return FeatureResult(score, state, {"vol_ratio_5d": vol_ratio},
                         [f"突破 60 日高点, 量比 {vol_ratio:.2f}" if vol_ratio else "突破 60 日高点"])


def _spring_upthrust(df: pl.DataFrame, params: dict) -> FeatureResult:
    """区间上下沿假突破: 跌破下沿后 3 日内收回 = Spring; 反之 = Upthrust。"""
    lookback = int(params.get("range_lookback", 20))
    close = _col(df, "close")
    high = _col(df, "high")
    low = _col(df, "low")
    if len(close) < lookback + 4:
        return FeatureResult(None, state="数据不足")
    window_low = min(v for v in low[-lookback - 3:-3] if v is not None)
    window_high = max(v for v in high[-lookback - 3:-3] if v is not None)
    recent_low = min(v for v in low[-3:] if v is not None)
    recent_high = max(v for v in high[-3:] if v is not None)
    last_close = close[-1]
    if last_close is None:
        return FeatureResult(None, state="数据不足")
    if recent_low < window_low and last_close > window_low:
        return FeatureResult(0.6, "Spring 假跌破收回",
                             {"range_low": window_low}, [f"跌破 {window_low:.2f} 后收回"])
    if recent_high > window_high and last_close < window_high:
        return FeatureResult(-0.6, "Upthrust 假突破回落",
                             {"range_high": window_high}, [f"突破 {window_high:.2f} 后回落"])
    return FeatureResult(0.0, "无假突破事件")


def _range_context(df: pl.DataFrame, params: dict) -> FeatureResult:
    """区间判定 + 区间前趋势 → 吸筹/派发候选 (威科夫阶段)。"""
    lookback = int(params.get("range_lookback", 20))
    close = [v for v in _col(df, "close") if v is not None]
    if len(close) < lookback * 2:
        return FeatureResult(None, state="数据不足")
    recent = close[-lookback:]
    prior = close[-lookback * 2:-lookback]
    band = (max(recent) - min(recent)) / (sum(recent) / len(recent))
    in_range = band < 0.12
    prior_trend = prior[-1] / prior[0] - 1 if prior[0] else 0
    if in_range and prior_trend < -0.08:
        return FeatureResult(0.2, "下跌后区间整理·吸筹候选",
                             {"band": round(band * 100, 2), "prior_trend": round(prior_trend * 100, 2)})
    if in_range and prior_trend > 0.08:
        return FeatureResult(-0.2, "上涨后区间整理·派发候选",
                             {"band": round(band * 100, 2), "prior_trend": round(prior_trend * 100, 2)})
    if not in_range and prior_trend > 0:
        return FeatureResult(0.5, "上升趋势延续", {"band": round(band * 100, 2)})
    if not in_range and prior_trend < 0:
        return FeatureResult(-0.5, "下降趋势延续", {"band": round(band * 100, 2)})
    return FeatureResult(0.0, "震荡", {"band": round(band * 100, 2)})


# ================================================================
# 树干 / 逻辑质量
# ================================================================

def _sector_strength(df: pl.DataFrame, params: dict) -> FeatureResult:
    rs_pct = params.get("sector_rs_pct")
    phase = params.get("sector_phase")
    if rs_pct is None:
        return FeatureResult(None, state="无板块数据")
    score = _clamp((float(rs_pct) - 0.5) * 1.6)
    if phase in {"发酵", "高潮"}:
        score += 0.2
    elif phase in {"退潮", "冰点"}:
        score -= 0.2
    state = f"板块强度分位 {float(rs_pct):.2f}" + (f"·{phase}" if phase else "")
    return FeatureResult(_clamp(score), state, {"sector_rs_pct": rs_pct, "sector_phase": phase})


def _fundamental_floor(df: pl.DataFrame, params: dict) -> FeatureResult:
    fins = params.get("financials") or {}
    metrics = (fins.get("metrics") or [])[:2]
    income = (fins.get("income") or [])[:2]
    if not metrics and not income:
        return FeatureResult(None, state="无财务数据")
    # 字段名为数据源标准化后的规范口径 (见 plugins/*/provider 的 metrics/income 映射,
    # 与 backtest/fundamentals 同源): roe / net_income_yoy / revenue_yoy 均为百分点数值。
    row = metrics[0] if metrics else {}
    inc = income[0] if income else {}
    roe = _num(row.get("roe"))
    yoy = _num(row.get("net_income_yoy"))
    revenue_yoy = _num(row.get("revenue_yoy"))
    profit = _num(inc.get("net_income_attributable"))
    if profit is None:
        profit = _num(inc.get("net_income"))
    issues = []
    if profit is not None and profit < 0:
        issues.append("净利润为负")
    if yoy is not None and yoy < -30:
        issues.append(f"净利同比 {yoy:.0f}%")
    if roe is not None and roe < 3:
        issues.append(f"ROE {roe:.1f}% 偏低")
    if not issues:
        state, score = ("盈利稳健", 0.4) if (roe is None or roe >= 8) else ("基本面一般", 0.0)
    else:
        state, score = "基本面薄弱", -0.4
    return FeatureResult(score, state, {
        "roe": roe, "net_profit_yoy": yoy, "revenue_yoy": revenue_yoy,
    }, issues or ["未见明显财务瑕疵"])


def _narrative_rule(df: pl.DataFrame, params: dict) -> FeatureResult:
    heat = _num(params.get("concept_heat"))       # 0..1 所属概念热度分位
    limit_ups = _num(params.get("recent_limit_ups")) or 0
    phase = params.get("sector_phase")
    if heat is None and not limit_ups:
        return FeatureResult(None, state="无叙事线索")
    score = _clamp((heat or 0.3) * 1.2 + min(limit_ups, 5) * 0.08)
    if phase in {"发酵", "高潮"}:
        score = _clamp(score + 0.1)
    state = f"叙事热度 {int((heat or 0) * 100)}%" + (f"·近{int(limit_ups)}次涨停" if limit_ups else "")
    return FeatureResult(score, state, {"concept_heat": heat, "recent_limit_ups": limit_ups})


# ================================================================
# 注册
# ================================================================

_BUILTIN: tuple[FeatureSpec, ...] = (
    FeatureSpec("swing_structure", "摆动结构", "trend", _swing_structure,
                {"swing_lookback": 5}, factor=False),
    FeatureSpec("ma_alignment", "均线排列", "trend", _ma_alignment, {}, factor=False),
    FeatureSpec("relative_strength", "相对强弱", "momentum", _relative_strength, {}, factor=False),
    FeatureSpec("macd_momentum", "MACD 动量", "momentum", _macd_momentum, {}, factor=False),
    FeatureSpec("new_high_proximity", "新高距离", "momentum", _new_high_proximity, {}, factor=False),
    FeatureSpec("vp_confirmation", "量价配合", "volume_price", _vp_confirmation, {}, factor=False),
    FeatureSpec("effort_result", "努力与结果", "volume_price", _effort_result, {}, factor=False),
    FeatureSpec("consensus_state", "分歧/一致", "volume_price", _consensus_state, {}, factor=False),
    FeatureSpec("level_proximity", "关键价位距离", "key_levels", _level_proximity, {}, factor=False),
    FeatureSpec("breakout_quality", "突破质量", "key_levels", _breakout_quality, {}, factor=False),
    FeatureSpec("spring_upthrust", "假突破事件", "key_levels", _spring_upthrust,
                {"range_lookback": 20}, factor=False),
    FeatureSpec("range_context", "区间与阶段", "phase", _range_context,
                {"range_lookback": 20}, factor=False),
    FeatureSpec("sector_strength", "板块强度", "structure", _sector_strength, {}, factor=False),
    FeatureSpec("fundamental_floor", "基本面底线", "logic", _fundamental_floor, {}, factor=False),
    FeatureSpec("narrative_rule", "叙事规则分", "logic", _narrative_rule, {}, factor=False),
)


def register_builtin_features() -> None:
    for spec in _BUILTIN:
        register_feature(spec)


register_builtin_features()
