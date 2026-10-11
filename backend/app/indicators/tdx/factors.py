"""自定义指标 ↔ 因子注册表桥接 (kind="indicator")。

每个已保存指标的每条数值输出 (非 STICKLINE) 注册一个因子:
  id = ind_{indicator_id}_{output.key}, label = {指标名}.{输出名}, group = 自定义指标
供自定义信号条件 / 策略评分 / 回测引用; 物化走 strategy/scoring.py 的
materialize_scoring_columns → materialize_indicator_columns (同一指标一次求值)。
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Collection
from pathlib import Path

import polars as pl

from app.factors.registry import FactorSpec, get_factor, register_factor, unregister_factor
from app.indicators.tdx.compiler import compile_indicator, compile_indicator_cached, evaluate

logger = logging.getLogger(__name__)

GROUP = "自定义指标"
FACTOR_PREFIX = "ind_"

_lock = threading.Lock()
_registered: dict[str, frozenset[str]] = {}  # indicator id → 已注册因子 id


def factor_id(indicator_id: str, key: str) -> str:
    return f"{FACTOR_PREFIX}{indicator_id}_{key}"


def to_specs(definition: dict) -> list[FactorSpec]:
    """指标定义 → FactorSpec 列表; 编译失败抛 ValueError。"""
    indicator_id = str(definition["id"])
    formula = str(definition.get("formula", ""))
    compiled = compile_indicator(formula)
    if not compiled.ok:
        first = compiled.errors[0]
        raise ValueError(f"公式无效 [{first.code}]: {first.message}")
    name = str(definition.get("name") or indicator_id)
    return [
        FactorSpec(
            id=factor_id(indicator_id, output.key),
            label=f"{name}.{output.name}",
            group=GROUP,
            formula_text=formula,
            kind="indicator",
            dependencies=compiled.dependencies,
            warmup_bars=compiled.warmup_bars,
            scale_free=False,
            incremental_safe=False,
            indicator_ref=(indicator_id, output.key),
        )
        for output in compiled.outputs
        if output.expr is not None
    ]


def factor_ids_for(definition: dict) -> frozenset[str]:
    try:
        return frozenset(spec.id for spec in to_specs(definition))
    except ValueError:
        return frozenset()


def register_indicator(definition: dict) -> list[str]:
    """(重新) 注册一个指标的全部输出因子: 先注销旧集合再注册新集合。

    与非指标因子 id 冲突时拒绝 (ValueError), 不覆盖他人注册。
    """
    specs = to_specs(definition)
    indicator_id = str(definition["id"])
    with _lock:
        previous = _registered.get(indicator_id, frozenset())
        for spec in specs:
            existing = get_factor(spec.id)
            if existing is not None and existing.kind != "indicator":
                raise ValueError(f"因子 id 冲突: {spec.id}")
            if existing is not None and spec.id not in previous:
                raise ValueError(f"因子 id 已被其他指标占用: {spec.id}")
        for fid in previous:
            unregister_factor(fid)
        for spec in specs:
            unregister_factor(spec.id)
            register_factor(spec)
        _registered[indicator_id] = frozenset(spec.id for spec in specs)
    return [spec.id for spec in specs]


def unregister_indicator(indicator_id: str) -> list[str]:
    with _lock:
        removed = sorted(_registered.pop(indicator_id, frozenset()))
        for fid in removed:
            unregister_factor(fid)
    return removed


def load_into_registry(data_dir: Path) -> list[str]:
    """启动期注册全部已保存指标; 单个失败只跳过并告警。"""
    from app.indicators.tdx import store

    loaded: list[str] = []
    for definition in store.load_all(data_dir):
        try:
            register_indicator(definition)
            loaded.append(str(definition["id"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("indicator 注册失败 %s: %s", definition.get("id"), exc)
    return loaded


def materialize_indicator_columns(frame: pl.DataFrame, names: Collection[str]) -> pl.DataFrame:
    """把请求的 ind_* 因子列补到帧上; 同一指标只求值一次。

    降级 (告警跳过, 不抛异常; 缺列由下游 inject/评分按"不可计算"处理):
      - 依赖列缺失;
      - 单日快照 (每个 symbol 至多 1 行): 时序指标在单日帧上无意义
        (EMA 会退化为当日价), 与"因子依赖信号在盘中单日路径缺列跳过"同语义。
    """
    groups: dict[str, list[FactorSpec]] = {}
    for name in names:
        name = str(name)
        if name in frame.columns:
            continue
        spec = get_factor(name)
        if spec is None or spec.kind != "indicator" or len(spec.indicator_ref) != 2:
            continue
        groups.setdefault(spec.formula_text, []).append(spec)
    if not groups or frame.is_empty():
        return frame
    if "symbol" not in frame.columns or "date" not in frame.columns:
        logger.warning("indicator factors skipped: frame lacks symbol/date")
        return frame
    if frame.height <= frame["symbol"].n_unique():
        logger.warning("indicator factors skipped on single-day snapshot: %s", sorted(s.id for g in groups.values() for s in g))
        return frame
    for formula, specs in groups.items():
        compiled = compile_indicator_cached(formula)
        if not compiled.ok:
            continue
        missing = sorted(compiled.dependencies - set(frame.columns))
        if missing:
            logger.warning("indicator factors %s missing dependencies: %s", [s.id for s in specs], missing)
            continue
        wanted = {spec.indicator_ref[1]: spec.id for spec in specs}
        columns = {output.column: wanted[output.key] for output in compiled.outputs if output.key in wanted}
        evaluated = evaluate(compiled, frame)
        produced = [column for column in evaluated.columns if column not in frame.columns]
        frame = evaluated.rename(columns).drop([c for c in produced if c not in columns])
    return frame
