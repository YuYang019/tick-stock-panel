"""心法配置 —— schema 校验、出厂心法、用户心法读写。

出厂心法: backend/app/analysis/methods/fusion-v1.json (只读, 随版本升级)。
用户心法: data/user_data/ta_methods/{id}.json; 保存时写历史版本 (.history/)。
选择: preferences.ta_method = {"active": id, "per_scenario": {...}}。

校验失败一律回退出厂心法 (fail-closed), 不阻断分析流程。
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METHODS_DIR = Path(__file__).parent / "methods"
FACTORY_ID = "fusion-v1"
MAX_PRINCIPLE_CHARS = 8000
MAX_HISTORY_VERSIONS = 5

_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{2,40}$")
_SCOPES = frozenset({"market", "stock"})
_LEVELS = frozenset({"worldview", "method", "structure", "market", "sector", "stock"})
_HOOK_SOURCES = ("feature:", "signal:")
_HOOK_REQUIRES = ("sector_gate", "market_gate")
_HOOK_OPS = frozenset({"eq", "in", "gte", "lte", "gt", "lt"})
_SCENARIOS = ("stock_analysis", "watchlist_review", "market_review")


class MethodValidationError(ValueError):
    pass


class MethodConflictError(MethodValidationError):
    """目标 id 已存在 (复制/新建不得覆盖已有心法)。"""


# ================================================================
# 校验
# ================================================================

def validate(method: dict[str, Any]) -> dict[str, Any]:
    """校验并规范化心法配置; 非法时抛 MethodValidationError。

    规范化只做补默认值, 不丢弃用户内容; 结构性错误 (模块/特征 id 未注册、
    权重非法、场景未知) 直接拒绝。
    """
    from app.analysis.features import get_feature

    if not isinstance(method, dict):
        raise MethodValidationError("心法必须是 JSON 对象")
    mid = str(method.get("id") or "").strip()
    if not _ID_RE.match(mid):
        raise MethodValidationError(f"心法 id 非法: {mid!r}")
    out = {
        "id": mid,
        "name": str(method.get("name") or mid)[:60],
        "version": int(method.get("version") or 1),
        "based_on": method.get("based_on"),
        "modules": [],
        "maxims": [],
        "gates": copy.deepcopy(method.get("gates") or {}),
        "logic_quality": copy.deepcopy(method.get("logic_quality") or {"enabled": True}),
        "underlying_logic_questions": list(method.get("underlying_logic_questions") or [
            "谁在买?", "为什么买?", "能持续多久?",
        ])[:5],
        "synthesis": copy.deepcopy(method.get("synthesis") or {}),
        "scenarios": copy.deepcopy(method.get("scenarios") or {}),
    }

    modules = method.get("modules")
    if not isinstance(modules, list) or not modules:
        raise MethodValidationError("modules 不能为空")
    seen_modules: set[str] = set()
    market_modules = 0
    for mod in modules:
        if not isinstance(mod, dict):
            raise MethodValidationError("module 必须是对象")
        mod_id = str(mod.get("id") or "").strip()
        if not mod_id or mod_id in seen_modules:
            raise MethodValidationError(f"模块 id 为空或重复: {mod_id!r}")
        seen_modules.add(mod_id)
        scope = str(mod.get("scope") or "stock")
        if scope not in _SCOPES:
            raise MethodValidationError(f"模块 {mod_id} scope 非法: {scope}")
        if scope == "market":
            market_modules += 1
        weight = mod.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise MethodValidationError(f"模块 {mod_id} 权重非法: {weight!r}")
        features = []
        for feat in mod.get("features") or []:
            if isinstance(feat, str):
                feat = {"id": feat}
            if not isinstance(feat, dict) or not feat.get("id"):
                raise MethodValidationError(f"模块 {mod_id} 的特征配置非法")
            if get_feature(str(feat["id"])) is None:
                raise MethodValidationError(f"未注册的特征: {feat['id']}")
            features.append({"id": str(feat["id"]), "params": dict(feat.get("params") or {})})
        out["modules"].append({
            "id": mod_id,
            "name": str(mod.get("name") or mod_id)[:40],
            "scope": scope,
            "role": str(mod.get("role") or ""),
            "enabled": bool(mod.get("enabled", True)),
            "weight": float(weight),
            "features": features,
            "principles": str(mod.get("principles") or "")[:MAX_PRINCIPLE_CHARS],
            "checklist": [str(c)[:200] for c in (mod.get("checklist") or [])][:10],
            "phase_overrides": copy.deepcopy(mod.get("phase_overrides") or {}),
        })
    if market_modules > 1:
        raise MethodValidationError("scope=market 的模块最多 1 个")

    for maxim in method.get("maxims") or []:
        if not isinstance(maxim, dict) or not maxim.get("id"):
            raise MethodValidationError("maxim 缺少 id")
        level = str(maxim.get("level") or "stock")
        if level not in _LEVELS:
            raise MethodValidationError(f"心法层级非法: {level}")
        hooks = []
        for hook in maxim.get("hooks") or []:
            hooks.append(_validate_hook(hook))
        out["maxims"].append({
            "id": str(maxim["id"])[:40],
            "title": str(maxim.get("title") or maxim["id"])[:60],
            "level": level,
            "enabled": bool(maxim.get("enabled", True)),
            "explanation": str(maxim.get("explanation") or "")[:2000],
            "hooks": hooks,
            "prompt_principle": str(maxim.get("prompt_principle") or "")[:2000],
            "checklist": [str(c)[:200] for c in (maxim.get("checklist") or [])][:10],
            "scenarios": [s for s in (maxim.get("scenarios") or list(_SCENARIOS)) if s in _SCENARIOS],
        })

    _validate_gates(out["gates"])
    scenarios = out["scenarios"]
    for name in scenarios:
        if name not in _SCENARIOS:
            raise MethodValidationError(f"未知场景: {name}")
    return out


def _validate_hook(hook: Any) -> dict:
    if not isinstance(hook, dict):
        raise MethodValidationError("hook 必须是对象")
    source = str(hook.get("source") or "")
    if not source.startswith(_HOOK_SOURCES):
        raise MethodValidationError(f"hook.source 非法: {source!r}")
    when = hook.get("when") or {}
    if not isinstance(when, dict) or not when:
        raise MethodValidationError("hook.when 不能为空")
    for op in when:
        if op not in _HOOK_OPS:
            raise MethodValidationError(f"hook.when 运算符非法: {op}")
    require = [str(r) for r in (hook.get("require") or [])]
    for item in require:
        if item not in _HOOK_REQUIRES and not item.startswith("cycle_in:"):
            raise MethodValidationError(f"hook.require 非法: {item}")
    mul = hook.get("confidence_mul", 1.0)
    if not isinstance(mul, (int, float)) or not 0 < mul <= 1:
        raise MethodValidationError(f"confidence_mul 非法: {mul}")
    return {
        "source": source,
        "when": {k: v for k, v in when.items()},
        "require": require,
        "tag": str(hook.get("tag") or "")[:40],
        "confidence_mul": float(mul),
    }


def _validate_gates(gates: dict) -> None:
    if not isinstance(gates, dict):
        raise MethodValidationError("gates 必须是对象")
    for name in ("market", "sector"):
        gate = gates.get(name)
        if gate is None:
            continue
        if not isinstance(gate, dict):
            raise MethodValidationError(f"gate {name} 必须是对象")
        mul = gate.get("confidence_mul", 1.0)
        if not isinstance(mul, (int, float)) or not 0 < mul <= 1:
            raise MethodValidationError(f"gate {name} confidence_mul 非法")


# ================================================================
# 文件读写
# ================================================================

def _user_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "user_data" / "ta_methods"


def factory_method() -> dict:
    """出厂心法 (校验通过后返回; 出厂文件损坏时回退最小可用配置)。"""
    path = METHODS_DIR / f"{FACTORY_ID}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return validate(raw)
    except Exception as e:  # noqa: BLE001
        logger.error("出厂心法加载失败 (%s): %s", path, e)
        return _minimal_method()


def _minimal_method() -> dict:
    return validate({
        "id": FACTORY_ID,
        "name": "融合心法 v1",
        "modules": [
            {"id": "trend", "name": "趋势结构", "scope": "stock", "weight": 1.0,
             "features": ["swing_structure", "ma_alignment"], "principles": "先看趋势结构。"},
        ],
    })


def list_methods(data_dir: Path) -> list[dict]:
    """出厂 + 用户心法列表 (不含正文, 供列表页)。"""
    out = [_summary(factory_method(), factory=True)]
    for path in sorted(_user_dir(data_dir).glob("*.json")):
        try:
            method = validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            logger.warning("用户心法解析失败 %s: %s", path, e)
            continue
        out.append(_summary(method, factory=False))
    return out


def _summary(method: dict, *, factory: bool) -> dict:
    return {
        "id": method["id"],
        "name": method["name"],
        "version": method["version"],
        "factory": factory,
        "module_count": len(method["modules"]),
        "maxim_count": len(method.get("maxims") or []),
        "description": next(
            (m.get("principles", "")[:80] for m in method["modules"] if m.get("principles")),
            "",
        ),
    }


def load_method(method_id: str, data_dir: Path) -> dict | None:
    if method_id == FACTORY_ID:
        return factory_method()
    # id 来自 URL / 偏好设置, 先过白名单再拼路径, 防止 ../ 读到目录之外的文件
    if not _ID_RE.match(str(method_id or "")):
        return None
    path = _user_dir(data_dir) / f"{method_id}.json"
    if not path.exists():
        return None
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:  # noqa: BLE001
        logger.warning("心法 %s 校验失败, 回退出厂心法: %s", method_id, e)
        return None


def save_method(data_dir: Path, method: dict) -> dict:
    """保存用户心法 (校验后写盘, 并保留最近 5 个历史版本)。"""
    from app.services.fs_utils import atomic_write_text

    validated = validate(method)
    if validated["id"] == FACTORY_ID:
        raise MethodValidationError("出厂心法不可覆盖, 请先复制为新 id")
    target = _user_dir(data_dir) / f"{validated['id']}.json"
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
            prev_version = int(previous.get("version") or 1)
        except Exception:  # noqa: BLE001
            prev_version = 0
        if prev_version >= validated["version"]:
            validated["version"] = prev_version + 1
        _archive(data_dir, validated["id"], previous)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(validated, ensure_ascii=False, indent=2))
    return validated


def delete_method(data_dir: Path, method_id: str) -> None:
    if method_id == FACTORY_ID:
        raise MethodValidationError("出厂心法不可删除")
    if not _ID_RE.match(method_id):
        raise MethodValidationError(f"心法 id 非法: {method_id!r}")
    path = _user_dir(data_dir) / f"{method_id}.json"
    path.unlink(missing_ok=True)


def _archive(data_dir: Path, method_id: str, content: dict) -> None:
    from app.services.fs_utils import atomic_write_text

    history = _user_dir(data_dir) / ".history"
    history.mkdir(parents=True, exist_ok=True)
    version = int(content.get("version") or 1)
    atomic_write_text(history / f"{method_id}.{version}.json",
                      json.dumps(content, ensure_ascii=False, indent=2))
    versions = sorted(history.glob(f"{method_id}.*.json"))
    for old in versions[:-MAX_HISTORY_VERSIONS]:
        old.unlink(missing_ok=True)


def copy_method(data_dir: Path, source_id: str, new_id: str, name: str | None = None) -> dict:
    if not _ID_RE.match(str(new_id or "")):
        raise MethodValidationError(f"心法 id 非法: {new_id!r}")
    if new_id == FACTORY_ID or (_user_dir(data_dir) / f"{new_id}.json").exists():
        raise MethodConflictError(f"心法 id 已存在: {new_id}")
    source = load_method(source_id, data_dir)
    if source is None:
        raise MethodValidationError(f"心法不存在: {source_id}")
    clone = copy.deepcopy(source)
    clone["id"] = new_id
    clone["name"] = name or f"{source['name']} 副本"
    clone["version"] = 1
    clone["based_on"] = source_id
    return save_method(data_dir, clone)


def resolve_for_scenario(data_dir: Path, scenario: str, selection: dict | None) -> dict:
    """按场景选择心法: per_scenario > active > 出厂; 加载失败回退出厂。"""
    selection = selection or {}
    per = selection.get("per_scenario") or {}
    method_id = per.get(scenario) or selection.get("active") or FACTORY_ID
    method = load_method(method_id, data_dir)
    if method is None:
        logger.warning("心法 %s 不可用, 回退出厂心法", method_id)
        method = factory_method()
    return method


def suggest_copy_id(data_dir: Path, source_id: str) -> str:
    """为复制操作生成未占用的 id: {source}-copy, {source}-copy-2, ..."""
    base = f"{source_id}-copy"[:34]
    existing = {p.stem for p in _user_dir(data_dir).glob("*.json")} | {FACTORY_ID}
    if base not in existing:
        return base
    for i in range(2, 1000):
        candidate = f"{base}-{i}"
        if candidate not in existing:
            return candidate
    raise MethodConflictError("无可用的副本 id")
