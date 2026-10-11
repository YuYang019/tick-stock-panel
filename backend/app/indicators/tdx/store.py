"""自定义指标存储 — data/user_data/indicators/{id}.json。

镜像 factors/store.py / custom_signals 的持久化写法: 原子写, 单文件损坏只跳过
并告警。id 正则 + 目录边界校验防穿越; 保存前必须编译通过 (fail-closed)。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from app.indicators.tdx.compiler import compile_indicator
from app.services.fs_utils import atomic_write_text

logger = logging.getLogger(__name__)

ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
PANES = frozenset({"main", "sub"})
SEED_MARKER = ".seeded_v1"

ZHIXING_FORMULA = (
    "知行短期趋势线:EMA(EMA(C,10),10),COLORWHITE,LINETHICK1;\n"
    "知行多空线:(MA(CLOSE,14)+MA(CLOSE,28)+MA(CLOSE,57)+MA(CLOSE,114))/4,COLORYELLOW;\n"
)

SEED_INDICATORS: tuple[dict, ...] = (
    {
        "id": "zhixing",
        "name": "知行",
        "formula": ZHIXING_FORMULA,
        "pane": "main",
        "description": "知行短期趋势线 = EMA(EMA(C,10),10); 知行多空线 = 14/28/57/114 日均线均值",
        "builtin": True,
    },
)


def _dir(data_dir: Path) -> Path:
    directory = Path(data_dir) / "user_data" / "indicators"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _path(data_dir: Path, indicator_id: str) -> Path:
    """id → 文件路径; id 非法或解析后越出目录即 ValueError (防穿越)。"""
    if not isinstance(indicator_id, str) or not ID_RE.match(indicator_id):
        raise ValueError(f"指标 id 非法 (仅小写字母数字下划线, 1-40 字符): {indicator_id!r}")
    directory = _dir(data_dir).resolve()
    target = (directory / f"{indicator_id}.json").resolve()
    if target.parent != directory:
        raise ValueError(f"指标路径越界: {indicator_id!r}")
    return target


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_all(data_dir: Path) -> list[dict]:
    """读取全部指标定义 (按 id 排序); 损坏/非法文件跳过并告警。"""
    out: list[dict] = []
    for file in sorted(_dir(data_dir).glob("*.json")):
        try:
            definition = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("indicator load failed %s: %s", file.name, exc)
            continue
        if not isinstance(definition, dict) or definition.get("id") != file.stem or not ID_RE.match(file.stem):
            logger.warning("indicator file skipped (id mismatch) %s", file.name)
            continue
        out.append(definition)
    return out


def exists(data_dir: Path, indicator_id: str) -> bool:
    return _path(data_dir, indicator_id).exists()


def load_one(data_dir: Path, indicator_id: str) -> dict | None:
    try:
        target = _path(data_dir, indicator_id)
    except ValueError:
        return None
    if not target.exists():
        return None
    try:
        definition = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("indicator load failed %s: %s", target.name, exc)
        return None
    return definition if isinstance(definition, dict) else None


def validate(definition: dict) -> None:
    """schema + 编译校验; 非法抛 ValueError (含首个编译错误)。"""
    indicator_id = definition.get("id")
    if not isinstance(indicator_id, str) or not ID_RE.match(indicator_id):
        raise ValueError(f"指标 id 非法 (仅小写字母数字下划线, 1-40 字符): {indicator_id!r}")
    name = definition.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("指标名称不能为空")
    if definition.get("pane") not in PANES:
        raise ValueError("pane 必须是 main 或 sub")
    compiled = compile_indicator(str(definition.get("formula", "")))
    if not compiled.ok:
        first = compiled.errors[0]
        raise ValueError(f"公式无效 [{first.code}] 第{first.line}行第{first.col}列: {first.message}")


def save_one(data_dir: Path, definition: dict) -> None:
    validate(definition)
    atomic_write_text(_path(data_dir, definition["id"]), json.dumps(definition, ensure_ascii=False, indent=2))


def delete_one(data_dir: Path, indicator_id: str) -> bool:
    target = _path(data_dir, indicator_id)
    if target.exists():
        target.unlink()
        return True
    return False


def seed_if_empty(data_dir: Path) -> int:
    """预设种子: 首次启动写入内置预设; .seeded_v1 标记后不再种入 (幂等, 不复活)。"""
    directory = _dir(data_dir)
    marker = directory / SEED_MARKER
    if marker.exists():
        return 0
    written = 0
    for seed in SEED_INDICATORS:
        if _path(data_dir, seed["id"]).exists():
            continue
        stamp = now()
        save_one(data_dir, {**seed, "created_at": stamp, "updated_at": stamp})
        written += 1
    marker.write_text(f"seeded: {written}\n", encoding="utf-8")
    return written
