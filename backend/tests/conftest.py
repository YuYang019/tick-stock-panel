"""测试全局隔离 — 禁止任何测试写到真实的 data/user_data/preferences.json。

教训: 恢复出厂类接口 (如 PUT /api/sentiment-cycle/config, config=null) 会绕过
preferences.save 直接原子写盘; 测试只 monkeypatch load/save 时仍会覆盖用户真实
偏好。这里统一把 preferences 的文件路径指到每个测试独立的临时目录, 测试内
再 monkeypatch settings.data_dir 也不会冲突 (后者只影响其他文件)。
"""
from __future__ import annotations

import pytest

from app.services import preferences


@pytest.fixture(autouse=True)
def _isolate_preferences_file(tmp_path, monkeypatch):
    path = tmp_path / "user_data" / "preferences.json"

    def _isolated_path():
        # 与真实 _path() 一致: 调用时才创建父目录 (save 会写同目录下的 .tmp)。
        # 不在 fixture 里预先创建, 否则会与测试自己的 mkdir 冲突。
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(preferences, "_path", _isolated_path)
    preferences._invalidate_cache()
    yield path
    preferences._invalidate_cache()
