"""AI 策略 META 规范化回归测试。"""
from __future__ import annotations

import pytest

from app.api.strategy import _normalize_build_result, _normalize_strategy_meta
from app.strategy.ai_generator import AIStrategyGenerator

RAW_CODE = '''"""模型返回的策略"""
import polars as pl

META = {
    "id": "custom_wrong_id",
    "name": "English Placeholder",
    "description": "model desc",
    "tags": ["AI"],
    "params": [],
    "scoring": {},
}

ENTRY_SIGNALS = []
EXIT_SIGNALS = []
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 20

def filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    return pl.lit(True)
'''


def test_normalize_strategy_meta_forces_ai_id_and_chinese_name():
    code = _normalize_strategy_meta(
        RAW_CODE,
        "ai_test123",
        name="断板反包",
        description="中文描述",
    )

    assert '"id": "ai_test123"' in code
    assert '"name": "断板反包"' in code
    assert '"description": "中文描述"' in code
    assert "custom_wrong_id" not in code
    assert "English Placeholder" not in code


def test_normalize_build_result_updates_code_and_meta():
    result = {"code": RAW_CODE, "meta": {}, "valid": True, "error": None}

    normalized = _normalize_build_result(
        result,
        "ai_from_frontend",
        name="中文策略名",
        description="前端描述",
    )

    assert normalized["valid"] is True
    assert normalized["meta"]["id"] == "ai_from_frontend"
    assert normalized["meta"]["name"] == "中文策略名"
    assert normalized["meta"]["description"] == "前端描述"
    assert '"id": "ai_from_frontend"' in normalized["code"]


def test_normalize_strategy_meta_inserts_missing_name_fields():
    raw = '''import polars as pl

META = {
    "id": "wrong",
    "tags": []
}

def filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    return pl.lit(True)
'''

    code = _normalize_strategy_meta(raw, "ai_inserted", name="中文名", description="描述")

    compile(code, "<strategy>", "exec")
    assert '"id": "ai_inserted"' in code
    assert '"name": "中文名"' in code
    assert '"description": "描述"' in code


# --- 回归: LLM 偏移写法 -------------------------------------------------
# 模型常给 META 加类型注解 (META: dict = {...}, ast.AnnAssign 节点)。
# 旧版匹配器只遍历 ast.Assign, 漏掉注解形式 → 报「找不到 META 字典」。

ANNOTATED_CODE = '''"""模型返回的策略 (带类型注解的 META — LLM 常见偏移)"""
import polars as pl

META: dict = {
    "id": "annotated_wrong_id",
    "name": "Placeholder",
    "description": "model desc",
    "tags": ["AI"],
    "params": [],
    "scoring": {},
}

def filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    return pl.lit(True)
'''


def test_find_meta_dict_accepts_type_annotated_form():
    """META: dict = {...} 必须能被识别 (旧版会抛「找不到 META 字典」)。"""
    from app.api.strategy import _find_meta_dict

    node = _find_meta_dict(ANNOTATED_CODE)
    assert node is not None  # 能找到就说明没抛异常


def test_extract_meta_accepts_type_annotated_form():
    from app.strategy.ai_generator import AIStrategyGenerator

    meta = AIStrategyGenerator._extract_meta(ANNOTATED_CODE)
    assert meta["id"] == "annotated_wrong_id"
    assert meta["name"] == "Placeholder"


def test_normalize_strategy_meta_works_on_annotated_form():
    """端到端: AI 生成注解形式 META 时, 规范化不再报「规范化 META 失败」。"""
    code = _normalize_strategy_meta(
        ANNOTATED_CODE,
        "ai_annotated_ok",
        name="断板反包",
        description="中文描述",
    )
    assert '"id": "ai_annotated_ok"' in code
    assert '"name": "断板反包"' in code
    assert '"description": "中文描述"' in code
    assert "annotated_wrong_id" not in code


def test_normalize_build_result_succeeds_on_annotated_form():
    """模拟前端 /build/stream 的完整结果路径 (之前报错的入口)。"""
    result = {"code": ANNOTATED_CODE, "meta": {}, "valid": True, "error": None}

    normalized = _normalize_build_result(result, "ai_build_ok")

    assert normalized["valid"] is True
    assert normalized["error"] is None
    assert normalized["meta"]["id"] == "ai_build_ok"


@pytest.mark.parametrize("alias", ["STRATEGY_META", "meta"])
def test_normalize_strategy_meta_accepts_common_aliases(alias):
    from app.strategy.ai_generator import AIStrategyGenerator

    raw = RAW_CODE.replace("META =", f"{alias} =", 1)

    code = _normalize_strategy_meta(raw, "ai_alias_ok", name="别名策略")

    compile(code, "<strategy>", "exec")
    assert "META =" in code
    assert f"{alias} =" not in code
    assert AIStrategyGenerator._extract_meta(code)["id"] == "ai_alias_ok"


def test_validate_code_rejects_missing_meta():
    from app.strategy.ai_generator import AIStrategyGenerator

    code = """import polars as pl

def filter(df, params):
    return pl.lit(True)
"""

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is False
    assert "找不到 META 字典" in result["error"]


def test_validate_code_ignores_meta_inside_function():
    from app.strategy.ai_generator import AIStrategyGenerator

    code = """import polars as pl

def filter(df, params):
    META = {"id": "nested"}
    return pl.lit(True)
"""

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is False
    assert "找不到 META 字典" in result["error"]


def test_validate_code_rejects_missing_strategy_entrypoint():
    from app.strategy.ai_generator import AIStrategyGenerator

    result = AIStrategyGenerator().validate_code('META = {"id": "no_filter"}')

    assert result["valid"] is False
    assert result["error"] == "找不到策略入口函数 filter() 或 filter_history()"


def test_validate_code_accepts_matrix_strategy_entrypoint():
    from app.strategy.ai_generator import AIStrategyGenerator

    code = '''META = {"id": "matrix", "execution_backend": "matrix_native"}
EXECUTION_BACKEND = "matrix_native"
MATRIX_STRATEGY = object()
'''

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is True
    assert result["error"] is None


def test_validate_code_accepts_controlled_virtual_scoring_field():
    code = RAW_CODE.replace(
        '"scoring": {},',
        '"scoring": {"ma20_bias": 0.6, "vol_ratio_5d": 0.4},',
    )

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is True


def test_validate_code_rejects_unknown_scoring_field():
    code = RAW_CODE.replace(
        '"scoring": {},',
        '"scoring": {"close_above_ma20": 1.0},',
    )

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is False
    assert "close_above_ma20" in result["error"]


def test_validate_code_rejects_param_without_id():
    code = RAW_CODE.replace(
        '"params": [],',
        '"params": [{"name": "volume_ratio", "default": 1.5}],',
    )

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is False
    assert result["error"] == "META.params[0] 缺少非空 id"


def test_validate_code_rejects_missing_matrix_strategy_entrypoint():
    from app.strategy.ai_generator import AIStrategyGenerator

    code = '''META = {"id": "matrix", "execution_backend": "matrix_native"}
EXECUTION_BACKEND = "matrix_native"
'''

    result = AIStrategyGenerator().validate_code(code)

    assert result["valid"] is False
    assert result["error"] == "找不到 Matrix 策略入口 MATRIX_STRATEGY"


def test_extract_code_block_prefers_complete_strategy():
    from app.strategy.ai_generator import AIStrategyGenerator

    content = f"""```python
print("draft")
```

```python
{RAW_CODE}
```
"""

    assert AIStrategyGenerator._extract_code_block(content) == RAW_CODE.strip()


# --- 回归: pprint 风格 META (迭代修改时 current_code 常见) -----------------------
# `META = {'id': ...,` 首键与 `{` 同行、续行单空格缩进、`}` 跟在末项同行。
# 旧版行首正则认不出首行键, 插入点落在 `}` 之外 → "unexpected indent"。

PPRINT_CODE = '''"""知行选股"""
import polars as pl

META = {'id': 'ai_b128c4e7',
 'name': '知行选股',
 'description': '旧描述',
 'basic_filter': {'price_min': 3,
                  'exclude_st': True},
 'scoring': {'momentum_20d': 0.5, 'ma20_bias': 0.5, 'id': 'nested_not_meta_id'},
 'research_only': False}

EXECUTION_BACKEND = "python_history_legacy"
LOOKBACK_DAYS = 130


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    return df
'''


def test_normalize_pprint_style_meta_keeps_code_valid():
    code = _normalize_strategy_meta(PPRINT_CODE, "ai_b128c4e7", name="新名字", description="新描述")
    compile(code, "<strategy>", "exec")
    meta = AIStrategyGenerator._extract_meta(code)
    assert meta["id"] == "ai_b128c4e7"
    assert meta["name"] == "新名字"
    assert meta["description"] == "新描述"
    assert meta["scoring"]["id"] == "nested_not_meta_id"  # 嵌套同名键不被误改
    assert meta["basic_filter"] == {"price_min": 3, "exclude_st": True}


def test_normalize_pprint_style_meta_inserts_missing_id():
    raw = PPRINT_CODE.replace("META = {'id': 'ai_b128c4e7',\n 'name'", "META = {'name'")
    result = _normalize_build_result({"code": raw, "meta": {}, "valid": True, "error": None}, "ai_new")
    assert result["valid"] is True, result["error"]
    assert result["meta"]["id"] == "ai_new"


@pytest.mark.parametrize("raw", [
    'META = {"name": "n", "tags": []}\ndef filter(df, params):\n    return True\n',
    'META = {\n    "name": "n",\n    "tags": [],\n}\ndef filter(df, params):\n    return True\n',
    'META = {\n    "name": "n",\n    "tags": []\n}\ndef filter(df, params):\n    return True\n',
])
def test_normalize_inserts_id_across_layouts(raw):
    code = _normalize_strategy_meta(raw, "ai_layout")
    compile(code, "<strategy>", "exec")
    assert AIStrategyGenerator._extract_meta(code)["id"] == "ai_layout"
