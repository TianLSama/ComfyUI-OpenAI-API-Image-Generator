"""cfoapi.cache 的 RED 阶段测试。

模块 cfoapi.cache 尚未实现——本文件应在导入处即失败（collection error）。
所有缓存调用都显式传 path=（指向 pytest tmp_path），只有一个环境变
量测试通过 monkeypatch 重定向缓存目录——绝不触碰真实插件缓存。
"""
from __future__ import annotations

import json

import pytest

from cfoapi.cache import get_all_models, get_api_key, get_models, load_cache, save_api_key, save_models


@pytest.fixture()
def cache_path(tmp_path):
    """本测试专用的缓存文件路径（父目录存在，文件尚不存在）。"""
    return tmp_path / "openapi_cache.json"


def test_save_then_get_models_roundtrip(cache_path):
    """保存模型列表后按同一 base_url 读回。"""
    save_models("https://api.x.com/v1", ["a", "b"], path=cache_path)
    assert get_models("https://api.x.com/v1", path=cache_path) == ["a", "b"]


def test_base_url_normalization_shares_entry(cache_path):
    """尾斜杠差异经 normalize 后是同一个条目。"""
    save_models("https://a.b/", ["m1"], path=cache_path)
    assert get_models("https://a.b", path=cache_path) == ["m1"]


def test_api_key_roundtrip_and_unknown_none(cache_path):
    """api_key 读写往返；未知 base_url 返回 None。"""
    save_api_key("https://api.x.com/v1", "sk-test-123", path=cache_path)
    assert get_api_key("https://api.x.com/v1", path=cache_path) == "sk-test-123"
    assert get_api_key("https://unknown.example", path=cache_path) is None


def test_get_models_unknown_base_returns_empty(cache_path):
    """未知 base_url 返回 [] 且不抛异常。"""
    save_models("https://known.example", ["m"], path=cache_path)
    assert get_models("https://other.example", path=cache_path) == []


def test_get_models_on_missing_file_returns_empty(tmp_path):
    """缓存文件完全不存在时返回 [] 且不抛异常。"""
    missing = tmp_path / "nope" / "openapi_cache.json"
    assert get_models("https://any.example", path=missing) == []


def test_save_models_preserves_existing_api_key(cache_path):
    """先写 key 再写 models：两者都可见（save_models 不清除 api_key）。"""
    save_api_key("https://api.x.com/v1", "sk-keep", path=cache_path)
    save_models("https://api.x.com/v1", ["m1", "m2"], path=cache_path)
    assert get_api_key("https://api.x.com/v1", path=cache_path) == "sk-keep"
    assert get_models("https://api.x.com/v1", path=cache_path) == ["m1", "m2"]


def test_save_api_key_preserves_existing_models(cache_path):
    """先写 models 再写 key：两者都可见（save_api_key 不清除 models）。"""
    save_models("https://api.x.com/v1", ["m1"], path=cache_path)
    save_api_key("https://api.x.com/v1", "sk-new", path=cache_path)
    assert get_models("https://api.x.com/v1", path=cache_path) == ["m1"]
    assert get_api_key("https://api.x.com/v1", path=cache_path) == "sk-new"


def test_load_cache_recovers_from_corrupt_file(cache_path):
    """损坏 JSON → load_cache 返回 {}；随后 save_models 可正常覆盖恢复。"""
    cache_path.write_text("!!! not json !!!", encoding="utf-8")
    assert load_cache(cache_path) == {}
    save_models("https://api.x.com/v1", ["x"], path=cache_path)
    assert get_models("https://api.x.com/v1", path=cache_path) == ["x"]


def test_load_cache_missing_path_returns_empty(tmp_path):
    """路径不存在时 load_cache 返回 {} 且不抛异常。"""
    assert load_cache(tmp_path / "absent.json") == {}


def test_sequential_saves_yield_last_valid_json(cache_path):
    """连续两次 save_models：读回第二次的列表，且文件本身是合法 JSON dict。"""
    save_models("https://api.x.com/v1", ["first"], path=cache_path)
    save_models("https://api.x.com/v1", ["second", "third"], path=cache_path)
    assert get_models("https://api.x.com/v1", path=cache_path) == ["second", "third"]
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_env_var_cache_dir_location(tmp_path, monkeypatch):
    """未传 path 时，缓存文件落在 COMFYUI_OPENAPI_CACHE_DIR 目录下。"""
    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    save_models("https://env.example", ["m-env"])
    assert (tmp_path / "openapi_cache.json").exists()
    assert get_models("https://env.example", path=tmp_path / "openapi_cache.json") == ["m-env"]


# ---------- get_all_models: 跨端点并集 (ComfyUI 无参校验语义修复, T15 QA 发现) ----------
# 背景: ComfyUI 在 /prompt 校验与 /object_info 生成时以无参方式调用 INPUT_TYPES(),
# 节点无法得知工作流里 base_url 控件的当前值; 若 COMBO 选项按单一 base_url 精确
# 查询, 用户经「获取模型列表」按钮缓存的模型永远无法通过服务端校验 (value_not_in_list)。
# 契约修复: 无 base_url 时返回所有缓存条目模型的并集 (首见顺序, 去重, 宽容畸形条目)。

def test_get_all_models_unions_entries_in_first_seen_order(cache_path):
    save_models("https://a.example/v1", ["m1", "m2"], path=cache_path)
    save_models("https://c.example/v1", ["m2", "m3"], path=cache_path)
    assert get_all_models(path=cache_path) == ["m1", "m2", "m3"]


def test_get_all_models_missing_file_returns_empty(tmp_path):
    assert get_all_models(path=tmp_path / "nonexistent.json") == []


def test_get_all_models_skips_malformed_entries(tmp_path):
    p = tmp_path / "mixed.json"
    p.write_text(
        '{"https://a.example": {"models": ["m1"]}, "junk": "not-a-dict", "bad-models": {"models": 42}}',
        encoding="utf-8",
    )
    assert get_all_models(path=p) == ["m1"]
