"""URL 归一化纯单元测试（T3 · RED 阶段）。

被测量模块 cfoapi.urls / cfoapi.errors 尚未实现，本文件在导入期即应
失败（ModuleNotFoundError/ImportError），这是 TDD 红灯的预期形态。
本文件不涉及任何 HTTP，全部为纯函数断言。
"""
from __future__ import annotations

import pytest

from cfoapi.errors import OpenAPIConfigError
from cfoapi.urls import api_root, normalize_base_url


# ---------- normalize_base_url ----------


def test_normalize_lowercases_host_and_keeps_v1_strips_trailing_slash() -> None:
    # Given: 大写 host、尾部 /v1/ 的原始地址
    # When: 归一化
    # Then: scheme+host 小写、保留 /v1、去掉尾部斜杠
    assert normalize_base_url("https://X.com/v1/") == "https://x.com/v1"


def test_normalize_strips_only_trailing_slash() -> None:
    # Then: 仅有尾部斜杠时去掉它
    assert normalize_base_url("https://a.b/") == "https://a.b"


def test_normalize_preserves_localhost_with_port() -> None:
    # Then: 带端口的 localhost 原样保留（含端口号）
    assert normalize_base_url("http://localhost:8000") == "http://localhost:8000"


def test_normalize_rejects_empty_string() -> None:
    # Then: 空串视为非法配置
    with pytest.raises(OpenAPIConfigError):
        normalize_base_url("")


def test_normalize_rejects_missing_scheme_with_http_hint() -> None:
    # When/Then: 无 scheme 抛错，且错误信息含 "http" 以提示期望格式
    with pytest.raises(OpenAPIConfigError) as exc_info:
        normalize_base_url("x.com")
    assert "http" in str(exc_info.value)


def test_normalize_rejects_non_http_scheme() -> None:
    # Then: 非 http(s) 协议（如 ftp）被拒绝
    with pytest.raises(OpenAPIConfigError):
        normalize_base_url("ftp://x")


def test_normalize_strips_surrounding_whitespace() -> None:
    # Then: 先去除首尾空白再归一化
    assert normalize_base_url("  https://a.b/ ") == "https://a.b"


# ---------- api_root ----------


def test_api_root_appends_v1_when_absent() -> None:
    # Then: 无 /v1 时追加
    assert api_root("https://a.b") == "https://a.b/v1"


def test_api_root_idempotent_when_v1_present() -> None:
    # Then: 已带 /v1 时原样返回（幂等）
    assert api_root("https://a.b/v1") == "https://a.b/v1"


def test_api_root_appends_v1_to_proxy_prefix() -> None:
    # Then: 多级路径前缀同样追加 /v1
    assert api_root("https://a.b/proxy/openai") == "https://a.b/proxy/openai/v1"


# ---------- dashscope_endpoint（v0.2 · W1 RED：符号尚未实现） ----------
#
# 说明：不在模块级导入行追加 dashscope_endpoint——cfoapi.urls 已存在，
# 模块级 ImportError 会连带拖红本文件全部既有测试，违反「其余既有测试
# 保持绿色」的验收约束；改用惰性导入辅助，使每个新用例单独以
# ImportError: cannot import name 'dashscope_endpoint' 精准报红。

_NATIVE_SUFFIX = "/api/v1/services/aigc/multimodal-generation/generation"


def _endpoint(base: str) -> str:
    """惰性调用 dashscope_endpoint（RED 期 ImportError；GREEN 期真实拼接）。"""
    from cfoapi.urls import dashscope_endpoint

    return dashscope_endpoint(base)


def test_dashscope_endpoint_plain_host_appends_suffix() -> None:
    # Given: 裸 host（无任何路径）
    # When/Then: 直接拼接原生完整路径
    assert _endpoint("https://x.com") == f"https://x.com{_NATIVE_SUFFIX}"


def test_dashscope_endpoint_strips_compatible_mode_v1() -> None:
    # Given: OpenAI 兼容模式端点 → When/Then: 剥离后拼原生路径
    assert (
        _endpoint("https://dashscope.aliyuncs.com/compatible-mode/v1")
        == f"https://dashscope.aliyuncs.com{_NATIVE_SUFFIX}"
    )


def test_dashscope_endpoint_trailing_slash_variant() -> None:
    # Given: 尾部斜杠变体（normalize_base_url 先行裁剪）
    assert (
        _endpoint("https://dashscope.aliyuncs.com/compatible-mode/v1/")
        == f"https://dashscope.aliyuncs.com{_NATIVE_SUFFIX}"
    )


def test_dashscope_endpoint_keeps_path_prefix_when_stripping() -> None:
    # Given: 反代挂载在多级路径前缀下的兼容模式端点
    # Then: 仅剥离一段已知后缀，前缀 /dashscope-mock 保留
    assert (
        _endpoint("http://127.0.0.1:18099/dashscope-mock/compatible-mode/v1")
        == f"http://127.0.0.1:18099/dashscope-mock{_NATIVE_SUFFIX}"
    )


def test_dashscope_endpoint_strips_api_v1_without_doubling() -> None:
    # Given: 已是原生 /api/v1 前缀 → Then: 剥离后不出现双重 /api/v1
    assert _endpoint("https://x.com/api/v1") == f"https://x.com{_NATIVE_SUFFIX}"


def test_dashscope_endpoint_strips_bare_v1() -> None:
    # Given: 仅 /v1 后缀 → Then: 同样剥离一段
    assert _endpoint("https://x.com/v1") == f"https://x.com{_NATIVE_SUFFIX}"


def test_dashscope_endpoint_rejects_non_http_scheme() -> None:
    # Given/Then: normalize_base_url 先行 → 非法 scheme 抛配置错误
    with pytest.raises(OpenAPIConfigError):
        _endpoint("ftp://x")
