"""T(W1 · RED)：cfoapi.dashscope 原生协议纯函数契约测试。

被测模块 cfoapi.dashscope 此时尚不存在，整个文件应在收集阶段即报
ModuleNotFoundError: cfoapi.dashscope（collection error），即预期红灯。
本文件锁定 v0.2-W2 的实现契约：模型目录、原生请求体组装
（multimodal-generation messages 形状）与响应图像抽取（含 200-带错误
信封识别）。全部为纯函数断言，不涉及任何 HTTP。
"""
from __future__ import annotations

import pytest

# 首要导入：模块缺失时错误信息必须指向 cfoapi.dashscope
from cfoapi.dashscope import (
    DASHSCOPE_IMAGE_MODELS,
    build_dashscope_request,
    extract_dashscope_image_items,
)
from cfoapi.errors import MalformedResponseError, OpenAPIConfigError
from cfoapi.parsing import ImageItem


def _native_payload(urls: list[str]) -> dict[str, object]:
    """构造原生成功响应：每个 URL 包成一个 choice 的 message.content[].image。"""
    return {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": [{"image": u}]},
                }
                for u in urls
            ]
        },
        "request_id": "mock-native",
    }


# ---------- DASHSCOPE_IMAGE_MODELS 目录 ----------


def test_catalog_is_non_empty_tuple_of_unique_strings() -> None:
    # Then: 目录为 tuple、非空、全部非空字符串且互不重复
    assert isinstance(DASHSCOPE_IMAGE_MODELS, tuple)
    assert DASHSCOPE_IMAGE_MODELS
    assert all(isinstance(m, str) and m for m in DASHSCOPE_IMAGE_MODELS)
    assert len(set(DASHSCOPE_IMAGE_MODELS)) == len(DASHSCOPE_IMAGE_MODELS)


def test_catalog_contains_locked_model_ids() -> None:
    # Then: 两个契约模型名必须在目录中（mock 服务器按精确名路由，改名即破）
    assert "qwen-image-3.0-pro" in DASHSCOPE_IMAGE_MODELS
    assert "qwen-image-plus" in DASHSCOPE_IMAGE_MODELS


# ---------- build_dashscope_request ----------


def test_build_empty_params_locks_full_body_shape() -> None:
    # Given: params_json 为空串
    # When: 组装原生请求体
    body = build_dashscope_request("a red square", "qwen-image-3.0-pro", "")
    # Then: 整体结构逐键锁定，且绝不出现 "parameters" 键
    assert body == {
        "model": "qwen-image-3.0-pro",
        "input": {"messages": [{"role": "user", "content": [{"text": "a red square"}]}]},
    }
    assert "parameters" not in body


def test_build_valid_params_attached_without_touching_locked_keys() -> None:
    # When: params 为合法 JSON 对象
    body = build_dashscope_request("p", "qwen-image-plus", '{"size":"64*64","n":2}')
    # Then: parameters 原样附带；model/input 两键结构不受 params 影响
    assert body["parameters"] == {"size": "64*64", "n": 2}
    assert body["model"] == "qwen-image-plus"
    assert body["input"] == {"messages": [{"role": "user", "content": [{"text": "p"}]}]}


def test_build_invalid_json_raises_with_params_hint() -> None:
    # When/Then: 非法 JSON → 配置错误，消息含 "params" 以便用户定位输入件
    with pytest.raises(OpenAPIConfigError) as exc_info:
        build_dashscope_request("p", "qwen-image-3.0-pro", "{bad json")
    assert "params" in str(exc_info.value)


def test_build_non_object_json_raises_with_params_hint() -> None:
    # When/Then: 合法 JSON 但非对象（数组）同样是配置错误
    with pytest.raises(OpenAPIConfigError) as exc_info:
        build_dashscope_request("p", "qwen-image-3.0-pro", "[1,2]")
    assert "params" in str(exc_info.value)


# ---------- extract_dashscope_image_items ----------


def test_extract_returns_url_items_in_order() -> None:
    # Given: 两个 choice 各带一张图
    payload = _native_payload(["http://a/img.png", "http://a/img2.png"])
    # When/Then: 按 choices 原顺序提取，kind 恒为 url
    assert extract_dashscope_image_items(payload) == [
        ImageItem(kind="url", value="http://a/img.png"),
        ImageItem(kind="url", value="http://a/img2.png"),
    ]


def test_extract_skips_non_image_entries() -> None:
    # Given: choices 含非 dict 条目、content 含 text 与非字符串 image、空 content
    payload: dict[str, object] = {
        "output": {
            "choices": [
                "junk",
                {
                    "message": {
                        "content": [
                            {"text": "这是描述文字"},
                            {"image": "http://a/i.png"},
                            {"image": 42},
                        ]
                    }
                },
                {"message": {"content": []}},
                {},
            ]
        }
    }
    # When/Then: 只收 str 型 image，其余全部跳过
    assert extract_dashscope_image_items(payload) == [
        ImageItem(kind="url", value="http://a/i.png")
    ]


def test_extract_error_code_raises_with_code_and_message() -> None:
    # Given: HTTP 200 但顶层非空 code（状态码成功、实为错误）
    payload: dict[str, object] = {
        "request_id": "r",
        "code": "InvalidParameter",
        "message": "n must be 1",
    }
    with pytest.raises(MalformedResponseError) as exc_info:
        extract_dashscope_image_items(payload)
    # Then: 错误消息同时包含 code 与 message，供节点原样透出排障
    text = str(exc_info.value)
    assert "InvalidParameter" in text
    assert "n must be 1" in text


def test_extract_empty_code_does_not_trigger_error_path() -> None:
    # Given: code 为空串 → 不视为错误，回落正常遍历
    payload = {"code": "", **_native_payload(["http://a/img.png"])}
    assert extract_dashscope_image_items(payload) == [
        ImageItem(kind="url", value="http://a/img.png")
    ]


def test_extract_non_dict_payload_raises() -> None:
    # When/Then: 顶层非 dict → 畸形响应
    with pytest.raises(MalformedResponseError):
        extract_dashscope_image_items(["not", "a", "dict"])


def test_extract_missing_output_raises() -> None:
    # When/Then: dict 但缺 output → 畸形响应
    with pytest.raises(MalformedResponseError):
        extract_dashscope_image_items({"request_id": "r"})


def test_extract_output_without_images_raises() -> None:
    # Given: 结构齐全但没有任何 image 字段 → 无可用图像
    payload: dict[str, object] = {
        "output": {"choices": [{"message": {"content": [{"text": "没有图"}]}}]}
    }
    with pytest.raises(MalformedResponseError):
        extract_dashscope_image_items(payload)
