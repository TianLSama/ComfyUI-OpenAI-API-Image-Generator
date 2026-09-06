"""T6 RED：cfoapi.nodes 的 ComfyUI 节点类契约测试。

cfoapi.nodes 模块此时尚未实现，整个文件应在导入阶段即报
ModuleNotFoundError: cfoapi.nodes（collection error），即预期红灯。

契约要点（详见计划）：
- OpenAPIImageGenerator 为纯 Python V1 节点，可在 ComfyUI 进程外导入；
- generate_image 校验 → 组请求 → 调 client → 解析 → 解码 → 返回
  (pil_to_tensor([...]),)，形状 [N,H,W,C] float32、取值 [0,1]；
- 校验错误抛 ValueError("OpenAPI Image Generator: <msg>")，且发生在任何网络调用之前。
"""
from __future__ import annotations

import math

import pytest
import requests
import torch

# 首要导入：模块缺失时错误信息必须指向 cfoapi.nodes
from cfoapi.nodes import NO_MODEL_SENTINEL, OpenAPIImageGenerator

TIMEOUT = 15  # 所有对 mock 服务器的调试请求统一超时


def _last_request(server) -> dict | None:
    """读取 mock 服务器记录的最近一次 generations 请求体（无则为 None）。"""
    resp = requests.get(f"{server.base_url}/debug/last_request", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fingerprint(body: dict | None) -> tuple | None:
    """只比较请求体的稳定关键字段，避免整 dict 对比带来的脆弱性。"""
    if body is None:
        return None
    return (body.get("model"), body.get("prompt"))


# ---------- 类属性契约 ----------


def test_class_attributes_match_v1_node_contract():
    assert OpenAPIImageGenerator.RETURN_TYPES == ("IMAGE",)
    assert OpenAPIImageGenerator.RETURN_NAMES == ("IMAGE",)
    assert OpenAPIImageGenerator.FUNCTION == "generate_image"
    # v0.2.2: 单段中文分类(仿 llm_party) → 根菜单
    assert OpenAPIImageGenerator.CATEGORY == "外部文生图 (OpenAPI)"
    assert isinstance(OpenAPIImageGenerator.DESCRIPTION, str)
    assert OpenAPIImageGenerator.DESCRIPTION  # 非空


def test_is_changed_returns_nan_each_call():
    # nan != nan，两次调用必不相等 → ComfyUI 每次执行都重跑，绝不缓存远端结果
    first = OpenAPIImageGenerator.IS_CHANGED()
    second = OpenAPIImageGenerator.IS_CHANGED()
    assert math.isnan(first)
    assert math.isnan(second)
    assert first != second


# ---------- generate_image 正常路径 ----------


def test_generate_image_b64_returns_single_image_tensor(mock_server):
    gen = OpenAPIImageGenerator()
    result = gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="mock-b64",
        prompt="a red square",
        system_prompt="",
        params="",
    )
    # 返回值是 1-tuple，元素为 [N,H,W,C] float32 张量
    assert isinstance(result, tuple)
    assert len(result) == 1
    t = result[0]
    assert tuple(t.shape) == (1, 64, 64, 3)
    assert t.dtype == torch.float32
    # 纯色红图：像素值归一化后必落在 [0,1]
    assert float(t.min()) >= 0.0
    assert float(t.max()) <= 1.0


def test_generate_image_url_model_downloads_and_decodes(mock_server):
    # 走 data[0].url → resolve_image_bytes 下载 → 解码 的端到端路径
    gen = OpenAPIImageGenerator()
    (t,) = gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="mock-url",
        prompt="a red square",
        system_prompt="",
        params="",
    )
    assert tuple(t.shape) == (1, 64, 64, 3)


def test_generate_image_siliconflow_polymorphic_parsing(mock_server):
    # 走顶层 images[] 形状的多态解析路径
    gen = OpenAPIImageGenerator()
    (t,) = gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="mock-siliconflow",
        prompt="a red square",
        system_prompt="",
        params="",
    )
    assert tuple(t.shape) == (1, 64, 64, 3)


# ---------- 请求体组装 ----------


def test_system_prompt_is_prepended_to_prompt(mock_server):
    gen = OpenAPIImageGenerator()
    gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="mock-b64",
        prompt="a red square",
        system_prompt="style: flat",
        params="",
    )
    body = _last_request(mock_server)
    assert body is not None
    assert body["prompt"] == "style: flat\na red square"


def test_params_json_merged_into_request_body(mock_server):
    gen = OpenAPIImageGenerator()
    gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="mock-b64",
        prompt="a red square",
        system_prompt="",
        params='{"size":"64x64"}',
    )
    body = _last_request(mock_server)
    assert body is not None
    assert body["size"] == "64x64"
    assert body["model"] == "mock-b64"


# ---------- 校验错误（必须先于任何网络调用） ----------


def test_sentinel_model_raises_fetch_models_hint_without_network(mock_server):
    before = _fingerprint(_last_request(mock_server))
    gen = OpenAPIImageGenerator()
    with pytest.raises(ValueError) as exc_info:
        gen.generate_image(
            base_url=mock_server.base_url,
            api_key="k",
            model=NO_MODEL_SENTINEL,
            prompt="a red square",
            system_prompt="",
            params="",
        )
    assert "OpenAPI Image Generator" in str(exc_info.value)
    assert "Fetch Models" in str(exc_info.value)
    # 无网络副作用：last_request 保持不变
    after = _fingerprint(_last_request(mock_server))
    assert after == before


def test_blank_prompt_raises_before_any_network(mock_server):
    before = _fingerprint(_last_request(mock_server))
    gen = OpenAPIImageGenerator()
    with pytest.raises(ValueError) as exc_info:
        gen.generate_image(
            base_url=mock_server.base_url,
            api_key="k",
            model="mock-b64",
            prompt="   ",
            system_prompt="",
            params="",
        )
    assert "OpenAPI Image Generator" in str(exc_info.value)
    after = _fingerprint(_last_request(mock_server))
    assert after == before


def test_invalid_params_json_raises_with_params_hint(mock_server):
    gen = OpenAPIImageGenerator()
    with pytest.raises(ValueError) as exc_info:
        gen.generate_image(
            base_url=mock_server.base_url,
            api_key="k",
            model="mock-b64",
            prompt="a red square",
            system_prompt="",
            params="{bad json",
        )
    assert "params" in str(exc_info.value)


# ---------- INPUT_TYPES 与模型缓存联动 ----------


def test_input_types_empty_cache_shows_sentinel(tmp_path, monkeypatch):
    # v0.2 规格演进（有意为之）：空缓存不再「仅哨兵」——哨兵仍居首，
    # 但 DashScope 目录模型并入选项，否则 /prompt 校验会拒绝合法的原生协议模型。
    from cfoapi.dashscope import DASHSCOPE_IMAGE_MODELS

    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    types = OpenAPIImageGenerator.INPUT_TYPES()
    required = types["required"]
    # COMBO 的选项列表是元组首元素：哨兵第一、目录模型全部在场
    options = required["model"][0]
    assert options[0] == NO_MODEL_SENTINEL
    assert all(m in options for m in DASHSCOPE_IMAGE_MODELS)
    # 文本输入件存在且类型正确
    assert required["base_url"][0] == "STRING"
    assert required["api_key"][0] == "STRING"
    # 三个多行文本件必须带 multiline: True
    assert required["prompt"][1]["multiline"] is True
    assert required["system_prompt"][1]["multiline"] is True
    assert required["params"][1]["multiline"] is True


def test_input_types_populated_cache_lists_models(mock_server, tmp_path, monkeypatch):
    # 预置缓存后，COMBO 选项应精确等于缓存的模型列表
    from cfoapi import cache

    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    cache.save_models(mock_server.base_url, ["mock-b64", "mock-url"], path=None)
    types = OpenAPIImageGenerator.INPUT_TYPES(base_url=mock_server.base_url)
    assert types["required"]["model"][0] == ["mock-b64", "mock-url"]


def test_input_types_no_kwargs_unions_all_cached_bases(mock_server, tmp_path, monkeypatch):
    """无参 INPUT_TYPES() 必须返回全部缓存端点模型的并集 (T15 QA 修复)。

    ComfyUI 的 /prompt 校验与 /object_info 均以无参方式调用 INPUT_TYPES(),
    看不到工作流里 base_url 控件的值; 若选项按单一 base_url 精确查询,
    任何已缓存模型都会被判 value_not_in_list 而拒绝执行。
    """
    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    from cfoapi import cache

    cache.save_models("https://one.example/v1", ["m-one"], path=None)
    cache.save_models(mock_server.base_url, ["mock-b64", "mock-url"], path=None)

    # v0.2 规格演进（有意为之）：无参 INPUT_TYPES 在缓存并集后并入 DashScope
    # 目录模型，否则 dashscope 协议下的合法模型会被 ComfyUI 的
    # value_not_in_list 校验拒绝执行。
    from cfoapi.dashscope import DASHSCOPE_IMAGE_MODELS

    required = OpenAPIImageGenerator.INPUT_TYPES()["required"]
    # 并集按缓存条目首见顺序, 去重; 目录模型作为增量追加在后
    assert required["model"][0] == ["m-one", "mock-b64", "mock-url"] + [
        m for m in DASHSCOPE_IMAGE_MODELS if m not in ("m-one", "mock-b64", "mock-url")
    ]


# ---------- DashScope 原生协议分支（v0.2 · W1 RED：protocol 参数尚未实现） ----------


def _last_native_request(server) -> dict | None:
    """读取 mock 服务器记录的最近一次原生 generation 请求体（无则为 None）。"""
    resp = requests.get(f"{server.base_url}/debug/last_native_request", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def test_generate_image_dashscope_happy_path_and_body_shape(mock_server):
    gen = OpenAPIImageGenerator()
    result = gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="qwen-image-3.0-pro",
        prompt="a red square",
        system_prompt="",
        params='{"size":"64*64","n":1}',
        protocol="dashscope",
    )
    # Then: 与 openai 分支同构的 [N,H,W,C] float32 张量（64x64 红图）
    assert isinstance(result, tuple) and len(result) == 1
    (t,) = result
    assert tuple(t.shape) == (1, 64, 64, 3)
    assert t.dtype == torch.float32
    # Then: 原生请求体结构锁定 —— model / input.messages / parameters 透传
    body = _last_native_request(mock_server)
    assert body is not None
    assert body["model"] == "qwen-image-3.0-pro"
    assert body["input"]["messages"][0]["content"][0]["text"] == "a red square"
    assert body["parameters"] == {"size": "64*64", "n": 1}


def test_generate_image_dashscope_prepends_system_prompt(mock_server):
    gen = OpenAPIImageGenerator()
    gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="qwen-image-3.0-pro",
        prompt="a red square",
        system_prompt="style",
        params="",
        protocol="dashscope",
    )
    body = _last_native_request(mock_server)
    # Then: system_prompt 前置拼接发生在 build 之前，落在 text 字段上
    assert body["input"]["messages"][0]["content"][0]["text"] == "style\na red square"


def test_generate_image_dashscope_n2_returns_two_images(mock_server):
    gen = OpenAPIImageGenerator()
    (t,) = gen.generate_image(
        base_url=mock_server.base_url,
        api_key="k",
        model="qwen-image-3.0-pro",
        prompt="a red square",
        system_prompt="",
        params='{"n":2}',
        protocol="dashscope",
    )
    # Then: 两张图（img2.png 为 32x48，pil_to_tensor 以首张 64x64 为基准对齐）
    assert tuple(t.shape) == (2, 64, 64, 3)


def test_generate_image_dashscope_404_envelope_survives_node_wrapping(mock_server):
    gen = OpenAPIImageGenerator()
    with pytest.raises(ValueError) as exc_info:
        gen.generate_image(
            base_url=mock_server.base_url,
            api_key="k",
            model="qwen-image-404",
            prompt="a red square",
            system_prompt="",
            params="",
            protocol="dashscope",
        )
    # Then: 固定前缀 + 原生信封消息原样透出（用户可直接排障）
    msg = str(exc_info.value)
    assert "OpenAPI Image Generator" in msg
    assert "Model not exist" in msg


def test_generate_image_dashscope_200_error_envelope_survives_wrapping(mock_server):
    gen = OpenAPIImageGenerator()
    with pytest.raises(ValueError) as exc_info:
        gen.generate_image(
            base_url=mock_server.base_url,
            api_key="k",
            model="qwen-image-200err",
            prompt="a red square",
            system_prompt="",
            params="",
            protocol="dashscope",
        )
    # Then: HTTP 200 但带 code 的响应必须被抽取层拒绝，消息透出到 ValueError
    assert "n must be 1" in str(exc_info.value)


def test_generate_image_rejects_unknown_protocol():
    # Given: 非法协议值 + 不可达 base_url（校验必须先于任何网络调用）
    gen = OpenAPIImageGenerator()
    with pytest.raises(ValueError) as exc_info:
        gen.generate_image(
            base_url="http://127.0.0.1:1",
            api_key="k",
            model="some-model",
            prompt="a red square",
            system_prompt="",
            params="",
            protocol="anthropic",
        )
    assert "OpenAPI Image Generator" in str(exc_info.value)
    assert "protocol" in str(exc_info.value)


def test_input_types_protocol_widget_exists_and_is_last_key():
    required = OpenAPIImageGenerator.INPUT_TYPES()["required"]
    # Then: protocol COMBO 选项固定为 ["openai","dashscope"]（openai 在前）
    assert required["protocol"][0] == ["openai", "dashscope"]
    # Then: 必须是最后一个键——控件排布向后兼容（旧工作流字段顺序不变）
    assert list(required)[-1] == "protocol"


def test_input_types_dashscope_protocol_lists_catalog(tmp_path, monkeypatch):
    from cfoapi.dashscope import DASHSCOPE_IMAGE_MODELS

    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    types = OpenAPIImageGenerator.INPUT_TYPES(
        base_url="https://dashscope.aliyuncs.com", protocol="dashscope"
    )
    # Then: dashscope 分支下空缓存回落到原生目录，选项精确等于目录
    assert types["required"]["model"][0] == list(DASHSCOPE_IMAGE_MODELS)


def test_input_types_no_args_empty_cache_exact_union(tmp_path, monkeypatch):
    from cfoapi.dashscope import DASHSCOPE_IMAGE_MODELS

    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    options = OpenAPIImageGenerator.INPUT_TYPES()["required"]["model"][0]
    # Then: 空缓存无参路径 = 哨兵 + 全部目录模型（精确顺序：哨兵第一，目录随后）
    assert options == [NO_MODEL_SENTINEL, *DASHSCOPE_IMAGE_MODELS]


def test_input_types_openai_base_url_no_catalog_leak(mock_server, tmp_path, monkeypatch):
    # Given: base_url 分支（默认 protocol=openai）+ 已缓存端点模型
    from cfoapi import cache

    monkeypatch.setenv("COMFYUI_OPENAPI_CACHE_DIR", str(tmp_path))
    cache.save_models(mock_server.base_url, ["mock-b64", "mock-url"], path=None)
    types = OpenAPIImageGenerator.INPUT_TYPES(base_url=mock_server.base_url)
    # Then: 选项精确等于缓存列表——base_url+openai 分支绝不混入 DashScope 目录
    assert types["required"]["model"][0] == ["mock-b64", "mock-url"]
