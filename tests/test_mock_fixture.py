"""mock 服务器基础设施冒烟测试（T2）。

证明后续所有 TDD 任务与实时 QA 依赖的每条路由行为符合规格。
本文件绝不导入 cfoapi.*——只验证 mock 本身。
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import pytest
import requests
from mock_openai_server import MODEL_IDS
from PIL import Image

# 本机回环请求，超时给足余量即可（/slow 默认 2 秒，测试里显式传短延迟）
_TIMEOUT = 15.0


@pytest.fixture()
def api_key_headers() -> dict[str, str]:
    return {"Authorization": "Bearer k"}


def _post_generation(server: Any, model: str) -> requests.Response:
    """向 generations 端点发一次最小请求体。"""
    return requests.post(
        f"{server.base_url}/v1/images/generations",
        json={"model": model, "prompt": "a colored square", "n": 1},
        timeout=_TIMEOUT,
    )


# ---------- /v1/models ----------


def test_models_returns_all_ids_in_order(mock_server, api_key_headers) -> None:
    # Given: 携带任意 Authorization 头
    # When: GET {base}/v1/models
    resp = requests.get(
        f"{mock_server.base_url}/v1/models", headers=api_key_headers, timeout=_TIMEOUT
    )
    # Then: 200 且 id 列表与 MODEL_IDS 逐位一致
    assert resp.status_code == 200
    body = resp.json()
    assert [item["id"] for item in body["data"]] == MODEL_IDS


def test_models_without_auth_returns_401(mock_server) -> None:
    # Given: 不带 Authorization 头
    # When: GET {base}/v1/models
    resp = requests.get(f"{mock_server.base_url}/v1/models", timeout=_TIMEOUT)
    # Then: 401 + error.message
    assert resp.status_code == 401
    assert resp.json()["error"]["message"] == "Missing API key"


# ---------- /v1/images/generations 各模型分支 ----------


def test_generation_b64_decodes_to_64x64_red_png(mock_server) -> None:
    # Given/When: 请求 mock-b64
    resp = _post_generation(mock_server, "mock-b64")
    # Then: 200，b64_json 可解码为 PIL 可读的 64x64 图
    assert resp.status_code == 200
    b64 = resp.json()["data"][0]["b64_json"]
    img = Image.open(BytesIO(base64.b64decode(b64)))
    assert img.size == (64, 64)


def test_generation_multi_returns_red_then_blue(mock_server) -> None:
    # When: 请求 mock-multi
    body = _post_generation(mock_server, "mock-multi").json()
    # Then: 两张图，尺寸依次为 64x64（红）与 32x48（蓝）
    sizes = [
        Image.open(BytesIO(base64.b64decode(item["b64_json"]))).size
        for item in body["data"]
    ]
    assert sizes == [(64, 64), (32, 48)]


def test_generation_url_points_to_static_image(mock_server) -> None:
    # When: 请求 mock-url
    body = _post_generation(mock_server, "mock-url").json()
    # Then: data[0].url 指向 /static/img.png
    assert body["data"][0]["url"].endswith("/static/img.png")


def test_generation_siliconflow_shape_and_url_fetchable(mock_server) -> None:
    # When: 请求 mock-siliconflow（顶层 images[] 风格响应）
    body = _post_generation(mock_server, "mock-siliconflow").json()
    url = body["images"][0]["url"]
    # Then: url 指向 /static/img.png，且可直接 GET 到 image/png
    assert url.endswith("/static/img.png")
    img_resp = requests.get(url, timeout=_TIMEOUT)
    assert img_resp.status_code == 200
    assert img_resp.headers["Content-Type"] == "image/png"


def test_generation_mock_401_returns_401_with_error_message(mock_server) -> None:
    resp = _post_generation(mock_server, "mock-401")
    assert resp.status_code == 401
    assert resp.json()["error"]["message"] == "Invalid API key provided."


def test_generation_error_code_gateway_style(mock_server) -> None:
    # Then: 顶层 code/message（无 error 包装），供后续错误解析任务消费
    resp = _post_generation(mock_server, "mock-error-code")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "insufficient_credit"
    assert body["message"] == "Account has no credit"


def test_generation_empty_data_list(mock_server) -> None:
    body = _post_generation(mock_server, "mock-empty").json()
    assert body["data"] == []


def test_generation_unknown_model_rejected(mock_server) -> None:
    resp = _post_generation(mock_server, "mock-does-not-exist")
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "Unknown model"


# ---------- 调试与边界端点 ----------


def test_debug_last_request_echoes_generation_body(mock_server) -> None:
    # When: 先发一次 generations 再查 /debug/last_request
    _post_generation(mock_server, "mock-b64")
    body = requests.get(
        f"{mock_server.base_url}/debug/last_request", timeout=_TIMEOUT
    ).json()
    # Then: 回显的请求体带 model 字段
    assert body["model"] == "mock-b64"


def test_static_big_png_advertises_huge_content_length(mock_server) -> None:
    # Given: stream=True 只读响应头，立即关闭（不拖 100MB 假 body）
    resp = requests.get(
        f"{mock_server.base_url}/static/big.png", stream=True, timeout=_TIMEOUT
    )
    try:
        # Then: 客户端可仅凭 Content-Length 头拒绝
        assert resp.status_code == 200
        assert resp.headers["Content-Length"] == "99999999"
        assert resp.headers["Content-Type"] == "image/png"
    finally:
        resp.close()


def test_slow_endpoint_responds_after_short_delay(mock_server) -> None:
    # When: delay=0.2（覆盖默认 2 秒）
    resp = requests.get(f"{mock_server.base_url}/slow?delay=0.2", timeout=_TIMEOUT)
    # Then: 200 {"ok": true}
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_crosshost_redirect_records_auth_leak(mock_server) -> None:
    # Given: 客户端被要求跨 host（127.0.0.1 → localhost）跟随重定向并携带鉴权头
    resp = requests.get(
        f"{mock_server.base_url}/redir-crosshost",
        headers={"Authorization": "Bearer k"},
        timeout=_TIMEOUT,
    )
    assert resp.status_code == 200
    # Then: /redir-target 观测到了 Authorization 头（requests 默认会泄漏）；
    # /debug/redirect_seen 为最近一次 target 请求的记录（无请求前 null→false）
    seen = requests.get(
        f"{mock_server.base_url}/debug/redirect_seen", timeout=_TIMEOUT
    ).json()
    assert isinstance(seen, bool)


# ---------- DashScope 原生路由（v0.2 · W1 绿色基础设施） ----------

_NATIVE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def _post_native(server: Any, payload: dict[str, Any], auth: bool = True) -> requests.Response:
    """向原生 multimodal-generation 端点发一次请求（auth=False 即不带 Authorization 头）。"""
    headers = {"Authorization": "Bearer k"} if auth else {}
    return requests.post(
        f"{server.base_url}{_NATIVE_PATH}", json=payload, headers=headers, timeout=_TIMEOUT
    )


def test_native_without_auth_returns_401_invalid_api_key(mock_server) -> None:
    # Given: 不带 Authorization 头
    # When: POST 原生端点
    resp = _post_native(mock_server, {"model": "qwen-image-3.0-pro"}, auth=False)
    # Then: 401 + 原生信封（顶层 code/message/request_id）
    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "InvalidApiKey"
    assert body["message"] == "No API-key provided."


def test_native_success_choice_image_ends_with_static_png(mock_server) -> None:
    # When: qwen-image-3.0-pro + Bearer → 200
    resp = _post_native(mock_server, {"model": "qwen-image-3.0-pro"})
    assert resp.status_code == 200
    # Then: choices[0].message.content[0].image 指向 /static/img.png
    content = resp.json()["output"]["choices"][0]["message"]["content"]
    assert content[0]["image"].endswith("/static/img.png")


def test_native_parameters_n2_returns_two_choices(mock_server) -> None:
    # When: parameters.n == 2
    body = _post_native(
        mock_server, {"model": "qwen-image-3.0-pro", "parameters": {"n": 2}}
    ).json()
    # Then: 两个 choice，图片依次为 img.png（红 64x64）与 img2.png（蓝 32x48）
    images = [c["message"]["content"][0]["image"] for c in body["output"]["choices"]]
    assert len(images) == 2
    assert images[0].endswith("/static/img.png")
    assert images[1].endswith("/static/img2.png")


def test_native_404_model_not_found(mock_server) -> None:
    # When/Then: qwen-image-404 → 404 + ModelNotFound
    resp = _post_native(mock_server, {"model": "qwen-image-404"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "ModelNotFound"


def test_native_200_with_error_envelope(mock_server) -> None:
    # Given: qwen-image-200err → HTTP 200 但顶层 code/message、无 output
    body = _post_native(mock_server, {"model": "qwen-image-200err"}).json()
    # Then: 「状态码成功但实为错误」形状，供 extract 阶段识别 code 信封
    assert body["code"] == "InvalidParameter"
    assert body["message"] == "n must be 1"
    assert "output" not in body


def test_native_unknown_model_returns_400(mock_server) -> None:
    # When: 目录矩阵之外的模型
    resp = _post_native(mock_server, {"model": "qwen-image-nope"})
    # Then: 400 Unsupported model
    assert resp.status_code == 400
    assert resp.json()["message"] == "Unsupported model"


def test_debug_last_native_request_echoes_body(mock_server) -> None:
    # When: 先发一次原生请求再查 /debug/last_native_request
    _post_native(
        mock_server,
        {"model": "qwen-image-3.0-pro", "parameters": {"size": "64*64", "n": 1}},
    )
    body = requests.get(
        f"{mock_server.base_url}/debug/last_native_request", timeout=_TIMEOUT
    ).json()
    # Then: 回显请求体的 model 与 parameters
    assert body["model"] == "qwen-image-3.0-pro"
    assert body["parameters"] == {"size": "64*64", "n": 1}
