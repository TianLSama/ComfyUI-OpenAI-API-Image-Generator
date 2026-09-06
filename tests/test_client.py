"""HTTP 客户端集成测试（T5 · RED 阶段）。

被测模块 cfoapi.client（及其依赖的 cfoapi.errors / cfoapi.parsing）
尚未实现，本文件应在导入期即失败（ModuleNotFoundError/ImportError），
符合 TDD 红灯预期。测试通过 session 级 mock_server 夹具发起真实 HTTP
请求，覆盖：模型列表拉取、重试策略、零重试计费安全、图片字节解析、
凭证泄漏防护与大小/类型防线。
"""
from __future__ import annotations

import socket
import time
from io import BytesIO

import pytest
import requests
from PIL import Image

from cfoapi import client
from cfoapi.client import fetch_models, generate_images, resolve_image_bytes
from cfoapi.errors import (
    MalformedResponseError,
    OpenAPIConfigError,
    OpenAPIError,
    RemoteAPIError,
)
from cfoapi.parsing import ImageItem
from mock_openai_server import MODEL_IDS, MockOpenAIServer

# 测试侧 HTTP 统一带鉴权头与超时（mock 只检查头是否存在）
_auth = {"Authorization": "Bearer test-key"}
_TIMEOUT = 15.0


# ---------- fetch_models ----------


def test_fetch_models_returns_all_ids_in_order(mock_server: MockOpenAIServer) -> None:
    # Given: mock /v1/models 携带任意 Bearer 即返回 7 个模型
    # When: 拉取模型列表
    ids = fetch_models(mock_server.base_url, "test-key")
    # Then: 顺序与数量与 MODEL_IDS 完全一致
    assert ids == MODEL_IDS


def test_fetch_models_retries_twice_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: 一个已停止（端口死亡）的服务器 → 连接错误路径
    srv = MockOpenAIServer()
    srv.start()
    dead_base = srv.base_url
    srv.stop()
    # 拦截 client 模块使用的 time.sleep，记录每次退避调用
    sleep_calls: list[float] = []
    monkeypatch.setattr(client.time, "sleep", lambda s: sleep_calls.append(s))
    # When: 拉取模型列表必然失败
    started = time.monotonic()
    with pytest.raises((OpenAPIError, requests.RequestException)):
        fetch_models(dead_base, "k", timeout=1.0, backoff=0.05)
    elapsed = time.monotonic() - started
    # Then: 恰好 2 次退避睡眠（首次 + 2 次重试 = 3 次尝试），且总耗时受控
    assert sleep_calls == [0.05, 0.05]
    assert elapsed < 2.0


# ---------- generate_images ----------


def test_generate_images_returns_parsed_dict(mock_server: MockOpenAIServer) -> None:
    # Given/When: POST mock-b64 生成请求
    resp = generate_images(
        mock_server.base_url, "test-key", {"model": "mock-b64", "prompt": "p"}
    )
    # Then: 返回解析后的 JSON dict，含 data[].b64_json
    assert isinstance(resp, dict)
    assert resp["data"][0]["b64_json"]


def test_generate_images_maps_401_with_envelope_message(
    mock_server: MockOpenAIServer,
) -> None:
    # When: mock-401 模型固定返回 401 信封
    with pytest.raises(RemoteAPIError) as exc_info:
        generate_images(
            mock_server.base_url, "bad-key", {"model": "mock-401", "prompt": "p"}
        )
    # Then: 状态码透出，消息取自 error.message 信封
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value)


def test_generate_images_timeout_never_retries() -> None:
    # 论证（中文注释）：generate_images 按契约把 URL 组合为
    # api_root(base_url) + "/v1/images/generations"，而 mock 服务器
    # 只按「后缀」匹配路由，无法借道 /slow 构造真正的慢响应
    # （追加后的路径不再以 /slow 结尾）。等价证明：监听一个从不 accept、
    # 也永不返回数据的本地 socket——三次握手由内核 backlog 完成，请求发出
    # 后只会等读超时。若实现存在重试（哪怕 3 次 × timeout=(0.4,0.6)），
    # 总耗时必然 ≥ 1.2s；断言 < 1.0s 即证明计费路径零重试。
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)  # 只排队，永不 accept，服务端不会返回任何响应
    port = listener.getsockname()[1]
    try:
        started = time.monotonic()
        with pytest.raises(RemoteAPIError):
            generate_images(
                f"http://127.0.0.1:{port}",
                "test-key",
                {"model": "x", "prompt": "p"},
                timeout=(0.4, 0.6),
            )
        elapsed = time.monotonic() - started
    finally:
        listener.close()
    # Then: 单次尝试即抛出，墙时远小于任何「重试+退避」的可能耗时
    assert elapsed < 1.0


# ---------- resolve_image_bytes: b64 ----------


def test_resolve_image_bytes_b64_decodes_to_png(
    mock_server: MockOpenAIServer,
) -> None:
    # Given: 通过真实 generations 调用取得 64x64 红图的 b64_json
    resp = generate_images(
        mock_server.base_url, "test-key", {"model": "mock-b64", "prompt": "p"}
    )
    value = str(resp["data"][0]["b64_json"])
    # When: 解码 b64 图片项
    data = resolve_image_bytes(ImageItem(kind="b64", value=value))
    # Then: 得到可被 PIL 打开的 64x64 PNG 字节
    with Image.open(BytesIO(data)) as img:
        assert img.size == (64, 64)


def test_resolve_image_bytes_b64_rejects_invalid_base64() -> None:
    # When/Then: 含非法字符的 b64（validate=True 路径）→ 畸形响应错误
    with pytest.raises(MalformedResponseError):
        resolve_image_bytes(ImageItem(kind="b64", value="not-valid-base64!!!"))


# ---------- resolve_image_bytes: url ----------


def test_resolve_image_bytes_url_downloads_png(mock_server: MockOpenAIServer) -> None:
    # Given: 指向 mock 静态图的 url 图片项
    item = ImageItem(kind="url", value=f"{mock_server.base_url}/static/img.png")
    # When
    data = resolve_image_bytes(item)
    # Then: 64x64 PNG
    with Image.open(BytesIO(data)) as img:
        assert img.size == (64, 64)


def test_resolve_image_bytes_rejects_file_scheme() -> None:
    # When/Then: 非 http/https scheme → 配置错误（防本地文件读取）
    with pytest.raises(OpenAPIConfigError):
        resolve_image_bytes(
            ImageItem(kind="url", value="file:///c:/windows/system32/whatever.png")
        )


def test_resolve_image_bytes_enforces_content_length_cap(
    mock_server: MockOpenAIServer,
) -> None:
    # Given: /static/big.png 只声明超大 Content-Length 就断开
    assert client.MAX_IMAGE_BYTES == 50_000_000
    item = ImageItem(kind="url", value=f"{mock_server.base_url}/static/big.png")
    # When/Then: 流式读取前即凭响应头拒绝，错误消息提及大小上限
    with pytest.raises(RemoteAPIError) as exc_info:
        resolve_image_bytes(item)
    msg = str(exc_info.value)
    assert "50" in msg or "size" in msg or "too large" in msg


def test_resolve_image_bytes_never_leaks_auth_across_redirect(
    mock_server: MockOpenAIServer,
) -> None:
    # Given: /redir-crosshost 302 → http://localhost:{port}/redir-target
    # （localhost 与绑定的 127.0.0.1 是不同 host 字符串，专门钓跨主机泄漏）
    item = ImageItem(kind="url", value=f"{mock_server.base_url}/redir-crosshost")
    # When: 手动跟随重定向下载
    data = resolve_image_bytes(item)
    with Image.open(BytesIO(data)) as img:
        assert img.size == (64, 64)
    # Then: 目标端点观测到的请求绝不能携带 Authorization（下载永不鉴权）
    seen = requests.get(
        f"{mock_server.base_url}/debug/redirect_seen",
        headers=_auth,
        timeout=_TIMEOUT,
    ).json()
    assert seen is False


def test_resolve_image_bytes_rejects_non_image_content_type(
    mock_server: MockOpenAIServer,
) -> None:
    # Given: 先发一次 generations，确保 /debug/last_request 返回非空 JSON
    generate_images(
        mock_server.base_url, "test-key", {"model": "mock-b64", "prompt": "p"}
    )
    item = ImageItem(kind="url", value=f"{mock_server.base_url}/debug/last_request")
    # When/Then: Content-Type 为 application/json（非 image/*）→ 拒绝
    with pytest.raises(RemoteAPIError):
        resolve_image_bytes(item)


# ---------- generate_images_dashscope（v0.2 · W1 RED：符号尚未实现） ----------
#
# 说明：不在既有模块级导入行追加 generate_images_dashscope——cfoapi.client
# 已存在，模块级 ImportError 会连带拖红本文件全部既有测试；改用惰性导入，
# 每个新用例单独以 ImportError 精准报红。

# 最小合法原生请求体（model + input.messages 结构，契约见 cfoapi.dashscope）
_NATIVE_BODY: dict[str, object] = {
    "model": "qwen-image-3.0-pro",
    "input": {"messages": [{"role": "user", "content": [{"text": "a red square"}]}]},
}


def _native(
    base_url: str,
    api_key: str,
    body: dict[str, object],
    timeout: tuple[float, float] = (10.0, 600.0),
) -> dict[str, object]:
    """惰性调用 generate_images_dashscope（RED 期 ImportError；GREEN 期真实请求）。"""
    from cfoapi.client import generate_images_dashscope

    return generate_images_dashscope(base_url, api_key, body, timeout=timeout)


def test_generate_images_dashscope_success_returns_dict(
    mock_server: MockOpenAIServer,
) -> None:
    # Given/When: POST 原生端点（qwen-image-3.0-pro）
    payload = _native(mock_server.base_url, "test-key", _NATIVE_BODY)
    # Then: 返回 dict，choices[0] 的 image 指向 /static/img.png
    assert isinstance(payload, dict)
    image = payload["output"]["choices"][0]["message"]["content"][0]["image"]
    assert image.endswith("/static/img.png")


def test_generate_images_dashscope_404_maps_status_and_message(
    mock_server: MockOpenAIServer,
) -> None:
    # When: qwen-image-404 → 404 ModelNotFound
    with pytest.raises(RemoteAPIError) as exc_info:
        _native(mock_server.base_url, "k", {"model": "qwen-image-404", "input": {}})
    # Then: 状态码透出，消息取自原生信封
    assert exc_info.value.status_code == 404
    assert "Model not exist" in str(exc_info.value)


def test_generate_images_dashscope_unknown_model_message(
    mock_server: MockOpenAIServer,
) -> None:
    # When: 目录外模型 → 400
    with pytest.raises(RemoteAPIError) as exc_info:
        _native(mock_server.base_url, "k", {"model": "qwen-image-nope", "input": {}})
    # Then: 信封消息透出
    assert "Unsupported model" in str(exc_info.value)


def test_generate_images_dashscope_n2_returns_two_choices(
    mock_server: MockOpenAIServer,
) -> None:
    # When: parameters.n == 2
    payload = _native(mock_server.base_url, "test-key", {**_NATIVE_BODY, "parameters": {"n": 2}})
    # Then: 两个 choice
    assert len(payload["output"]["choices"]) == 2


def test_generate_images_dashscope_timeout_never_retries() -> None:
    # 计费安全证明（与 generate_images 同法）：监听一个从不 accept 的
    # 本地 socket，请求只会在读超时上失败；timeout=(0.4,0.6) 下若存在
    # 任何重试（≥2 次尝试），墙时必然 ≥ 1.2s，断言 < 1.0s 即证明零重试。
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)  # 只排队，永不 accept，服务端不会返回任何响应
    port = listener.getsockname()[1]
    try:
        started = time.monotonic()
        with pytest.raises(RemoteAPIError):
            _native(
                f"http://127.0.0.1:{port}",
                "test-key",
                _NATIVE_BODY,
                timeout=(0.4, 0.6),
            )
        elapsed = time.monotonic() - started
    finally:
        listener.close()
    # Then: 单次尝试即抛出，墙时远小于任何「重试+退避」的可能耗时
    assert elapsed < 1.0
