"""OpenAI 兼容图像 API 的 HTTP 传输层。

本模块是全插件唯一发起网络调用的地方，封装 requests、统一异常映射，
并把三条互不相同的可靠性策略锁死在函数边界内：

- fetch_models：只读接口，连接错误 / 5xx 允许最多 2 次退避重试；
- generate_images：计费接口，任何情况下零重试（重试可能重复扣费）；
- resolve_image_bytes：图片下载，永不携带 Authorization（防跨主机重定向
  泄漏凭证），并对大小与 Content-Type 设双重防线。

错误消息一律使用英文（ComfyUI 前端会直接展示）。
"""
from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from typing import Final, assert_never
from urllib.parse import urljoin, urlsplit

import requests

from .errors import MalformedResponseError, OpenAPIConfigError, RemoteAPIError
from .parsing import ImageItem, parse_error_envelope, parse_model_list
from .urls import api_root, dashscope_endpoint, normalize_base_url

# 单张图片字节数硬上限：Content-Length 预检与流式累计两条防线共用
MAX_IMAGE_BYTES: Final[int] = 50_000_000

# fetch_models 的最大尝试次数（1 次首发 + 2 次重试）；time.sleep 必须
# 经模块属性调用（import time），测试依赖 monkeypatch client.time.sleep
# 来精确统计退避次数
_MAX_ATTEMPTS: Final[int] = 3

# 图片下载允许手动跟随的重定向跳数与分块读取的缓冲区大小
_MAX_REDIRECTS: Final[int] = 3
_CHUNK_SIZE: Final[int] = 64 * 1024

# 下载超时（连接 10s / 读取 60s）：图片端点不受生成端点的长超时约束
_DOWNLOAD_TIMEOUT: Final[tuple[float, float]] = (10.0, 60.0)

# 每一跳都只放行 http/https，杜绝 file: 等本地/危险协议
_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# 需要手动处理的Redirect状态码（303 语义上应改发 GET，本下载器恒为 GET）
_REDIRECT_STATUSES: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})


def fetch_models(
    base_url: str,
    api_key: str,
    timeout: float = 30.0,
    backoff: float = 0.5,
) -> list[str]:
    """拉取 GET {api_root}/models 并解析出模型 id 列表（保持原顺序）。

    这是唯一允许重试的入口：连接错误（含连接超时）与 5xx 各退避 backoff
    秒后重试，总计最多 3 次尝试；4xx 与读超时立刻失败、绝不重试。
    """
    response = _get_with_retry(_api_url(base_url, "/models"), _bearer(api_key), timeout, backoff)
    _ensure_ok(response)
    return parse_model_list(_json_payload(response))


def generate_images(
    base_url: str,
    api_key: str,
    body: Mapping[str, object],
    timeout: tuple[float, float] = (10.0, 600.0),
) -> dict[str, object]:
    """POST {api_root}/images/generations，返回解析后的 JSON dict。

    计费安全契约：零重试。超时、连接失败与非 2xx 一律第一时间透出为
    RemoteAPIError，由上层（节点）决定是否让用户重新发起，绝不自动补发。
    """
    url = _api_url(base_url, "/images/generations")
    try:
        response = requests.post(url, headers=_bearer(api_key), json=body, timeout=timeout)
    except requests.Timeout as exc:
        raise RemoteAPIError(0, f"POST {url} timed out (timeout={timeout}): {exc}") from exc
    except requests.RequestException as exc:
        raise RemoteAPIError(0, f"POST {url} failed without retry (billing safety): {exc}") from exc
    _ensure_ok(response)
    payload = _json_payload(response)
    if not isinstance(payload, dict):
        raise MalformedResponseError("Images response is not a JSON object")
    return payload


def generate_images_dashscope(
    base_url: str,
    api_key: str,
    body: Mapping[str, object],
    timeout: tuple[float, float] = (10.0, 600.0),
) -> dict[str, object]:
    """POST DashScope 原生 multimodal-generation 端点，返回解析后的 JSON dict。

    计费安全契约与 generate_images 完全一致：零重试。请求体已由
    dashscope.build_dashscope_request 组装完毕，本函数不做任何结构校验；
    非 2xx 经 _ensure_ok 映射为 RemoteAPIError（消息取自原生信封）。
    """
    url = dashscope_endpoint(base_url)
    try:
        response = requests.post(url, headers=_bearer(api_key), json=body, timeout=timeout)
    except requests.Timeout as exc:
        raise RemoteAPIError(0, f"POST {url} timed out (timeout={timeout}): {exc}") from exc
    except requests.RequestException as exc:
        raise RemoteAPIError(0, f"POST {url} failed without retry (billing safety): {exc}") from exc
    _ensure_ok(response)
    payload = _json_payload(response)
    if not isinstance(payload, dict):
        raise MalformedResponseError("DashScope response is not a JSON object")
    return payload


def resolve_image_bytes(item: ImageItem) -> bytes:
    """把一个 ImageItem 物化为图片字节。

    b64 条目本地解码；url 条目走匿名下载（手动跟随重定向、逐跳校验
    scheme、大小与类型双重防线、全程不发送任何鉴权头）。
    """
    match item.kind:
        case "b64":
            return _decode_b64(item.value)
        case "url":
            return _download_image(item.value)
        case unreachable:
            assert_never(unreachable)


# ---------- 内部辅助：URL 与响应映射 ----------


def _api_url(base_url: str, suffix: str) -> str:
    """组合 v1 端点最终 URL：归一化 Base URL → 补全 /v1 → 拼接路径后缀。"""
    return api_root(normalize_base_url(base_url)) + suffix


def _bearer(api_key: str) -> dict[str, str]:
    """构造 OpenAI 风格的 Bearer 鉴权头（仅此一处，下载路径刻意不用）。"""
    return {"Authorization": f"Bearer {api_key}"}


def _ensure_ok(response: requests.Response) -> None:
    """把任意非 2xx 状态统一映射为 RemoteAPIError，消息取自错误信封。"""
    if 200 <= response.status_code < 300:
        return
    raise RemoteAPIError(
        response.status_code,
        parse_error_envelope(response.status_code, _error_payload(response)),
    )


def _error_payload(response: requests.Response) -> object:
    """尽力把错误响应体解析为 JSON；不可解析时退回原始文本供信封兜底。"""
    try:
        return response.json()
    except ValueError:
        return response.text


def _json_payload(response: requests.Response) -> object:
    """解析 2xx 响应体为 JSON；非法 JSON 视为畸形响应而非网络故障。"""
    try:
        return response.json()
    except ValueError as exc:
        raise MalformedResponseError(f"Response from {response.url} is not valid JSON") from exc


# ---------- 内部辅助：带退避的 GET ----------


def _get_with_retry(
    url: str,
    headers: dict[str, str],
    timeout: float,
    backoff: float,
) -> requests.Response:
    """带有限退避的 GET：仅对连接错误与 5xx 重试，最多 _MAX_ATTEMPTS 次。

    读超时不参与重试（契约只锁定 ConnectionError 与 5xx）；重试耗尽后
    返回的最后一次响应可能仍是 5xx，由调用方 _ensure_ok 决定如何抛出。

    连接预算均摊：把 timeout 拆为 (timeout/尝试次数, timeout) 的
    (connect, read) 二元组，使「含重试的全部连接尝试」总墙时不超出
    timeout —— 在 SYN 被静默丢弃（而非立即 RST 拒绝）的网络上，这是
    拉取路径总耗时可控的唯一办法；读超时本就只发生一次，无需拆分。
    """
    attempt = 0
    connect_timeout = timeout / _MAX_ATTEMPTS
    while True:
        attempt += 1
        try:
            response = requests.get(url, headers=headers, timeout=(connect_timeout, timeout))
        except requests.ConnectionError as exc:
            # ConnectTimeout 同为 ConnectionError 与 Timeout 的子类；
            # 语义上属于「连接失败」，必须先于 Timeout 捕获以进入重试分支
            if attempt >= _MAX_ATTEMPTS:
                raise RemoteAPIError(
                    0,
                    f"GET {url} failed to connect after {attempt} attempts: {exc}",
                ) from exc
        except requests.Timeout as exc:
            raise RemoteAPIError(0, f"GET {url} timed out after {timeout}s: {exc}") from exc
        else:
            if response.status_code < 500 or attempt >= _MAX_ATTEMPTS:
                return response
        # 走到这里说明本次尝试属于「可重试」分支：退避后再来一轮
        time.sleep(backoff)


# ---------- 内部辅助：匿名图片下载 ----------


def _download_image(url: str) -> bytes:
    """下载远端图片：手动跟随重定向（最多 3 跳）、零凭证、双重大小防线。

    跳数预算与重定向环：每消费一跳 hops_left 严格递减，即便 Location
    指回已访问过的 URL 也最迟在第 4 个请求处终止，不存在无限循环。
    """
    target = _validated_url(url)
    hops_left = _MAX_REDIRECTS
    while True:
        try:
            # 刻意不传 headers：下载请求一旦带上 Authorization，
            # 跨主机重定向就会把凭证泄漏给第三方
            response = requests.get(
                target,
                allow_redirects=False,
                stream=True,
                timeout=_DOWNLOAD_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RemoteAPIError(0, f"Image download from {target} failed: {exc}") from exc
        with response:
            status = response.status_code
            if status in _REDIRECT_STATUSES:
                if hops_left == 0:
                    raise RemoteAPIError(
                        status,
                        f"Image URL redirects more than {_MAX_REDIRECTS} times",
                    )
                location = response.headers.get("Location")
                if not location:
                    raise RemoteAPIError(
                        status,
                        f"HTTP {status} redirect is missing the Location header",
                    )
                # 相对 Location 基于当前 URL 解析；解析结果再逐跳校验 scheme
                target = _validated_url(urljoin(target, location))
                hops_left -= 1
                continue
            _ensure_ok(response)
            _enforce_image_headers(response)
            return _read_capped(response)


def _validated_url(url: str) -> str:
    """强制 http/https scheme：file: 等协议一律按非法配置拒绝（防本地文件读取）。"""
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise OpenAPIConfigError(f"Image URL scheme must be http or https, got: {url!r}")
    return url


def _enforce_image_headers(response: requests.Response) -> None:
    """读 body 之前的响应头防线：Content-Length 上限优先，其次 Content-Type 白名单。"""
    try:
        # 头缺失或不可解析时按 0 处理，交给流式累计这条第二防线兜底
        declared = int(response.headers.get("Content-Length") or "")
    except ValueError:
        declared = 0
    if declared > MAX_IMAGE_BYTES:
        raise RemoteAPIError(
            200,
            f"Image exceeds {MAX_IMAGE_BYTES} bytes (Content-Length: {declared})",
        )
    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    if content_type.startswith("image/") or content_type == "application/octet-stream":
        return
    raise RemoteAPIError(200, f"Image response has disallowed content type: {content_type!r}")


def _read_capped(response: requests.Response) -> bytes:
    """分块流式读取 body；累计字节一旦越限立即中止（无 Content-Length 时的防线）。"""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(_CHUNK_SIZE):
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise RemoteAPIError(200, f"Image exceeds {MAX_IMAGE_BYTES} bytes while streaming")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_b64(value: str) -> bytes:
    """严格模式解码 b64_json；非法字符（如填充错位、字母表外符号）→ 畸形响应。"""
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        # binascii.Error 是 ValueError 的子类，故单一 except 即可覆盖两者
        raise MalformedResponseError(f"b64_json payload is not valid base64: {exc}") from exc
