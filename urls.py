"""Base URL 归一化与 API 根路径拼接。

纯字符串处理，无任何网络调用；错误消息为英文（用户可见）。
"""
from __future__ import annotations

from urllib.parse import urlsplit

from .errors import OpenAPIConfigError

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def normalize_base_url(raw: str) -> str:
    """把用户输入的 Base URL 归一化为规范形式。

    规则：去除首尾空白；scheme 与 host 小写、path 保留原大小写；
    去掉全部尾部 "/"；空串 / 缺 scheme / 非 http(s) 协议均视为非法配置。
    """
    candidate = raw.strip()
    if not candidate:
        raise OpenAPIConfigError(
            "Base URL is empty. Expected an http(s) URL, e.g. https://api.example.com/v1"
        )
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise OpenAPIConfigError(f"Base URL is not a valid http(s) URL: {candidate!r} ({exc})") from exc

    scheme = parts.scheme.lower()
    if not scheme:
        raise OpenAPIConfigError(
            f"Base URL is missing a scheme. Expected http:// or https://, got: {candidate!r}"
        )
    if scheme not in _ALLOWED_SCHEMES:
        raise OpenAPIConfigError(
            f"Base URL scheme must be http or https, got: {scheme}:// (in {candidate!r})"
        )
    if not parts.netloc:
        raise OpenAPIConfigError(
            f"Base URL is missing a host. Expected http:// or https://, got: {candidate!r}"
        )

    # host:port（含 userinfo）整体小写；path 保留大小写，仅裁剪尾部 "/"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def api_root(normalized: str) -> str:
    """在已归一化的 Base URL 后补全 "/v1"（幂等）。"""
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


# DashScope 原生 multimodal-generation 端点路径（拼接在剥离兼容后缀之后的根上）
_DASHSCOPE_NATIVE_SUFFIX = "/api/v1/services/aigc/multimodal-generation/generation"

# 可剥离的已知端点后缀，按长度降序排列：/api/v1 必须先于 /v1 匹配，
# 否则 "https://x.com/api/v1" 只剥掉 "/v1" 会残留 "/api" 导致路径翻倍。
_DASHSCOPE_STRIPPABLE = ("/compatible-mode/v1", "/api/v1", "/v1")


def dashscope_endpoint(base_url: str) -> str:
    """把任意形态的 DashScope Base URL 组合为原生生成端点完整 URL。

    规则：先 normalize_base_url 归一化（非法 scheme 在此即抛错），再剥离
    **一段**已知后缀（/compatible-mode/v1、/api/v1、/v1 之一；多级反代
    前缀如 /dashscope-mock 保留），最后拼接原生完整路径。
    """
    normalized = normalize_base_url(base_url)
    for suffix in _DASHSCOPE_STRIPPABLE:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized + _DASHSCOPE_NATIVE_SUFFIX
