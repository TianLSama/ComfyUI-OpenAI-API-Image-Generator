"""响应解析与请求体构建的纯函数集合。

只做「解析」不做「传输」：输入均为已解码的 JSON 载荷（object），
非法结构统一抛 MalformedResponseError；用户可见错误消息为英文。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, Literal

from .errors import MalformedResponseError, OpenAPIConfigError

# params JSON 中不允许覆盖的基础键（由节点输入权威提供）
_LOCKED_BODY_KEYS: Final[frozenset[str]] = frozenset({"model", "prompt"})


@dataclass(frozen=True, slots=True)
class ImageItem:
    """一张生成结果图像：b64 内联数据或远端 URL。"""

    kind: Literal["b64", "url"]
    value: str


def parse_model_list(payload: object) -> list[str]:
    """解析 GET /v1/models 响应，按 data[] 原顺序提取字符串 id。"""
    if not isinstance(payload, dict):
        raise MalformedResponseError("Models response is not a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise MalformedResponseError("Models response is missing the 'data' array")
    models: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str):
            models.append(model_id)
    return models


def parse_error_envelope(status_code: int, payload: object) -> str:
    """从非 2xx 响应体中尽力提取人类可读错误消息。

    兼容三种形状：OpenAI 的 {"error":{"message":m}}、网关风格的
    {"code":c,"message":m}，以及完全无法解析时的通用兜底文案（含状态码）。
    """
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = payload.get("message")
        if isinstance(message, str) and message:
            code = payload.get("code")
            if isinstance(code, str | int) and str(code):
                return f"{code}: {message}"
            return message
    return f"Remote API returned HTTP {status_code} with an unrecognized error body"


def extract_image_items(payload: object) -> list[ImageItem]:
    """从图像生成响应中提取图像列表。

    优先读 OpenAI 的 data[]（条目内 b64_json 优于 url）；data 缺失或
    为空时回退到 SiliconFlow 风格的顶层 images[].url；两者皆无则视为畸形响应。
    """
    if not isinstance(payload, dict):
        raise MalformedResponseError("Images response is not a JSON object")

    items: list[ImageItem] = []
    data = payload.get("data")
    if isinstance(data, list):
        for entry in data:
            item = _image_item_from_entry(entry)
            if item is not None:
                items.append(item)
    if not items:
        images = payload.get("images")
        if isinstance(images, list):
            for entry in images:
                if not isinstance(entry, dict):
                    continue
                url = entry.get("url")
                if isinstance(url, str) and url:
                    items.append(ImageItem(kind="url", value=url))
    if not items:
        raise MalformedResponseError("Images response contains no usable image data")
    return items


def _image_item_from_entry(entry: object) -> ImageItem | None:
    """解析 data[] 单条目：b64_json 优先于 url，均无则返回 None。"""
    if not isinstance(entry, dict):
        return None
    b64 = entry.get("b64_json")
    if isinstance(b64, str) and b64:
        return ImageItem(kind="b64", value=b64)
    url = entry.get("url")
    if isinstance(url, str) and url:
        return ImageItem(kind="url", value=url)
    return None


def apply_system_prompt(prompt: str, system_prompt: str) -> str:
    """把 system prompt 前置拼接到用户 prompt；纯空白视为未提供。"""
    if system_prompt.strip():
        return f"{system_prompt}\n{prompt}"
    return prompt


def build_request_body(
    prompt: str,
    model: str,
    params_json: str,
    response_format_default: str = "b64_json",
) -> dict[str, object]:
    """合并节点输入与用户附加 params JSON，生成最终请求体。

    model/prompt 两键被锁定不可覆盖；response_format 允许 params 覆盖；
    params 非法（非 JSON 或非对象）时抛 OpenAPIConfigError。
    """
    body: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format_default,
    }
    if not params_json:
        return body
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise OpenAPIConfigError(f"Extra params JSON is invalid: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OpenAPIConfigError("Extra params must be a JSON object")
    for key, value in parsed.items():
        if key in _LOCKED_BODY_KEYS:
            continue
        body[key] = value
    return body
