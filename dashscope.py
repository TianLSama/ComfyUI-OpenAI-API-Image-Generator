"""DashScope（阿里云百炼）原生 multimodal-generation 协议的纯函数集合。

本模块只做「请求组装 / 响应抽取」，不做任何传输：
- build_dashscope_request：把节点输入拼成原生请求体（model + input.messages
  形状，params 落入 parameters 键）；
- extract_dashscope_image_items：从 output.choices[].message.content[].image
  按序提取图像 URL，并识别「HTTP 200 但顶层带非空 code」的错误信封。

用户可见错误消息一律英文（ComfyUI 前端原样展示）。
"""
from __future__ import annotations

import json
from typing import Final

from .errors import MalformedResponseError, OpenAPIConfigError
from .parsing import ImageItem

# 原生图像生成模型目录（顺序即节点 COMBO 展示顺序）。
# qwen-image-3.0-pro / qwen-image-plus 为测试与 mock 服务器锁定的契约名，
# 其余为百炼平台公开的通用图像生成模型 id；目录为静态常量，不做远程拉取。
DASHSCOPE_IMAGE_MODELS: Final[tuple[str, ...]] = (
    "qwen-image-3.0-pro",
    "qwen-image-max",
    "qwen-image-plus",
    "qwen-image",
    "wan2.5-t2i-preview",
    "wan2.2-t2i-plus",
    "wan2.2-t2i-flash",
)


def build_dashscope_request(prompt: str, model: str, params_json: str) -> dict[str, object]:
    """组装原生 multimodal-generation 请求体。

    锁定形状：model 顶层 + input.messages 单条 user 消息、content 内仅
    text 一项；params_json 非空时必须是 JSON 对象，原样落入
    "parameters" 键（空串时绝不出现该键）。params 非法抛 OpenAPIConfigError，
    消息含 "params" 以便用户定位输入件。
    """
    body: dict[str, object] = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
    }
    if not params_json:
        return body
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise OpenAPIConfigError(f"Extra params JSON is invalid: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OpenAPIConfigError("Extra params must be a JSON object")
    body["parameters"] = parsed
    return body


def extract_dashscope_image_items(payload: object) -> list[ImageItem]:
    """从原生响应中按序提取图像条目。

    防线顺序：顶层非 dict → 畸形；顶层非空 "code"（200-带错误信封）→
    畸形且消息同时携带 code 与 message；缺 output → 畸形；
    choices 内非 dict / 非 image / 非字符串条目全部跳过；一张图都没有 → 畸形。
    """
    if not isinstance(payload, dict):
        raise MalformedResponseError("DashScope response is not a JSON object")

    code = payload.get("code")
    if isinstance(code, str) and code:
        message = payload.get("message")
        detail = message if isinstance(message, str) and message else "<no message>"
        raise MalformedResponseError(f"DashScope returned error code '{code}': {detail}")

    output = payload.get("output")
    if not isinstance(output, dict):
        raise MalformedResponseError("DashScope response is missing the 'output' object")

    items: list[ImageItem] = []
    choices = output.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            items.extend(_images_from_choice(choice))
    if not items:
        raise MalformedResponseError("DashScope response contains no usable image URL")
    return items


def _images_from_choice(choice: object) -> list[ImageItem]:
    """解析单个 choice：仅收 message.content[] 内字符串型 image 字段。"""
    if not isinstance(choice, dict):
        return []
    message = choice.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    items: list[ImageItem] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        image = part.get("image")
        if isinstance(image, str) and image:
            items.append(ImageItem(kind="url", value=image))
    return items
