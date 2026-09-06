"""DashScope（阿里云百炼）原生 multimodal-generation 协议的纯函数集合。

本模块只做「请求组装 / 响应抽取」，不做任何传输：
- build_dashscope_request：把节点输入拼成原生请求体（model + input.messages
  形状，params 落入 parameters 键）；
- extract_dashscope_image_items：从 output.choices[].message.content[].image
  按序提取图像 URL，并识别「HTTP 200 但顶层带非空 code」的错误信封。

v0.3 起本模块还承载 dashscope 侧「交互式基本参数」的纯语义：协议适配
校验（openai 专属控件 / size 'auto' 拒绝）、size 归一为 W*H、控件名→
parameters 子键映射（共享纯函数复用 cfoapi.parsing）。

用户可见错误消息一律英文（ComfyUI 前端原样展示）。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .errors import MalformedResponseError, OpenAPIConfigError
from .parsing import ImageItem, collect_set_widgets, normalize_size, parse_extra_params

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

# openai 专属控件：dashscope 协议下一旦被设置即视为协议错配，必须点名报错，
# 提示用户清空或改走 params JSON（原生 parameters 键不受限）。
_OPENAI_ONLY_WIDGETS: Final[tuple[str, ...]] = ("quality", "output_format", "background", "moderation")

# dashscope 协议：控件名 → "parameters" 子键（count 映射到原生的 n；
# quality 等 openai 专属控件不在表内）
_DASHSCOPE_PARAM_KEYS: Final[Mapping[str, str]] = {
    "size": "size",
    "negative_prompt": "negative_prompt",
    "count": "n",
}


def _dashscope_widget_entries(basic_params: Mapping[str, object] | None) -> dict[str, object]:
    """DashScope 侧基本参数语义：协议适配校验 + size 归一(*) + 控件名→parameters 子键映射。

    openai 专属控件（quality/output_format/background/moderation）一旦被设置
    即拒绝并点名；size 的正典形式为 W*H 且 "auto" 仅 openai 支持。
    """
    set_widgets = collect_set_widgets(basic_params)
    offending = [name for name in _OPENAI_ONLY_WIDGETS if name in set_widgets]
    if offending:
        raise OpenAPIConfigError(
            f"OpenAI-only option(s) set while protocol is 'dashscope': {', '.join(offending)}. "
            "Clear these fields, or pass equivalent parameters via the params JSON "
            "(they land in 'parameters') if your gateway supports them."
        )
    entries: dict[str, object] = {}
    for widget, param_key in _DASHSCOPE_PARAM_KEYS.items():
        if widget not in set_widgets:
            continue
        value = set_widgets[widget]
        if widget == "size":
            size = normalize_size(str(value), "*")
            if size == "auto":
                raise OpenAPIConfigError(
                    "size 'auto' is only supported by the openai protocol; "
                    "dashscope requires an explicit resolution like 1024*1024."
                )
            entries[param_key] = size
        else:
            entries[param_key] = value
    return entries


def build_dashscope_request(
    prompt: str,
    model: str,
    params_json: str,
    basic_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """组装原生 multimodal-generation 请求体。

    锁定形状：model 顶层 + input.messages 单条 user 消息、content 内仅
    text 一项；params_json 非空时必须是 JSON 对象，原样（逐字节）落入
    "parameters" 键（空串且无控件值时绝不出现该键）。params 非法抛
    OpenAPIConfigError，消息含 "params" 以便用户定位输入件。

    basic_params 为交互式基本参数控件映射（键见 parsing.BASIC_PARAM_WIDGETS）：
    控件派生键（size 归一为 W*H、negative_prompt、count→n）先落入
    "parameters"，随后 params JSON 同名键覆盖——JSON 是高级通道、优先级
    最高。basic_params 为 None/全默认时请求体与 v0.2 逐键一致。
    """
    body: dict[str, object] = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
    }
    widget_entries = _dashscope_widget_entries(basic_params)
    if not params_json:
        if widget_entries:
            body["parameters"] = widget_entries
        return body
    parsed = parse_extra_params(params_json)
    body["parameters"] = {**widget_entries, **parsed} if widget_entries else parsed
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
