"""响应解析与请求体构建的纯函数集合。

只做「解析」不做「传输」：输入均为已解码的 JSON 载荷（object），
非法结构统一抛 MalformedResponseError；用户可见错误消息为英文。

v0.3 起本模块同时承载「交互式基本参数」的纯语义：控件是否已设置的判定
（collect_set_widgets）、size 归一化（normalize_size）与 openai 侧的
协议适配校验 + 键名映射（_openai_widget_entries）；dashscope 侧的对应
语义在 cfoapi.dashscope 中实现并复用本模块的共享纯函数。
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from .errors import MalformedResponseError, OpenAPIConfigError

# params JSON 中不允许覆盖的基础键（由节点输入权威提供）
_LOCKED_BODY_KEYS: Final[frozenset[str]] = frozenset({"model", "prompt"})

# 交互式基本参数控件名（INPUT_TYPES 追加顺序）：两个协议的组装函数共用
# 同一映射形状（值来自节点 kwargs），未设置的控件绝不出现于映射语义中。
BASIC_PARAM_WIDGETS: Final[tuple[str, ...]] = (
    "size",
    "quality",
    "output_format",
    "background",
    "moderation",
    "negative_prompt",
    "count",
)

# size 控件可接受形式：2-5 位数字 × 2-5 位数字，分隔符支持
# x / X / * / × / ✕ / ✖，分隔符两侧允许空白；整串在 strip 后全匹配。
_SIZE_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{2,5})\s*[xX*×✕✖]\s*(\d{2,5})$")

# openai 协议：控件名 → 请求体顶层键（count 映射到 OpenAI 的 n；
# negative_prompt 不在表内——它是 DashScope 专属控件）
_OPENAI_WIDGET_KEYS: Final[Mapping[str, str]] = {
    "size": "size",
    "quality": "quality",
    "output_format": "output_format",
    "background": "background",
    "moderation": "moderation",
    "count": "n",
}


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


def collect_set_widgets(basic_params: Mapping[str, object] | None) -> dict[str, object]:
    """从控件映射中提取「已设置」的基本参数（值已 trim）。

    判定语义（nodes.py 只原样传入控件值，不做任何判断）：
    字符串控件 trim 后非空才算已设置；count 仅 > 1 才算已设置；
    其余类型 / 未设置控件一概不出现在结果中——未设置的控件对请求体
    零贡献，保证全默认时与 v0.2 请求体逐键一致。
    """
    entries: dict[str, object] = {}
    if not basic_params:
        return entries
    for name in BASIC_PARAM_WIDGETS:
        raw = basic_params.get(name)
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                entries[name] = stripped
        elif isinstance(raw, int) and not isinstance(raw, bool) and raw > 1:
            entries[name] = raw
    return entries


def normalize_size(raw: str, separator: str) -> str:
    """把 size 控件输入归一化为协议正典形式 "W<separator>H"；"auto" 原样返回。

    接受 x / X / * / × / ✕ / ✖ 六种分隔符与两侧可选空白，数字每段 2-5 位；
    "auto"（大小写不敏感）交由各协议的组装函数决定是否放行；不匹配且
    非 auto → OpenAPIConfigError 并附格式提示。仅作用于控件通道——
    params JSON 的 size 逐字节透传，绝不经过本函数（中转网关可能要求
    * 形式，用户自己的 JSON 才是权威）。
    """
    value = raw.strip()
    if value.lower() == "auto":
        return "auto"
    m = _SIZE_RE.match(value)
    if m is None:
        raise OpenAPIConfigError(
            f"Invalid size {raw!r}: expected WIDTH{separator}HEIGHT with 2-5 digit dimensions "
            f"(e.g. 1024{separator}1024, 1664{separator}928), or 'auto'"
        )
    return f"{m.group(1)}{separator}{m.group(2)}"


def parse_extra_params(params_json: str) -> dict[str, object]:
    """解析附加 params JSON 为 dict：空串 → 空 dict；非法 JSON 或非对象 →
    OpenAPIConfigError（消息含 "params" 便于用户定位输入件）。
    两个协议的组装函数共用，保证错误文案与 v0.2 完全一致。
    """
    if not params_json:
        return {}
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise OpenAPIConfigError(f"Extra params JSON is invalid: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OpenAPIConfigError("Extra params must be a JSON object")
    return parsed


def _openai_widget_entries(basic_params: Mapping[str, object] | None) -> dict[str, object]:
    """OpenAI 侧基本参数语义：协议适配校验 + size 归一(x) + 控件名→顶层键映射。

    negative_prompt 为 DashScope 专属控件，在 openai 协议下一旦被设置即拒绝
    （JSON 通道不受限，用户仍可经 params 传给支持它的网关）。
    """
    set_widgets = collect_set_widgets(basic_params)
    if "negative_prompt" in set_widgets:
        raise OpenAPIConfigError(
            "negative_prompt is a DashScope-only widget and is not sent under the openai protocol. "
            "Clear the field, or pass it via the params JSON if your gateway supports it."
        )
    entries: dict[str, object] = {}
    for widget, body_key in _OPENAI_WIDGET_KEYS.items():
        if widget not in set_widgets:
            continue
        value = set_widgets[widget]
        entries[body_key] = normalize_size(str(value), "x") if widget == "size" else value
    return entries


def build_request_body(
    prompt: str,
    model: str,
    params_json: str,
    response_format_default: str = "b64_json",
    basic_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """合并节点输入与用户附加 params JSON，生成最终请求体。

    model/prompt 两键被锁定不可覆盖；response_format 允许 params 覆盖；
    params 非法（非 JSON 或非对象）时抛 OpenAPIConfigError。

    basic_params 为交互式基本参数控件映射（键见 BASIC_PARAM_WIDGETS）：
    控件派生键先写入请求体（size 归一为 WxH、count→n 等），随后
    params JSON 的同名键覆盖控件值——JSON 是高级通道、优先级最高
    （留空即不发送；与 params JSON 同名时 JSON 优先）。
    basic_params 为 None/全默认时请求体与 v0.2 逐键一致。
    """
    body: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format_default,
    }
    body.update(_openai_widget_entries(basic_params))
    if not params_json:
        return body
    for key, value in parse_extra_params(params_json).items():
        if key in _LOCKED_BODY_KEYS:
            continue
        body[key] = value
    return body
