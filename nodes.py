"""OpenAI 兼容图像生成 API 的 ComfyUI 节点（V1 node API）。

本模块刻意不导入任何 ComfyUI 宿主模块（server / comfy / folder_paths），
使其可以在 ComfyUI 进程之外被纯 pytest 直接导入与测试。全部业务逻辑
委托给包内纯模块：cache（模型列表）、parsing（请求体/响应解析）、
client（唯一的 HTTP 传输层）、imaging（解码与 IMAGE 张量转换）。

面向用户的错误统一为 ValueError("OpenAPI Image Generator: <detail>")：
前缀为固定英文（ComfyUI 前端原样展示，供排障检索），detail 可中英混排。
"""
from __future__ import annotations

from typing import Final

import torch

from . import cache, client, dashscope, imaging, parsing
from .errors import OpenAPIError

# 模型缓存为空时 model COMBO 的唯一占位项（按钮文案与测试断言均依赖该常量）
NO_MODEL_SENTINEL: Final[str] = "(no models — press Fetch Models)"

# 协议 COMBO 的固定选项（openai 在前，向后兼容默认值）；同一列表对象直接
# 交给 ComfyUI，前端按下拉渲染。
_PROTOCOLS: Final[list[str]] = ["openai", "dashscope"]

# 所有节点级用户可见错误的固定英文前缀（不得本地化、不得改写）
_ERROR_PREFIX: Final[str] = "OpenAPI Image Generator: "

# base_url 控件在缓存未命中时展示的官方默认端点
_DEFAULT_BASE_URL: Final[str] = "https://api.openai.com/v1"


def _user_error(detail: str) -> ValueError:
    """把细节消息包装为带固定前缀、供 ComfyUI 展示的 ValueError。"""
    return ValueError(f"{_ERROR_PREFIX}{detail}")


class OpenAPIImageGenerator:
    """经由 OpenAI 兼容 /images/generations 端点生成图像的 V1 节点。

    输入为 Base URL / API Key / 模型 / 提示词 / 系统提示词 / 附加 params
    JSON；输出为标准 IMAGE 张量 [N, H, W, C] float32、取值 [0, 1]。
    """

    RETURN_TYPES: Final[tuple[str, ...]] = ("IMAGE",)
    RETURN_NAMES: Final[tuple[str, ...]] = ("IMAGE",)
    FUNCTION: Final[str] = "generate_image"
    CATEGORY: Final[str] = "外部文生图 (OpenAPI)"  # 单段中文分类(仿 llm_party 菜单样式) → 显示在添加节点根菜单
    DESCRIPTION: Final[str] = (
        "通过 OpenAI 兼容的图像生成 API（/images/generations）生成图像。"
        "先在「获取模型列表」按钮拉取模型，再填写提示词运行；"
        "附加参数以 JSON 形式填入 params 字段，model/prompt 两键不可覆盖。"
        "v0.2 起支持 protocol 协议选择：openai（兼容端点）或 dashscope（阿里云百炼原生 multimodal-generation）。"
    )

    @classmethod
    def INPUT_TYPES(
        cls,
        base_url: str = "",
        api_key: str = "",
        protocol: str = "openai",
        **kwargs: object,
    ) -> dict[str, dict[str, tuple[object, ...]]]:
        """声明节点输入；model COMBO 按协议与缓存三态驱动。"""
        # 三态取选项：
        # 1) dashscope 协议：原生端点没有 OpenAI /models，缓存未命中时回落到
        #    本插件锁定的静态目录；
        # 2) 带 base_url（前端刷新单节点定义）→ 按端点精确查询，绝不混入目录；
        # 3) 无参调用（ComfyUI /prompt 校验与 /object_info 的固定行为）→ 取全部
        #    缓存条目的并集并并入 DashScope 目录（缓存在前、去重），否则
        #    value_not_in_list 会拒绝任何已缓存或原生协议下的合法模型。
        if protocol == "dashscope":
            options: list[str] = cache.get_models(base_url) or list(dashscope.DASHSCOPE_IMAGE_MODELS)
        elif base_url:
            options = cache.get_models(base_url) or [NO_MODEL_SENTINEL]
        else:
            options = cache.get_all_models() or [NO_MODEL_SENTINEL]
            seen = set(options)
            options.extend(m for m in dashscope.DASHSCOPE_IMAGE_MODELS if m not in seen)
        required: dict[str, tuple[object, ...]] = {
            "base_url": (
                "STRING",
                {"default": base_url or _DEFAULT_BASE_URL, "placeholder": "https://api.example.com/v1"},
            ),
            "api_key": (
                "STRING",
                {"default": api_key, "password": True, "placeholder": "sk-..."},
            ),
            "model": (options,),
            "prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": "提示词（可连线输入）",
                },
            ),
            "system_prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": "系统提示词（可选）",
                },
            ),
            "params": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": 'JSON 参数，如 {"size":"1024x1024"}',
                },
            ),
            # protocol 必须是最后一个键：旧工作流的 widgets_values 按索引映射，
            # 新控件追加在尾部才能保证向后兼容。
            "protocol": (_PROTOCOLS, {"default": protocol or "openai"}),
        }
        return {"required": required}

    @staticmethod
    def IS_CHANGED(*args: object, **kwargs: object) -> float:
        """恒为 NaN（nan != nan），令 ComfyUI 每次执行都重跑、绝不缓存远端结果。"""
        return float("nan")

    def generate_image(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: str,
        params: str,
        protocol: str = "openai",
    ) -> tuple[torch.Tensor]:
        """校验 → 按协议组请求体 → 调远端 → 解析解码 → 返回 IMAGE 张量 1-tuple。

        全部本地校验必须先于任何网络调用发生（含非法 protocol 的即时拒绝）；
        网络之后，OpenAPIError 家族（含远端 401、200-带错误等）一律转译为
        带固定前缀的 ValueError，原始英文消息原样透出以便用户排障。
        """
        if protocol not in _PROTOCOLS:
            raise _user_error(f"不支持的协议 / unsupported protocol: {protocol!r}，可选 {_PROTOCOLS}")
        if model == NO_MODEL_SENTINEL:
            raise _user_error(
                "尚未选择模型：请先点击「获取模型列表」按钮 / no model selected, press Fetch Models first"
            )
        if not prompt or not prompt.strip():
            raise _user_error("提示词为空：请描述要生成的图像 / prompt is empty")

        try:
            full_prompt = parsing.apply_system_prompt(prompt, system_prompt)
            if protocol == "dashscope":
                # 原生分支：仅复用 apply_system_prompt 与末尾解码/张量转换，
                # 请求体组装与响应抽取走 cfoapi.dashscope 纯函数。
                body = dashscope.build_dashscope_request(full_prompt, model, params)
                response = client.generate_images_dashscope(base_url, api_key, body)
                items = dashscope.extract_dashscope_image_items(response)
            else:
                body = parsing.build_request_body(full_prompt, model, params)
                response = client.generate_images(base_url, api_key, body)
                items = parsing.extract_image_items(response)
            images = [imaging.decode_image_bytes(client.resolve_image_bytes(item)) for item in items]
            tensor = imaging.pil_to_tensor(images)
        except OpenAPIError as exc:
            raise _user_error(str(exc)) from exc
        return (tensor,)
