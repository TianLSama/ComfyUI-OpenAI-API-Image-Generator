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
    JSON / 协议 + 交互式基本参数控件（size / quality / output_format /
    background / moderation / negative_prompt / count，留空即不发送）；
    输出为标准 IMAGE 张量 [N, H, W, C] float32、取值 [0, 1]。
    """

    RETURN_TYPES: Final[tuple[str, ...]] = ("IMAGE",)
    RETURN_NAMES: Final[tuple[str, ...]] = ("IMAGE",)
    FUNCTION: Final[str] = "generate_image"
    CATEGORY: Final[str] = "外部文生图 (OpenAPI)"  # 单段中文分类(仿 llm_party 菜单样式) → 显示在添加节点根菜单
    DESCRIPTION: Final[str] = (
        "外部文生图：经 OpenAI 兼容 /images/generations 或阿里百炼原生接口生成图像。"
        "用法：填 base_url 与 api_key → 点「获取模型列表」选模型 → 填提示词运行。"
        "基础参数用交互控件（图片尺寸/质量/输出格式/背景/审核/负面词/数量），留空不发送；"
        "size 写法 x/*/× 均可，按协议自动归一（openai→x，dashscope→*，auto 仅 openai）。"
        "其余参数写 params JSON（同名键以 JSON 为准、原样透传；model/prompt 不可覆盖）。"
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
            # 向后兼容规则（自基本参数控件加入后修订）：protocol 固定保持在
            # 第 7 位（索引 6）——旧工作流的 widgets_values 按索引映射，前 7
            # 键的顺序永不可变；基本参数控件一律追加在 protocol 之后，未来
            # 新增控件只能继续追加在 count 之后。
            "protocol": (_PROTOCOLS, {"default": protocol or "openai"}),
            # ↓ 交互式基本参数：默认值全为空/1 = 不发送任何键，请求体与
            #   v0.2 逐键一致；与 params JSON 同名时 JSON 优先（高级通道）。
            "size": (
                "STRING",
                {"default": "", "placeholder": "如 1024x1024 / 1664*928 / auto；留空=走 params JSON"},
            ),
            "quality": (["", "auto", "high", "medium", "low"], {"default": ""}),
            "output_format": (["", "png", "jpeg", "webp"], {"default": ""}),
            "background": (["", "auto", "transparent", "opaque"], {"default": ""}),
            "moderation": (["", "auto", "low"], {"default": ""}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
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
        size: str = "",
        quality: str = "",
        output_format: str = "",
        background: str = "",
        moderation: str = "",
        negative_prompt: str = "",
        count: int = 1,
    ) -> tuple[torch.Tensor]:
        """校验 → 按协议组请求体 → 调远端 → 解析解码 → 返回 IMAGE 张量 1-tuple。

        全部本地校验必须先于任何网络调用发生（含非法 protocol 的即时拒绝）；
        基本参数控件的「是否已设置」判定、协议适配校验与 size 归一化全部
        下沉在 parsing / dashscope 的纯函数内（同样先于网络调用），本方法
        只把控件值原样打包传递；网络之后，OpenAPIError 家族（含远端 401、
        200-带错误等）一律转译为带固定前缀的 ValueError，原始英文消息原样
        透出以便用户排障。
        """
        if protocol not in _PROTOCOLS:
            raise _user_error(f"不支持的协议 / unsupported protocol: {protocol!r}，可选 {_PROTOCOLS}")
        if model == NO_MODEL_SENTINEL:
            raise _user_error(
                "尚未选择模型：请先点击「获取模型列表」按钮 / no model selected, press Fetch Models first"
            )
        if not prompt or not prompt.strip():
            raise _user_error("提示词为空：请描述要生成的图像 / prompt is empty")

        # 控件值原样打包：语义（trim 判定 / 协议适配 / size 归一 / 键名映射 /
        # JSON 覆盖优先级）全部由协议侧纯函数负责，本方法不解释任何值。
        basic_params: dict[str, object] = {
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "background": background,
            "moderation": moderation,
            "negative_prompt": negative_prompt,
            "count": count,
        }
        try:
            full_prompt = parsing.apply_system_prompt(prompt, system_prompt)
            if protocol == "dashscope":
                # 原生分支：仅复用 apply_system_prompt 与末尾解码/张量转换，
                # 请求体组装与响应抽取走 cfoapi.dashscope 纯函数。
                body = dashscope.build_dashscope_request(full_prompt, model, params, basic_params)
                response = client.generate_images_dashscope(base_url, api_key, body)
                items = dashscope.extract_dashscope_image_items(response)
            else:
                body = parsing.build_request_body(full_prompt, model, params, basic_params=basic_params)
                response = client.generate_images(base_url, api_key, body)
                items = parsing.extract_image_items(response)
            images = [imaging.decode_image_bytes(client.resolve_image_bytes(item)) for item in items]
            tensor = imaging.pil_to_tensor(images)
        except OpenAPIError as exc:
            raise _user_error(str(exc)) from exc
        return (tensor,)
