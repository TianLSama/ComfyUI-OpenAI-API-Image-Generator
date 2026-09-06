"""ComfyUI-OpenAPI：把外部 OpenAI 兼容图像 API 接入 ComfyUI 的自定义节点插件。

导入本包即暴露节点映射（NODE_CLASS_MAPPINGS）并挂载 /comfyui_openapi 代理路由。
"""
from . import routes  # 导入即注册 PromptServer 路由（F401 已由 pyproject per-file-ignores 豁免）
from .nodes import OpenAPIImageGenerator

NODE_CLASS_MAPPINGS = {"OpenAPIImageGenerator": OpenAPIImageGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAPIImageGenerator": "OpenAPI Image Generator (外部文生图)"}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
