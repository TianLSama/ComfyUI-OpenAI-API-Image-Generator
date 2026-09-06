"""插件的统一异常层级。

所有面向用户的错误消息一律使用英文（ComfyUI 前端节点报错会直接展示
这些字符串）；本模块不依赖任何第三方库，可被 urls / parsing / cache 等
纯模块安全导入。
"""
from __future__ import annotations


class OpenAPIError(Exception):
    """插件全部自定义异常的基类。"""


class OpenAPIConfigError(OpenAPIError):
    """用户配置非法（URL、params JSON 等）时抛出。"""


class RemoteAPIError(OpenAPIError):
    """远端 API 返回非 2xx 状态码时抛出，携带 HTTP 状态码。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class MalformedResponseError(OpenAPIError):
    """响应结构不符合预期（缺字段 / 类型错误）无法解析时抛出。"""
