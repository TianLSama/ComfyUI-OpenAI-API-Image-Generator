"""ComfyUI-OpenAPI 的 aiohttp 代理路由（浏览器 → 本插件 → 远端 API）。

两条路由挂在 PromptServer 的 /comfyui_openapi 前缀下，与
web/js/openapi.js 中的 ROUTE_FETCH / ROUTE_CACHE 常量逐字对应；
请求/响应字段名（base_url / api_key / ok / models / error）
即为前后端锁定契约，任何一侧改名都会破坏前端回填逻辑。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote

from aiohttp import web
from server import PromptServer

from . import cache, client
from .dashscope import DASHSCOPE_IMAGE_MODELS
from .errors import OpenAPIConfigError, RemoteAPIError
from .urls import normalize_base_url

# 部署自检：ComfyUI 0.34 + frontend 1.49 的 /extensions 列表会对插件目录名做
# percent-encode（空格→%20），而 aiohttp 静态路由按【原目录名】前缀匹配，
# 编码后的请求必然 404 —— 前端永远加载不到本插件 JS（症状：节点可用但
# 「获取模型列表」按钮与中文标签永不出现，且清缓存/换设备无效）。
# 目录名含空格等需编码字符时在启动控制台打印明确指引，避免再次踩坑。
_PLUGIN_DIR_NAME = Path(__file__).resolve().parent.name
if quote(_PLUGIN_DIR_NAME) != _PLUGIN_DIR_NAME:
    print(
        "[ComfyUI-OpenAPI] 警告：插件目录名含需 URL 编码的字符（如空格）："
        f"{_PLUGIN_DIR_NAME!r}。ComfyUI 0.34+/frontend 1.49+ 下前端扩展 JS 将 404，"
        "「获取模型列表」按钮与中文标签无法加载。请把 custom_nodes 下的本插件目录"
        "重命名为不含空格的名字（建议 ComfyUI-OpenAI-API-Image-Generator）后重启。"
    )


@PromptServer.instance.routes.post("/comfyui_openapi/fetch_models")
async def handle_fetch_models(request: web.Request) -> web.Response:
    """POST /comfyui_openapi/fetch_models —— 供 openapi.js fetchModels() 调用。

    JS 契约：请求体为 {"base_url": str, "api_key": str, "protocol": str?}
    （protocol 缺省为 "openai"，未知值 → 400）；成功响应
    {"ok": true, "models": string[], "base_url": str}，失败一律
    {"ok": false, "error": str}。错误映射：请求体不是合法 JSON → 400；
    OpenAPIConfigError（URL 非法等）与未知 protocol → 400；RemoteAPIError
    （远端非 2xx / 网络失败）→ 502；其余未知异常 → 500。阻塞的 requests
    调用经 asyncio.to_thread 移出事件循环，避免卡死 ComfyUI 主服务。
    """
    try:
        body = await request.json()
    except ValueError:
        # aiohttp 在请求体缺失或非法 JSON 时抛 json.JSONDecodeError（ValueError 子类）
        return web.json_response({"ok": False, "error": "请求体必须是 JSON"}, status=400)

    try:
        base_url = str(body.get("base_url") or "").strip()
        api_key = str(body.get("api_key") or "").strip()
        protocol = str(body.get("protocol") or "openai")
        if protocol not in ("openai", "dashscope"):
            return web.json_response(
                {"ok": False, "error": f"Unsupported protocol: {protocol!r} (expected 'openai' or 'dashscope')"},
                status=400,
            )
        normalized = normalize_base_url(base_url)
        if protocol == "dashscope":
            # 原生协议没有 OpenAI 风格的 /models 列表端点：直接缓存本插件
            # 锁定的静态目录，零远程调用、零计费风险。
            models = list(DASHSCOPE_IMAGE_MODELS)
        else:
            models = await asyncio.to_thread(client.fetch_models, normalized, api_key)
        cache.save_models(normalized, models)
        if api_key:
            cache.save_api_key(normalized, api_key)
    except OpenAPIConfigError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except RemoteAPIError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)

    return web.json_response({"ok": True, "models": models, "base_url": normalized})


@PromptServer.instance.routes.get("/comfyui_openapi/cache")
async def handle_cache_lookup(request: web.Request) -> web.Response:
    """GET /comfyui_openapi/cache?base_url=... —— 供 openapi.js onConfigure 预填 api_key。

    JS 契约：本路由是尽力而为（best-effort）的缓存读取，**永远**返回 200
    {"ok": true, "api_key": str|null, "models": string[]}；base_url 查询参数
    缺失/非法或任何缓存读取错误都退化为 {"ok": true, "api_key": null,
    "models": []}，绝不返回 4xx/5xx——前端对预填失败是静默容错的。
    cache.get_api_key / get_models 本身对非法 URL 宽容，外层再兜底任何异常。
    """
    base_url = request.rel_url.query.get("base_url") or ""
    try:
        api_key = cache.get_api_key(base_url)
        models = cache.get_models(base_url)
        return web.json_response({"ok": True, "api_key": api_key, "models": models})
    except Exception:
        return web.json_response({"ok": True, "api_key": None, "models": []})
