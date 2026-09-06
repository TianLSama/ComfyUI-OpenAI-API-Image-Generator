"""模型列表 / API Key 的本地 JSON 文件缓存。

文件形状：{"<normalized_base_url>": {"models": [...], "api_key": "..."}}。
路径解析优先级：显式 path 参数 > COMFYUI_OPENAPI_CACHE_DIR 目录环境变量 >
<插件根>/cache/openapi_cache.json。所有读取对缺失/损坏文件宽容返回空值，
保证节点 INPUT_TYPES 阶段（base_url 可能为空串）永不抛异常。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import OpenAPIConfigError
from .urls import normalize_base_url

_ENV_CACHE_DIR = "COMFYUI_OPENAPI_CACHE_DIR"
_CACHE_FILENAME = "openapi_cache.json"
# __file__ 位于插件根目录，parent 即插件根
_PLUGIN_ROOT = Path(__file__).resolve().parent


def _resolve_path(path: Path | str | None) -> Path:
    """按三级优先级解析缓存文件的最终路径。"""
    if path is not None:
        return Path(path)
    env_dir = os.environ.get(_ENV_CACHE_DIR, "").strip()
    if env_dir:
        return Path(env_dir) / _CACHE_FILENAME
    return _PLUGIN_ROOT / "cache" / _CACHE_FILENAME


def load_cache(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """读取缓存文件；文件缺失、JSON 损坏或顶层形状非法一律返回 {}。"""
    file = _resolve_path(path)
    try:
        raw = file.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    # 仅保留形状正确的条目（str 键 → dict 值），脏条目直接丢弃
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, dict)}


def save_models(base_url: str, models: list[str], path: Path | str | None = None) -> None:
    """写入某 base_url 的模型列表，合并保留既有条目（含 api_key）。"""
    _save_fields(base_url, path, {"models": list(models)})


def save_api_key(base_url: str, api_key: str, path: Path | str | None = None) -> None:
    """写入某 base_url 的 API Key，合并保留既有条目（含 models）。"""
    _save_fields(base_url, path, {"api_key": api_key})


def get_models(base_url: str, path: Path | str | None = None) -> list[str]:
    """读取模型列表；base_url 非法、条目未知或形状错误均返回 []。"""
    entry = _lookup_entry(base_url, path)
    if entry is None:
        return []
    models = entry.get("models")
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, str)]


def get_api_key(base_url: str, path: Path | str | None = None) -> str | None:
    """读取 API Key；base_url 非法、条目未知或值非字符串均返回 None。"""
    entry = _lookup_entry(base_url, path)
    if entry is None:
        return None
    api_key = entry.get("api_key")
    return api_key if isinstance(api_key, str) else None


def get_all_models(path: Path | str | None = None) -> list[str]:
    """返回所有缓存条目的模型并集（首见顺序，去重，宽容畸形条目）。

    ComfyUI 在 /prompt 校验与 /object_info 生成时以无参方式调用 INPUT_TYPES()，
    节点无法得知工作流里 base_url 控件的当前值；若 COMBO 选项按单一 base_url
    精确查询，用户经「获取模型列表」按钮缓存的模型将永远无法通过服务端校验
    （value_not_in_list）。并集保证任何已缓存端点的模型都能通过校验；
    前端 JS 在点击按钮后仍按当前 base_url 精确替换下拉选项，UX 不受影响。
    """
    merged: list[str] = []
    seen: set[str] = set()
    for entry in load_cache(path).values():
        models = entry.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if isinstance(model, str) and model not in seen:
                seen.add(model)
                merged.append(model)
    return merged


def _lookup_entry(base_url: str, path: Path | str | None) -> dict[str, Any] | None:
    """归一化 base_url 后查条目；空/非法 URL 在读取路径上宽容为 None。"""
    try:
        key = normalize_base_url(base_url)
    except OpenAPIConfigError:
        return None
    entry = load_cache(path).get(key)
    return entry if isinstance(entry, dict) else None


def _save_fields(base_url: str, path: Path | str | None, fields: dict[str, Any]) -> None:
    """读-改-写合并入口：更新归一化键对应的条目字段并原子落盘。"""
    key = normalize_base_url(base_url)
    file = _resolve_path(path)
    cache = load_cache(file)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        entry = {}
    entry.update(fields)
    cache[key] = entry
    _atomic_write(file, cache)


def _atomic_write(file: Path, payload: dict[str, Any]) -> None:
    """同目录临时文件 + os.replace 原子写入，父目录自动创建。"""
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, file)
