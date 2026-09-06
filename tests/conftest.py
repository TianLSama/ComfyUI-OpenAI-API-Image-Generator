"""pytest 全局夹具。

1. 以「合成包」方式注册插件根目录模块（包名 cfoapi），且绝不执行插件的
   __init__.py——它日后会导入仅存在于 ComfyUI 进程内的模块。
   这样 `import cfoapi.urls` 会解析到 <root>/urls.py，模块内部的
   相对导入（from .errors import ...）也能正常工作。
2. 提供 session 级 mock_server 夹具（临时端口，避免与并行任务冲突）。

本文件与 test_mock_fixture.py 都不导入任何 cfoapi.* 模块——
插件源码在各自任务中才会逐步出现，conftest 必须先于它们可导入。
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from mock_openai_server import MockOpenAIServer

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _register_synthetic_package() -> None:
    """把 cfoapi 注册为指向插件根目录的合成包（幂等）。"""
    if "cfoapi" in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec("cfoapi", loader=None, is_package=True)
    # 关键：注入搜索路径，让 cfoapi.* 子模块从插件根目录解析
    spec.submodule_search_locations = [str(PLUGIN_ROOT)]
    module = importlib.util.module_from_spec(spec)
    sys.modules["cfoapi"] = module


_register_synthetic_package()


@pytest.fixture(scope="session")
def mock_server() -> Iterator[MockOpenAIServer]:
    """整个测试会话共享一个 mock 服务器（port=0 取临时端口）。"""
    server = MockOpenAIServer()
    server.start()
    yield server
    server.stop()
