"""QA 专用 CLI：以固定端口启动 mock OpenAI 图像 API 服务器。

复用 tests/mock_openai_server.py 中的 MockOpenAIServer 实现（标准库
http.server + PIL），供 qa/smoke.ps1（T14）与实时 QA（T15，对接
127.0.0.1:18188 的真实 ComfyUI）共同使用。

协议：启动成功后向 stdout 打印恰好一行 `MOCK_READY http://<host>:<port>`
（QA 脚本 grep 该行判定就绪），随后阻塞直到 Ctrl+C，finally 中关闭服务器。
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# 插件根目录 = qa/ 的上一级；把 tests/ 插入 sys.path 以复用 mock 实现
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "tests"))

from mock_openai_server import MockOpenAIServer  # noqa: E402  (依赖上一行的 sys.path 注入)


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 QA 用 mock OpenAI 图像 API 服务器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=18099, help="监听端口，默认 18099")
    args = parser.parse_args()

    server = MockOpenAIServer(host=args.host, port=args.port)
    server.start()
    # 就绪信号行：flush 保证 QA 脚本能立即 grep 到
    print(f"MOCK_READY http://{server.host}:{server.port}", flush=True)
    try:
        threading.Event().wait()  # 无限阻塞主线程，直到 KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Ctrl+C 属正常退出路径
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
