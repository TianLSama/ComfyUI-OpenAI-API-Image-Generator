"""OpenAI 图像接口兼容的本地 mock 服务器。

仅依赖标准库 http.server + PIL（不引入 aiohttp/flask），供 pytest 与
qa/run_mock_server.py（实时 QA，固定端口 18099）共同复用。
所有路由以「后缀匹配」实现，允许挂载在任意 base path 前缀之下。
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

# 与真实 OpenAI/兼容网关对齐的 mock 模型清单（顺序即 /v1/models 返回顺序）
MODEL_IDS = [
    "mock-b64",
    "mock-url",
    "mock-multi",
    "mock-siliconflow",
    "mock-401",
    "mock-error-code",
    "mock-empty",
]


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """在内存中生成纯色 PNG，作为固定图像夹具。"""
    buf = BytesIO()
    Image.new("RGB", (width, height), rgb).save(buf, format="PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    """bytes → ASCII base64 字符串（OpenAI b64_json 字段格式）。"""
    return base64.b64encode(data).decode("ascii")


# DashScope 原生多模态生成端点后缀（与 /v1/images/generations 后缀互斥，无碰撞）
_NATIVE_SUFFIX = "/api/v1/services/aigc/multimodal-generation/generation"


class MockOpenAIServer:
    """可嵌入 pytest / QA 的 mock OpenAI 图像 API 服务器。

    用法::

        srv = MockOpenAIServer()          # port=0 → 临时端口
        srv.start()
        ...  # srv.base_url / srv.port
        srv.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port  # start() 之后（port=0 时）会被替换为实际绑定端口
        self._listen_port = port
        self._httpd: _MockServer | None = None
        self._thread: threading.Thread | None = None
        # 启动时一次性生成 PNG 夹具：64x64 纯红、32x48 纯蓝
        self.png_red = _make_png(64, 64, (255, 0, 0))
        self.png_blue = _make_png(32, 48, (0, 0, 255))
        # 线程安全状态：最近一次 generations 请求体、最近一次重定向是否携带鉴权头
        self._lock = threading.Lock()
        self._last_request: dict[str, Any] | None = None
        self._last_native_request: dict[str, Any] | None = None
        self.redirect_had_auth = False

    # ---------- 生命周期 ----------

    def start(self) -> None:
        if self._httpd is not None:
            raise RuntimeError("MockOpenAIServer already started")
        httpd = _MockServer((self.host, self._listen_port), _Handler)
        httpd.mock = self  # 注入回指，供 handler 使用
        self._httpd = httpd
        if self._listen_port == 0:
            # 临时端口：取内核实际分配的端口
            self.port = httpd.server_address[1]
        self._thread = threading.Thread(
            target=httpd.serve_forever, name="mock-openai-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ---------- 供 _Handler 调用的共享状态 ----------

    @property
    def last_request(self) -> dict[str, Any] | None:
        with self._lock:
            return self._last_request  # 无请求时返回 None（/debug/last_request → null）

    def record_generation_request(self, body: dict[str, Any]) -> None:
        with self._lock:
            self._last_request = body

    @property
    def last_native_request(self) -> dict[str, Any] | None:
        with self._lock:
            return self._last_native_request  # 无请求时 None（/debug/last_native_request → null）

    def record_native_request(self, body: dict[str, Any]) -> None:
        with self._lock:
            self._last_native_request = body

    def record_redirect_auth(self, auth_header: str | None) -> None:
        # 「未收到请求」与「收到但无鉴权头」都记为 False（null→false 语义）
        self.redirect_had_auth = auth_header is not None

    def generation_response(self, model: str) -> tuple[int, Any]:
        """按 model 字段返回 (status_code, json_payload)。"""
        match model:
            case "mock-b64":
                return 200, {"created": 1, "data": [{"b64_json": _b64(self.png_red)}]}
            case "mock-url":
                return 200, {
                    "created": 1,
                    "data": [{"url": f"{self.base_url}/static/img.png"}],
                }
            case "mock-multi":
                return 200, {
                    "created": 1,
                    "data": [
                        {"b64_json": _b64(self.png_red)},
                        {"b64_json": _b64(self.png_blue)},
                    ],
                }
            case "mock-siliconflow":
                # SiliconFlow 风格：顶层 images[] + timings + seed
                return 200, {
                    "images": [{"url": f"{self.base_url}/static/img.png"}],
                    "timings": {"inference": 0.1},
                    "seed": 42,
                }
            case "mock-401":
                return 401, {"error": {"message": "Invalid API key provided."}}
            case "mock-error-code":
                # 网关风格：顶层 code/message（无 error 包装）
                return 400, {
                    "code": "insufficient_credit",
                    "message": "Account has no credit",
                }
            case "mock-empty":
                return 200, {"created": 1, "data": []}
            case _:
                return 400, {"error": {"message": "Unknown model"}}

    def native_response(self, has_auth: bool, body: dict[str, Any]) -> tuple[int, Any]:
        """按 DashScope 原生 multimodal-generation 协议返回 (status_code, json_payload)。"""
        if not has_auth:
            return 401, {"code": "InvalidApiKey", "message": "No API-key provided.", "request_id": "mock-native"}
        match str(body.get("model")):
            case "qwen-image-3.0-pro":
                # n=2 → 两张（img.png 64x64 / img2.png 32x48），否则一张
                files = ["img.png", "img2.png"] if body.get("parameters", {}).get("n") == 2 else ["img.png"]
                return 200, {
                    "output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                                 "content": [{"image": f"{self.base_url}/static/{f}"}]}} for f in files]},
                    "usage": {"image_count": len(files), "width": 64, "height": 64}, "request_id": "mock-native"}
            case "qwen-image-404" | "qwen-image-plus":
                # qwen-image-plus 也在目录内且被 QA S7 用作 404 场景模型:
                # 错误传播工作流的 model 值必须能通过 ComfyUI 的 combo 校验, 必须是目录成员
                return 404, {"code": "ModelNotFound", "message": "Model not exist.", "request_id": "mock-native"}
            case "qwen-image-200err":
                # HTTP 200 携带顶层 code/message（无 output）：状态码成功但实为错误
                return 200, {"request_id": "mock-native", "code": "InvalidParameter", "message": "n must be 1"}
            case _:
                return 400, {"code": "InvalidParameter", "message": "Unsupported model", "request_id": "mock-native"}


class _MockServer(ThreadingHTTPServer):
    """持有 MockOpenAIServer 回指引用的 HTTP 服务，供 handler 经 self.server.mock 访问。"""

    daemon_threads = True  # 工作线程随进程退出，避免阻塞会话 teardown
    allow_reuse_address = True
    mock: MockOpenAIServer  # 由 start() 注入（前向引用，运行时不求值）


class _Handler(BaseHTTPRequestHandler):
    """把请求路由到 MockOpenAIServer 实例（经 self.server.mock 访问共享状态）。"""

    protocol_version = "HTTP/1.1"  # 所有响应显式带 Content-Length，保证 keep-alive 正确
    server: _MockServer

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # 静默访问日志，避免污染 pytest 输出

    # ---------- 响应辅助 ----------

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # 客户端中途断开：丢弃写入错误即可，绝不允许拖垮服务线程
            pass

    def _send_png(self, png: bytes) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ---------- GET 路由 ----------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path.endswith("/v1/models"):
            self._handle_models()
        elif path.endswith("/static/img.png"):
            self._send_png(self.server.mock.png_red)
        elif path.endswith("/static/img2.png"):
            self._send_png(self.server.mock.png_blue)
        elif path.endswith("/static/big.png"):
            self._handle_big_png()
        elif path.endswith("/debug/last_request"):
            self._send_json(200, self.server.mock.last_request)
        elif path.endswith("/debug/last_native_request"):
            self._send_json(200, self.server.mock.last_native_request)
        elif path.endswith("/debug/redirect_seen"):
            self._send_json(200, self.server.mock.redirect_had_auth)
        elif path.endswith("/redir-crosshost"):
            self._handle_redir_crosshost()
        elif path.endswith("/redir-target"):
            self._handle_redir_target()
        elif path.endswith("/slow"):
            self._handle_slow(parsed.query)
        else:
            self._send_json(404, {"error": {"message": f"Unknown path {path}"}})

    def _handle_models(self) -> None:
        # 仅检查 Authorization 头是否存在（不校验具体值）
        if self.headers.get("Authorization") is None:
            self._send_json(401, {"error": {"message": "Missing API key"}})
            return
        data = [
            {"id": m, "object": "model", "created": 0, "owned_by": "mock"}
            for m in MODEL_IDS
        ]
        self._send_json(200, {"object": "list", "data": data})

    def _handle_big_png(self) -> None:
        # 只声明超大 Content-Length 就立即断开：客户端必须凭响应头拒绝
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "99999999")
            self.end_headers()
            self.close_connection = True  # 发送头之后不再发任何 body
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_redir_crosshost(self) -> None:
        # 刻意换成 localhost（与绑定的 127.0.0.1 是不同 host 字符串），
        # 用于测试客户端是否会跨主机泄漏 Authorization 头。
        try:
            self.send_response(302)
            self.send_header(
                "Location", f"http://localhost:{self.server.mock.port}/redir-target"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_redir_target(self) -> None:
        self.server.mock.record_redirect_auth(self.headers.get("Authorization"))
        self._send_png(self.server.mock.png_red)

    def _handle_slow(self, query: str) -> None:
        delay_raw = parse_qs(query).get("delay", ["2"])[0]
        try:
            delay = float(delay_raw)
        except ValueError:
            self._send_json(400, {"error": {"message": "bad delay"}})
            return
        time.sleep(delay)
        self._send_json(200, {"ok": True})

    # ---------- POST 路由 ----------

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.endswith(_NATIVE_SUFFIX):  # 原生分支先匹配（后缀互斥，无碰撞）
            self._handle_native()
            return
        if not path.endswith("/v1/images/generations"):
            self._send_json(404, {"error": {"message": f"Unknown path {path}"}})
            return
        body = self._read_json_body()
        if body is None:
            return
        self.server.mock.record_generation_request(body)
        status, payload = self.server.mock.generation_response(str(body.get("model")))
        self._send_json(status, payload)

    def _read_json_body(self) -> dict[str, Any] | None:
        """读取并解析 JSON 对象请求体；失败时已自行发出 400 并返回 None。"""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": {"message": "Invalid JSON body"}})
            return None
        if not isinstance(body, dict):
            self._send_json(400, {"error": {"message": "Body must be a JSON object"}})
            return None
        return body

    def _handle_native(self) -> None:
        """DashScope 原生路由：记录请求体 → 鉴权检查 → 按 model 分支响应。"""
        body = self._read_json_body()
        if body is None:
            return
        self.server.mock.record_native_request(body)
        status, payload = self.server.mock.native_response(self.headers.get("Authorization") is not None, body)
        self._send_json(status, payload)
