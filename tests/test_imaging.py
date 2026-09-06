"""cfoapi.imaging 的 RED 阶段测试。

模块 cfoapi.imaging 尚未实现——本文件应在导入处即失败（collection error），
这是 TDD 红灯阶段的预期现象。
"""
from __future__ import annotations

import io

import PIL.Image
import pytest
import torch

from cfoapi.imaging import decode_image_bytes, pil_to_tensor


def _png(w: int, h: int, color, mode: str = "RGB") -> PIL.Image.Image:
    """构造纯色图像并经 PNG 编码回读，返回 PIL.Image.Image。

    注意：PIL 的 size 是 (width, height)，而 ComfyUI IMAGE 张量是
    [N, H, W, C]，断言时不要混淆宽高。
    """
    img = PIL.Image.new(mode, (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return PIL.Image.open(buf)


def _png_bytes(w: int, h: int, color, mode: str = "RGB") -> bytes:
    """返回一张纯色图的 PNG 字节，供 decode_image_bytes 使用。"""
    img = PIL.Image.new(mode, (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_pil_to_tensor_single_image_shape_and_range():
    """单张 2x3（w=2,h=3）红色图 → 张量形状 (1,3,2,3)，float32 且值域 [0,1]。"""
    img = _png(2, 3, (255, 0, 0))
    t = pil_to_tensor([img])
    assert t.shape == (1, 3, 2, 3)
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    assert t.min() >= 0.0
    assert t.max() <= 1.0


def test_pil_to_tensor_red_pixel_exact():
    """纯红图左上角像素归一化后应恰为 [1.0, 0.0, 0.0]。"""
    img = _png(4, 4, (255, 0, 0))
    t = pil_to_tensor([img])
    expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(t[0, 0, 0], expected, atol=1e-6)


def test_pil_to_tensor_rgba_drops_alpha():
    """RGBA 输入应转为 RGB：输出恰好 3 通道。"""
    img = _png(5, 6, (255, 0, 0, 128), mode="RGBA")
    t = pil_to_tensor([img])
    assert t.shape == (1, 6, 5, 3)


def test_pil_to_tensor_grayscale_expands_to_three_channels():
    """mode="L" 灰度图应扩展为 3 通道。"""
    img = _png(5, 4, 128, mode="L")
    t = pil_to_tensor([img])
    assert t.shape == (1, 4, 5, 3)


def test_pil_to_tensor_batch_order_preserved():
    """两张同尺寸图批量：形状 (2,h,w,3) 且顺序保持（红在前、绿在后）。"""
    first = _png(4, 4, (255, 0, 0))
    second = _png(4, 4, (0, 255, 0))
    t = pil_to_tensor([first, second])
    assert t.shape == (2, 4, 4, 3)
    assert torch.allclose(t[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(t[1, 0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_pil_to_tensor_resizes_to_first_image_size():
    """尺寸不一致时，后续图 resize 到第一张的尺寸：64x64 + 32x48 → (2,64,64,3)。"""
    big_red = _png(64, 64, (255, 0, 0))
    small_blue = _png(32, 48, (0, 0, 255))
    t = pil_to_tensor([big_red, small_blue])
    assert t.shape == (2, 64, 64, 3)
    # resize 后纯色应保持：第一条纯红、第二条纯蓝
    assert torch.allclose(t[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(t[1, 0, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_decode_image_bytes_rejects_garbage():
    """非图像字节应抛出异常。"""
    with pytest.raises(Exception):
        decode_image_bytes(b"this is not an image")


def test_decode_image_bytes_returns_pil_image():
    """合法 PNG 字节应解出 PIL.Image.Image 实例。"""
    raw = _png_bytes(3, 2, (0, 128, 255))
    img = decode_image_bytes(raw)
    assert isinstance(img, PIL.Image.Image)
