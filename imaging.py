"""图像解码与 ComfyUI IMAGE 张量转换。

输入输出均为纯 PIL / numpy / torch 对象，不涉及任何 HTTP 或 ComfyUI 模块。
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
import PIL.Image
import torch


def decode_image_bytes(raw: bytes) -> PIL.Image.Image:
    """校验并解码图像字节为可用的 PIL 图像。

    先 open + verify() 做格式校验（verify 会使对象失效），再重新打开并
    load() 返回可继续处理的图像；非图像字节让 PIL 异常直接上抛。
    """
    probe = PIL.Image.open(BytesIO(raw))
    probe.verify()
    image = PIL.Image.open(BytesIO(raw))
    image.load()
    return image


def pil_to_tensor(images: list[PIL.Image.Image]) -> torch.Tensor:
    """把 PIL 图像列表转为 ComfyUI IMAGE 张量 [N, H, W, C] float32 CPU。

    全部统一为 RGB；尺寸以第一张为基准，后续不一致的双线性 resize 对齐。
    """
    if not images:
        raise ValueError("pil_to_tensor requires at least one image")

    reference_size: tuple[int, int] | None = None
    frames: list[torch.Tensor] = []
    for image in images:
        rgb = image.convert("RGB")
        if reference_size is None:
            reference_size = rgb.size
        elif rgb.size != reference_size:
            rgb = rgb.resize(reference_size, PIL.Image.Resampling.BILINEAR)
        arr = np.array(rgb, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr)[None,])
    return torch.cat(frames, dim=0)
