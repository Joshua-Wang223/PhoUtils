"""16-bit PNG writer (RGB, big-endian samples) — Pillow 不支持 RGB 16bit 原生模式，用 zlib 手写。

仅覆盖本工具所需：非隔行、无 ancillary chunk（可选 eXIf）。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    """PNG chunk：len(4) + type(4) + data + CRC32(4)。"""
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_png16(
    path: str,
    rgb: np.ndarray,
    exif_payload: Optional[bytes] = None,
) -> None:
    """
    写 16-bit RGB PNG。

    Args:
        path: 输出路径
        rgb: H×W×3 数组。float 输入按 [0,1] 线性值缩放并截断到 [0,65535]；
             整数输入直接用（uint16 语义）
        exif_payload: 原始 EXIF 载荷（含 'Exif\\0\\0' 头）→ eXIf chunk
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"需要 H×W×3 数组，得到 {rgb.shape}")
    h, w = rgb.shape[:2]

    if np.issubdtype(rgb.dtype, np.floating):
        # float [0,1] → 16bit（允许 >1 的 HDR 值，截断到 65535）
        u16 = np.clip(rgb[:, :, :3] * 65535.0 + 0.5, 0, 65535).astype(">u2")
    else:
        u16 = np.clip(rgb[:, :, :3], 0, 65535).astype(">u2")

    out = bytearray()
    out += PNG_SIGNATURE

    # IHDR: width, height, bitdepth=16, colortype=2 (RGB), 0, 0, 0
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))

    # eXIf chunk（可选）
    if exif_payload:
        payload = exif_payload
        if not payload.startswith(b"Exif\x00\x00"):
            payload = b"Exif\x00\x00" + payload
        out += _chunk(b"eXIf", payload)

    # IDAT：每行 filter byte(0) + RGB16 大端数据
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter: None
        raw += u16[y].tobytes()
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), 6))

    out += _chunk(b"IEND", b"")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(bytes(out))