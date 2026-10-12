"""Pure-Python EXR I/O (uncompressed, half-float RGB).

不依赖 OpenEXR/OpenCV——保持项目"Pillow + numpy"的薄依赖风格。
仅支持：scanline、NO_COMPRESSION、HALF 像素、RGB 通道（足够线性 HDR 图像交换用）。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

MAGIC = 20000630  # 0x76 0x2F 0x31 0x01
VERSION = 2       # scanline, single-part, no tiling


def _attr(name: str, atype: str, data: bytes) -> bytes:
    """序列化单个 header 属性。"""
    return name.encode("ascii") + b"\x00" + atype.encode("ascii") + b"\x00" \
        + struct.pack("<i", len(data)) + data


def _chlist(channel_names: Tuple[str, ...]) -> bytes:
    """chlist：每通道 name\\0 + ptype(1=HALF) + pLinear(0) + pad(3) + sampling(1,1)。"""
    out = b""
    for ch_name in channel_names:
        out += ch_name.encode("ascii") + b"\x00"
        out += struct.pack("<i", 1)      # HALF
        out += struct.pack("<B", 0)      # pLinear
        out += b"\x00\x00\x00"           # reserved
        out += struct.pack("<ii", 1, 1)  # xSampling, ySampling
    return out + b"\x00"                 # 列表结束


def _header(width: int, height: int, string_attrs: Dict[str, str]) -> bytes:
    channels = _chlist(("B", "G", "R"))  # EXR 要求通道按字母序
    header = b""
    header += _attr("channels", "chlist", channels)
    header += _attr("compression", "compression", struct.pack("<B", 0))  # NO_COMPRESSION
    header += _attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += _attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += _attr("lineOrder", "lineOrder", struct.pack("<B", 0))      # INCREASING_Y
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    # 自定义字符串属性（EXIF 字段等）
    for key, value in string_attrs.items():
        encoded = value.encode("utf-8")
        header += _attr(key, "string", encoded + b"\x00")
    header += b"\x00"  # header 结束
    return header


def write_exr(path: str, rgb: np.ndarray, string_attrs: Optional[Dict[str, str]] = None) -> None:
    """
    写 half-float RGB EXR（线性值，无压缩）。

    Args:
        path: 输出路径
        rgb: H×W×3 float32/float64 数组（R,G,B 顺序，线性值）
        string_attrs: EXR header 字符串属性（如 EXIF 字段）
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"需要 H×W×3 数组，得到 {rgb.shape}")
    h, w = rgb.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("图像尺寸无效")

    attrs = string_attrs or {}
    # 单属性大小上限检查（EXR 属性值 < 1GB，这里 < 64KB 足够安全）
    for k, v in attrs.items():
        if len(v.encode("utf-8")) + 1 > 0xFFFF:
            attrs[k] = v[:32000]  # 超长截断，避免文件损坏

    out = bytearray()
    out += struct.pack("<I", MAGIC)
    out += struct.pack("<I", VERSION)
    out += _header(w, h, attrs)

    # 每行一个 chunk（NO_COMPRESSION 时 linesInBuffer=1）
    n_chunks = h
    offset_table_pos = len(out)
    out += b"\x00" * (8 * n_chunks)  # 占位，稍后回填

    # 通道数据：EXR 存 B,G,R 顺序，且 scanline 内按通道连续排列（先全部 B，再 G，再 R）
    half = np.empty((h, w, 3), dtype="<f2")
    half[:, :, 0] = rgb[:, :, 2]  # B
    half[:, :, 1] = rgb[:, :, 1]  # G
    half[:, :, 2] = rgb[:, :, 0]  # R

    offsets = []
    for y in range(n_chunks):
        offsets.append(len(out))
        line = np.ascontiguousarray(half[y].T).tobytes()  # (3, W) → B整行,G整行,R整行
        out += struct.pack("<ii", y, len(line))
        out += line

    # 回填 offset table
    for i, off in enumerate(offsets):
        struct.pack_into("<Q", out, offset_table_pos + 8 * i, off)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(bytes(out))


def read_exr(path: str) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    读 half-float RGB EXR（仅支持 NO_COMPRESSION scanline）。

    Returns:
        (rgb float32 H×W×3 [R,G,B], header 字符串属性)
    """
    data = Path(path).read_bytes()
    pos = 0

    magic, version = struct.unpack_from("<II", data, pos)
    pos += 8
    if magic != MAGIC:
        raise ValueError(f"不是有效的 EXR 文件（magic={magic:#x}）")
    if version != VERSION:
        raise ValueError(f"不支持的 EXR version={version}（仅支持无压缩 scanline）")

    # 解析 header
    attrs: Dict[str, str] = {}
    width = height = None
    channels = []
    while True:
        name_end = data.index(b"\x00", pos)
        name = data[pos:name_end].decode("ascii")
        pos = name_end + 1
        if not name:
            break  # header 结束
        type_end = data.index(b"\x00", pos)
        atype = data[pos:type_end].decode("ascii")
        pos = type_end + 1
        size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        value = data[pos:pos + size]
        pos += size

        if name == "channels" and atype == "chlist":
            channels = _parse_chlist(value)
        elif name == "dataWindow" and atype == "box2i":
            xmin, ymin, xmax, ymax = struct.unpack("<iiii", value)
            width, height = xmax - xmin + 1, ymax - ymin + 1
        elif atype == "string":
            attrs[name] = value.rstrip(b"\x00").decode("utf-8", errors="replace")

    if width is None or height is None:
        raise ValueError("EXR header 缺少 dataWindow")

    # offset table
    offset_table = np.frombuffer(data, dtype="<u8", count=height, offset=pos)
    pos += 8 * height

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        chunk_off = int(offset_table[y])
        _y, data_size = struct.unpack_from("<ii", data, chunk_off)
        pixel_bytes = data[chunk_off + 8:chunk_off + 8 + data_size]
        # 通道顺序按 header 中的 chlist 排列（本例 B,G,R）
        bytes_per = 2
        for ci, ch_name in enumerate(channels):
            start = ci * width * bytes_per
            end = (ci + 1) * width * bytes_per
            arr = np.frombuffer(pixel_bytes[start:end], dtype="<f2").astype(np.float32)
            if ch_name == "R":
                rgb[y, :, 0] = arr
            elif ch_name == "G":
                rgb[y, :, 1] = arr
            elif ch_name == "B":
                rgb[y, :, 2] = arr
    return rgb, attrs


def _parse_chlist(value: bytes) -> list:
    """解析 chlist 属性为通道名列表。"""
    names = []
    pos = 0
    while pos < len(value):
        end = value.index(b"\x00", pos)
        name = value[pos:end].decode("ascii")
        pos = end + 1
        if not name:
            break
        # ptype(4) + pLinear(1) + reserved(3) + xSampling(4) + ySampling(4) = 16
        pos += 16
        names.append(name)
    return names