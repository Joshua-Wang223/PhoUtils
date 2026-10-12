"""EXIF utilities: extraction, injection, sidecar, EXR header attributes (Option A+C).

策略：
- JPEG/AVIF/PNG → 原生 EXIF 注入（JPEG 在 SOI 后插 EXIF APP1）
- EXR → 关键字段写 header 属性 + 完整 EXIF JSON 存 exifData 属性
- 所有格式 → .exif.json sidecar（--no-sidecar 可关闭）
"""

from __future__ import annotations

import base64
import io
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import numpy as np
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow 和 numpy：pip install pillow numpy")


# ── EXIF 提取 ────────────────────────────────────────────────────────────────

def extract_exif_bytes(input_path: str) -> Optional[bytes]:
    """
    从源文件提取原始 EXIF APP1 载荷（含 'Exif\\0\\0' 头）。

    对 JPEG：直接扫描 APP1 段取原始字节。
    对其他格式：用 Pillow 的 img.getexif() 序列化为 TIFF 结构。
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    # JPEG 系：直接扫描原始 APP1 段（零损失）
    if suffix in (".jpg", ".jpeg"):
        data = path.read_bytes()
        exif_app1 = _find_jpeg_exif_app1(data)
        if exif_app1 is not None:
            return exif_app1

    # 其他格式：Pillow 重建 TIFF 结构的 EXIF 块
    try:
        img = Image.open(path)
        exif_obj = img.getexif()
        if exif_obj and len(exif_obj) > 0:
            payload = exif_obj.tobytes()
            # Pillow 的 tobytes() 输出含 'Exif\\0\\0' 头（视版本而定），统一处理
            if payload.startswith(b"Exif\x00\x00"):
                return payload
            return b"Exif\x00\x00" + payload
    except Exception:
        pass
    return None


def _find_jpeg_exif_app1(data: bytes) -> Optional[bytes]:
    """扫描 JPEG 段，返回第一个 EXIF APP1 载荷（含 'Exif\\0\\0' 头）。"""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            # 非 marker 字节（异常文件），向后找
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xD9:  # EOI
            break
        if marker == 0xDA:  # SOS，头部结束
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        seg_data = data[pos + 4:pos + 2 + seg_len]
        if marker == 0xE1 and seg_data.startswith(b"Exif\x00\x00"):
            return seg_data
        pos += 2 + seg_len
    return None


# ── EXIF APP1 段构建与注入 ──────────────────────────────────────────────────

def build_exif_app1(exif_payload: Optional[bytes]) -> Optional[bytes]:
    """
    把 EXIF 载荷包装成完整 APP1 段（FF E1 + len + payload）。
    payload 必须含 'Exif\\0\\0' 头。
    """
    if not exif_payload:
        return None
    if not exif_payload.startswith(b"Exif\x00\x00"):
        exif_payload = b"Exif\x00\x00" + exif_payload
    total_len = len(exif_payload) + 2
    if total_len > 0xFFFF:
        return None  # EXIF 段超过 64KB，无法放入单个 APP1
    return b"\xff\xe1" + struct.pack(">H", total_len) + exif_payload


def inject_exif_into_jpeg(jpeg_bytes: bytes, exif_app1: bytes) -> bytes:
    """
    在 JPEG 的 SOI 之后插入 EXIF APP1 段。
    适用于普通 JPEG 与 UltraHDR（SOI + EXIF + XMP + MPF + ... 布局）。
    """
    if not jpeg_bytes.startswith(b"\xff\xd8"):
        raise ValueError("不是有效的 JPEG（缺少 SOI）")
    return jpeg_bytes[:2] + exif_app1 + jpeg_bytes[2:]


# ── EXIF 字段解析（用于 EXR header / sidecar） ───────────────────────────────

# 关键 IFD0/ExifIFD 标签 → 友好字段名
_TAG_NAMES = {
    0x010F: "cameraMake",
    0x0110: "cameraModel",
    0x0112: "orientation",
    0x0132: "modifyDate",
    0x8769: "exifIfdPointer",
    0x9003: "captureTimeOriginal",
    0x9004: "captureTimeDigitized",
    0x829A: "exposureTime",
    0x829D: "fNumber",
    0x8827: "isoSpeed",
    0x920A: "focalLength",
    0xA405: "focalLength35mm",
    0x0131: "software",
    0x011B: "artist",
    0x8298: "copyright",
    0x8825: "gpsIfdPointer",
}


def parse_exif_fields(exif_payload: Optional[bytes]) -> Dict[str, Any]:
    """
    解析 EXIF 关键字段为可 JSON 序列化字典（不追求完整 TIFF 解析，
    只提取 A+C 方案中 EXR header 需要的常见标量字段）。
    """
    fields: Dict[str, Any] = {}
    if not exif_payload:
        return fields

    body = exif_payload[6:] if exif_payload.startswith(b"Exif\x00\x00") else exif_payload
    if len(body) < 8:
        return fields

    try:
        # TIFF 头：字节序 + magic + IFD0 偏移（偏移相对 TIFF 头起点 = body 起点）
        byte_order = body[:2]
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            return fields
        ifd_offset = struct.unpack(endian + "I", body[4:8])[0]

        def read_ifd(ifd_off: int) -> Tuple[List, Optional[int]]:
            """读一个 IFD，返回 ([(tag, value), ...], ExifIFD 指针)。"""
            entries = []
            sub_ifd = None
            if ifd_off + 2 > len(body):
                return entries, sub_ifd
            n = struct.unpack(endian + "H", body[ifd_off:ifd_off + 2])[0]
            for i in range(min(n, 256)):
                e = ifd_off + 2 + i * 12
                if e + 12 > len(body):
                    break
                tag, typ, count = struct.unpack(endian + "HHI", body[e:e + 8])
                value_raw = body[e + 8:e + 12]
                val = _read_ifd_value(body, endian, typ, count, value_raw)
                if tag == 0x8769:  # ExifIFD 指针
                    sub_ifd = val if isinstance(val, int) else None
                else:
                    entries.append((tag, val))
            return entries, sub_ifd

        ifd0_entries, exif_ifd_offset = read_ifd(ifd_offset)
        for tag, val in ifd0_entries:
            name = _TAG_NAMES.get(tag)
            if name and val is not None:
                fields[name] = val

        # ExifIFD（拍摄参数所在）
        if exif_ifd_offset:
            exif_entries, _ = read_ifd(exif_ifd_offset)
            for tag, val in exif_entries:
                name = _TAG_NAMES.get(tag)
                if name and val is not None:
                    fields[name] = val
    except Exception:
        pass
    return fields


def _read_ifd_value(body: bytes, endian: str, typ: int, count: int, value_raw: bytes) -> Any:
    """读取 IFD 条目的值（>4 字节时按 TIFF 头起算的偏移）。"""
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
    size = type_sizes.get(typ, 1) * count
    if size <= 4:
        data = value_raw[:size]
    else:
        off = struct.unpack(endian + "I", value_raw)[0]
        data = body[off:off + size] if off + size <= len(body) else b""

    try:
        if typ == 2:  # ASCII
            return data.rstrip(b"\x00").decode("utf-8", errors="replace")
        if typ == 3 and count == 1:  # SHORT
            return struct.unpack(endian + "H", data[:2])[0]
        if typ == 3 and count == 2:  # SHORT 对 = 简化的 rational（Pillow tobytes() 如此编码）
            num, den = struct.unpack(endian + "HH", data[:4])
            return round(num / den, 6) if den else 0.0
        if typ == 4 and count == 1:  # LONG
            return struct.unpack(endian + "I", data[:4])[0]
        if typ == 5 and count == 1:  # RATIONAL
            num, den = struct.unpack(endian + "II", data[:8])
            return round(num / den, 6) if den else 0.0
        if typ == 10 and count == 1:  # SRATIONAL
            num, den = struct.unpack(endian + "ii", data[:8])
            return round(num / den, 6) if den else 0.0
        if typ == 9 and count == 1:  # SLONG
            return struct.unpack(endian + "i", data[:4])[0]
    except Exception:
        return None
    return None


# ── EXR header 属性 ─────────────────────────────────────────────────────────

def exr_header_attributes(exif_fields: Dict[str, Any]) -> Dict[str, str]:
    """
    把 EXIF 关键字段转为 EXR header 字符串属性。
    键名遵循 OpenEXR 惯例（cameraMake / isoSpeed 等）。
    """
    mapping = {
        "cameraMake": "cameraMake",
        "cameraModel": "cameraModel",
        "isoSpeed": "isoSpeed",
        "exposureTime": "exposureTime",
        "fNumber": "fNumber",
        "focalLength": "focalLength",
        "focalLength35mm": "focalLengthIn35mm",
        "software": "software",
        "artist": "artist",
        "copyright": "copyright",
        "orientation": "orientation",
        "captureTimeOriginal": "captureTime",
    }
    attrs = {}
    for src, dst in mapping.items():
        if src in exif_fields and exif_fields[src] not in (None, ""):
            attrs[dst] = str(exif_fields[src])
    return attrs


# ── Sidecar（.exif.json） ───────────────────────────────────────────────────

def write_exif_sidecar(output_path: str, exif_payload: Optional[bytes],
                       fields: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    写 .exif.json sidecar：完整 EXIF 字段 + base64 原始块。
    返回 sidecar 路径；无 EXIF 时返回 None。
    """
    if not exif_payload and not fields:
        return None
    import base64
    out = Path(output_path)
    sidecar = out.with_name(out.stem + ".exif.json")
    data = {
        "sourceFormat": "exif",
        "fields": fields if fields is not None else parse_exif_fields(exif_payload),
        "exifBase64": base64.b64encode(exif_payload).decode("ascii") if exif_payload else None,
    }
    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(sidecar)


# ── 组合入口 ────────────────────────────────────────────────────────────────

class ExifCarrier:
    """一次提取，多格式落地（A+C）。"""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.payload: Optional[bytes] = extract_exif_bytes(input_path)
        self.fields: Dict[str, Any] = parse_exif_fields(self.payload) if self.payload else {}
        self.app1: Optional[bytes] = build_exif_app1(self.payload)

    @property
    def has_exif(self) -> bool:
        return self.payload is not None

    def apply_jpeg(self, jpeg_bytes: bytes) -> bytes:
        """JPEG/UltraHDR：SOI 后注入 EXIF APP1。"""
        if self.app1:
            return inject_exif_into_jpeg(jpeg_bytes, self.app1)
        return jpeg_bytes

    def write_sidecar(self, output_path: str) -> Optional[str]:
        """旁车 JSON 兜底。"""
        return write_exif_sidecar(output_path, self.payload, self.fields)

    def exr_attributes(self) -> Dict[str, str]:
        """EXR header 属性。"""
        return exr_header_attributes(self.fields)