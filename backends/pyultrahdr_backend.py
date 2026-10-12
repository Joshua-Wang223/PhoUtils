"""pyultrahdr backend — 纯 Python SDR → Ultra HDR JPEG 转换器。

算法 vendor 自 hanfeisun/pyultrahdr (MIT License, ~230行)：
  - sRGB EOTF → 线性光
  - Inverse TMO（单调提升曲线）：boost(L) = shadow_boost + (peak_boost - shadow_boost)·L^γ
  - BT.709 → BT.2020 色域扩展（线性光）
  - per-pixel log2 增益图（单通道亮度）
  - Ultra HDR 组装：SOI + XMP(hdrgm:) + MPF(APP2) + baseJPEG + gainmapJPEG

本后端扩展（超出原项目）：
  - EXIF 注入（A+C 方案）：SOI + EXIF_APP1 + XMP + MPF + ...
  - 多输出格式：UltraHDR JPEG（原生）/ EXR（half-float 线性）/ PNG16
  - --gain-map-method: auto / linear / log / fixed
  - .exif.json sidecar
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import SDRToHDRBackend, BackendRegistry
from utils.exif import ExifCarrier
from utils.exr_io import write_exr
from utils.png16 import write_png16
from utils.logging import get_logger
from utils.itm import (
    srgb_to_linear, linear_to_srgb, luminance,
    inverse_tone_map, compute_gain_map, to_bt2020,
    sdr_uint8_to_linear, ITM_METHODS,
    BT709_TO_BT2020, BT2020_TO_BT709,
)

logger = get_logger()


# ── XMP / MPF / JPEG 组装（vendor 自 pyultrahdr） ──────────────────────────

def _xmp_hdrgm(gain_min: float, gain_max: float, hdr_capacity: float) -> bytes:
    """构建 hdrgm:1.0 XMP APP1 段（与 Google 参考样本格式一致）。"""
    xml = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.5.0">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"\n'
        '   hdrgm:Version="1.0"\n'
        f'   hdrgm:GainMapMin="{gain_min:.6f}"\n'
        f'   hdrgm:GainMapMax="{gain_max:.6f}"\n'
        f'   hdrgm:HDRCapacityMin="0.000000"\n'
        f'   hdrgm:HDRCapacityMax="{hdr_capacity:.6f}"\n'
        '   hdrgm:OffsetHDR="0.000000"\n'
        '   hdrgm:OffsetSDR="0.000000"/>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    payload = b'http://ns.adobe.com/xap/1.0/\x00' + xml.encode("utf-8")
    return b"\xff\xe1" + _be16(len(payload) + 2) + payload


def _be16(value: int) -> bytes:
    import struct
    return struct.pack(">H", value)


def _mpf_app2(primary_size: int, gainmap_size: int, gainmap_offset: int) -> bytes:
    """构建 90 字节 MPF APP2 段（CIPA DC-007，小端 TIFF IFD）。"""
    import struct
    TIFF_HDR = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    MP_ENTRIES_TIFF_OFF = 50
    e_ver = struct.pack("<HHI4s", 0xB000, 7, 4, b"0100")
    e_num = struct.pack("<HHII", 0xB001, 4, 1, 2)
    e_entry = struct.pack("<HHII", 0xB002, 7, 32, MP_ENTRIES_TIFF_OFF)
    ifd = struct.pack("<H", 3) + e_ver + e_num + e_entry + struct.pack("<I", 0)
    mp1 = struct.pack("<IIIHH", 0x00030000, primary_size, 0, 0, 0)
    mp2 = struct.pack("<IIIHH", 0x00000000, gainmap_size, gainmap_offset, 0, 0)
    data = b"MPF\x00" + TIFF_HDR + ifd + mp1 + mp2
    assert len(data) == 86
    return b"\xff\xe2" + struct.pack(">H", 88) + data


def encode_gain_map(gain_log2: np.ndarray, gain_min: float, gain_max: float,
                    quality: int = 85) -> bytes:
    """量化 log2 增益为 8-bit 灰度 JPEG。"""
    gain_range = (gain_max - gain_min) or 1.0
    gain_u8 = np.clip((gain_log2 - gain_min) / gain_range * 255 + 0.5, 0, 255).astype(np.uint8)
    if gain_u8.ndim == 3:  # (H,W,1) → (H,W)
        gain_u8 = gain_u8[:, :, 0]
    buf = io.BytesIO()
    Image.fromarray(gain_u8, "L").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_ultra_hdr(
    sdr_uint8: np.ndarray,
    gain_log2: np.ndarray,
    gain_min: float,
    gain_max: float,
    peak_nits: float,
    sdr_white: float,
    base_quality: int = 92,
    gainmap_quality: int = 85,
    exif_app1: Optional[bytes] = None,
) -> bytes:
    """
    组装 Ultra HDR JPEG。

    布局：SOI + [EXIF_APP1] + XMP(hdrgm:) + MPF(APP2) + baseJPEG + gainmapJPEG
    EXIF 段插在 XMP 之前（EXIF 传统位置，兼容性更好）。
    """
    hdr_capacity = math.log2(peak_nits / sdr_white)
    xmp_app1 = _xmp_hdrgm(gain_min, gain_max, hdr_capacity)

    buf = io.BytesIO()
    Image.fromarray(sdr_uint8, "RGB").save(
        buf, format="JPEG", quality=base_quality, subsampling=0)
    base_jpeg = buf.getvalue()

    gm_raw = encode_gain_map(gain_log2, gain_min, gain_max, gainmap_quality)
    gainmap_jpeg = gm_raw[:2] + xmp_app1 + gm_raw[2:]  # 增益图 JPEG 内也注入同一 XMP

    A = len(xmp_app1)
    B = len(base_jpeg)
    G = len(gainmap_jpeg)
    E = len(exif_app1) if exif_app1 else 0
    primary_size = 2 + E + A + 90 + (B - 2)
    # 增益图 SOI 文件偏移（含 EXIF 段后的布局）
    gainmap_file_offset = 2 + E + A + 90 + (B - 2)
    # MP Entry 偏移从 TIFF 头起算：TIFF 头文件偏移 = 2 + E + A + 4(FFE2+len) + 4(MPF\0)
    tiff_header_file_offset = 2 + E + A + 4 + 4
    gainmap_offset = gainmap_file_offset - tiff_header_file_offset
    mpf_app2 = _mpf_app2(primary_size, G, gainmap_offset)

    parts = [b"\xff\xd8"]
    if exif_app1:
        parts.append(exif_app1)
    parts += [xmp_app1, mpf_app2, base_jpeg[2:], gainmap_jpeg]
    return b"".join(parts)


# ── 后端实现 ───────────────────────────────────────────────────────────────

@BackendRegistry.register
class PyUltraHDRBackend(SDRToHDRBackend):
    name = "pyultrahdr"
    description = "纯 Python Ultra HDR 编码器（vendor 自 hanfeisun/pyultrahdr，MIT）"
    required_deps: List[str] = []  # 仅 numpy + Pillow（核心依赖）
    supports_batch = True
    # 注意：jpg 输出本身就是 UltraHDR（SOI+EXIF+XMP+MPF+base+gainmap）
    output_formats = ["jpg", "ultrahdr", "exr", "png"]

    def check_available(self) -> Tuple[bool, str]:
        # 纯算法后端：只要 numpy/Pillow 可用即可
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
            return True, "numpy + Pillow 可用"
        except ImportError as e:
            return False, f"缺少依赖: {e}"

    def prepare(self, device: str, **kwargs) -> None:
        # 算法后端无模型加载；记录设备（API 一致性）
        self._device = device
        self._prepared = True
        logger.info(f"[{self.name}] prepared (algorithmic backend, device={device})")

    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "jpg",
        colorspace: str = "linear-rec709",
        keep_exif: bool = True,
        sidecar: bool = True,
        target_nits: int = 1000,
        sdr_white: float = 203.0,
        gain_map_method: str = "auto",
        base_quality: int = 92,
        gainmap_quality: int = 85,
        **_ignored: Any,
    ) -> bool:
        """
        单张转换。

        output_format: jpg/ultrahdr（UltraHDR JPEG）/ exr（half-float 线性）/ png（16bit 线性）
        """
        try:
            # 1. 读源图（RGB uint8）
            img = Image.open(input_path)
            img.load()
            sdr_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
            img.close()

            sdr_lin = srgb_to_linear(sdr_uint8.astype(np.float32) / 255.0)

            # 2. EXIF 提取（A+C）
            carrier = ExifCarrier(input_path) if keep_exif else ExifCarrier.__new__(ExifCarrier)
            if not keep_exif:
                carrier.input_path = input_path
                carrier.payload = None
                carrier.fields = {}
                carrier.app1 = None

            # 3. Inverse TMO（BT.709 线性，nits）
            hdr_bt709_nits = inverse_tone_map(
                sdr_lin, sdr_white, float(target_nits), method=gain_map_method)

            # 4. BT.709 → BT.2020（供增益图计算）
            hdr_bt2020 = np.clip(
                hdr_bt709_nits @ BT709_TO_BT2020.T, 0.0, float(target_nits))

            # 5. 增益图
            gain_log2, gain_min, gain_max = compute_gain_map(
                sdr_lin, hdr_bt2020, sdr_white)

            # 6. 按目标格式写盘
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = output_format.lower().lstrip(".")

            if fmt in ("jpg", "jpeg", "ultrahdr"):
                ultra = build_ultra_hdr(
                    sdr_uint8, gain_log2, gain_min, gain_max,
                    float(target_nits), sdr_white,
                    base_quality, gainmap_quality,
                    exif_app1=carrier.app1 if keep_exif else None,
                )
                out_path.write_bytes(ultra)
            elif fmt == "exr":
                attrs = {
                    "software": "photo_sdr_to_hdr / pyultrahdr",
                    "peakNits": str(target_nits),
                    "gainMapMethod": gain_map_method,
                    "colorspace": "linear Rec.709 (nits)",
                }
                if keep_exif:
                    attrs.update(carrier.exr_attributes())
                write_exr(str(out_path), hdr_bt709_nits.astype(np.float32), attrs)
            elif fmt == "png":
                # PNG16 语义：[0,1] 归一化（峰值=1.0），16bit 线性
                normalized = hdr_bt709_nits / float(target_nits)
                write_png16(
                    str(out_path),
                    normalized.astype(np.float32),
                    exif_payload=carrier.payload if keep_exif else None,
                )
            else:
                raise ValueError(
                    f"pyultrahdr 不支持的输出格式: {output_format}"
                    f"（支持: {', '.join(self.output_formats)}）")

            # 7. Sidecar（C：旁车兜底）
            if sidecar and keep_exif:
                carrier.write_sidecar(str(out_path))

            return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False