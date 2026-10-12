"""hdrconv backend — hdr-conversion 库封装（ISO 21496-1 / UltraHDR 规范级封装）。

与 pyultrahdr 后端的分工：
- pyultrahdr：手写 UltraHDR 组装（SOI+XMP+MPF+base+gainmap），零第三方依赖
- hdrconv：调用维护中的 hdr-conversion 库（pip install hdr-conversion），
  由其完成规范级增益图元数据编码（hdr_to_gainmap + write_ultrahdr/write_21496）

ITM（逆色调映射）仍用共享 utils.itm 的曲线——hdr-conversion 是格式库而非重建模型，
"HDR 内容从哪来"由我们决定；它负责"如何规范地封装"。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import SDRToHDRBackend, BackendRegistry
from utils.exif import ExifCarrier
from utils.exr_io import write_exr
from utils.png16 import write_png16
from utils.logging import get_logger
from utils.itm import sdr_uint8_to_linear, inverse_tone_map, compute_gain_map, to_bt2020

logger = get_logger()


@BackendRegistry.register
class HdrConvBackend(SDRToHDRBackend):
    name = "hdrconv"
    description = "hdr-conversion 库：ISO 21496-1 / UltraHDR 规范封装（pip install hdr-conversion）"
    required_deps: List[str] = ["hdrconv"]
    supports_batch = True
    # ultrahdr = Google UltraHDR 容器；gainmap = ISO 21496-1 容器（container kwarg 选择）
    output_formats = ["jpg", "ultrahdr", "gainmap", "exr", "png"]

    # container 选择：ultrahdr（Google，设备兼容性广）/ iso21496（ISO 标准）
    CONTAINERS = ("ultrahdr", "iso21496")

    def check_available(self) -> Tuple[bool, str]:
        try:
            import hdrconv  # noqa: F401
            return True, "hdr-conversion 已安装"
        except ImportError:
            return False, "未安装：pip install hdr-conversion"

    def prepare(self, device: str, **kwargs) -> None:
        self._device = device
        self._prepared = True
        logger.info(f"[{self.name}] prepared (library backend, device={device})")

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
        container: str = "ultrahdr",
        baseline_quality: int = 95,
        gainmap_quality: int = 95,
        **_ignored: Any,
    ) -> bool:
        """
        单张转换。

        container: ultrahdr（Google UltraHDR）/ iso21496（ISO 21496-1 Gainmap JPEG）
        output_format: jpg/ultrahdr/gainmap（走增益图容器）/ exr / png
        """
        try:
            # 1. 读源图
            img = Image.open(input_path)
            img.load()
            sdr_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
            img.close()

            # 2. EXIF 提取（A+C）
            carrier = ExifCarrier(input_path) if keep_exif else None

            # 3. ITM → HDR 线性 nits（BT.709）
            sdr_lin = sdr_uint8_to_linear(sdr_uint8)
            hdr_bt709_nits = inverse_tone_map(
                sdr_lin, sdr_white, float(target_nits), method=gain_map_method)
            hdr_bt2020 = to_bt2020(hdr_bt709_nits, float(target_nits))

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = output_format.lower().lstrip(".")

            if fmt in ("jpg", "jpeg", "ultrahdr", "gainmap"):
                self._write_gainmap_container(
                    sdr_uint8, sdr_lin, hdr_bt709_nits, hdr_bt2020,
                    out_path, container, float(target_nits), sdr_white,
                    baseline_quality, gainmap_quality,
                    carrier=carrier if keep_exif else None,
                )
            elif fmt == "exr":
                attrs = {
                    "software": f"photo_sdr_to_hdr / {self.name}",
                    "peakNits": str(target_nits),
                    "gainMapMethod": gain_map_method,
                    "colorspace": "linear Rec.709 (nits)",
                    "container": container,
                }
                if keep_exif and carrier and carrier.fields:
                    attrs.update(carrier.exr_attributes())
                write_exr(str(out_path), hdr_bt709_nits.astype(np.float32), attrs)
            elif fmt == "png":
                normalized = hdr_bt709_nits / float(target_nits)
                write_png16(
                    str(out_path), normalized.astype(np.float32),
                    exif_payload=(carrier.payload if keep_exif and carrier else None),
                )
            else:
                raise ValueError(
                    f"hdrconv 不支持的输出格式: {output_format}"
                    f"（支持: {', '.join(self.output_formats)} + ultrahdr/gainmap 容器）")

            # Sidecar（C：旁车兜底）
            if sidecar and keep_exif and carrier:
                carrier.write_sidecar(str(out_path))

            return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False

    def _write_gainmap_container(
        self,
        sdr_uint8: np.ndarray,
        sdr_lin: np.ndarray,
        hdr_bt709_nits: np.ndarray,
        hdr_bt2020: np.ndarray,
        out_path: Path,
        container: str,
        peak_nits: float,
        sdr_white: float,
        baseline_quality: int,
        gainmap_quality: int,
        carrier: Optional[ExifCarrier],
    ) -> None:
        """用 hdrconv 库写 UltraHDR / ISO21496 增益图容器，再注入 EXIF。"""
        from hdrconv.convert import hdr_to_gainmap
        from hdrconv.io import write_ultrahdr, write_21496, insert_segments

        # hdr_to_gainmap 期望：hdr 数据 + baseline（线性 [0,1] float32）
        # 约定 SDR 白 = 1.0（与 UltraHDR HDRCapacityMax 惯例一致）
        hdr_normalized = (hdr_bt709_nits / sdr_white).astype(np.float32)
        hdr_image = {
            "data": hdr_normalized,
            "transfer_function": "linear",
            "color_space": "srgb",
            "icc_profile": None,
        }
        gainmap_image = hdr_to_gainmap(
            hdr_image, baseline=sdr_lin.astype(np.float32), gamma=1.0)

        # 写容器（先写临时文件，再读回字节注入 EXIF）
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if container == "iso21496":
                write_21496(gainmap_image, str(tmp_path),
                            baseline_quality=baseline_quality,
                            gainmap_quality=gainmap_quality)
            else:
                write_ultrahdr(gainmap_image, str(tmp_path),
                               baseline_quality=baseline_quality,
                               gainmap_quality=gainmap_quality)

            data = tmp_path.read_bytes()
            # EXIF 注入（A 方案主通道）
            if carrier is not None and carrier.app1:
                data = insert_segments(data, [carrier.app1])
            out_path.write_bytes(data)
        finally:
            tmp_path.unlink(missing_ok=True)