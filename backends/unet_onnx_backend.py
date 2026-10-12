"""unet-onnx backend — SDR-to-HDR U-Net (ONNX Runtime)。

vendor 自 zhuiber/SDR-to-HDR-Reconstruction-System：
  - 模型：color_unet/Model_onnx/unet4k.onnx（7.4MB，首次运行自动下载）
  - 输入：SDR 图像 resize 到 2160×3840，归一化 [0,1]，NCHW
  - 输出：同尺寸 float32 HDR（线性域，[0,1] 归一化到峰值）
  - 后处理（参考原项目）：色彩变换 + 线性光叠加

硬件要求（重要）：模型固定 4K 输入，ORT 激活内存数 GB——
  - CPU 推理需 ≥5GB 可用内存（8GB 机器实测 OOM，已加预检）
  - 推荐 CUDA：pip install onnxruntime-gpu（CPU 版自动回退）
  - 单张 4K 推理 CPU 约数分钟，CUDA 约秒级
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import SDRToHDRBackend, BackendRegistry
from utils.exif import ExifCarrier
from utils.exr_io import write_exr
from utils.png16 import write_png16
from utils.logging import get_logger
from utils.itm import srgb_to_linear, inverse_tone_map

logger = get_logger()

MODEL_W, MODEL_H = 3840, 2160  # unet4k 固定输入尺寸
MODEL_FILENAME = "unet4k.onnx"
MODEL_URLS = [
    # fastly.jsdelivr.net（GitHub 文件 CDN，国内可达性最好）
    "https://fastly.jsdelivr.net/gh/zhuiber/SDR-to-HDR-Reconstruction-System@main/color_unet/Model_onnx/unet4k.onnx",
    "https://cdn.jsdelivr.net/gh/zhuiber/SDR-to-HDR-Reconstruction-System@main/color_unet/Model_onnx/unet4k.onnx",
]
MODEL_MIN_SIZE = 1024 * 1024  # 1MB，校验下载完整性

# 4K U-Net 的 ORT 激活内存预估（实测 8GB 机器 OOM；OS OOM killer 的
# SIGKILL 无法被 Python 捕获，必须前置预检）
MODEL_MIN_AVAIL_GB = 5.0


def _available_memory_gb() -> Optional[float]:
    """探测可用内存（GB）。"""
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return None


@BackendRegistry.register
class UnetOnnxBackend(SDRToHDRBackend):
    name = "unet-onnx"
    description = "SDR-to-HDR U-Net ONNX 模型（unet4k.onnx, zhuiber/SDR-to-HDR-Reconstruction-System）"
    required_deps: List[str] = ["onnxruntime"]
    supports_batch = True
    output_formats = ["exr", "png", "jpg"]

    def __init__(self, model_dir: Optional[Path] = None, **kwargs):
        super().__init__(model_dir=model_dir, **kwargs)
        self._session = None
        self._input_name: Optional[str] = None

    # ── 模型管理 ──────────────────────────────────────────────────────────

    @property
    def model_path(self) -> Path:
        return self.model_dir / MODEL_FILENAME

    def ensure_model(self) -> bool:
        """确保模型存在（缺失则自动下载）。"""
        p = self.model_path
        if p.exists() and p.stat().st_size >= MODEL_MIN_SIZE:
            return True
        import urllib.request
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for url in MODEL_URLS:
            try:
                logger.info(f"[{self.name}] 下载模型: {url}")
                urllib.request.urlretrieve(url, p)
                if p.exists() and p.stat().st_size >= MODEL_MIN_SIZE:
                    logger.info(f"[{self.name}] 模型就绪: {p} "
                                f"({p.stat().st_size / 1024 / 1024:.1f} MB)")
                    return True
            except Exception as e:
                logger.warning(f"[{self.name}] 下载失败 ({url}): {e}")
        return False

    # ── 后端接口 ──────────────────────────────────────────────────────────

    def check_available(self) -> Tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "未安装 onnxruntime：pip install onnxruntime"
        if not self.model_path.exists():
            ok = self.ensure_model()
            if not ok:
                return False, "模型不可用（下载失败，需手动放置 unet4k.onnx）"
        # 内存预检（模型固定 4K 输入，激活内存数 GB；OOM SIGKILL 不可捕获）
        avail = _available_memory_gb()
        if avail is not None and avail < MODEL_MIN_AVAIL_GB:
            return False, (
                f"内存不足：unet4k 固定 {MODEL_W}×{MODEL_H} 推理需 ~{MODEL_MIN_AVAIL_GB:.0f}GB "
                f"可用内存（当前 {avail:.1f}GB）。建议使用 CUDA (onnxruntime-gpu) "
                f"或更大内存的机器")
        return True, f"onnxruntime + {MODEL_FILENAME}"

    def prepare(self, device: str, **kwargs) -> None:
        import onnxruntime as ort

        if not self.ensure_model():
            raise RuntimeError(f"[{self.name}] 模型不可用")

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.warning(
                    f"[{self.name}] CUDA 不可用（需 onnxruntime-gpu），回退 CPU")

        so = ort.SessionOptions()
        so.log_severity_level = 3  # 抑制 ORT 的逐层日志
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._device = device
        self._prepared = True
        logger.info(f"[{self.name}] prepared (device={device}, "
                    f"providers={self._session.get_providers()})")

    def get_memory_estimate(self) -> Tuple[float, float]:
        # 4K float32 中间张量：输入/输出各 2160*3840*3*4 ≈ 100MB，ORT 预留若干倍
        return 1.0, 2.5

    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "exr",
        colorspace: str = "linear-rec709",
        keep_exif: bool = True,
        sidecar: bool = True,
        target_nits: int = 1000,
        **_ignored: Any,
    ) -> bool:
        if not self._prepared or self._session is None:
            self.prepare(self._device or "cpu")
        try:
            # 1. 读图并预处理（resize 到模型固定尺寸）
            img = Image.open(input_path)
            img.load()
            sdr_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
            img.close()

            resized = Image.fromarray(sdr_uint8).resize(
                (MODEL_W, MODEL_H), Image.LANCZOS)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            nchw = np.transpose(arr, (2, 0, 1))[None, ...]  # (1,3,H,W)

            # 2. 推理
            outputs = self._session.run(
                [self._session.get_outputs()[0].name], {self._input_name: nchw})
            out = outputs[0][0]  # (3,H,W)
            hdr_hwc = np.transpose(out, (1, 2, 0))  # (H,W,3)

            # 3. 语义标定：模型输出 [0,1] → 线性 HDR nits（缩放到目标峰值）
            hdr_nits = np.clip(hdr_hwc, 0.0, 1.0) * float(target_nits)

            # 4. 还原到原图尺寸（HDR 与原图同尺寸更方便比对）
            hdr_full = np.asarray(
                Image.fromarray(hdr_hwc.astype(np.float32)).resize(
                    (sdr_uint8.shape[1], sdr_uint8.shape[0]), Image.LANCZOS))
            hdr_nits_full = np.clip(hdr_full, 0.0, 1.0) * float(target_nits)

            # 5. EXIF（A+C）
            carrier = ExifCarrier(input_path) if keep_exif else None

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = output_format.lower().lstrip(".")

            if fmt in ("jpg", "jpeg"):
                # JPEG 路径：把线性 HDR tonemap 回 SDR 外观再存（本后端非 UltraHDR 原生）
                from utils.itm import linear_to_srgb
                from utils.hdr_metrics import tonemap_reinhard
                sdr_view = (linear_to_srgb(tonemap_reinhard(hdr_nits_full)) * 255
                            + 0.5).clip(0, 255).astype(np.uint8)
                im = Image.fromarray(sdr_view)
                im.save(out_path, format="JPEG", quality=95,
                        icc_profile=None)
                # EXIF 注入
                if carrier and carrier.app1:
                    from utils.exif import inject_exif_into_jpeg
                    data = out_path.read_bytes()
                    out_path.write_bytes(inject_exif_into_jpeg(data, carrier.app1))
            elif fmt == "exr":
                attrs = {
                    "software": "photo_sdr_to_hdr / unet-onnx",
                    "peakNits": str(target_nits),
                    "model": MODEL_FILENAME,
                    "colorspace": "linear Rec.709 (nits)",
                }
                if keep_exif and carrier and carrier.fields:
                    attrs.update(carrier.exr_attributes())
                write_exr(str(out_path), hdr_nits_full.astype(np.float32), attrs)
            elif fmt == "png":
                write_png16(
                    str(out_path),
                    (hdr_nits_full / float(target_nits)).astype(np.float32),
                    exif_payload=(carrier.payload if keep_exif and carrier else None))
            else:
                raise ValueError(
                    f"unet-onnx 不支持的输出格式: {output_format}")

            if sidecar and keep_exif and carrier:
                carrier.write_sidecar(str(out_path))
            return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False