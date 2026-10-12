"""Subprocess backend base — 外部 CLI 工具的统一封装。

适用于无法 import 的外部工具（Rust CLI、C++ CLI 等）：
  - 二进制发现（PATH / 显式路径 / 环境变量）
  - 超时控制、错误捕获、stderr 诊断
  - 临时目录管理
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import SDRToHDRBackend, BackendRegistry
from utils.exif import ExifCarrier
from utils.logging import get_logger

logger = get_logger()


class SubprocessBackend(SDRToHDRBackend):
    """外部 CLI 后端基类。"""

    # 子类覆盖
    binary_names: List[str] = []        # 可执行文件名候选（PATH 查找）
    env_vars: List[str] = []            # 环境变量名（显式指定路径）
    default_timeout: int = 300          # 单次调用超时（秒）

    def __init__(self, model_dir: Optional[Path] = None, binary_path: Optional[str] = None,
                 **kwargs):
        super().__init__(model_dir=model_dir, **kwargs)
        self._binary: Optional[str] = None
        self._explicit_binary = binary_path

    def find_binary(self) -> Optional[str]:
        """定位外部可执行文件。"""
        if self._explicit_binary:
            if os.path.isfile(self._explicit_binary) and os.access(self._explicit_binary, os.X_OK):
                return self._explicit_binary
            logger.warning(f"[{self.name}] 指定的二进制不可执行: {self._explicit_binary}")
            return None
        # 环境变量
        for var in self.env_vars:
            p = os.environ.get(var)
            if p and os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        # PATH
        for name in self.binary_names:
            p = shutil.which(name)
            if p:
                return p
        return None

    def check_available(self) -> Tuple[bool, str]:
        self._binary = self.find_binary()
        if self._binary is None:
            hint = ", ".join(self.binary_names) or "binary"
            return False, f"未找到 {hint}（PATH/环境变量 {self.env_vars or 'N/A'}）"
        return True, f"binary: {self._binary}"

    def prepare(self, device: str, **kwargs) -> None:
        self._device = device
        if self._binary is None:
            self._binary = self.find_binary()
        self._prepared = True

    def run_command(
        self,
        args: List[str],
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        执行外部命令。

        Returns:
            (success, stdout, stderr)
        """
        if self._binary is None:
            raise RuntimeError(f"[{self.name}] 二进制未初始化（先调用 prepare）")
        cmd = [self._binary] + args
        logger.debug(f"[{self.name}] 执行: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout or self.default_timeout, cwd=cwd)
            return proc.returncode == 0, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"超时（>{timeout or self.default_timeout}s）"
        except Exception as e:
            return False, "", f"{type(e).__name__}: {e}"


@BackendRegistry.register
class UltraHDRCliBackend(SubprocessBackend):
    """libultrahdr 官方 CLI（ultrahdr_app，需自行编译安装）。

    libultrahdr 的 ultrahdr_app 输入 HDR raw（P010/rgba1010102）+ SDR JPEG，
    计算增益图并封装 UltraHDR。本后端负责：
      1. 用共享 ITM 把 SDR 提升为 HDR 线性值
      2. 编码为 ultrahdr_app 接受的 raw 格式
      3. 调用 ultrahdr_app 封装
    未安装 CLI 时优雅降级（check_available 返回 False，bench 跳过）。
    """

    name = "ultrahdr"
    description = "libultrahdr 官方 CLI（ultrahdr_app，需编译安装；Google 参考实现）"
    required_deps: List[str] = []
    supports_batch = False
    output_formats = ["jpg"]
    binary_names = ["ultrahdr_app", "ultrahdr"]
    env_vars = ["ULTRHDR_APP_PATH"]

    def convert(self, input_path: str, output_path: str, **kwargs) -> bool:
        if self._binary is None:
            ok, why = self.check_available()
            if not ok:
                logger.error(f"[{self.name}] 不可用: {why}")
                return False
        try:
            import numpy as np
            from PIL import Image
            from utils.itm import sdr_uint8_to_linear, inverse_tone_map
            from utils.exif import ExifCarrier

            img = Image.open(input_path)
            img.load()
            sdr_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
            img.close()

            # ITM 提升（BT.709 线性 nits）
            sdr_lin = sdr_uint8_to_linear(sdr_uint8)
            hdr_nits = inverse_tone_map(
                sdr_lin, 203.0, float(kwargs.get("target_nits", 1000)))

            # ultrahdr_app 期望 rgba1010102 raw（RGBA 各 10bit，32bpp）
            # 线性 [0,1] → 10bit；alpha=1.0
            hdr10 = np.clip(hdr_nits / 1000.0, 0.0, 1.0)
            rgba10 = np.zeros((*hdr10.shape[:2], 4), dtype=np.uint32)
            rgba10[..., :3] = (hdr10 * 1023.0 + 0.5).astype(np.uint32)
            rgba10[..., 3] = 1023
            # rgba1010102 内存布局：每通道 10bit 打包进 32bit 字（小端 R,G,B,A 顺序字）
            raw = rgba10.astype("<u4").tobytes()

            with tempfile.TemporaryDirectory(prefix="uhdr_") as tmp:
                raw_path = Path(tmp) / "input.rgba1010102"
                raw_path.write_bytes(raw)
                out_path = Path(tmp) / "output.jpg"
                # ultrahdr_app 用法（参考 examples/ultrahdr_app.cpp）：
                #   ultrahdr_app -m 1 -p P010/rgba1010102 -i <raw> -i <sdr.jpg> -o out.jpg
                args = [
                    "-m", "1",
                    "-p", "rgba1010102",
                    "-i", str(raw_path),
                    "-i", str(input_path),
                    "-o", str(out_path),
                ]
                ok, stdout, stderr = self.run_command(args)
                if not ok or not out_path.exists():
                    logger.error(f"[{self.name}] CLI 失败: {stderr[:300]}")
                    return False

                data = out_path.read_bytes()
                # EXIF 注入（A+C）
                carrier = ExifCarrier(input_path) if kwargs.get("keep_exif", True) else None
                if carrier and carrier.app1:
                    from utils.exif import inject_exif_into_jpeg
                    data = inject_exif_into_jpeg(data, carrier.app1)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(data)
                if kwargs.get("sidecar", True) and carrier:
                    carrier.write_sidecar(output_path)
                return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False


@BackendRegistry.register
class Sdr2HdrBackend(SDRToHDRBackend):
    """sdr2hdr 后端（guofei9987/sdr2hdr，Rust 项目的纯 Python 复刻）。

    原项目通过嵌入特殊 ICC 配置文件，使普通 PNG/JPEG 在 HDR 显示器上
    呈现更亮效果（`embed_icc_file_with_type`）。本后端 vendor 了该项目的
    两个 ICC 配置文件（models/sdr2hdr/icc1.icc / icc2.icc），用 Pillow 原生
    icc_profile 参数嵌入，无需安装 Rust 工具链。

    局限（原项目同理）：这是 ICC 显示技巧而非真实 HDR 重建——
    不产生增益图/线性 HDR 数据，仅在支持 HDR 显示的设备上观感更亮。
    mode 1/2 对应原项目的两种 ICC 配置。
    """

    name = "sdr2hdr"
    description = "sdr2hdr ICC 嵌入（vendor 自 guofei9987/sdr2hdr，MIT；纯 Python 无需 Rust）"
    required_deps: List[str] = []
    supports_batch = True
    output_formats = ["png", "jpg"]

    ICC_SOURCES = {
        1: ("icc1.icc",
            "https://raw.githubusercontent.com/guofei9987/sdr2hdr/main/src/icc/icc1.icc"),
        2: ("icc2.icc",
            "https://raw.githubusercontent.com/guofei9987/sdr2hdr/main/src/icc/icc2.icc"),
    }

    def _icc_path(self, mode: int) -> Path:
        name = self.ICC_SOURCES.get(mode, self.ICC_SOURCES[1])[0]
        return self.model_dir / name

    def _ensure_icc(self, mode: int) -> Optional[Path]:
        """确保 ICC 文件存在（缺失时尝试下载）。"""
        path = self._icc_path(mode)
        if path.exists() and path.stat().st_size > 100:
            return path
        # 尝试下载（codeload tarball 路径，raw.githubusercontent 不可达）
        try:
            import urllib.request
            url = f"https://codeload.github.com/guofei9987/sdr2hdr/tar.gz/refs/heads/main"
            logger.info(f"[{self.name}] ICC 缺失，从 {url} 获取")
            import tarfile
            import io
            with urllib.request.urlopen(url, timeout=60) as resp:
                tarball = resp.read()
            name = self.ICC_SOURCES[mode][0]
            with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(f"src/icc/{name}"):
                        self.model_dir.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(tf.extractfile(member).read())
                        return path
        except Exception as e:
            logger.warning(f"[{self.name}] ICC 下载失败: {e}")
        return None

    def check_available(self) -> Tuple[bool, str]:
        ok = self._ensure_icc(1) is not None and self._ensure_icc(2) is not None
        if ok:
            return True, f"ICC 配置就绪 ({self.model_dir})"
        return False, "ICC 配置文件缺失且无法下载（需手动放置 icc1.icc/icc2.icc）"

    def prepare(self, device: str, **kwargs) -> None:
        self._device = device
        self._prepared = True

    def convert(
        self,
        input_path: str,
        output_path: str,
        mode: int = 1,
        keep_exif: bool = True,
        sidecar: bool = True,
        output_format: str = "png",
        **_ignored,
    ) -> bool:
        """嵌入 ICC 配置文件（mode 1/2）。"""
        try:
            from PIL import Image
            # 格式能力校验（不支持显式报错，避免写出错误扩展名的文件）
            fmt = output_format.lower().lstrip(".")
            if fmt not in self.output_formats:
                logger.error(
                    f"[{self.name}] 不支持的输出格式: {output_format}"
                    f"（支持: {', '.join(self.output_formats)}）")
                return False

            icc = self._ensure_icc(int(mode))
            if icc is None:
                logger.error(f"[{self.name}] ICC mode={mode} 不可用")
                return False

            img = Image.open(input_path)
            img.load()
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # EXIF 提取（A+C）
            carrier = ExifCarrier(input_path) if keep_exif else None
            icc_bytes = icc.read_bytes()

            if fmt in ("jpg", "jpeg"):
                img.convert("RGB").save(
                    out_path, format="JPEG", quality=95,
                    icc_profile=icc_bytes)
                # EXIF 注入（SOI 后）
                if carrier and carrier.app1:
                    from utils.exif import inject_exif_into_jpeg
                    data = out_path.read_bytes()
                    out_path.write_bytes(inject_exif_into_jpeg(data, carrier.app1))
            else:
                # PNG：icc_profile + exif 一起传（Pillow 写 eXIf chunk）
                save_kwargs = {"format": "PNG", "icc_profile": icc_bytes}
                if carrier and carrier.payload:
                    save_kwargs["exif"] = carrier.payload
                img.save(out_path, **save_kwargs)

            if sidecar and keep_exif and carrier:
                carrier.write_sidecar(output_path)
            return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False