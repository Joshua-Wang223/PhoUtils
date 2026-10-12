"""LumiPic backends — SDR → HDR via diffusion LoRA（三个变体）。

vendor 自 oumad/LumiPic（MIT）：
  - lumi-qwen:    Qwen-Image-Edit-2511 + LumiPic LoRA（production，需 ~24GB VRAM bf16）
  - lumi-klein4b: FLUX.2-klein-base-4B + LoRA（alpha，Apache 2.0，~12GB）
  - lumi-klein9b: FLUX.2-klein-base-9B + LoRA（alpha，gated，~20GB）

管线：diffusers Pipeline + LoRA → 输出 LogC3 编码 [0,1] 图像 →
      LogC3 解压 → 场景线性 HDR（Rec.709 原色，0..~55.1）→ EXR half-float

模型获取（本仓库不内置数 GB~数十 GB 权重）：
  - HuggingFace 直连不可达时自动使用镜像 HF_ENDPOINT=https://hf-mirror.com
  - 首次使用前需下载基础模型与 LoRA，例如：
      HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \\
        Qwen/Qwen-Image-Edit-2511
      HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \\
        oumoumad/LumiPic v5b_step2000.safetensors
  - 或将权重放到 models/lumipic/<变体>/ 下（本地路径优先）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import SDRToHDRBackend, BackendRegistry
from utils.exif import ExifCarrier
from utils.exr_io import write_exr
from utils.png16 import write_png16
from utils.logging import get_logger

logger = get_logger()

# HF 镜像（huggingface.co 不可达时的替代；实测国内可达）
HF_MIRROR = "https://hf-mirror.com"

# 各变体配置
VARIANTS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "base_model": "Qwen/Qwen-Image-Edit-2511",
        "pipeline_cls": "QwenImageEditPipeline",
        "pipeline_module": "diffusers",
        "lora_repo": "oumoumad/LumiPic",
        "lora_weight": "v5b_step2000.safetensors",
        "vram_gb_bf16": 24,
        "gated": False,
        "steps": 40,
    },
    "klein4b": {
        "base_model": "black-forest-labs/FLUX.2-klein-base-4B",
        "pipeline_cls": "Flux2KleinPipeline",
        "pipeline_module": "diffusers",
        "lora_repo": "oumoumad/LumiPic",
        "lora_weight": "klein4b_alpha_step1750.safetensors",
        "vram_gb_bf16": 12,
        "gated": False,
        "steps": 25,
    },
    "klein9b": {
        "base_model": "black-forest-labs/FLUX.2-klein-base-9B",
        "pipeline_cls": "Flux2KleinPipeline",
        "pipeline_module": "diffusers",
        "lora_repo": "oumoumad/LumiPic",
        "lora_weight": "klein9b_alpha_step2000.safetensors",
        "vram_gb_bf16": 20,
        "gated": True,
        "steps": 25,
    },
}


# ── LogC3（vendor 自 LumiPic logc3.py，ARRI EI 800） ─────────────────────

class LogC3:
    """ARRI LogC3 (EI 800)：场景线性 [0, ~55.1] ↔ LogC3 [0, 1]。"""

    A, B, C, D, E, F = 5.555556, 0.052272, 0.247190, 0.385537, 0.092809, 5.367655
    CUT = 0.010591

    def decompress_numpy(self, logc: np.ndarray) -> np.ndarray:
        """LogC3 [0,1] → 场景线性。"""
        logc = np.clip(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        linear_from_log = (np.power(10.0, (logc - self.D) / self.C) - self.B) / self.A
        linear_from_lin = (logc - self.F) / self.E
        return np.where(logc >= cut_log, linear_from_log, linear_from_lin)


# 场景线性 → nits 的缩放：1.0（漫射白）→ 203 nits（BT.2408 SDR 参考白）
LINEAR_TO_NITS = 203.0


class LumiPicBackend(SDRToHDRBackend):
    """LumiPic 基类（子类按 variant 区分）。"""

    variant: str = "qwen"
    required_deps: List[str] = ["torch", "diffusers", "transformers", "peft"]
    supports_batch = False  # 扩散模型逐张推理（管线内含批处理但不实用）
    output_formats = ["exr", "png", "jpg"]

    def __init__(self, model_dir: Optional[Path] = None, **kwargs):
        super().__init__(model_dir=model_dir, **kwargs)
        self._pipe = None
        self._logc3 = LogC3()
        cfg = VARIANTS[self.variant]
        self._cfg = cfg
        self._local_dir = self.model_dir / self.variant

    # ── 模型管理 ──────────────────────────────────────────────────────────

    def _setup_hf_mirror(self) -> None:
        """huggingface.co 不可达时切换到镜像。"""
        if os.environ.get("HF_ENDPOINT"):
            return
        try:
            import socket
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return  # 直连可用
        except Exception:
            pass
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        logger.info(f"[{self.name}] huggingface.co 不可达，已切换镜像: {HF_MIRROR}")

    def _find_cached_model(self) -> Optional[Path]:
        """查找本地缓存的基础模型目录。"""
        # 1) models/lumipic/<variant>/（用户显式放置）
        if self._local_dir.is_dir() and any(self._local_dir.iterdir()):
            return self._local_dir
        # 2) HF hub 缓存（~/.cache/huggingface/hub/models--<org>--<name>）
        base = self._cfg["base_model"].replace("/", "--")
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        for prefix in (f"models--{base}", base):
            candidate = hf_cache / prefix
            if candidate.is_dir():
                return None  # 存在缓存但不返回本地路径（让 from_pretrained 自己解析）
        # 3) HF_HOME 环境变量
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidate = Path(hf_home) / "hub" / f"models--{base}"
            if candidate.is_dir():
                return None
        return None

    def _lora_local_path(self) -> Optional[Path]:
        """查找本地 LoRA 文件。"""
        candidate = self._local_dir / self._cfg["lora_weight"]
        if candidate.is_file():
            return candidate
        return None

    def check_available(self) -> Tuple[bool, str]:
        # 1. Python 依赖
        for mod in self.required_deps:
            try:
                __import__(mod)
            except ImportError:
                return False, f"缺少依赖: {mod}（pip install {' '.join(self.required_deps)}）"

        # 2. 管线类存在性（diffusers 版本可能没有 Flux2KleinPipeline）
        try:
            module = __import__(self._cfg["pipeline_module"], fromlist=[self._cfg["pipeline_cls"]])
            if not hasattr(module, self._cfg["pipeline_cls"]):
                return False, (f"diffusers 无 {self._cfg['pipeline_cls']}"
                               f"（需更新版 diffusers 或该变体已更名）")
        except ImportError:
            return False, f"缺少依赖: {self._cfg['pipeline_module']}"

        # 3. 模型缓存
        base = self._cfg["base_model"].replace("/", "--")
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{base}"
        hf_home = os.environ.get("HF_HOME")
        cached = hf_cache.is_dir() or (hf_home and (Path(hf_home) / "hub" / f"models--{base}").is_dir())
        local = self._local_dir.is_dir() and any(self._local_dir.iterdir())
        if not (cached or local):
            gated = "（gated 模型，需先申请授权）" if self._cfg["gated"] else ""
            return False, (
                f"基础模型未下载: {self._cfg['base_model']}{gated}。"
                f"下载: HF_ENDPOINT={HF_MIRROR} huggingface-cli download {self._cfg['base_model']}"
            )
        if not (self._lora_local_path() or self._lora_cached()):
            return False, (
                f"LoRA 未下载: {self._cfg['lora_repo']}/{self._cfg['lora_weight']}。"
                f"下载: HF_ENDPOINT={HF_MIRROR} huggingface-cli download "
                f"{self._cfg['lora_repo']} {self._cfg['lora_weight']}"
            )
        return True, "deps + models ready"

    def _lora_cached(self) -> bool:
        """LoRA 是否在 HF 缓存中。"""
        repo = self._cfg["lora_repo"].replace("/", "--")
        for root in (Path.home() / ".cache" / "huggingface" / "hub",
                     Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None):
            if root and (root / f"models--{repo}").is_dir():
                return True
        return False

    # ── 后端接口 ──────────────────────────────────────────────────────────

    def prepare(self, device: str, quantization: str = "none", **kwargs) -> None:
        self._setup_hf_mirror()
        import torch

        self._device = device
        self._prepared = True
        logger.info(f"[{self.name}] prepared (variant={self.variant}, "
                    f"device={device}, quantization={quantization})")

    def _load_pipeline(self, quantization: str = "none"):
        """懒加载扩散管线（含 LoRA）。"""
        if self._pipe is not None:
            return self._pipe
        self._setup_hf_mirror()
        import torch

        cfg = self._cfg
        module = __import__(cfg["pipeline_module"], fromlist=[cfg["pipeline_cls"]])
        pipeline_cls = getattr(module, cfg["pipeline_cls"])

        dtype = torch.bfloat16 if quantization == "none" else torch.float16
        model_id = str(self._local_dir) if self._local_dir.is_dir() and any(
            self._local_dir.iterdir()) else cfg["base_model"]

        load_kwargs = {"torch_dtype": dtype}
        # 量化（需 bitsandbytes）
        if quantization == "8bit":
            load_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            load_kwargs["load_in_4bit"] = True

        logger.info(f"[{self.name}] 加载基础模型: {model_id} (dtype={dtype}, quant={quantization})")
        try:
            pipe = pipeline_cls.from_pretrained(model_id, **load_kwargs)
        except Exception as e:
            if quantization != "none" and "bitsandbytes" in str(e).lower():
                raise RuntimeError(
                    f"量化需要 bitsandbytes：pip install bitsandbytes（{e}）")
            raise

        # LoRA
        lora_local = self._lora_local_path()
        if lora_local:
            logger.info(f"[{self.name}] 加载本地 LoRA: {lora_local}")
            pipe.load_lora_weights(str(lora_local))
        else:
            logger.info(f"[{self.name}] 加载 HF LoRA: {cfg['lora_repo']}/{cfg['lora_weight']}")
            pipe.load_lora_weights(cfg["lora_repo"], weight_name=cfg["lora_weight"])

        pipe = pipe.to(self._device or "cpu")
        self._pipe = pipe
        return pipe

    def get_memory_estimate(self) -> Tuple[float, float]:
        cfg = self._cfg
        return cfg["vram_gb_bf16"] * 0.9, cfg["vram_gb_bf16"] * 0.3

    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "exr",
        colorspace: str = "linear-rec709",
        keep_exif: bool = True,
        sidecar: bool = True,
        target_nits: int = 1000,
        quantization: str = "none",
        steps: Optional[int] = None,
        guidance: float = 3.0,
        seed: int = 42,
        **_ignored: Any,
    ) -> bool:
        try:
            import torch

            if not self._prepared:
                self.prepare(self._device or "cpu", quantization=quantization)
            pipe = self._load_pipeline(quantization)

            img = Image.open(input_path)
            img.load()
            image = img.convert("RGB")
            img.close()

            n_steps = steps or self._cfg["steps"]
            generator = torch.Generator(device=self._device or "cpu").manual_seed(seed)

            logger.info(f"[{self.name}] 推理中 (steps={n_steps}, guidance={guidance}, seed={seed}) ...")
            with torch.inference_mode():
                output = pipe(
                    prompt="Convert this image to HDR",
                    image=image,
                    num_inference_steps=n_steps,
                    guidance_scale=guidance,
                    generator=generator,
                    output_type="pt",
                )
                pixel = output.images[0].float()
                logc3_data = torch.clamp(pixel, 0, 1).permute(1, 2, 0).cpu().numpy()

            # LogC3 → 场景线性 → nits（Rec.709 原色）
            hdr_linear = self._logc3.decompress_numpy(logc3_data)
            hdr_nits = np.clip(hdr_linear, 0.0, None) * LINEAR_TO_NITS

            # EXIF（A+C）
            carrier = ExifCarrier(input_path) if keep_exif else None

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = output_format.lower().lstrip(".")

            if fmt in ("jpg", "jpeg"):
                from utils.itm import linear_to_srgb
                from utils.hdr_metrics import tonemap_reinhard
                sdr_view = (linear_to_srgb(tonemap_reinhard(hdr_nits)) * 255
                            + 0.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(sdr_view).save(out_path, format="JPEG", quality=95)
                if carrier and carrier.app1:
                    from utils.exif import inject_exif_into_jpeg
                    data = out_path.read_bytes()
                    out_path.write_bytes(inject_exif_into_jpeg(data, carrier.app1))
            elif fmt == "exr":
                attrs = {
                    "software": f"photo_sdr_to_hdr / {self.name}",
                    "peakNits": str(target_nits),
                    "variant": self.variant,
                    "baseModel": self._cfg["base_model"],
                    "colorspace": "linear Rec.709 (nits, LogC3-decoded)",
                    "linearToNits": str(LINEAR_TO_NITS),
                }
                if keep_exif and carrier and carrier.fields:
                    attrs.update(carrier.exr_attributes())
                write_exr(str(out_path), hdr_nits.astype(np.float32), attrs)
            elif fmt == "png":
                write_png16(
                    str(out_path),
                    (hdr_nits / float(target_nits)).astype(np.float32),
                    exif_payload=(carrier.payload if keep_exif and carrier else None))
            else:
                raise ValueError(f"LumiPic 不支持的输出格式: {output_format}")

            if sidecar and keep_exif and carrier:
                carrier.write_sidecar(str(out_path))
            return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False


@BackendRegistry.register
class LumiPicQwenBackend(LumiPicBackend):
    name = "lumi-qwen"
    description = "LumiPic Qwen-Image-Edit LoRA（production，bf16 需 ~24GB 显存）"
    variant = "qwen"


@BackendRegistry.register
class LumiPicKlein4bBackend(LumiPicBackend):
    name = "lumi-klein4b"
    description = "LumiPic FLUX.2-klein-4B LoRA（alpha, Apache 2.0, ~12GB 显存）"
    variant = "klein4b"


@BackendRegistry.register
class LumiPicKlein9bBackend(LumiPicBackend):
    name = "lumi-klein9b"
    description = "LumiPic FLUX.2-klein-9B LoRA（alpha, gated 需授权, ~20GB 显存）"
    variant = "klein9b"