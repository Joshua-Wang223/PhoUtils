"""gmnet backend — GMNet (ICLR 2025) Gain Map 预测。

vendor 自 qtlark/GMNet（Learning Gain Map for Inverse Tone Mapping）：
  - 双分支网络：local branch（全分辨率 SDR）+ global branch（256×256 缩略图）
  - 输出：半分辨率增益图（1 通道）+ qmax（每图标量）
  - 渲染（复刻 scripts/render_sdr_gm_to_linear_hdr.py）：
        gm_log2 = qmax * gainmap
        hdr_linear = sdr_linear * 2^gm_log2
  - 检查点：models/gmnet/G_realworld.pth（7.7MB，首次运行自动下载）
    （另有 G_synthetic.pth，--gmnet-variant synthetic 切换）

依赖：torch（无需 BasicSR 完整框架，架构已 vendor）。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import SDRToHDRBackend, BackendRegistry
from utils.exif import ExifCarrier
from utils.exr_io import write_exr
from utils.png16 import write_png16
from utils.logging import get_logger
from utils.itm import srgb_to_linear

logger = get_logger()

THUMBNAIL_SIZE = 256  # global branch 输入尺寸（数据集约定）
CHECKPOINTS = {
    "realworld": ("G_realworld.pth",
                  "https://fastly.jsdelivr.net/gh/qtlark/GMNet@main/checkpoints/G_realworld.pth"),
    "synthetic": ("G_synthetic.pth",
                  "https://fastly.jsdelivr.net/gh/qtlark/GMNet@main/checkpoints/G_synthetic.pth"),
}


# ── 架构（vendor 自 GMNet codes/models/modules/） ─────────────────────────

def _initialize_weights(net_l, scale=1):
    import torch
    import torch.nn as nn
    import torch.nn.init as init
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


def _make_layer(block, n_layers):
    import torch.nn as nn
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class _DownConv:
    """延迟构建（import torch 时才需要 nn.Module 基类）。"""
    pass


def _build_gmnet(variant: str = "realworld"):
    """构建 GMNet 网络（torch 导入在函数内，保持模块可无 torch 导入）。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlockNoBN(nn.Module):
        def __init__(self, nf=64):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            _initialize_weights([self.conv1, self.conv2], 0.1)

        def forward(self, x):
            identity = x
            out = F.relu(self.conv1(x), inplace=True)
            out = self.conv2(out)
            return identity + out

    class DownConv(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 2, 1, bias=True), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, 1, 1, bias=True), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, 1, 1, bias=True), nn.ReLU())
            _initialize_weights([self.conv], 0.1)

        def forward(self, x):
            return self.conv(x)

    class SQModule(nn.Module):
        def __init__(self, size, out_nc, nf=64):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d([size, size])
            self.conv = nn.Sequential(
                nn.Conv2d(nf, nf * 2, 1, 1, 0, bias=True), nn.ReLU(),
                nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True), nn.ReLU(),
                nn.Conv2d(nf, out_nc, 1, 1, 0, bias=True))
            _initialize_weights([self.conv], 0.1)

        def forward(self, x):
            return self.conv(self.pool(x))

    class GMNet(nn.Module):
        def __init__(self, in_nc=3, out_nc=1, nf=64, nb=16, act_type='relu'):
            super().__init__()
            res_block = functools.partial(ResidualBlockNoBN, nf=nf)
            # global branch
            self.down1 = DownConv(in_nc, nf // 4)
            self.down2 = DownConv(nf // 4, nf)
            self.res_y = _make_layer(res_block, 5)
            self.sq_ker = SQModule(3, nf)
            self.sq_chn = SQModule(1, nf)
            self.sq_qmax = SQModule(1, 1)
            self.mask_est = nn.Sequential(
                nn.Conv2d(nf, nf, 3, 1, 1, bias=True), nn.ReLU(),
                nn.Conv2d(nf, 1, 3, 1, 1, bias=True), nn.Sigmoid())
            self.att_est = nn.Sigmoid()
            # local branch
            self.down_x = DownConv(in_nc, nf)
            self.res1 = _make_layer(res_block, 5)
            self.res2 = _make_layer(res_block, 5)
            self.res3 = _make_layer(res_block, 5)
            # tail
            self.upconv = nn.Conv2d(nf, 4 * nf, 3, 1, 1, bias=True)
            self.upsampler = nn.PixelShuffle(2)
            self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
            self.act = nn.ReLU(inplace=True) if act_type == 'relu' \
                else nn.LeakyReLU(negative_slope=0.1, inplace=True)
            _initialize_weights([self.mask_est, self.upconv, self.HRconv, self.conv_last], 0.1)

        def forward(self, x):
            # global
            y = x[1]
            y = self.down1(y)
            y = self.down2(y)
            y = self.res_y(y)
            kernel = self.sq_ker(y)
            cse = self.sq_chn(y)
            qmax = self.sq_qmax(y)
            # local
            x = x[0]
            x = self.down_x(x)
            B, C, H, W = x.shape
            x = self.res1(x)
            mask = F.conv2d(
                x.view(1, B * C, H, W), kernel.view(B * C, 1, 3, 3),
                stride=1, padding=1, dilation=1, groups=B * C).view(B, C, H, W)
            mask = self.mask_est(mask)
            x = x * mask
            x = self.res2(x)
            x = x * self.att_est(cse)
            x = self.res3(x)
            out = self.act(self.upsampler(self.upconv(x)))
            out = self.conv_last(self.act(self.HRconv(out)))
            return out, qmax * out

    return GMNet()


@BackendRegistry.register
class GMNetBackend(SDRToHDRBackend):
    name = "gmnet"
    description = "GMNet (ICLR2025) 双分支增益图预测（vendor 自 qtlark/GMNet）"
    required_deps: List[str] = ["torch"]
    supports_batch = True
    output_formats = ["exr", "png", "jpg"]

    def __init__(self, model_dir: Optional[Path] = None, variant: str = "realworld",
                 **kwargs):
        super().__init__(model_dir=model_dir, **kwargs)
        self._variant = variant if variant in CHECKPOINTS else "realworld"
        self._net = None
        self._device = "cpu"

    # ── 模型管理 ──────────────────────────────────────────────────────────

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / CHECKPOINTS[self._variant][0]

    def ensure_checkpoint(self) -> bool:
        p = self.checkpoint_path
        if p.exists() and p.stat().st_size > 1024 * 1024:
            return True
        import urllib.request
        self.model_dir.mkdir(parents=True, exist_ok=True)
        url = CHECKPOINTS[self._variant][1]
        try:
            logger.info(f"[{self.name}] 下载检查点: {url}")
            urllib.request.urlretrieve(url, p)
            return p.exists() and p.stat().st_size > 1024 * 1024
        except Exception as e:
            logger.warning(f"[{self.name}] 检查点下载失败: {e}")
            return False

    def check_available(self) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "未安装 torch：pip install torch"
        if not self.checkpoint_path.exists():
            if not self.ensure_checkpoint():
                return False, f"检查点不可用（下载失败，需手动放置 {CHECKPOINTS[self._variant][0]}）"
        return True, f"torch + {CHECKPOINTS[self._variant][0]}"

    # ── 后端接口 ──────────────────────────────────────────────────────────

    def prepare(self, device: str, **kwargs) -> None:
        import torch

        if not self.ensure_checkpoint():
            raise RuntimeError(f"[{self.name}] 检查点不可用")

        self._net = _build_gmnet(self._variant)
        state = torch.load(str(self.checkpoint_path), map_location="cpu",
                           weights_only=True)
        # 处理可能的 key 前缀（如 'module.'）
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        missing, unexpected = self._net.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"[{self.name}] 检查点缺少层: {len(missing)} 个（前3: {missing[:3]}）")
        if unexpected:
            logger.warning(f"[{self.name}] 检查点多余键: {len(unexpected)} 个")
        self._net.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            self._net = self._net.to(device)
        self._device = device
        self._prepared = True
        logger.info(f"[{self.name}] prepared (variant={self._variant}, device={device})")

    def get_memory_estimate(self) -> Tuple[float, float]:
        return 0.8, 1.5

    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "exr",
        colorspace: str = "linear-rec709",
        keep_exif: bool = True,
        sidecar: bool = True,
        target_nits: int = 1000,
        sdr_white: float = 203.0,
        max_size: int = 2160,
        **_ignored: Any,
    ) -> bool:
        try:
            import torch

            if not self._prepared or self._net is None:
                self.prepare(self._device)

            # 1. 读图
            img = Image.open(input_path)
            img.load()
            sdr_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
            img.close()
            h, w = sdr_uint8.shape[:2]

            # 尺寸上限（local branch 全分辨率推理，超大图限制规模）
            scale = min(1.0, max_size / max(h, w))
            if scale < 1.0:
                nw, nh = int(w * scale), int(h * scale)
                sdr_uint8 = np.asarray(
                    Image.fromarray(sdr_uint8).resize((nw, nh), Image.LANCZOS))
                h, w = nh, nw

            # 2. 网络输入：local（全分辨率）+ global（256×256 缩略图）
            thumb = np.asarray(
                Image.fromarray(sdr_uint8).resize(
                    (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS),
                dtype=np.float32) / 255.0
            local = sdr_uint8.astype(np.float32) / 255.0

            x_local = torch.from_numpy(np.transpose(local, (2, 0, 1)))[None]
            x_global = torch.from_numpy(np.transpose(thumb, (2, 0, 1)))[None]
            if self._device.startswith("cuda"):
                x_local, x_global = x_local.to(self._device), x_global.to(self._device)

            # 3. 推理：out=半分辨率增益图, qmax*out=log2 增益
            with torch.inference_mode():
                out, gm_log2 = self._net([x_local, x_global])
                gm_log2 = gm_log2[0, 0].float().cpu().numpy()  # (H/2, W/2)

            # 4. 增益图上采样到全分辨率，渲染线性 HDR（复刻 render 脚本）
            gm_full = np.asarray(
                Image.fromarray(gm_log2).resize((w, h), Image.BILINEAR),
                dtype=np.float32)
            sdr_lin = srgb_to_linear(sdr_uint8.astype(np.float32) / 255.0)
            hdr_linear = sdr_lin * np.power(2.0, np.maximum(gm_full, 0.0))[..., None]
            hdr_nits = hdr_linear * sdr_white

            # 5. EXIF（A+C）
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
                    "software": "photo_sdr_to_hdr / gmnet",
                    "variant": self._variant,
                    "model": CHECKPOINTS[self._variant][0],
                    "gainMapMaxStops": f"{float(gm_full.max()):.3f}",
                    "colorspace": "linear Rec.709 (nits)",
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
                raise ValueError(f"gmnet 不支持的输出格式: {output_format}")

            if sidecar and keep_exif and carrier:
                carrier.write_sidecar(str(out_path))
            return True
        except Exception as e:
            logger.error(f"[{self.name}] 转换失败 {input_path}: {type(e).__name__}: {e}")
            return False