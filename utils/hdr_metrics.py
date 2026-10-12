"""HDR 客观质量指标（objective quality metrics）。

对标视频质量评价的 VMAF 思路，为 SDR→HDR 转换提供可量化的质量维度：

全参考（有 GT HDR 时）：
  - ΔICtCp PSNR（ITU-R BT.2124，HDR 标准化官方指标，单位 dB）
  - PU21 PSNR / PU21 SSIM（Mantiuk & Azimi, PCS 2021 —— 感知均匀编码后用 SDR 指标）
  - VMAF（ffmpeg libvmaf，对 tonemap(HDR) vs tonemap(GT) 计算）
  - Linear PSNR（线性光域）

无参考（SDR→HDR 生成任务的固有情形）：
  - Round-trip 一致性：tonemap(HDR) → SDR 后与原图比（VMAF / SSIM / PSNR）
    —— 衡量 HDR 是否"忠于原图"，过度/不足提升都会扣分
  - 动态范围扩展（stops）、高光截断率、阴影提升量

依赖：numpy + Pillow；VMAF 需 ffmpeg（含 libvmaf，缺失时自动跳过）。
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .itm import luminance

# ── PQ（SMPTE ST 2084 / BT.2100）传递函数 ─────────────────────────────────

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0

PQ_PEAK_NITS = 10000.0  # PQ 域 1.0 = 10000 nits


def pq_encode(luminance_nits: np.ndarray) -> np.ndarray:
    """线性亮度（nits）→ PQ 编码 [0,1]。"""
    L = np.clip(luminance_nits / PQ_PEAK_NITS, 0.0, 1.0)
    num = PQ_C1 + PQ_C2 * L ** PQ_M1
    den = 1.0 + PQ_C3 * L ** PQ_M1
    return (num / den) ** PQ_M2


def pq_decode(v: np.ndarray) -> np.ndarray:
    """PQ 编码 [0,1] → 线性亮度（nits）。"""
    v = np.clip(v, 0.0, 1.0)
    vp = v ** (1.0 / PQ_M2)
    num = np.maximum(vp - PQ_C1, 0.0)
    den = PQ_C2 - PQ_C3 * vp
    return PQ_PEAK_NITS * (num / den) ** (1.0 / PQ_M1)


# ── ICtCp（ITU-R BT.2124）矩阵（取自 colour-science 权威实现） ────────────

MATRIX_BT2020_TO_LMS = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
], dtype=np.float64) / 4096.0

MATRIX_LMS_P_TO_ICTCP = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
], dtype=np.float64) / 4096.0

# BT.709 → BT.2020 原色矩阵（ICtCp 要求 BT.2020 原色输入）
MATRIX_BT709_TO_BT2020 = np.array([
    [0.6274040, 0.3292820, 0.0433136],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955950],
], dtype=np.float64)


def rgb_linear_to_ictcp(rgb_linear_nits: np.ndarray) -> np.ndarray:
    """
    线性 RGB（BT.709 原色，nits）→ ICtCp。

    Returns:
        (H, W, 3) float64，通道顺序 I, Ct, Cp
    """
    rgb = rgb_linear_nits.astype(np.float64)
    # BT.709 原色 → BT.2020 原色（矩阵作用于线性值，缩放不变，可后做）
    rgb_2020 = rgb @ MATRIX_BT709_TO_BT2020.T
    # → LMS（ cones）
    lms = rgb_2020 @ MATRIX_BT2020_TO_LMS.T
    # PQ 编码 cone 响应（注意：LMS 值需按亮度缩放，PQ 以 10000 nits 为 1.0）
    lms_norm = np.clip(lms / PQ_PEAK_NITS, 0.0, 1.0)
    lms_p = pq_encode(lms_norm)
    # → ICtCp
    return lms_p @ MATRIX_LMS_P_TO_ICTCP.T


# ── PU21 感知均匀编码（Mantiuk & Azimi, PCS 2021，banding_glare 变体） ────

PU21_PARAMS = (
    0.353487901,        # p0
    0.3734658629,       # p1
    8.277049286e-05,    # p2
    0.9062562627,       # p3 (Y 指数)
    0.09150303166,      # p4 (缩放)
    0.9099517204,       # p5 (t1/t2 指数)
    596.3148142,        # p6 (缩放)
)
PU21_L_MIN = 0.005     # nits
PU21_L_MAX = 10000.0   # nits
PU21_NITS_AT_256 = 100.0  # 100 nits → 256 PU（对齐 SDR 指标输入域）


def pu21_encode(luminance_nits: np.ndarray) -> np.ndarray:
    """
    线性亮度（nits）→ PU21 编码（0 ≈ 595，100 nits → 256）。

    公式（与 gfxdisp/pu21 MATLAB 参考实现逐字一致）：
        V = max( p6 * (((p0 + p1*Y^p3) / (1 + p2*Y^p3))^p4 - p5), 0 )
    """
    p0, p1, p2, p3, p4, p5, p6 = PU21_PARAMS
    Y = np.clip(luminance_nits.astype(np.float64), PU21_L_MIN, PU21_L_MAX)
    y_pow = Y ** p3
    t1 = p0 + p1 * y_pow
    t2 = 1.0 + p2 * y_pow
    return np.maximum(p6 * ((t1 / t2) ** p4 - p5), 0.0)


# ── 基础指标 ───────────────────────────────────────────────────────────────

def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    """PSNR（dB）。peak 为数据域最大值。"""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)


def _box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """可分离盒式滤波（cumsum 实现，O(n)）。"""
    n = 2 * radius + 1
    if n <= 1:
        return img.copy()
    # 水平
    cum = np.cumsum(np.insert(img, 0, 0.0, axis=1), axis=1)
    out_h = (cum[:, n:] - cum[:, :-n]) / n
    # 垂直
    cum = np.cumsum(np.insert(out_h, 0, 0.0, axis=0), axis=0)
    out = (cum[n:, :] - cum[:-n, :]) / n
    return out


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """
    标准 SSIM（Wang et al. 2004），单通道灰度输入，8×8 窗、步长 1。
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    radius = 4

    mu_a = _box_filter(a, radius)
    mu_b = _box_filter(b, radius)
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a2 = _box_filter(a * a, radius) - mu_a2
    sigma_b2 = _box_filter(b * b, radius) - mu_b2
    sigma_ab = _box_filter(a * b, radius) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / \
               ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return float(np.mean(ssim_map))


# ── HDR 全参考指标 ─────────────────────────────────────────────────────────

def delta_ictcp_psnr(
    hdr_a_nits: np.ndarray,
    hdr_b_nits: np.ndarray,
    peak_nits: float = 10000.0,
) -> float:
    """
    ΔICtCp PSNR（ITU-R BT.2124 精神）：ICtCp 域 MSE，I 通道按 PQ 域归一。

    这是 HDR 图像/视频质量评价的标准客观指标（BT.2124 / Dolby 白皮书）。
    输入为线性 RGB（BT.709 原色，nits）。
    """
    ictcp_a = rgb_linear_to_ictcp(hdr_a_nits)
    ictcp_b = rgb_linear_to_ictcp(hdr_b_nits)
    # MSE 在 ICtCp（PQ 编码）域，域峰值 1.0
    mse = float(np.mean((ictcp_a - ictcp_b) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def pu21_metrics(
    hdr_a_nits: np.ndarray,
    hdr_b_nits: np.ndarray,
) -> Dict[str, float]:
    """
    PU21 PSNR / SSIM：PU21 编码后按 SDR 指标口径计算（100 nits → 256）。
    Mantiuk & Azimi (PCS 2021) 推荐的 HDR 指标适配方法。
    """
    ya = luminance(hdr_a_nits.astype(np.float64))
    yb = luminance(hdr_b_nits.astype(np.float64))
    va = pu21_encode(ya)
    vb = pu21_encode(yb)
    mse = float(np.mean((va - vb) ** 2))
    p = 10.0 * np.log10(256.0 ** 2 / mse) if mse > 0 else float("inf")
    # SSIM 需要 2D 输入（PU 域）
    return {
        "pu21_psnr": p,
        "pu21_ssim": ssim(va[:, :, 0], vb[:, :, 0], data_range=595.0),
    }


def linear_psnr(hdr_a_nits: np.ndarray, hdr_b_nits: np.ndarray,
                peak_nits: float = 10000.0) -> float:
    """线性光域 PSNR（nits 域，peak 为目标峰值亮度）。"""
    return psnr(hdr_a_nits, hdr_b_nits, peak=peak_nits)


# ── 无参考指标（SDR→HDR 生成任务） ─────────────────────────────────────────

def dynamic_range_stats(hdr_nits: np.ndarray) -> Dict[str, float]:
    """
    动态范围统计：以中位亮度为基准的 stops 扩展量。
    返回 p50/p90/max 相对中位数的 log2 提升（HDR headroom）。
    """
    y = luminance(hdr_nits.astype(np.float64))
    positive = y[y > 1e-4]
    if positive.size == 0:
        return {"dr_stops_p50": 0.0, "dr_stops_p90": 0.0, "dr_stops_max": 0.0}
    median = float(np.median(positive))
    log2y = np.log2(positive / median)
    return {
        "dr_stops_p50": float(np.percentile(log2y, 50)),
        "dr_stops_p90": float(np.percentile(log2y, 90)),
        "dr_stops_max": float(log2y.max()),
    }


def clip_rate(hdr_nits: np.ndarray, peak_nits: float) -> Dict[str, float]:
    """截断统计：达到/超过目标峰值的像素比例。"""
    y = luminance(hdr_nits.astype(np.float64))
    total = y.size
    at_peak = float(np.mean(y >= peak_nits * 0.999))
    over = float(np.mean(y > peak_nits))
    return {"clip_at_peak": at_peak, "clip_over": over}


def boost_stats(hdr_nits: np.ndarray, sdr_linear: np.ndarray,
                sdr_white: float = 203.0) -> Dict[str, float]:
    """
    提升量统计：HDR 相对 SDR 的 log2 增益分布（分暗部/中间调/高光）。
    衡量 ITM 是否在全域单调可感知提升（暗部无提升 = 观感发灰）。
    """
    y_sdr = luminance(sdr_linear.astype(np.float64)) * sdr_white
    y_hdr = luminance(hdr_nits.astype(np.float64))
    ratio = y_hdr / np.maximum(y_sdr, 1e-4)
    log2_boost = np.log2(np.maximum(ratio, 1e-6))
    # 按 SDR 亮度分层
    lo = y_sdr < np.percentile(y_sdr, 25)
    mid = (y_sdr >= np.percentile(y_sdr, 25)) & (y_sdr <= np.percentile(y_sdr, 75))
    hi = y_sdr > np.percentile(y_sdr, 75)
    return {
        "boost_shadows": float(np.mean(log2_boost[lo])) if lo.any() else 0.0,
        "boost_mids": float(np.mean(log2_boost[mid])) if mid.any() else 0.0,
        "boost_highlights": float(np.mean(log2_boost[hi])) if hi.any() else 0.0,
    }


# ── 色调映射（回环/预览用） ────────────────────────────────────────────────

def tonemap_reinhard(hdr_nits: np.ndarray, peak_nits: float = 1000.0,
                     sdr_white: float = 203.0) -> np.ndarray:
    """
    Reinhard 扩展色调映射 → SDR sRGB [0,1]。

    display-referred：SDR 白（203 nits）映射到 1.0，>1 的部分 Reinhard 压缩，
    最后过 sRGB OETF。用于回环一致性：与原始 sRGB 图同域比较。
    """
    from .itm import linear_to_srgb
    x = hdr_nits / sdr_white
    mapped = x / (1.0 + x)
    return np.clip(linear_to_srgb(mapped), 0.0, 1.0)


def tonemap_hable(hdr_nits: np.ndarray, peak_nits: float = 1000.0,
                  sdr_white: float = 203.0) -> np.ndarray:
    """Hable/Uncharted2 色调映射 → SDR sRGB [0,1]（带 Shoulder/Toe，电影感）。"""
    from .itm import linear_to_srgb
    a, b, c, d, e = 0.15, 0.50, 0.10, 0.20, 0.02
    f, w = 0.50, peak_nits / 1000.0  # exposure 缩放

    def curve(x):
        return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f

    x = hdr_nits / sdr_white * f
    white_scale = 1.0 / curve(np.array([w]))[0]
    return np.clip(linear_to_srgb(curve(x) * white_scale), 0.0, 1.0)


def sdr_display_render(hdr_nits: np.ndarray, sdr_white: float = 203.0) -> np.ndarray:
    """
    SDR 显示器渲染 HDR 的结果（硬截断在 SDR 白，过 sRGB OETF）。

    这是 UltraHDR 生态里"同一文件在 SDR 屏上的样子"的保守下界估计：
    截断法不做任何高光滚落，衡量 HDR 是否在 SDR 域保持原图外观。
    """
    from .itm import linear_to_srgb
    x = np.clip(hdr_nits / sdr_white, 0.0, 1.0)
    return np.clip(linear_to_srgb(x), 0.0, 1.0)


# ── VMAF（ffmpeg libvmaf） ────────────────────────────────────────────────

def ffmpeg_vmaf_available() -> bool:
    """检测 ffmpeg 是否带 libvmaf。"""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30)
        return "libvmaf" in out.stdout
    except Exception:
        return False


def run_vmaf(reference: np.ndarray, distorted: np.ndarray,
             frames: int = 30) -> Optional[float]:
    """
    对两张 uint8 RGB 图计算 VMAF 分数（0-100）。reference 为参考，distorted 为被测。

    实现要点（ffmpeg 9.0.2 + libvmaf 实测结论，全部已验证）：
    1. **必须多帧输入**：单帧图像直接喂 libvmaf，帧同步行为异常
       （分数不可信）。用 rawvideo 把静态图展开为 30 帧后行为一致。
    2. **方向敏感（已知模型特性）**：VMAF 对全局色调/亮度偏移不对称
       （实测同一对交换 ref/distorted：100.0 vs 64.3）。故本工具统一约定
       **reference = 原始 SDR，distorted = tonemap(HDR)**——衡量"HDR 的 SDR
       渲染相对原图有无感知损失"，100 = 无感知损失。
    3. **对低频率失真不敏感（设计使然）**：平滑色调变化（如 tonemap 的提升曲线）
       即使 PSNR 只有 20dB 也可能得 100——VMAF 是感知指标，此时应结合
       SSIM/PSNR（像素级）与 PU21/ΔICtCp（HDR 级）一起解读。
    4. **静态内容地板**：完全相同的静态图得 ~97.43 而非 100（模型以运动内容训练），
       解读时 ≥97.4 视为完美保真。
    5. **可检测真实失真**：局部损坏 +60 → 86.5，JPEG q30 → 85.6（有区分度）。

    大图会自动降采样到 ≤960×540（VMAF 标准做法，同时控制 raw 临时文件大小）。
    需要 ffmpeg + libvmaf；不可用时返回 None。
    """
    if not ffmpeg_vmaf_available():
        return None

    from PIL import Image  # noqa: F401

    ref = np.asarray(reference, dtype=np.uint8)
    dis = np.asarray(distorted, dtype=np.uint8)
    if ref.shape != dis.shape:
        return None
    h, w = ref.shape[:2]

    # 降采样控制（VMAF 标准实践 + raw 文件大小控制）
    max_w, max_h = 960, 540
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        from PIL import Image as _Img
        ref = np.asarray(_Img.fromarray(ref).resize((nw, nh), _Img.LANCZOS))
        dis = np.asarray(_Img.fromarray(dis).resize((nw, nh), _Img.LANCZOS))
        h, w = nh, nw

    tmpdir = Path(tempfile.mkdtemp(prefix="vmaf_"))
    try:
        raw_a = tmpdir / "a.raw"
        raw_b = tmpdir / "b.raw"
        raw_a.write_bytes(np.tile(ref[None], (frames, 1, 1, 1)).tobytes())
        raw_b.write_bytes(np.tile(dis[None], (frames, 1, 1, 1)).tobytes())

        cmd = [
            "ffmpeg", "-hide_banner",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-framerate", "30", "-i", str(raw_a),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-framerate", "30", "-i", str(raw_b),
            "-lavfi", "libvmaf",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in proc.stderr.splitlines():
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return None


def roundtrip_vmaf(
    hdr_nits: np.ndarray,
    sdr_ref_uint8: np.ndarray,
    tonemap: str = "reinhard",
) -> Optional[float]:
    """
    回环一致性 VMAF：tonemap(HDR) → SDR 8bit sRGB，与原 SDR 比 VMAF。

    这是 SDR→HDR 无 GT 情形下的核心客观指标（方向约定见 run_vmaf）：
      - 分数高（≈97.4+）→ HDR 在 SDR 显示器上呈现原图外观（提升保真）
      - 分数低 → 提升过程改变了 SDR 外观（过增强/色偏/细节丢失）

    注意 VMAF 对平滑色调变化不敏感（可能给 100），需结合 SSIM/PSNR 解读；
    详见 run_vmaf 文档中的实测结论。
    """
    if tonemap == "hable":
        sdr_float = tonemap_hable(hdr_nits)
    elif tonemap == "clip":
        sdr_float = sdr_display_render(hdr_nits)
    else:
        sdr_float = tonemap_reinhard(hdr_nits)
    test_uint8 = np.clip(sdr_float * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return run_vmaf(np.asarray(sdr_ref_uint8, dtype=np.uint8), test_uint8)


def extract_ultrahdr_base(jpeg_path: str) -> Optional[np.ndarray]:
    """
    提取 UltraHDR/ISO21496 JPEG 的基础（SDR）图像为 uint8 RGB 数组。

    MPF 主图即 SDR 回退图——直接解码到 SOS 前的 baseline 部分。
    Pillow 打开 UltraHDR JPEG 默认就返回 baseline（忽略增益图），
    因此这里用 Pillow 解码即可（与 HDR 显示器忽略附加数据的行为一致）。
    """
    from PIL import Image
    try:
        img = Image.open(jpeg_path)
        img.load()
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def base_image_fidelity(jpeg_path: str, sdr_ref_uint8: np.ndarray) -> Dict[str, float]:
    """
    UltraHDR 的 SDR 回退保真度：提取 baseline 与原图比（SSIM/PSNR）。

    这是 UltraHDR 格式的核心产品指标——SDR 设备上看到的就是 baseline，
    它必须与原图一致（我们的 ITM 不碰 baseline，理论满分）。
    """
    base = extract_ultrahdr_base(jpeg_path)
    if base is None:
        return {"base_SSIM": 0.0, "base_PSNR_dB": 0.0}
    if base.shape != sdr_ref_uint8.shape:
        # MPF 里 baseline 可能被重编码但尺寸不变；尺寸不一致时报 0
        return {"base_SSIM": 0.0, "base_PSNR_dB": 0.0}
    s = ssim(base[:, :, 0].astype(np.float64), sdr_ref_uint8[:, :, 0].astype(np.float64))
    p = psnr(base.astype(np.float64), sdr_ref_uint8.astype(np.float64), 255.0)
    return {"base_SSIM": round(s, 4), "base_PSNR_dB": round(p, 2)}


# ── 综合入口 ───────────────────────────────────────────────────────────────

def full_reference_metrics(
    hdr_test_nits: np.ndarray,
    hdr_ref_nits: np.ndarray,
    peak_nits: float = 1000.0,
) -> Dict[str, float]:
    """全参考指标组（有 GT HDR）。"""
    out: Dict[str, float] = {
        "dICtCp_PSNR_dB": delta_ictcp_psnr(hdr_test_nits, hdr_ref_nits),
        "linear_PSNR_dB": linear_psnr(hdr_test_nits, hdr_ref_nits, peak_nits),
    }
    out.update(pu21_metrics(hdr_test_nits, hdr_ref_nits))
    return out


def no_reference_metrics(
    hdr_nits: np.ndarray,
    sdr_uint8: np.ndarray,
    sdr_linear: np.ndarray,
    peak_nits: float = 1000.0,
    sdr_white: float = 203.0,
    with_vmaf: bool = True,
) -> Dict[str, float]:
    """
    无参考指标组（SDR→HDR 生成任务默认情形）：
    回环一致性 + 动态范围 + 截断 + 提升分布。

    hdr_nits 用于 EXR/PNG 路径（无内嵌 SDR 基图，需 tonemap 回环）。
    JPEG/UltraHDR 路径请用 base_image_fidelity()（baseline 即 SDR 回退）。
    """
    out: Dict[str, float] = {}
    # 回环：tonemap(HDR) vs 原 SDR（Reinhard，display-referred + sRGB OETF）
    tm = tonemap_reinhard(hdr_nits, peak_nits, sdr_white)
    rt_ssim = ssim(
        tm[:, :, 0] * 255.0,
        sdr_uint8[:, :, 0].astype(np.float64), data_range=255.0)
    out["roundtrip_SSIM"] = rt_ssim
    rt_mse = float(np.mean((tm * 255.0 - sdr_uint8.astype(np.float64)) ** 2))
    out["roundtrip_PSNR_dB"] = 10.0 * np.log10(255.0 ** 2 / rt_mse) if rt_mse > 0 else float("inf")
    if with_vmaf:
        v = roundtrip_vmaf(hdr_nits, sdr_uint8)
        if v is not None:
            out["roundtrip_VMAF"] = v
    # 动态范围与截断
    out.update(dynamic_range_stats(hdr_nits))
    out.update(clip_rate(hdr_nits, peak_nits))
    # 提升分布
    out.update(boost_stats(hdr_nits, sdr_linear, sdr_white))
    return out