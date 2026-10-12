"""共享的逆色调映射（ITM）与色彩工具。

所有后端的 SDR→HDR 数学核心都从这里取，保证：
- 不同后端在同一输入下产生可比较的 HDR 输出
- 增益图方法语义一致（auto/linear/log/fixed）
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# BT.709 → BT.2020（线性光），用于增益图计算的色域扩展
BT709_TO_BT2020 = np.array([
    [0.6274040, 0.3292820, 0.0433136],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955950],
], dtype=np.float32)

BT2020_TO_BT709 = np.linalg.inv(BT709_TO_BT2020.astype(np.float64)).astype(np.float32)

_LUM_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# ITM 方法名（CLI --gain-map-method 可选值）
ITM_METHODS = ("auto", "linear", "log", "fixed")


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB EOTF 逆函数（[0,1] → 线性 [0,1]）。"""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """sRGB OETF（线性 [0,1] → [0,1]）。"""
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 亮度（H×W×3 → H×W×1）。"""
    return (rgb * _LUM_WEIGHTS).sum(-1, keepdims=True)


def inverse_tone_map(
    rgb_lin: np.ndarray,
    sdr_white: float = 203.0,
    peak_nits: float = 1000.0,
    method: str = "auto",
    shadow_boost: float = 1.3,
    hi_gamma: float = 1.0,
) -> np.ndarray:
    """
    SDR 线性光 → HDR (nits)，保持 BT.709 原色域。

    method:
      auto   — Banterle 式：boost(L) = shadow_boost + (peak_boost-shadow_boost)·L^γ（默认）
      linear — 纯比例提升：boost(L) = peak_boost·L（暗部提升小、无阴影地板）
      log    — 对数压缩：boost(L) = peak_boost·log2(1+L)（高光滚落更柔和）
      fixed  — 固定增益：boost = peak_boost（增益图平坦，等效整体曝光提升）
    """
    if method not in ITM_METHODS:
        raise ValueError(f"未知 ITM 方法: {method}（可选: {', '.join(ITM_METHODS)}）")
    L = np.clip(luminance(rgb_lin), 1e-6, 1.0)
    peak_boost = peak_nits / sdr_white

    if method == "linear":
        boost = peak_boost * L
    elif method == "log":
        boost = peak_boost * np.log2(1.0 + L)
    elif method == "fixed":
        boost = np.full_like(L, peak_boost)
    else:  # auto
        boost = shadow_boost + (peak_boost - shadow_boost) * L ** hi_gamma
    return rgb_lin * sdr_white * boost


def compute_gain_map(
    sdr_lin: np.ndarray,
    hdr_nits_bt2020: np.ndarray,
    sdr_white: float,
) -> Tuple[np.ndarray, float, float]:
    """
    单通道亮度增益图（log2 域）。

    Returns:
        (gain_log2 H×W float32, gain_min, gain_max)，约定 gain>=0（只提亮不压暗）
    """
    hdr_norm = hdr_nits_bt2020 / sdr_white
    L_sdr = np.clip(luminance(sdr_lin), 1e-4, None)
    L_hdr = np.clip(luminance(hdr_norm), 1e-4, None)
    gain = np.maximum(np.log2(L_hdr) - np.log2(L_sdr), 0.0)
    return gain.astype(np.float32), 0.0, float(gain.max())


def to_bt2020(hdr_bt709_nits: np.ndarray, peak_nits: float) -> np.ndarray:
    """BT.709 线性 nits → BT.2020 线性 nits（色域扩展，截断到峰值）。"""
    return np.clip(hdr_bt709_nits @ BT709_TO_BT2020.T, 0.0, peak_nits)


def sdr_uint8_to_linear(sdr_uint8: np.ndarray) -> np.ndarray:
    """uint8 sRGB 数组 → 线性 float32 [0,1]。"""
    return srgb_to_linear(sdr_uint8.astype(np.float32) / 255.0)