#!/usr/bin/env python3
"""bench_photo_sdr_to_hdr.py — photo_sdr_to_hdr 综合基准测试

覆盖维度：
  A. 后端可用性矩阵（自动跳过未安装依赖的后端）
  B. 输出格式矩阵（exr/jpg/png/ultrahdr/gainmap × 结构合法性 + 质量指标）
  C. 增益图方法（auto/linear/log/fixed）— 动态范围与提升分布对比
  D. 目标峰值亮度（500/1000/2000/4000 nits）— 峰值与截断验证
  E. EXIF 保留矩阵（keep/no-keep × sidecar on/off × 格式）
  F. 并行扩展（workers 1/4/auto — 吞吐对比）
  G. 干跑安全性（零文件写入）
  H. 确定性（同输入两次转换输出字节一致）
  I. 客观质量指标（无 GT）：
       - Round-trip VMAF（ffmpeg libvmaf，tonemap(HDR) vs 原 SDR）
       - Round-trip SSIM / PSNR
       - 动态范围 stops、截断率、分域提升量
  J. 客观质量指标（有合成 GT HDR）：
       - ΔICtCp PSNR（ITU-R BT.2124）
       - PU21 PSNR / SSIM（Mantiuk & Azimi, PCS 2021）
  K. 资源占用（子进程 ru_maxrss 峰值内存）

用法：
    python bench_photo_sdr_to_hdr.py                    # 全量（自动跳过不可用项）
    python bench_photo_sdr_to_hdr.py --quick            # 只跑冒烟子集
    python bench_photo_sdr_to_hdr.py --backends pyultrahdr hdrconv
    python bench_photo_sdr_to_hdr.py --report-dir ./bench_out
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from backends import BackendRegistry, create_backend, list_backends
from utils.exr_io import read_exr
from utils.itm import srgb_to_linear, inverse_tone_map, sdr_uint8_to_linear
from utils.hdr_metrics import (
    delta_ictcp_psnr, pu21_metrics, ssim, psnr,
    dynamic_range_stats, clip_rate, boost_stats,
    tonemap_reinhard, roundtrip_vmaf, ffmpeg_vmaf_available,
    base_image_fidelity,
)
from utils.logging import setup_logging, get_logger
from utils.logging import log as log_msg

setup_logging(log_file="-")
log = log_msg  # 模块级 log(msg) 函数（photo_dedup.py 同款风格）

CLI = str(Path(__file__).parent / "photo_sdr_to_hdr.py")


# ═══════════════════════════════════════════════════════════════════
#  测试素材生成（固定种子，确定性）
# ═══════════════════════════════════════════════════════════════════

def make_test_image(width: int = 512, height: int = 384, seed: int = 42) -> np.ndarray:
    """
    生成结构化测试图：渐变 + 亮斑（模拟太阳）+ 纹理噪声。
    固定种子保证确定性；同时带 EXIF 便于 E 组测试。

    尺寸 512×384 + σ=8 纹理：VMAF 对无纹理平滑内容特征退化（会虚高到 100），
    必须带足够纹理才能得到有区分度的分数（实测结论）。
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = x / (width - 1) * 255.0            # R 水平渐变
    img[..., 1] = y / (height - 1) * 255.0           # G 垂直渐变
    img[..., 2] = 110.0                              # B 常量
    # 太阳亮斑（右上）
    cx, cy = width * 0.75, height * 0.25
    r = min(width, height) * 0.15
    mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
    img[mask] = [255.0, 252.0, 235.0]
    # 传感器噪声纹理（VMAF 需要纹理才能区分失真）
    img += rng.normal(0, 8.0, img.shape)
    # 细密纹理（模拟细节）
    img += (np.sin(x * 1.7) * np.cos(y * 1.3) * 6.0)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def save_test_image(arr: np.ndarray, path: str, with_exif: bool = True) -> None:
    """保存测试图，可选带 EXIF。"""
    im = Image.fromarray(arr)
    if with_exif:
        exif = Image.Exif()
        exif[0x010F] = "BenchCam"
        exif[0x0110] = "Bench-Model"
        exif[0x8827] = 200
        exif[0x829A] = (1, 250)
        exif[0x829D] = (18, 10)
        exif[0x920A] = (35, 1)
        im.save(path, quality=92, exif=exif.tobytes())
    else:
        im.save(path, quality=92)


# ═══════════════════════════════════════════════════════════════════
#  结果收集
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    group: str          # 测试组（A~K）
    name: str           # 测试名
    backend: str
    passed: bool
    skipped: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    note: str = ""


class Bench:
    """基准执行器。"""

    def __init__(self, workdir: Path, quick: bool = False):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.quick = quick
        self.results: List[BenchResult] = []

    def run(self, group: str, name: str, backend: str, fn: Callable[[], Dict]) -> BenchResult:
        t0 = time.perf_counter()
        try:
            metrics = fn()
            passed = metrics.pop("_passed", True)
            skipped = metrics.pop("_skipped", False)
            result = BenchResult(group, name, backend, passed, skipped,
                                 metrics, time.perf_counter() - t0)
        except Exception as e:
            result = BenchResult(group, name, backend, False, False,
                                 {"error": f"{type(e).__name__}: {e}"},
                                 time.perf_counter() - t0)
            if os.environ.get("BENCH_TRACEBACK"):
                traceback.print_exc()
        status = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
        metric_str = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in list(result.metrics.items())[:6])
        log(f"[{status}] {group}.{name} [{backend}] {metric_str}"
            + (f"  ({result.duration_s:.2f}s)" if result.duration_s > 0.5 else ""))
        self.results.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════
#  测试组实现
# ═══════════════════════════════════════════════════════════════════

def group_A_availability(bench: Bench, backends: List[str]) -> Dict[str, Tuple[bool, str]]:
    """A. 后端可用性矩阵（不可用 = 环境缺依赖/模型，记 SKIP 而非 FAIL）。"""
    avail = {}
    for name in backends:
        def check(n=name):
            be = create_backend(n)
            ok, why = be.check_available()
            avail[n] = (ok, why)
            if ok:
                return {"_passed": True, "reason": why}
            return {"_skipped": True, "reason": why}
        bench.run("A", "availability", name, check)
    return avail


def _convert_via_backend(backend: str, input_path: str, output_path: str,
                         tmpdir: Path, **kwargs) -> bool:
    """直接调用后端 API 转换（不走 CLI 进程池）。"""
    be = create_backend(backend, model_dir=tmpdir / "models")
    ok, why = be.check_available()
    if not ok:
        raise RuntimeError(f"backend unavailable: {why}")
    be.prepare("cpu", **{k: v for k, v in kwargs.items()
                         if k in ("quantization", "variant")})
    return be.convert(input_path, output_path, **kwargs)


def group_B_formats(bench: Bench, backends: List[str], img_path: str,
                    tmpdir: Path) -> None:
    """B. 输出格式矩阵：结构合法性 + 基础质量指标。"""
    formats = ["exr", "jpg", "png", "ultrahdr", "gainmap"]
    for backend in backends:
        # 后端能力矩阵：不支持的格式跳过
        from backends import create_backend as _cb
        be_probe = _cb(backend)
        supported = set(be_probe.output_formats)
        for fmt in formats:
            if fmt not in supported:
                bench.run("B", f"format_{fmt}", backend,
                          lambda f=fmt: {"_skipped": True,
                                         "note": f"{backend} 不支持 {f}"})
                continue
            def check(b=backend, f=fmt):
                out = tmpdir / f"fmt_{f}.out"
                out.parent.mkdir(parents=True, exist_ok=True)
                # ultrahdr/gainmap 容器 → .jpg 扩展名
                ext = ".jpg" if f in ("ultrahdr", "gainmap") else f".{f}"
                out = out.with_suffix(ext)
                container = "iso21496" if f == "gainmap" else "ultrahdr"
                ok = _convert_via_backend(
                    b, img_path, str(out), tmpdir,
                    output_format=f, container=container,
                    keep_exif=True, sidecar=True, target_nits=1000,
                )
                if not ok:
                    return {"_passed": False, "note": "convert returned False"}
                if not out.exists():
                    return {"_passed": False, "note": "output missing"}

                metrics: Dict[str, Any] = {"size_kb": round(out.stat().st_size / 1024, 1)}

                if f == "exr":
                    rgb, attrs = read_exr(str(out))
                    metrics["exr_max_nits"] = round(float(rgb.max()), 1)
                    metrics["exr_shape"] = list(rgb.shape)
                    # 学习型后端峰值可能低于 ITM 目标，>250 nits 即真实 HDR
                    metrics["_passed"] = bool(rgb.max() > 250)
                elif f in ("jpg", "ultrahdr", "gainmap"):
                    data = out.read_bytes()
                    segments = _jpeg_segments(data)
                    has_xmp = any(k == "XMP APP1" for _, k, _ in segments)
                    has_mpf = any(k.startswith("MPF") for _, k, _ in segments)
                    has_exif = any(k == "EXIF APP1" for _, k, _ in segments)
                    metrics["has_xmp"] = has_xmp
                    metrics["has_mpf"] = has_mpf
                    metrics["has_exif"] = has_exif
                    if f == "jpg":
                        # 普通 JPEG：只需是合法 JPEG
                        metrics["_passed"] = True
                    elif f == "ultrahdr":
                        # UltraHDR：MPF + XMP(hdrgm)
                        metrics["_passed"] = bool(has_mpf and has_xmp)
                    else:  # gainmap = ISO 21496-1：MPF + ISO APP2（不用 XMP）
                        metrics["_passed"] = bool(has_mpf)
                elif f == "png":
                    im = Image.open(out)
                    metrics["png_mode"] = im.mode
                    metrics["_passed"] = im.mode in ("RGB", "I;16", "RGB;16B")

                # sidecar 检查
                sidecar = out.with_name(out.stem + ".exif.json")
                metrics["sidecar"] = sidecar.exists()
                return metrics
            bench.run("B", f"format_{fmt}", backend, check)


def _jpeg_segments(data: bytes) -> List[Tuple[int, str, int]]:
    """扫描 JPEG 段，返回 [(offset, kind, length)]。"""
    segs = []
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return segs
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        if marker in (0xD8, 0xD9, 0xDA) or 0xD0 <= marker <= 0xD7:
            if marker == 0xD9:
                break
            if marker == 0xDA:
                segs.append((pos, "SOS", 0))
                break
            pos += 2
            continue
        seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        seg_data = data[pos + 4:pos + 2 + seg_len]
        if marker == 0xE1:
            kind = "EXIF APP1" if seg_data.startswith(b"Exif\x00\x00") else "XMP APP1"
        elif marker == 0xE2:
            kind = "MPF APP2" if seg_data.startswith(b"MPF\x00") else "APP2"
        else:
            kind = f"marker {marker:#04x}"
        segs.append((pos, kind, seg_len))
        pos += 2 + seg_len
    return segs


def group_C_gain_methods(bench: Bench, backends: List[str], img_path: str,
                         tmpdir: Path) -> None:
    """C. 增益图方法对比（动态范围/提升分布）。"""
    methods = ["auto"] if bench.quick else ["auto", "linear", "log", "fixed"]
    for backend in backends:
        # 能力检查：C/D 组走 EXR，不支持的跳过
        from backends import create_backend as _cb
        if "exr" not in _cb(backend).output_formats:
            for method in methods:
                bench.run("C", f"method_{method}", backend,
                          lambda m=method: {"_skipped": True,
                                            "note": f"{backend} 不支持 exr 输出"})
            continue
        for method in methods:
            def check(b=backend, m=method):
                out = tmpdir / f"method_{m}.exr"
                ok = _convert_via_backend(
                    b, img_path, str(out), tmpdir,
                    output_format="exr", gain_map_method=m,
                    target_nits=1000, keep_exif=False, sidecar=False)
                if not ok:
                    return {"_passed": False}
                rgb, _ = read_exr(str(out))
                sdr_u8 = np.asarray(Image.open(img_path).convert("RGB"))
                sdr_lin = sdr_uint8_to_linear(sdr_u8)
                dr = dynamic_range_stats(rgb)
                bs = boost_stats(rgb, sdr_lin)
                return {
                    "_passed": True,
                    "peak_nits": round(float(rgb.max()), 0),
                    "dr_max_stops": round(dr["dr_stops_max"], 2),
                    "boost_mids": round(bs["boost_mids"], 2),
                }
            bench.run("C", f"method_{method}", backend, check)


def group_D_target_nits(bench: Bench, backends: List[str], img_path: str,
                        tmpdir: Path) -> None:
    """D. 目标峰值亮度验证。"""
    targets = [1000] if bench.quick else [500, 1000, 2000, 4000]
    for backend in backends:
        # 能力检查：D 组走 EXR
        from backends import create_backend as _cb
        if "exr" not in _cb(backend).output_formats:
            for nits in targets:
                bench.run("D", f"target_{nits}nits", backend,
                          lambda n=nits: {"_skipped": True,
                                          "note": f"{backend} 不支持 exr 输出"})
            continue
        for nits in targets:
            def check(b=backend, n=nits):
                out = tmpdir / f"nits_{n}.exr"
                ok = _convert_via_backend(
                    b, img_path, str(out), tmpdir,
                    output_format="exr", target_nits=n,
                    keep_exif=False, sidecar=False)
                if not ok:
                    return {"_passed": False}
                rgb, _ = read_exr(str(out))
                peak = float(rgb.max())
                cr = clip_rate(rgb, n)
                # 学习型后端（如 GMNet）不严格跟随 target_nits，
                # 只验证"产生了真实 HDR 提升"（>250 nits）且不超目标 10%
                return {
                    "_passed": 250 < peak <= n * 1.10,
                    "target": n,
                    "actual_peak": round(peak, 0),
                    "clip_at_peak": round(cr["clip_at_peak"], 4),
                }
            bench.run("D", f"target_{nits}nits", backend, check)


def group_E_exif(bench: Bench, backends: List[str], img_path: str,
                 tmpdir: Path) -> None:
    """E. EXIF 保留矩阵。"""
    cases = [("jpg", True, True), ("exr", True, True), ("png", True, True),
             ("jpg", False, False)]
    if bench.quick:
        cases = [("jpg", True, True), ("jpg", False, False)]
    for backend in backends:
        # 能力检查：跳过不支持的格式
        from backends import create_backend as _cb
        supported = set(_cb(backend).output_formats)
        cases_b = [(f, k, s) for f, k, s in cases
                   if f in supported or (f == "jpg" and "jpg" in supported)]
        for fmt, keep, sidecar in cases_b:
            def check(b=backend, f=fmt, k=keep, s=sidecar):
                out = tmpdir / f"exif_{f}_{k}_{s}.{f}"
                ok = _convert_via_backend(
                    b, img_path, str(out), tmpdir,
                    output_format=f, keep_exif=k, sidecar=s,
                    target_nits=1000)
                if not ok:
                    return {"_passed": False, "note": "convert failed"}

                sidecar_path = out.with_name(out.stem + ".exif.json")
                if k and s:
                    if not sidecar_path.exists():
                        return {"_passed": False, "note": "sidecar missing"}
                if not s and sidecar_path.exists():
                    return {"_passed": False, "note": "sidecar should not exist"}

                if not k:
                    # 不保留时：输出不应含 EXIF
                    if f == "exr":
                        _, attrs = read_exr(str(out))
                        if any(a in attrs for a in ("cameraMake", "isoSpeed")):
                            return {"_passed": False, "note": "EXIF leaked"}
                    return {"_passed": True, "note": "no-exif OK"}

                # 保留时：验证字段
                if f == "exr":
                    _, attrs = read_exr(str(out))
                    ok_fields = (attrs.get("cameraMake") == "BenchCam"
                                 and attrs.get("isoSpeed") == "200")
                    return {"_passed": ok_fields, "cameraMake": attrs.get("cameraMake", "?"),
                            "isoSpeed": attrs.get("isoSpeed", "?")}
                elif f == "png":
                    data = out.read_bytes()
                    pos, chunks = 8, []
                    while pos < len(data):
                        ln = struct.unpack(">I", data[pos:pos+4])[0]
                        chunks.append(data[pos+4:pos+8].decode("ascii"))
                        pos += 12 + ln
                    return {"_passed": "eXIf" in chunks, "chunks": chunks}
                else:  # jpg
                    data = out.read_bytes()
                    segs = _jpeg_segments(data)
                    has_exif = any(k2 == "EXIF APP1" for _, k2, _ in segs)
                    return {"_passed": has_exif, "has_exif": has_exif}
            bench.run("E", f"exif_{fmt}_keep{keep}_side{sidecar}", backend, check)


def group_F_workers(bench: Bench, img_dir: str, tmpdir: Path,
                    backend: str = "pyultrahdr") -> None:
    """F. 并行扩展：子进程 CLI 计时（workers 1/4/auto）。"""
    configs = [1] if bench.quick else [1, 4, 0]
    for workers in configs:
        def check(w=workers):
            out_dir = tmpdir / f"workers_{w}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            cmd = [
                sys.executable, CLI,
                "-i", img_dir, "-o", str(out_dir),
                "--models", backend, "--format", "exr",
                "--apply", "--workers", str(w), "--no-sidecar",
            ]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.perf_counter() - t0
            n_out = len(list(out_dir.glob("*.exr"))) if out_dir.exists() else 0
            return {
                "_passed": proc.returncode == 0 and n_out > 0,
                "workers": w,
                "elapsed_s": round(elapsed, 2),
                "outputs": n_out,
                "throughput": round(n_out / elapsed, 2),
            }
        bench.run("F", f"workers_{workers}", backend, check)


def group_G_dryrun(bench: Bench, img_dir: str, tmpdir: Path,
                   backend: str = "pyultrahdr") -> None:
    """G. 干跑安全性。"""
    def check():
        out_dir = tmpdir / "dryrun_out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        cmd = [sys.executable, CLI, "-i", img_dir, "-o", str(out_dir),
               "--models", backend, "--format", "exr"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        leaked = list(out_dir.rglob("*")) if out_dir.exists() else []
        return {
            "_passed": proc.returncode == 0 and not leaked,
            "rc": proc.returncode,
            "files_written": len(leaked),
        }
    bench.run("G", "dry_run_zero_writes", backend, check)


def group_H_determinism(bench: Bench, backends: List[str], img_path: str,
                        tmpdir: Path) -> None:
    """H. 确定性：同输入两次 → 字节一致。"""
    from backends import create_backend as _cb
    for backend in backends:
        if "exr" not in _cb(backend).output_formats:
            bench.run("H", "determinism", backend,
                      lambda: {"_skipped": True, "note": f"{backend} 不支持 exr 输出"})
            continue
        def check(b=backend):
            out1 = tmpdir / f"det1_{b}.exr"
            out2 = tmpdir / f"det2_{b}.exr"
            for o in (out1, out2):
                ok = _convert_via_backend(
                    b, img_path, str(o), tmpdir,
                    output_format="exr", keep_exif=False, sidecar=False,
                    target_nits=1000)
                if not ok:
                    return {"_passed": False, "note": "convert failed"}
            same = out1.read_bytes() == out2.read_bytes()
            return {"_passed": same, "bytes": out1.stat().st_size}
        bench.run("H", "determinism", backend, check)


def group_I_quality_noref(bench: Bench, backends: List[str], img_path: str,
                          tmpdir: Path) -> None:
    """I. 无参考客观质量：
    - EXR 路径：Round-trip VMAF + SSIM/PSNR + DR 统计（tonemap 回环）
    - JPEG/UltraHDR 路径：baseline 保真度（MPF 主图 vs 原图）
    """
    sdr_u8 = np.asarray(Image.open(img_path).convert("RGB"))
    has_vmaf = ffmpeg_vmaf_available()
    for backend in backends:
        # 能力检查：I/J 组走 EXR
        from backends import create_backend as _cb
        if "exr" not in _cb(backend).output_formats:
            bench.run("I", "quality_noref", backend,
                      lambda: {"_skipped": True, "note": f"{backend} 不支持 exr 输出"})
            continue
        def check(b=backend, hv=has_vmaf):
            # ── EXR 路径（无内嵌 SDR 基图，需 tonemap 回环） ──
            out = tmpdir / f"quality_{b}.exr"
            ok = _convert_via_backend(
                b, img_path, str(out), tmpdir,
                output_format="exr", keep_exif=False, sidecar=False,
                target_nits=1000)
            if not ok:
                return {"_passed": False, "note": "convert failed"}
            rgb, _ = read_exr(str(out))
            sdr_lin = sdr_uint8_to_linear(sdr_u8)

            tm = tonemap_reinhard(rgb)
            rt_ssim = ssim(tm[:, :, 0] * 255.0, sdr_u8[:, :, 0].astype(np.float64))
            rt_psnr = psnr(tm * 255.0, sdr_u8.astype(np.float64), 255.0)
            metrics: Dict[str, Any] = {
                "_passed": True,
                "rt_SSIM": round(rt_ssim, 4),
                "rt_PSNR_dB": round(rt_psnr, 2),
            }
            if hv:
                vmaf = roundtrip_vmaf(rgb, sdr_u8)
                if vmaf is not None:
                    metrics["rt_VMAF"] = round(vmaf, 2)
            dr = dynamic_range_stats(rgb)
            metrics["dr_stops_max"] = round(dr["dr_stops_max"], 2)
            bs = boost_stats(rgb, sdr_lin)
            metrics["boost_shadows"] = round(bs["boost_shadows"], 2)
            metrics["boost_highlights"] = round(bs["boost_highlights"], 2)

            # ── JPEG/UltraHDR 路径（baseline = SDR 回退，应保真） ──
            jpg_out = tmpdir / f"quality_{b}.jpg"
            ok2 = _convert_via_backend(
                b, img_path, str(jpg_out), tmpdir,
                output_format="ultrahdr", container="ultrahdr",
                keep_exif=False, sidecar=False, target_nits=1000)
            if ok2:
                fid = base_image_fidelity(str(jpg_out), sdr_u8)
                metrics["base_SSIM"] = fid["base_SSIM"]
                metrics["base_PSNR_dB"] = fid["base_PSNR_dB"]
            return metrics
        bench.run("I", "quality_noref", backend, check)


def group_J_quality_fullref(bench: Bench, backends: List[str], img_path: str,
                            tmpdir: Path) -> None:
    """J. 全参考客观质量（合成 GT = 已知 ITM 曲线）：ΔICtCp PSNR + PU21。"""
    sdr_u8 = np.asarray(Image.open(img_path).convert("RGB"))
    sdr_lin = sdr_uint8_to_linear(sdr_u8)
    # 合成 GT：共享 ITM auto 曲线 @1000nits（与 pyultrahdr/hdrconv 的 ITM 同源）
    gt_hdr = inverse_tone_map(sdr_lin, 203.0, 1000.0, method="auto")

    for backend in backends:
        # 能力检查：J 组走 EXR
        from backends import create_backend as _cb
        if "exr" not in _cb(backend).output_formats:
            bench.run("J", "quality_fullref", backend,
                      lambda: {"_skipped": True, "note": f"{backend} 不支持 exr 输出"})
            continue
        def check(b=backend):
            out = tmpdir / f"fullref_{b}.exr"
            ok = _convert_via_backend(
                b, img_path, str(out), tmpdir,
                output_format="exr", gain_map_method="auto",
                keep_exif=False, sidecar=False, target_nits=1000)
            if not ok:
                return {"_passed": False, "note": "convert failed"}
            rgb, _ = read_exr(str(out))
            d_psnr = delta_ictcp_psnr(rgb, gt_hdr)
            pu = pu21_metrics(rgb, gt_hdr)
            return {
                "_passed": True,
                "dICtCp_PSNR_dB": round(d_psnr, 2),
                "PU21_PSNR_dB": round(pu["pu21_psnr"], 2),
                "PU21_SSIM": round(pu["pu21_ssim"], 4),
            }
        bench.run("J", "quality_fullref", backend, check)


def group_K_memory(bench: Bench, img_dir: str, tmpdir: Path,
                   backend: str = "pyultrahdr") -> None:
    """K. 资源占用：子进程峰值 RSS（ru_maxrss）。"""
    def check():
        out_dir = tmpdir / "mem_out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        cmd = [sys.executable, CLI, "-i", img_dir, "-o", str(out_dir),
               "--models", backend, "--format", "exr", "--apply",
               "--workers", "2", "--no-sidecar"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        # Linux: KB；取峰值（ru_maxrss 是历史最大，非增量）
        peak_mb = after / 1024.0
        return {
            "_passed": proc.returncode == 0,
            "child_peak_rss_mb": round(peak_mb, 1),
        }
    bench.run("K", "peak_memory", backend, check)


# ═══════════════════════════════════════════════════════════════════
#  报告
# ═══════════════════════════════════════════════════════════════════

def _json_default(obj):
    """numpy → Python 原生类型转换（JSON 序列化兜底）。"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_report(results: List[BenchResult], report_dir: Path,
                 elapsed: float) -> Tuple[Path, Path]:
    """写 JSON + Markdown 报告。"""
    report_dir.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    # JSON
    json_data = {
        "summary": {"total": len(results), "passed": passed,
                    "failed": failed, "skipped": skipped,
                    "elapsed_s": round(elapsed, 1)},
        "results": [
            {"group": r.group, "name": r.name, "backend": r.backend,
             "passed": bool(r.passed), "skipped": bool(r.skipped),
             "duration_s": round(r.duration_s, 3), **r.metrics}
            for r in results
        ],
    }
    json_path = report_dir / "bench_report.json"
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2,
                                    default=_json_default),
                         encoding="utf-8")

    # Markdown
    lines = [
        "# photo_sdr_to_hdr Benchmark Report",
        "",
        f"- 总测试: {len(results)}｜通过: {passed}｜失败: {failed}｜跳过: {skipped}",
        f"- 总耗时: {elapsed:.1f}s",
        f"- VMAF (ffmpeg libvmaf): {'可用' if ffmpeg_vmaf_available() else '不可用'}",
        "",
        "| 组 | 测试 | 后端 | 结果 | 指标 | 耗时(s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "SKIP" if r.skipped else ("✅" if r.passed else "❌")
        metric_str = "<br>".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.metrics.items())
        lines.append(f"| {r.group} | {r.name} | {r.backend} | {status} "
                     f"| {metric_str} | {r.duration_s:.2f} |")
    lines.append("")
    md_path = report_dir / "bench_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="photo_sdr_to_hdr 综合基准测试")
    ap.add_argument("--quick", action="store_true", help="只跑冒烟子集")
    ap.add_argument("--backends", nargs="+", default=None,
                    help="限定测试的后端（默认全部）")
    ap.add_argument("--report-dir", default="./bench_out", help="报告输出目录")
    ap.add_argument("--keep-workdir", action="store_true", help="保留临时工作目录")
    args = ap.parse_args()

    t_start = time.perf_counter()

    # 所有注册后端
    all_backends = BackendRegistry.list_names()
    backends = args.backends or all_backends
    unknown = [b for b in backends if b not in all_backends]
    if unknown:
        log(f"未知后端: {unknown}（已注册: {all_backends}）")
        return 2

    # 工作目录与测试素材
    workdir = Path(tempfile.mkdtemp(prefix="bench_sdr2hdr_"))
    img_path = workdir / "test_sdr.jpg"
    img_dir = workdir / "batch"
    img_dir.mkdir()
    try:
        test_img = make_test_image()
        save_test_image(test_img, str(img_path))
        # 批处理目录（F/K 组用）：4 张
        for i in range(4):
            save_test_image(make_test_image(seed=100 + i),
                            str(img_dir / f"img_{i}.jpg"))

        bench = Bench(workdir / "outputs", quick=args.quick)

        log("=" * 60)
        log("photo_sdr_to_hdr Benchmark")
        log(f"后端: {backends}")
        log(f"模式: {'quick' if args.quick else 'full'}")
        log(f"工作目录: {workdir}")
        log("=" * 60)

        avail = group_A_availability(bench, backends)
        usable = [b for b, (ok, _) in avail.items() if ok]
        if not usable:
            log("没有可用后端，终止。")
            return 1
        log(f"可用后端: {usable}")
        unavailable = {b: why for b, (ok, why) in avail.items() if not ok}
        if unavailable:
            log("不可用后端（环境原因，已跳过）:")
            for b, why in unavailable.items():
                log(f"  {b}: {why[:90]}")

        if not args.quick:
            group_B_formats(bench, usable, str(img_path), workdir / "outputs")
        group_C_gain_methods(bench, usable, str(img_path), workdir / "outputs")
        group_D_target_nits(bench, usable, str(img_path), workdir / "outputs")
        group_E_exif(bench, usable, str(img_path), workdir / "outputs")
        # F/G/K 用代表性后端：优先 pyultrahdr（纯算法、快、无外部依赖）
        primary = "pyultrahdr" if "pyultrahdr" in usable else usable[0]
        group_F_workers(bench, str(img_dir), workdir / "outputs", backend=primary)
        group_G_dryrun(bench, str(img_dir), workdir / "outputs", backend=primary)
        group_H_determinism(bench, usable, str(img_path), workdir / "outputs")
        group_I_quality_noref(bench, usable, str(img_path), workdir / "outputs")
        group_J_quality_fullref(bench, usable, str(img_path), workdir / "outputs")
        group_K_memory(bench, str(img_dir), workdir / "outputs", backend=primary)

        elapsed = time.perf_counter() - t_start
        json_path, md_path = write_report(bench.results, Path(args.report_dir), elapsed)

        # 汇总
        passed = sum(1 for r in bench.results if r.passed and not r.skipped)
        failed = sum(1 for r in bench.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in bench.results if r.skipped)
        log("")
        log("=" * 60)
        log(f"完成: {passed} 通过, {failed} 失败, {skipped} 跳过 "
            f"(共 {len(bench.results)} 项, {elapsed:.1f}s)")
        log(f"报告: {json_path}")
        log(f"      {md_path}")
        if failed:
            log("")
            log("失败项:")
            for r in bench.results:
                if not r.passed and not r.skipped:
                    log(f"  {r.group}.{r.name} [{r.backend}]: {r.metrics}")
        return 1 if failed else 0
    finally:
        if args.keep_workdir:
            log(f"工作目录保留: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())