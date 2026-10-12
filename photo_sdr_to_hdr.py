#!/usr/bin/env python3
"""photo_sdr_to_hdr.py — SDR到HDR批量转换工具

支持多种后端模型：
- hdrconv: Python原生，HDR格式转换/封装 (pip install hdr-conversion)
- unet-onnx: ONNX Runtime, U-Net深度学习重建
- lumi-qwen/klein4b/klein9b: LumiPic扩散模型LoRA
- gmnet: GMNet ICLR2025 Gain Map预测
- pyultrahdr: 纯Python UltraHDR编码器 (~230行，无需编译)
- sdr2hdr: Rust CLI工具 (cargo install sdr2hdr)
- ultrahdr: libultrahdr CLI (需自行编译)

设计原则：
- 默认干跑 (--dry-run)，不写文件
- --apply 才真正执行
- 进程池并行，自动探测核数/内存/显存
- 详细日志与报告
- 模型自动下载到 models/ 子目录
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Local imports
from utils import (
    collect_input_files,
    make_output_path,
    resolve_output_root,
    get_input_root,
    format_to_ext,
    SUPPORTED_EXTS,
    setup_logging,
    ProgressTracker,
    ReportGenerator,
    ConversionResult,
    resolve_device,
    get_system_resources,
    format_resources,
    resolve_workers,
    estimate_worker_memory,
    ParallelConverter,
    run_parallel_backends,
)
from utils.logging import log as log_msg, get_logger
from backends import list_backends, BackendRegistry


def build_args() -> argparse.Namespace:
    """Build and parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="SDR到HDR批量转换工具：多后端支持、并行处理、自动模型管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    ap.add_argument(
        "-i", "--input",
        nargs="+", required=True,
        help="输入文件或目录路径（可多个）"
    )
    ap.add_argument(
        "-o", "--output",
        help="输出目录（默认：输入目录下的 _hdr_output/）"
    )

    # Model backends
    available_backends = BackendRegistry.list_names()
    ap.add_argument(
        "--models",
        nargs="+", default=["hdrconv"],
        choices=available_backends,
        help=f"选择转换后端（可多个，默认: hdrconv）。可用: {', '.join(available_backends)}"
    )

    # Parallelism
    ap.add_argument(
        "-w", "--workers", "--threads",
        type=int, default=0,
        help="并行进程数 (0=自动探测核数/内存/显存, 1=串行, N=指定)"
    )

    # Device
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="计算设备 (默认: auto 自动选择)"
    )

    # Output naming
    ap.add_argument(
        "--suffix",
        default="_hdr",
        help="输出文件名后缀 (默认: _hdr)"
    )
    ap.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归处理子目录"
    )

    # Execution mode
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="显式声明预览模式（默认行为即预览，无需加此参数）"
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="真正执行转换（不加则只预览，不写文件）"
    )

    # Output format
    ap.add_argument(
        "--format",
        choices=["exr", "jpg", "jpeg", "avif", "png", "ultrahdr", "gainmap"],
        default="exr",
        help="输出格式 (默认: exr 统一为 OpenEXR half-float 线性)"
    )

    # LumiPic quantization
    ap.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="LumiPic 模型量化 (默认: none/bf16, 显存不足时建议 4bit)"
    )

    # Colorspace
    ap.add_argument(
        "--colorspace",
        choices=["linear-rec709", "pq-rec2020", "hlg-rec2020"],
        default="linear-rec709",
        help="输出色彩空间 (默认: linear-rec709 线性 Rec.709)"
    )

    # Model directory
    ap.add_argument(
        "--model-dir",
        default="./models",
        help="模型存储目录 (默认: ./models/)"
    )

    # EXIF
    ap.add_argument(
        "--keep-exif",
        action="store_true",
        default=True,
        help="保留原图 EXIF 元数据 (默认: 保留)"
    )
    ap.add_argument(
        "--no-keep-exif",
        action="store_false",
        dest="keep_exif",
        help="不保留 EXIF"
    )
    ap.add_argument(
        "--sidecar",
        action="store_true",
        default=True,
        help="写 .exif.json 旁车文件 (默认: 写)"
    )
    ap.add_argument(
        "--no-sidecar",
        action="store_false",
        dest="sidecar",
        help="不写 .exif.json 旁车文件"
    )

    # pyultrahdr specific
    ap.add_argument(
        "--gain-map-method",
        choices=["auto", "linear", "log", "fixed"],
        default="auto",
        help="pyultrahdr 增益图计算方法 (默认: auto)"
    )
    ap.add_argument(
        "--target-nits",
        type=int, default=1000,
        help="目标峰值亮度 nits (UltraHDR/Gainmap 用，默认: 1000)"
    )

    # sdr2hdr specific
    ap.add_argument(
        "--sdr2hdr-mode",
        type=int, choices=[1, 2], default=1,
        help="sdr2hdr ICC 模式: 1=icc1, 2=icc2 (默认: 1)"
    )

    # ultrahdr CLI specific
    ap.add_argument(
        "--ultrahdr-binary",
        help="ultrahdr_app 可执行文件路径（默认从 PATH/ULTRHDR_APP_PATH 查找）"
    )

    # Logging
    ap.add_argument(
        "--log",
        default="-",
        help="日志文件路径 (- 为 stdout，默认: -)"
    )

    # Version
    ap.add_argument(
        "-v", "--version",
        action="version",
        version="photo_sdr_to_hdr 0.1.0"
    )

    args = ap.parse_args()

    # 验证：--dry-run 与 --apply 互斥；默认行为为预览（不加 --apply）
    if args.apply and args.dry_run:
        ap.error("--dry-run 和 --apply 不能同时使用（不加 --apply 即为预览）")

    # dry_run 语义：未指定 --apply 时默认预览
    args.dry_run = not args.apply

    return args


def print_banner(args: argparse.Namespace, input_files: List[str], output_root: str) -> None:
    """Print startup banner with configuration."""
    log_msg("=" * 60)
    log_msg("photo_sdr_to_hdr — SDR to HDR Batch Converter")
    log_msg("=" * 60)

    # System info
    resources = get_system_resources()
    log_msg(f"System: {format_resources(resources)}")

    # Input
    log_msg(f"Input: {len(input_files)} file(s)")
    if len(input_files) <= 5:
        for f in input_files:
            log_msg(f"  {f}")
    else:
        for f in input_files[:3]:
            log_msg(f"  {f}")
        log_msg(f"  ... and {len(input_files) - 3} more")

    # Output
    log_msg(f"Output: {output_root}")
    log_msg(f"Format: {args.format.upper()} | Suffix: {args.suffix} | Colorspace: {args.colorspace}")

    # Backends
    log_msg(f"Backends: {', '.join(args.models)}")
    for m in args.models:
        info = BackendRegistry.get(m)
        log_msg(f"  {m}: {info.description}")

    # Parallelism
    if args.workers > 0:
        log_msg(f"Workers: {args.workers} (user specified)")
    else:
        log_msg("Workers: auto (CPU/RAM/VRAM aware)")

    # Device
    device = resolve_device(args.device)
    log_msg(f"Device: {device}")

    # Mode
    mode = "DRY RUN (preview only)" if args.dry_run else "APPLY (will write files)"
    log_msg(f"Mode: {mode}")

    # Other options
    if args.quantization != "none":
        log_msg(f"Quantization: {args.quantization}")
    if args.model_dir != "./models":
        log_msg(f"Model Dir: {args.model_dir}")
    if not args.keep_exif:
        log_msg("EXIF: disabled")
    if args.gain_map_method != "auto":
        log_msg(f"Gain Map Method: {args.gain_map_method}")
    if args.target_nits != 1000:
        log_msg(f"Target Nits: {args.target_nits}")

    log_msg("")


def build_conversion_kwargs(args: argparse.Namespace, backend_name: str) -> Dict:
    """Build conversion kwargs for a specific backend."""
    kwargs = {
        "output_format": args.format,
        "colorspace": args.colorspace,
        "keep_exif": args.keep_exif,
        "sidecar": args.sidecar,
        "target_nits": args.target_nits,
    }

    # Backend-specific kwargs
    if backend_name.startswith("lumi-"):
        kwargs["quantization"] = args.quantization
    elif backend_name == "pyultrahdr":
        kwargs["gain_map_method"] = args.gain_map_method
    elif backend_name == "hdrconv":
        kwargs["gain_map_method"] = args.gain_map_method
        # format 决定容器：gainmap → ISO 21496-1；ultrahdr/jpg → Google UltraHDR
        kwargs["container"] = "iso21496" if args.format == "gainmap" else "ultrahdr"
    elif backend_name == "sdr2hdr":
        kwargs["mode"] = args.sdr2hdr_mode

    return kwargs


def build_backend_kwargs(args: argparse.Namespace, backend_name: str) -> Dict:
    """Build backend initialization kwargs."""
    kwargs = {}

    if backend_name.startswith("lumi-"):
        # LumiPic specific: variant selection
        if backend_name == "lumi-qwen":
            kwargs["variant"] = "qwen"
        elif backend_name == "lumi-klein4b":
            kwargs["variant"] = "klein4b"
        elif backend_name == "lumi-klein9b":
            kwargs["variant"] = "klein9b"
    elif backend_name == "ultrahdr":
        if args.ultrahdr_binary:
            kwargs["binary_path"] = args.ultrahdr_binary

    return kwargs


def main() -> int:
    """Main entry point."""
    args = build_args()

    # Setup logging
    setup_logging(log_file=args.log)

    start_time = time.time()

    # Collect input files
    input_files = collect_input_files(args.input, recursive=args.recursive)
    if not input_files:
        log_msg("ERROR: No input images found", logging.ERROR)
        return 1

    # Resolve output root
    output_root = resolve_output_root(args.output, input_files)
    input_root = get_input_root(input_files)

    # Print banner
    print_banner(args, input_files, output_root)

    if args.dry_run:
        log_msg("DRY RUN - No files will be written. Use --apply to execute.")
        log_msg("")
        # Show what would be done
        out_ext = format_to_ext(args.format)
        for backend_name in args.models:
            log_msg(f"[{backend_name}] Would process {len(input_files)} file(s)")
            for inp in input_files[:3]:
                out = make_output_path(
                    inp, output_root, args.suffix, input_root,
                    format_override=out_ext, backend_name=backend_name
                )
                log_msg(f"  {os.path.basename(inp)} -> {os.path.relpath(out, output_root)}")
            if len(input_files) > 3:
                log_msg(f"  ... and {len(input_files) - 3} more")
        log_msg("")
        log_msg("To execute, run with --apply")
        return 0

    # Ensure output directory exists
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Prepare conversion kwargs per backend
    conversion_kwargs = {
        name: build_conversion_kwargs(args, name) for name in args.models
    }
    backend_kwargs = {
        name: build_backend_kwargs(args, name) for name in args.models
    }

    # Run conversion
    all_results = []
    tracker = ProgressTracker(total=len(input_files) * len(args.models), desc="Total")

    def progress_cb(backend: str, completed: int, failed: int):
        pass  # 各 backend 的 ProgressTracker 已自行打印

    try:
        results = run_parallel_backends(
            backend_names=args.models,
            input_paths=input_files,
            output_dir=output_root,
            model_dir=args.model_dir,
            device=resolve_device(args.device),
            max_workers=args.workers,
            backend_kwargs=backend_kwargs,
            conversion_kwargs=conversion_kwargs,
            progress_callback=progress_cb,
            output_suffix=args.suffix,
            input_root=input_root,
            output_format=args.format,
        )
        all_results.extend(results)

    except KeyboardInterrupt:
        log_msg("\nInterrupted by user")
        return 130
    except Exception as e:
        log_msg(f"FATAL ERROR: {e}", logging.ERROR)
        import traceback
        log_msg(traceback.format_exc())
        return 1

    # Results already carry worker-constructed output paths
    final_results = [
        ConversionResult(
            input_path=r["input_path"],
            output_path=r["output_path"],
            backend=r["backend"],
            success=r["success"],
            error=r["error"],
            duration=r["duration"],
            output_size=r["output_size"],
        )
        for r in all_results
    ]

    # Generate report
    config = {
        "input": args.input,
        "output": output_root,
        "models": args.models,
        "format": args.format,
        "suffix": args.suffix,
        "recursive": args.recursive,
        "device": resolve_device(args.device),
        "workers": args.workers,
        "quantization": args.quantization,
        "colorspace": args.colorspace,
        "model_dir": args.model_dir,
        "keep_exif": args.keep_exif,
        "gain_map_method": args.gain_map_method,
        "target_nits": args.target_nits,
    }

    report_gen = ReportGenerator(output_root)
    report_gen.set_config(config)
    report_gen.start_time = start_time
    for r in final_results:
        report_gen.add_result(r)

    report_path = report_gen.generate()
    report_gen.print_summary()

    log_msg(f"\nReport saved to: {report_path}")
    log_msg(f"Total time: {time.time() - start_time:.1f}s")

    # Exit code based on failures
    failed = sum(1 for r in final_results if not r.success)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    import logging
    sys.exit(main())