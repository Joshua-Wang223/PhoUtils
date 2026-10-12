"""Utilities package for photo_sdr_to_hdr."""

from .file_io import (
    collect_input_files, make_output_path, SUPPORTED_EXTS,
    resolve_output_root, get_input_root, ensure_output_dir, format_to_ext,
    FORMAT_TO_EXT
)
from .logging import setup_logging, get_logger, ProgressTracker
from .report import ReportGenerator, ConversionResult, write_report
from .device import (
    DeviceInfo, SystemResources, resolve_device, get_system_resources,
    resolve_workers, estimate_worker_memory, format_resources
)
from .parallel import ParallelConverter, run_parallel_backends

__all__ = [
    "collect_input_files",
    "make_output_path",
    "SUPPORTED_EXTS",
    "resolve_output_root",
    "get_input_root",
    "ensure_output_dir",
    "format_to_ext",
    "FORMAT_TO_EXT",
    "setup_logging",
    "get_logger",
    "ProgressTracker",
    "ReportGenerator",
    "ConversionResult",
    "write_report",
    "DeviceInfo",
    "SystemResources",
    "resolve_device",
    "get_system_resources",
    "resolve_workers",
    "estimate_worker_memory",
    "format_resources",
    "ParallelConverter",
    "run_parallel_backends",
]