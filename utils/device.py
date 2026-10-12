"""Device detection and memory utilities."""

from __future__ import annotations

import os
import sys
import ctypes
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about compute device."""
    name: str
    type: str  # "cuda", "cpu", "mps"
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    compute_capability: Optional[str] = None


@dataclass
class SystemResources:
    """System resource information."""
    cpu_count: int
    cpu_source: str
    total_memory_gb: float
    available_memory_gb: float
    memory_source: str
    devices: list = None

    def __post_init__(self):
        if self.devices is None:
            self.devices = []


def detect_cpu() -> Tuple[int, str]:
    """Detect available CPU cores (logical)."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            parts = f.read().split()
            if len(parts) == 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, round(quota / period)), "cgroup v2"
    except Exception:
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return max(1, round(quota / period)), "cgroup v1"
    except Exception:
        pass

    # Linux sched_getaffinity
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n, "sched_getaffinity"
    except Exception:
        pass

    # Windows process affinity
    if sys.platform == "win32":
        try:
            k32 = ctypes.windll.kernel32
            k32.GetProcessAffinityMask.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t)
            ]
            mask, sysmask = ctypes.c_size_t(), ctypes.c_size_t()
            if k32.GetProcessAffinityMask(ctypes.c_void_p(-1),
                                           ctypes.byref(mask), ctypes.byref(sysmask)):
                n = bin(mask.value).count("1")
                if n > 0:
                    return n, "Windows affinity"
        except Exception:
            pass

    return max(1, os.cpu_count() or 1), "os.cpu_count"


def detect_memory_gb() -> Tuple[float, float, str]:
    """Detect total and available memory in GB."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            v = f.read().strip()
            if v != "max":
                limit = int(v) / 1024**3
                cur = 0.0
                try:
                    with open("/sys/fs/cgroup/memory.current", "r") as f2:
                        cur = int(f2.read()) / 1024**3
                except Exception:
                    pass
                return limit, max(0.0, limit - cur), "cgroup v2"
    except Exception:
        pass

    # Linux /proc/meminfo
    try:
        total = avail = 0.0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024**2
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024**2
                if total and avail:
                    break
        if total:
            return total, avail, "/proc/meminfo"
    except Exception:
        pass

    # Windows GlobalMemoryStatusEx
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return (st.ullTotalPhys / 1024**3, st.ullAvailPhys / 1024**3,
                        "GlobalMemoryStatusEx")
        except Exception:
            pass

    return 0.0, 0.0, "unknown"


def detect_cuda_devices() -> list:
    """Detect CUDA devices and their memory."""
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / 1024**3
                # Get free memory
                torch.cuda.set_device(i)
                free_mem = (torch.cuda.mem_get_info()[0] / 1024**3)
                devices.append(DeviceInfo(
                    name=props.name,
                    type="cuda",
                    total_memory_gb=total_mem,
                    free_memory_gb=free_mem,
                    compute_capability=f"{props.major}.{props.minor}"
                ))
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")
    return devices


def detect_mps() -> Optional[DeviceInfo]:
    """Detect Apple Metal Performance Shaders device."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose memory info easily
            return DeviceInfo(
                name="Apple Silicon GPU",
                type="mps",
                total_memory_gb=0.0,  # Unified memory
                free_memory_gb=0.0,
            )
    except Exception:
        pass
    return None


def get_system_resources() -> SystemResources:
    """Get comprehensive system resource information."""
    cpu_count, cpu_src = detect_cpu()
    total_mem, avail_mem, mem_src = detect_memory_gb()

    devices = []
    devices.extend(detect_cuda_devices())
    mps = detect_mps()
    if mps:
        devices.append(mps)

    return SystemResources(
        cpu_count=cpu_count,
        cpu_source=cpu_src,
        total_memory_gb=total_mem,
        available_memory_gb=avail_mem,
        memory_source=mem_src,
        devices=devices
    )


def resolve_device(requested: str) -> str:
    """
    Resolve device string to actual device.

    Args:
        requested: "auto", "cuda", "cpu", "mps", or "cuda:N"

    Returns:
        Resolved device string
    """
    if requested == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


def get_device_memory(device: str) -> Tuple[float, float]:
    """Get total and free memory for a device in GB."""
    try:
        import torch
        if device.startswith("cuda"):
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx < torch.cuda.device_count():
                torch.cuda.set_device(idx)
                free, total = torch.cuda.mem_get_info()
                return total / 1024**3, free / 1024**3
        elif device == "mps":
            # Unified memory - return system memory
            total, avail, _ = detect_memory_gb()
            return total, avail
    except Exception:
        pass
    return 0.0, 0.0


# Memory estimation constants (matching photo_dedup.py)
MAX_USABLE_MEM_RATIO = 0.85
MIN_SYSTEM_RESERVE_GB = 1.5
DEFAULT_MEM_PER_WORKER = 0.5  # GB, conservative default
PARALLEL_MIN_ITEMS = 8


def estimate_worker_memory(backend_name: str, model_size_gb: float = 0) -> float:
    """
    Estimate per-worker memory requirement in GB.

    Args:
        backend_name: Name of the backend
        model_size_gb: Model size if known

    Returns:
        Estimated GB per worker
    """
    estimates = {
        "hdrconv": 0.3,
        "pyultrahdr": 0.2,
        "unet-onnx": max(0.8, model_size_gb * 1.5),
        "lumi-qwen": 12.0,  # bf16
        "lumi-klein4b": 8.0,
        "lumi-klein9b": 16.0,
        "gmnet": 3.0,
        "sdr2hdr": 0.1,
        "ultrahdr": 0.1,
    }
    base = estimates.get(backend_name, DEFAULT_MEM_PER_WORKER)
    # Add model size if provided
    if model_size_gb > 0:
        base = max(base, model_size_gb * 1.2)
    return base


def resolve_workers(
    n_tasks: int,
    mem_per_worker_gb: float,
    requested_workers: int = 0
) -> Tuple[int, str, SystemResources]:
    """
    Resolve optimal worker count based on system resources.

    Args:
        n_tasks: Number of tasks to process
        mem_per_worker_gb: Estimated memory per worker (GB)
        requested_workers: User-requested workers (0 = auto)

    Returns:
        Tuple of (worker_count, reason_string, system_resources)
    """
    if n_tasks <= 0:
        return 0, "no tasks", get_system_resources()

    resources = get_system_resources()

    if requested_workers > 0:
        workers = max(1, min(requested_workers, n_tasks))
        return workers, f"user requested {requested_workers}", resources

    # Auto-calculate based on CPU
    by_cpu = resources.cpu_count

    # Auto-calculate based on memory
    if resources.total_memory_gb > 0:
        usable = max(0.1, min(
            resources.total_memory_gb * MAX_USABLE_MEM_RATIO,
            max(0.0, resources.available_memory_gb - MIN_SYSTEM_RESERVE_GB)
        ))
        by_mem = max(1, int(usable / max(0.05, mem_per_worker_gb)))
        limit_desc = f"memory limit {by_mem}"
    else:
        by_mem = by_cpu
        limit_desc = "memory unknown, using CPU count"

    workers = max(1, min(by_cpu, by_mem, n_tasks))

    if n_tasks < PARALLEL_MIN_ITEMS and requested_workers == 0:
        workers = 1
        reason = f"only {n_tasks} tasks, serial is faster"
    else:
        reason = f"CPU limit {by_cpu}, {limit_desc}, tasks {n_tasks} -> min = {workers}"

    return workers, reason, resources


def format_resources(resources: SystemResources) -> str:
    """Format system resources for display."""
    mem_str = (
        f"RAM {resources.total_memory_gb:.1f} GB (avail {resources.available_memory_gb:.1f} GB, {resources.memory_source})"
        if resources.total_memory_gb > 0
        else f"RAM unknown ({resources.memory_source})"
    )
    cpu_str = f"CPU {resources.cpu_count} ({resources.cpu_source})"

    dev_str = ""
    if resources.devices:
        dev_info = []
        for d in resources.devices:
            mem_info = f", {d.free_memory_gb:.1f}/{d.total_memory_gb:.1f} GB free" if d.total_memory_gb > 0 else ""
            dev_info.append(f"{d.name} ({d.type}{mem_info})")
        dev_str = f" | GPU: {', '.join(dev_info)}"

    return f"{cpu_str} | {mem_str}{dev_str}"