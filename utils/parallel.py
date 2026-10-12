"""Parallel processing utilities for photo_sdr_to_hdr."""

from __future__ import annotations

import multiprocessing as mp
import queue
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .logging import get_logger, ProgressTracker, fmt_time
from .device import (
    resolve_workers, estimate_worker_memory, format_resources, SystemResources
)

logger = get_logger()

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class WorkerConfig:
    """Configuration passed to worker processes."""
    backend_name: str
    device: str
    model_dir: str
    backend_kwargs: Dict[str, Any]
    conversion_kwargs: Dict[str, Any]


# Global worker state (per-process)
_WORKER_BACKEND = None
_WORKER_CONFIG: Optional[WorkerConfig] = None


def _worker_initializer(config: WorkerConfig) -> None:
    """Initialize worker process with backend."""
    global _WORKER_BACKEND, _WORKER_CONFIG
    _WORKER_CONFIG = config

    # Import here to avoid pickling issues
    from backends import create_backend

    try:
        model_dir = Path(config.model_dir) / config.backend_name
        _WORKER_BACKEND = create_backend(config.backend_name, model_dir=model_dir, **config.backend_kwargs)

        # Check availability
        available, reason = _WORKER_BACKEND.check_available()
        if not available:
            raise RuntimeError(f"Backend {config.backend_name} not available: {reason}")

        # Prepare backend
        _WORKER_BACKEND.prepare(config.device, **config.conversion_kwargs)
        logger.info(f"Worker initialized: {config.backend_name} on {config.device}")

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise


def _worker_convert(job: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Worker function to convert a single image.

    Args:
        job: Tuple of (input_path, output_path, extra_kwargs)

    Returns:
        Dict with result info
    """
    global _WORKER_BACKEND
    input_path, output_path, extra_kwargs = job

    start = time.time()
    try:
        # Merge conversion kwargs
        kwargs = dict(_WORKER_CONFIG.conversion_kwargs) if _WORKER_CONFIG else {}
        kwargs.update(extra_kwargs)

        success = _WORKER_BACKEND.convert(input_path, output_path, **kwargs)
        duration = time.time() - start

        output_size = 0
        if success:
            try:
                output_size = Path(output_path).stat().st_size
            except Exception:
                pass

        return {
            "input_path": input_path,
            "output_path": output_path,
            "backend": _WORKER_CONFIG.backend_name if _WORKER_CONFIG else "unknown",
            "success": success,
            "error": None,
            "duration": duration,
            "output_size": output_size,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "input_path": input_path,
            "output_path": output_path,
            "backend": _WORKER_CONFIG.backend_name if _WORKER_CONFIG else "unknown",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "duration": duration,
            "output_size": 0,
        }


class ParallelConverter:
    """Manages parallel conversion across multiple backends."""

    def __init__(
        self,
        model_dir: str,
        default_device: str = "auto",
        max_workers: int = 0,
        mem_per_worker_override: Optional[Dict[str, float]] = None
    ):
        self.model_dir = Path(model_dir)
        self.default_device = default_device
        self.max_workers = max_workers
        self.mem_per_worker_override = mem_per_worker_override or {}

    def convert_batch(
        self,
        backend_name: str,
        jobs: List[Tuple[str, str]],
        device: Optional[str] = None,
        backend_kwargs: Optional[Dict] = None,
        conversion_kwargs: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert a batch of images with a single backend.

        Args:
            jobs: List of (input_path, output_path) tuples
            device: Target device
            backend_kwargs: Backend initialization kwargs
            conversion_kwargs: Per-conversion kwargs
            progress_callback: Callback(completed, failed)

        Returns:
            List of result dicts
        """
        if not jobs:
            return []

        device = device or self.default_device
        backend_kwargs = backend_kwargs or {}
        conversion_kwargs = conversion_kwargs or {}

        # Estimate memory per worker
        mem_per_worker = self.mem_per_worker_override.get(
            backend_name,
            estimate_worker_memory(backend_name)
        )

        # Resolve worker count
        workers, reason, resources = resolve_workers(
            len(jobs), mem_per_worker, self.max_workers
        )
        logger.info(f"[{backend_name}] Parallel: {workers} workers ({reason})")
        logger.info(f"[{backend_name}] Resources: {format_resources(resources)}")

        # Prepare worker config
        config = WorkerConfig(
            backend_name=backend_name,
            device=device,
            model_dir=str(self.model_dir),
            backend_kwargs=backend_kwargs,
            conversion_kwargs=conversion_kwargs,
        )

        # Verify backend can be created
        from backends import create_backend
        test_backend = create_backend(backend_name, model_dir=self.model_dir / backend_name, **backend_kwargs)
        available, reason = test_backend.check_available()
        if not available:
            raise RuntimeError(f"Backend {backend_name} not available: {reason}")

        # Progress tracker
        tracker = ProgressTracker(total=len(jobs), desc=backend_name)

        # Prepare jobs with output paths
        job_tuples = [(inp, out, conversion_kwargs) for inp, out in jobs]

        results = []

        if workers == 1:
            # Serial execution (no multiprocessing overhead)
            logger.info(f"[{backend_name}] Running in serial mode")
            _worker_initializer(config)
            for job in job_tuples:
                result = _worker_convert(job)
                results.append(result)
                tracker.increment(result["success"])
                if progress_callback:
                    progress_callback(tracker._completed, tracker._failed)
        else:
            # Parallel execution
            logger.info(f"[{backend_name}] Running with {workers} processes")
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_initializer,
                initargs=(config,)
            ) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(_worker_convert, job): job[0]
                    for job in job_tuples
                }

                # Collect results
                for future in as_completed(future_to_job):
                    try:
                        result = future.result()
                        results.append(result)
                        tracker.increment(result["success"])
                        if progress_callback:
                            progress_callback(tracker._completed, tracker._failed)
                    except Exception as e:
                        inp = future_to_job[future]
                        logger.error(f"Job failed for {inp}: {e}")
                        results.append({
                            "input_path": inp,
                            "output_path": "",
                            "backend": backend_name,
                            "success": False,
                            "error": str(e),
                            "duration": 0,
                            "output_size": 0,
                        })
                        tracker.increment(False)
                        if progress_callback:
                            progress_callback(tracker._completed, tracker._failed)

        logger.info(f"[{backend_name}] {tracker.finish()}")
        return results


def run_parallel_backends(
    backend_names: List[str],
    input_paths: List[str],
    output_dir: str,
    model_dir: str,
    device: str = "auto",
    max_workers: int = 0,
    backend_kwargs: Optional[Dict[str, Dict]] = None,
    conversion_kwargs: Optional[Dict[str, Dict]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    output_suffix: str = "_hdr",
    input_root: Optional[str] = None,
    output_format: str = "exr"
) -> List[Dict[str, Any]]:
    """
    Run conversion with multiple backends (each backend gets its own process pool).

    Args:
        backend_names: List of backend names to run
        input_paths: Input image paths
        output_dir: Output directory
        model_dir: Model directory root
        device: Target device
        max_workers: Max workers per backend (0 = auto)
        backend_kwargs: Per-backend kwargs for backend creation
        conversion_kwargs: Per-backend kwargs for conversion
        progress_callback: Callback(backend_name, completed, failed)
        output_suffix: Output filename suffix
        input_root: Input root dir for relative path preservation
        output_format: Output format extension (e.g. "exr")

    Returns:
        Combined list of all results
    """
    from .file_io import make_output_path, format_to_ext

    all_results = []
    backend_kwargs = backend_kwargs or {}
    conversion_kwargs = conversion_kwargs or {}
    out_ext = format_to_ext(output_format)

    for backend_name in backend_names:
        converter = ParallelConverter(
            model_dir=model_dir,
            default_device=device,
            max_workers=max_workers,
        )

        def make_callback(bn):
            def cb(completed, failed):
                if progress_callback:
                    progress_callback(bn, completed, failed)
            return cb

        # Build (input, output) job pairs for this backend
        jobs = [
            (inp, make_output_path(
                inp, output_dir, output_suffix, input_root,
                format_override=out_ext, backend_name=backend_name
            ))
            for inp in input_paths
        ]

        results = converter.convert_batch(
            backend_name=backend_name,
            jobs=jobs,
            device=device,
            backend_kwargs=backend_kwargs.get(backend_name, {}),
            conversion_kwargs=conversion_kwargs.get(backend_name, {}),
            progress_callback=make_callback(backend_name),
        )
        all_results.extend(results)

    return all_results