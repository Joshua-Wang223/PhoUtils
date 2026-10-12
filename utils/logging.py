"""Logging and progress utilities for photo_sdr_to_hdr."""

from __future__ import annotations

import sys
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TextIO, List
from dataclasses import dataclass, field

# Global logger instance
_logger: Optional[logging.Logger] = None
_log_file: Optional[TextIO] = None


def setup_logging(
    log_file: str = "-",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup global logging configuration.

    Args:
        log_file: Log file path, "-" for stdout, or file path
        level: Logging level
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    global _logger, _log_file

    logger = logging.getLogger("photo_sdr_to_hdr")
    logger.setLevel(level)
    logger.handlers.clear()  # Clear any existing handlers

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # File handler
    if log_file and log_file != "-":
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        file_handler = logging.StreamHandler(_log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    elif log_file == "-":
        # Log to stdout is already handled by console handler
        pass

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def log(msg: str = "", level: int = logging.INFO) -> None:
    """Convenience function to log a message."""
    get_logger().log(level, msg)


@dataclass
class ProgressTracker:
    """Thread-safe progress tracker for batch operations."""

    total: int
    desc: str = "Processing"
    unit: str = "img"
    _completed: int = 0
    _failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _start_time: float = field(default_factory=time.time)
    _last_update: float = field(default_factory=time.time)
    _update_interval: float = 0.5  # Minimum seconds between updates

    def increment(self, success: bool = True) -> None:
        """Increment progress counter."""
        with self._lock:
            self._completed += 1
            if not success:
                self._failed += 1

    def update(self, completed: int, failed: int = 0) -> None:
        """Set absolute progress values."""
        with self._lock:
            self._completed = completed
            self._failed = failed

    def get_stats(self) -> tuple:
        """Get current (completed, failed, elapsed, rate)."""
        with self._lock:
            elapsed = time.time() - self._start_time
            rate = self._completed / elapsed if elapsed > 0 else 0
            return self._completed, self._failed, elapsed, rate

    def format_progress(self, width: int = 40) -> str:
        """Format progress bar string."""
        completed, failed, elapsed, rate = self.get_stats()
        pct = completed / self.total if self.total > 0 else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        eta = (self.total - completed) / rate if rate > 0 else 0
        return (
            f"{self.desc}: [{bar}] {completed}/{self.total} ({pct*100:.1f}%) "
            f"| {rate:.1f} {self.unit}/s | ETA: {eta:.0f}s"
            + (f" | Failed: {failed}" if failed else "")
        )

    def should_update(self) -> bool:
        """Check if enough time has passed for an update."""
        return time.time() - self._last_update >= self._update_interval

    def mark_update(self) -> None:
        """Mark that an update was displayed."""
        self._last_update = time.time()

    def finish(self) -> str:
        """Return final summary string."""
        completed, failed, elapsed, _ = self.get_stats()
        return (
            f"{self.desc} complete: {completed} {self.unit}(s)"
            + (f", {failed} failed" if failed else "")
            + f" in {elapsed:.1f}s"
        )


class LogCapture:
    """Capture log messages for report generation."""

    def __init__(self):
        self.messages: List[str] = []
        self._handler: Optional[logging.Handler] = None

    def start(self, logger: Optional[logging.Logger] = None) -> None:
        """Start capturing log messages."""
        target = logger or get_logger()
        self._handler = logging.Handler()
        self._handler.setLevel(logging.DEBUG)
        self._handler.emit = self._capture
        target.addHandler(self._handler)

    def stop(self, logger: Optional[logging.Logger] = None) -> None:
        """Stop capturing log messages."""
        if self._handler:
            target = logger or get_logger()
            target.removeHandler(self._handler)
            self._handler = None

    def _capture(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg:
            self.messages.append(msg)

    def get_logs(self) -> List[str]:
        """Get captured log messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear captured messages."""
        self.messages.clear()


@contextmanager
def capture_logs(logger: Optional[logging.Logger] = None):
    """Context manager to capture logs."""
    capture = LogCapture()
    capture.start(logger)
    try:
        yield capture
    finally:
        capture.stop(logger)


def fmt_time(seconds: float) -> str:
    """Format time duration for display."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"


def fmt_bytes(bytes_val: int) -> str:
    """Format byte count for display."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"