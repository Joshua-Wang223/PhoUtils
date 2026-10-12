"""Report generation for photo_sdr_to_hdr."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .logging import fmt_time, fmt_bytes


@dataclass
class ConversionResult:
    """Result of a single conversion."""
    input_path: str
    output_path: str
    backend: str
    success: bool
    error: Optional[str] = None
    duration: float = 0.0
    output_size: int = 0


@dataclass
class BackendStats:
    """Statistics for a single backend."""
    name: str
    total: int = 0
    success: int = 0
    failed: int = 0
    total_time: float = 0.0
    total_output_bytes: int = 0
    errors: Dict[str, int] = field(default_factory=dict)

    def add_result(self, result: ConversionResult) -> None:
        self.total += 1
        self.total_time += result.duration
        if result.success:
            self.success += 1
            self.total_output_bytes += result.output_size
        else:
            self.failed += 1
            if result.error:
                self.errors[result.error] = self.errors.get(result.error, 0) + 1

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total if self.total > 0 else 0.0


class ReportGenerator:
    """Generates conversion reports."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ConversionResult] = []
        self.backend_stats: Dict[str, BackendStats] = {}
        self.start_time = time.time()
        self.config: Dict = {}

    def set_config(self, config: Dict) -> None:
        """Store run configuration."""
        self.config = config

    def add_result(self, result: ConversionResult) -> None:
        """Add a conversion result."""
        self.results.append(result)
        if result.backend not in self.backend_stats:
            self.backend_stats[result.backend] = BackendStats(result.backend)
        self.backend_stats[result.backend].add_result(result)

    def generate(self, report_name: str = "photo_sdr_to_hdr_report.txt") -> str:
        """Generate and write report file."""
        report_path = self.output_dir / report_name
        lines = self._build_report_lines()
        content = "\n".join(lines) + "\n"
        report_path.write_text(content, encoding="utf-8")
        return str(report_path)

    def _build_report_lines(self) -> List[str]:
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("photo_sdr_to_hdr Conversion Report")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Duration: {fmt_time(time.time() - self.start_time)}")
        lines.append("")

        # Configuration
        if self.config:
            lines.append("-" * 60)
            lines.append("Configuration")
            lines.append("-" * 60)
            for key, value in self.config.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Backend summary
        lines.append("-" * 60)
        lines.append("Backend Summary")
        lines.append("-" * 60)
        total_tasks = len(self.results)
        total_success = sum(1 for r in self.results if r.success)
        total_failed = total_tasks - total_success
        total_output_bytes = sum(r.output_size for r in self.results if r.success)

        lines.append(f"  Total Tasks: {total_tasks}")
        lines.append(f"  Successful:  {total_success}")
        lines.append(f"  Failed:      {total_failed}")
        lines.append(f"  Output Size: {fmt_bytes(total_output_bytes)}")
        lines.append("")

        for backend_name in sorted(self.backend_stats.keys()):
            stats = self.backend_stats[backend_name]
            lines.append(f"  {backend_name}:")
            lines.append(f"    Tasks: {stats.total} | Success: {stats.success} | Failed: {stats.failed}")
            lines.append(f"    Time: {fmt_time(stats.total_time)} | Avg: {fmt_time(stats.avg_time)}")
            lines.append(f"    Output: {fmt_bytes(stats.total_output_bytes)}")
            if stats.errors:
                lines.append(f"    Errors:")
                for err, count in sorted(stats.errors.items(), key=lambda x: -x[1]):
                    lines.append(f"      {err}: {count}")
            lines.append("")

        # Detailed results
        if self.results:
            lines.append("-" * 60)
            lines.append("Detailed Results")
            lines.append("-" * 60)

            # Group by input file
            by_input: Dict[str, List[ConversionResult]] = {}
            for r in self.results:
                by_input.setdefault(r.input_path, []).append(r)

            for i, (inp, results) in enumerate(sorted(by_input.items()), 1):
                rel_inp = os.path.basename(inp)
                lines.append(f"  {i}. {rel_inp}")
                for r in sorted(results, key=lambda x: x.backend):
                    status = "✓" if r.success else "✗"
                    out_name = os.path.basename(r.output_path)
                    size_str = f" ({fmt_bytes(r.output_size)})" if r.success else ""
                    time_str = f" [{fmt_time(r.duration)}]"
                    lines.append(f"     {status} {r.backend}: {out_name}{size_str}{time_str}")
                    if not r.success and r.error:
                        lines.append(f"        Error: {r.error}")
                lines.append("")

        return lines

    def print_summary(self) -> None:
        """Print summary to console."""
        lines = self._build_report_lines()
        for line in lines:
            print(line)


def write_report(
    results: List[ConversionResult],
    config: Dict,
    output_dir: str,
    start_time: float,
    report_name: str = "photo_sdr_to_hdr_report.txt"
) -> str:
    """Convenience function to generate a report."""
    generator = ReportGenerator(output_dir)
    generator.set_config(config)
    generator.start_time = start_time
    for r in results:
        generator.add_result(r)
    return generator.generate(report_name)