"""File I/O utilities for photo_sdr_to_hdr."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

# Supported image extensions (matching photo_dedup.py)
SUPPORTED_EXTS = (
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    ".heic", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"
)

# 输出格式 → 文件扩展名（ultrahdr/gainmap 是 JPEG 容器，扩展名为 .jpg）
FORMAT_TO_EXT = {
    "exr": ".exr",
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "avif": ".avif",
    "png": ".png",
    "ultrahdr": ".jpg",
    "gainmap": ".jpg",
}


def format_to_ext(output_format: str) -> str:
    """输出格式名 → 文件扩展名（含点）。"""
    return FORMAT_TO_EXT.get(output_format.lower().lstrip("."), f".{output_format.lower().lstrip('.')}")


def collect_input_files(
    paths: List[str],
    recursive: bool = False,
    exts: Optional[Tuple[str, ...]] = None
) -> List[str]:
    """
    Collect image files from input paths (files or directories).

    Args:
        paths: List of file or directory paths
        recursive: Whether to recurse into subdirectories
        exts: File extensions to include (default: SUPPORTED_EXTS)

    Returns:
        Sorted list of unique absolute file paths
    """
    if exts is None:
        exts = SUPPORTED_EXTS

    exts_lower = tuple(e.lower() for e in exts)
    files = []
    seen = set()

    for path_str in paths:
        path = Path(path_str).resolve()

        if path.is_file():
            if path.suffix.lower() in exts_lower:
                abs_path = str(path)
                if abs_path not in seen:
                    seen.add(abs_path)
                    files.append(abs_path)

        elif path.is_dir():
            if recursive:
                walker = path.rglob("*")
            else:
                walker = path.glob("*")

            for item in walker:
                if item.is_file() and item.suffix.lower() in exts_lower:
                    abs_path = str(item)
                    if abs_path not in seen:
                        seen.add(abs_path)
                        files.append(abs_path)

        else:
            # Path doesn't exist - will be handled by caller
            pass

    return sorted(files)


def make_output_path(
    input_path: str,
    output_root: str,
    suffix: str = "_hdr",
    input_root: Optional[str] = None,
    format_override: Optional[str] = None,
    backend_name: Optional[str] = None
) -> str:
    """
    Generate output path preserving directory structure.

    Args:
        input_path: Absolute path to input file
        output_root: Root output directory
        suffix: Suffix to append to filename (before extension)
        input_root: Root input directory (for relative path calculation)
        format_override: Override output format/extension (e.g., "exr", "jpg")
        backend_name: Optional backend name to include in filename

    Returns:
        Absolute output path
    """
    in_path = Path(input_path)
    out_root = Path(output_root).resolve()

    # Determine relative path from input root
    if input_root:
        try:
            rel_path = in_path.relative_to(Path(input_root).resolve())
        except ValueError:
            # Input path not under input_root, use just filename
            rel_path = in_path.name
    else:
        rel_path = in_path.name

    # Build output filename
    stem = rel_path.stem
    if backend_name:
        # Sanitize backend name for filename
        safe_backend = backend_name.replace("/", "-").replace("\\", "-")
        stem = f"{stem}{suffix}_{safe_backend}"
    else:
        stem = f"{stem}{suffix}"

    # Determine extension
    if format_override:
        ext = format_override if format_override.startswith(".") else f".{format_override}"
    else:
        ext = in_path.suffix

    out_rel = rel_path.with_name(f"{stem}{ext}")
    return str(out_root / out_rel)


def ensure_output_dir(output_path: str) -> None:
    """Ensure output directory exists."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def get_input_root(paths: List[str]) -> Optional[str]:
    """
    Determine common input root directory from a list of paths.

    Returns the common parent directory if all paths share one,
    otherwise returns the first directory path found, or None.
    """
    if not paths:
        return None

    path_objs = [Path(p) for p in paths]

    # Check if all are under a common directory
    dirs = [p.parent for p in path_objs if p.is_file()]
    if not dirs:
        return None

    # Find common prefix
    common = Path(os.path.commonpath([str(d) for d in dirs]))
    return str(common) if common != Path("/") else None


def resolve_output_root(
    output_arg: Optional[str],
    input_paths: List[str],
    default_suffix: str = "_hdr_output"
) -> str:
    """
    Resolve output root directory from CLI argument.

    Args:
        output_arg: Explicit --output argument (or None)
        input_paths: List of input file paths
        default_suffix: Suffix for auto-generated output dir

    Returns:
        Absolute output directory path
    """
    if output_arg:
        return str(Path(output_arg).resolve())

    # Auto-generate based on first input
    if input_paths:
        first = Path(input_paths[0])
        if first.is_file():
            return str(first.parent / f"{first.stem}{default_suffix}")
        else:
            return str(first / default_suffix)

    return str(Path.cwd() / default_suffix)