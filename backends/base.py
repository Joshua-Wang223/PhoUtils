"""Abstract base class and registry for SDR-to-HDR backends."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class SDRToHDRBackend(abc.ABC):
    """Abstract base class for all SDR-to-HDR conversion backends."""

    # Class attributes to be overridden by subclasses
    name: str = "base"
    description: str = "Base SDR-to-HDR backend"
    required_deps: List[str] = []
    supports_batch: bool = False
    output_formats: List[str] = ["exr"]  # Default native output format

    def __init__(self, model_dir: Optional[Path] = None, **kwargs):
        """
        Initialize backend.

        Args:
            model_dir: Directory for model weights/storage
            **kwargs: Backend-specific configuration
        """
        self.model_dir = model_dir or Path("./models") / self.name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._prepared = False
        self._device: Optional[str] = None
        self._config = kwargs

    @abc.abstractmethod
    def check_available(self) -> Tuple[bool, str]:
        """
        Check if backend is available (dependencies installed, models present).

        Returns:
            Tuple of (available: bool, reason: str)
        """
        ...

    @abc.abstractmethod
    def prepare(self, device: str, **kwargs) -> None:
        """
        Prepare backend for inference (load models, initialize runtime).

        Args:
            device: Target device ("cuda", "cpu", "mps")
            **kwargs: Additional preparation options (quantization, etc.)
        """
        ...

    @abc.abstractmethod
    def convert(
        self,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> bool:
        """
        Convert a single SDR image to HDR.

        Args:
            input_path: Path to input SDR image
            output_path: Path for output HDR image
            **kwargs: Conversion options (format, colorspace, target_nits, etc.)

        Returns:
            True if successful, False otherwise
        """
        ...

    def convert_batch(
        self,
        input_paths: List[str],
        output_dir: str,
        **kwargs
    ) -> List[Tuple[str, bool]]:
        """
        Batch convert multiple images (default: serial fallback).

        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            **kwargs: Conversion options

        Returns:
            List of (output_path, success) tuples
        """
        results = []
        for inp in input_paths:
            stem = Path(inp).stem
            # Default to first supported format
            ext = f".{self.output_formats[0]}"
            out = str(Path(output_dir) / f"{stem}{ext}")
            success = self.convert(inp, out, **kwargs)
            results.append((out, success))
        return results

    def get_memory_estimate(self) -> Tuple[float, float]:
        """
        Estimate memory usage in GB.

        Returns:
            Tuple of (base_memory_gb, per_image_peak_gb)
        """
        return 0.5, 0.1  # Conservative defaults

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    @property
    def device(self) -> Optional[str]:
        return self._device


class BackendRegistry:
    """Registry for backend classes."""

    _backends: Dict[str, Type[SDRToHDRBackend]] = {}

    @classmethod
    def register(cls, backend_class: Type[SDRToHDRBackend]) -> Type[SDRToHDRBackend]:
        """Register a backend class."""
        if not issubclass(backend_class, SDRToHDRBackend):
            raise TypeError(f"{backend_class} must inherit from SDRToHDRBackend")
        if backend_class.name in cls._backends:
            logger.warning(f"Overriding backend: {backend_class.name}")
        cls._backends[backend_class.name] = backend_class
        return backend_class

    @classmethod
    def get(cls, name: str) -> Type[SDRToHDRBackend]:
        """Get backend class by name."""
        if name not in cls._backends:
            raise KeyError(f"Backend not found: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name]

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered backend names."""
        return sorted(cls._backends.keys())

    @classmethod
    def list_all(cls) -> Dict[str, Type[SDRToHDRBackend]]:
        """Get all registered backends."""
        return dict(cls._backends)


def create_backend(
    name: str,
    model_dir: Optional[Path] = None,
    **kwargs
) -> SDRToHDRBackend:
    """Factory function to create a backend instance."""
    backend_class = BackendRegistry.get(name)
    return backend_class(model_dir=model_dir, **kwargs)


def list_backends() -> List[Dict[str, Any]]:
    """List all available backends with metadata."""
    result = []
    for name, cls in BackendRegistry.list_all().items():
        result.append({
            "name": name,
            "description": cls.description,
            "required_deps": cls.required_deps,
            "supports_batch": cls.supports_batch,
            "output_formats": cls.output_formats,
        })
    return result