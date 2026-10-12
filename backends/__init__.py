"""SDR-to-HDR backends package."""

from .base import SDRToHDRBackend, BackendRegistry, create_backend, list_backends

# 导入后端实现（导入即注册到 BackendRegistry）
from . import pyultrahdr_backend  # noqa: F401
from . import hdrconv_backend     # noqa: F401
from . import subprocess_backend  # noqa: F401
from . import unet_onnx_backend   # noqa: F401
from . import lumipic_backend     # noqa: F401
from . import gmnet_backend       # noqa: F401

__all__ = [
    "SDRToHDRBackend",
    "BackendRegistry",
    "create_backend",
    "list_backends",
]