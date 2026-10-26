"""Preload the pip-installed CUDA 12 cuBLAS/cuDNN libraries so CTranslate2 finds them."""

from __future__ import annotations

import ctypes
import importlib
from pathlib import Path

_LIBRARIES: dict[str, tuple[str, ...]] = {
    "nvidia.cublas": ("libcublasLt.so.12", "libcublas.so.12"),
    "nvidia.cudnn": (
        "libcudnn_graph.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn.so.9",
    ),
}


def preload_cuda_libraries() -> list[str]:
    """Load the libraries globally; returns the ones that were found."""
    loaded: list[str] = []
    for package_name, library_names in _LIBRARIES.items():
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            continue
        library_dir = Path(next(iter(package.__path__))) / "lib"
        for library_name in library_names:
            library_path = library_dir / library_name
            if library_path.exists():
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(library_name)
    return loaded
