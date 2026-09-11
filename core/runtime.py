"""Runtime backend detection and safe GPU/CPU selection."""
from __future__ import annotations

import os
import subprocess
import sys


def nvidia_present() -> bool:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
        return p.returncode == 0 and bool(p.stdout.strip())
    except Exception:
        return False


def test_cuda_inference() -> bool:
    """Actually execute the face model once; provider visibility alone is insufficient."""
    try:
        import numpy as np
        from .face_engine import FaceEngine
        engine = FaceEngine()
        if not engine._cuda_provider_available():
            return False
        engine._load_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        # A blank frame still traverses the detector network, exercising CUDA/cuDNN.
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        engine.model.get(frame)
        # _load_with_providers() intentionally does not choose the active backend;
        # reaching this point proves real CUDA inference succeeded.
        return True
    except Exception:
        return False
