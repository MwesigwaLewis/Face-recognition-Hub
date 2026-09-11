"""InsightFace engine with automatic NVIDIA GPU -> CPU fallback."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np


class FaceEngine:
    def __init__(self, det_size=(800, 800)):
        self.model = None
        self.det_size = det_size
        self.det_thresh = 0.55
        self.providers = []
        self.active_backend = "UNINITIALIZED"
        self.last_gpu_error = None

    @staticmethod
    def _cuda_provider_available() -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    @staticmethod
    def gpu_available() -> bool:
        return FaceEngine._cuda_provider_available()

    def _configure_nvidia_dlls(self):
        try:
            import onnxruntime as ort
            site_packages = Path(ort.__file__).resolve().parent.parent
            root = site_packages / "nvidia"
            for name in ("cudnn", "cublas", "cuda_runtime"):
                d = root / name / "bin"
                if d.is_dir():
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            import onnxruntime as _ort
            if hasattr(_ort, "preload_dlls"):
                try:
                    _ort.preload_dlls(directory="")
                except TypeError:
                    _ort.preload_dlls()
        except Exception:
            # CPU mode does not need NVIDIA DLLs.
            pass

    def _load_with_providers(self, providers):
        import insightface
        self._configure_nvidia_dlls()
        model = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        model.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=self.det_size, det_thresh=self.det_thresh)
        self.model = model
        self.providers = list(providers)

    def _exercise_model(self):
        # Run an actual detector inference. This catches unsupported GPU kernels.
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.get(blank)

    def load_model(self):
        if self.model is not None:
            return

        import onnxruntime as ort
        cuda = "CUDAExecutionProvider" in ort.get_available_providers()

        if cuda:
            try:
                self._load_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
                self._exercise_model()
                self.active_backend = "CUDA"
                print("LEWIS: NVIDIA GPU inference enabled")
                return
            except Exception as exc:
                self.last_gpu_error = str(exc)
                self.model = None
                print(f"LEWIS: GPU test failed; falling back to CPU. {exc}")

        # CPU is always the final supported backend.
        self._load_with_providers(["CPUExecutionProvider"])
        self.active_backend = "CPU"
        print("LEWIS: CPU inference enabled")

    def get_faces(self, image):
        if self.model is None:
            self.load_model()
        return self.model.get(image)

    def extract_embedding(self, image):
        faces = self.get_faces(image)
        if len(faces) != 1:
            return None, "Exactly one face must be visible."
        face = faces[0]
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        area_ratio = area / float(w * h) if w and h else 0.0
        det_score = float(getattr(face, "det_score", 1.0))
        quality = det_score * min(1.0, area_ratio / 0.08)
        if quality < 0.45:
            return None, "Move closer and make sure your face is clearly visible."
        return self.normalize(face.embedding), "OK"

    @staticmethod
    def normalize(embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def compare_faces(self, embedding1, embedding2):
        return float(np.dot(self.normalize(embedding1), self.normalize(embedding2)))
