"""Camera discovery and capture, including USB webcams and RTSP streams."""

from __future__ import annotations

import cv2


class Camera:
    """Unified camera source. ``source`` may be a USB index or an RTSP URL."""

    def __init__(self, source=0):
        self.source = source
        self.index = int(source) if isinstance(source, int) or str(source).isdigit() else None
        self.camera = None
        self.open(source)

    @staticmethod
    def list_cameras(max_indices=10):
        found = []
        for index in range(max_indices):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
            cap.release()
        return found

    @staticmethod
    def is_rtsp(source):
        return isinstance(source, str) and source.lower().startswith("rtsp://")

    def open(self, source):
        if self.camera is not None:
            self.release()
        self.source = source

        if self.is_rtsp(source):
            self.index = None
            # FFmpeg is preferred for RTSP. CAP_PROP_BUFFERSIZE=1 helps keep
            # recognition close to real-time instead of processing stale frames.
            self.camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = cv2.VideoCapture(source)
            if not self.camera.isOpened():
                self.camera = None
                raise RuntimeError("RTSP stream could not be opened.")
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return

        self.index = int(source)
        self.camera = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(self.index)
        if not self.camera.isOpened():
            self.camera = None
            raise RuntimeError(f"Camera {self.index} could not be opened.")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self):
        if self.camera is None:
            return None
        success, frame = self.camera.read()
        return frame if success else None

    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
