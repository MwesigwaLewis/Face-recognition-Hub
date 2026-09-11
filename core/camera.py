"""Camera discovery and capture, including USB/external webcams."""

from __future__ import annotations

import cv2


class Camera:
    def __init__(self, index=0):
        self.index = int(index)
        self.camera = None
        self.open(self.index)

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

    def open(self, index):
        if self.camera is not None:
            self.release()
        self.index = int(index)
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
