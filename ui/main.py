"""Lewiscrypt HUB desktop application.

GPU recognition runs in a worker thread so camera/UI rendering stays responsive.
Enrollment uses a guided multi-angle capture flow rather than one instant snapshot.
"""

from __future__ import annotations

import sys
import time
import threading
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QMessageBox, QLabel, QPushButton, QLineEdit, QFormLayout,
    QFrame, QScrollArea, QGridLayout, QComboBox, QProgressBar, QSlider, QTabWidget, QSpinBox, QCheckBox, QSizePolicy
)

from .chrome_bars import TopBar, BottomNav
from .sidebar import Sidebar
from .video_panel import VideoPanel
from .recognition_interface import RecognitionEngine, FaceResult, DemoEngine
from core.face_engine import FaceEngine
from core import database
from core import settings as app_settings
from core.enrollment import enroll_person
from core.face_pose import estimate_head_pose, analyze_lighting, ring_segment
from core.camera import Camera
from core.surveillance import SurveillanceTracker


class StreamReader:
    """Independent RTSP reader that keeps only the newest frame.

    A stalled camera is isolated and periodically reconnected so one bad
    channel cannot block the rest of the camera wall.
    """
    def __init__(self, source):
        self.source = source
        self.camera = None
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error = ""
        self.connected = False
        self.last_frame_time = 0.0

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        failures = 0
        while not self._stop.is_set():
            try:
                if self.camera is None:
                    self.camera = Camera(self.source)
                    self.connected = True
                    self.error = ""
                    failures = 0
                frame = self.camera.get_frame()
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
                        self.last_frame_time = time.monotonic()
                    failures = 0
                else:
                    failures += 1
                    if failures >= 20:
                        self.error = "RTSP stream stopped delivering frames."
                        self.connected = False
                        if self.camera:
                            self.camera.release()
                            self.camera = None
                        failures = 0
                        self._stop.wait(1.0)
                    else:
                        self._stop.wait(0.02)
            except Exception as exc:
                self.error = str(exc)
                self.connected = False
                if self.camera:
                    self.camera.release()
                    self.camera = None
                self._stop.wait(1.0)
        if self.camera:
            self.camera.release()
            self.camera = None
        self.connected = False

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self.camera:
            self.camera.release()
            self.camera = None


class CameraGrid(QWidget):
    """Responsive surveillance wall with optional single-camera focus mode."""
    camera_double_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.panels = {}
        self._focused_index = None
        self._cols = 1
        self._rows = 1

    def set_camera_count(self, count):
        for panel in list(self.panels.values()):
            self.grid.removeWidget(panel)
            panel.deleteLater()
        self.panels.clear()
        self._focused_index = None

        count = max(1, int(count))
        self._cols, self._rows = self._layout_for_count(count)
        for i in range(count):
            panel = VideoPanel()
            panel.setObjectName("cameraFeed")
            panel.set_compact_mode(count > 1)
            panel.double_clicked.connect(lambda idx=i: self.camera_double_clicked.emit(idx))
            self.panels[i] = panel
            row, col = divmod(i, self._cols)
            self.grid.addWidget(panel, row, col)

        self._set_stretch(self._cols, self._rows)

    @staticmethod
    def _layout_for_count(count):
        # Surveillance-oriented layouts: 8 -> 4x2, 9 -> 3x3, 16 -> 4x4.
        layouts = {
            1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2),
            5: (3, 2), 6: (3, 2), 7: (4, 2), 8: (4, 2),
            9: (3, 3), 10: (4, 3), 11: (4, 3), 12: (4, 3),
            13: (4, 4), 14: (4, 4), 15: (4, 4), 16: (4, 4),
        }
        if count <= 16:
            return layouts[count]
        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        return cols, rows

    def _set_stretch(self, cols, rows):
        for col in range(max(cols, self.grid.columnCount())):
            self.grid.setColumnStretch(col, 0)
        for row in range(max(rows, self.grid.rowCount())):
            self.grid.setRowStretch(row, 0)
        for col in range(cols):
            self.grid.setColumnStretch(col, 1)
        for row in range(rows):
            self.grid.setRowStretch(row, 1)

    def panel(self, index):
        return self.panels.get(index)

    def set_focus(self, index=None):
        """Put one feed into the entire camera area, or restore the wall."""
        if index is None or index not in self.panels:
            self._focused_index = None
            for panel in self.panels.values():
                self.grid.removeWidget(panel)
                panel.show()
            for i, panel in self.panels.items():
                row, col = divmod(i, self._cols)
                self.grid.addWidget(panel, row, col)
            for panel in self.panels.values():
                panel.set_compact_mode(len(self.panels) > 1)
            self._set_stretch(self._cols, self._rows)
            return

        self._focused_index = index
        for panel in self.panels.values():
            self.grid.removeWidget(panel)
            panel.hide()
        focused = self.panels[index]
        focused.set_compact_mode(False)
        focused.show()
        self.grid.addWidget(focused, 0, 0, 1, 1)
        self._set_stretch(1, 1)


@dataclass
class EnrollmentFrame:
    """Result of analyzing one camera frame during free-form sweep capture.

    There is no fixed target pose here -- the user just moves their head
    naturally. This is produced for every submitted frame so the UI can keep
    a live tracking box on screen continuously; the drawer decides whether
    each good frame is a novel-enough angle to keep.
    """
    have_face: bool
    bbox: tuple | None = None          # (x, y, w, h) in frame coordinates
    landmarks: list | None = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    ring_segment: int | None = None    # 0..11 position on the capture ring, or None if centered/unavailable
    lighting_ok: bool = True
    lighting_label: str = ""
    pose_ok: bool = True               # basic sanity check, not a fixed target
    quality_ok: bool = True
    embedding: object = None           # set only when this frame is capture-worthy
    thumbnail: object = None
    message: str = ""

BASE = Path(__file__).resolve().parents[1]


class RealEngine(RecognitionEngine):
    """Recognition engine. The model is created in the recognition worker thread."""

    def __init__(self):
        self.engine = FaceEngine()
        self.engine.load_model()
        self.people = []
        self.refresh()

    def refresh(self):
        self.people = database.get_people_records()
        # Vectorize gallery comparison: one face embedding is compared against
        # the whole enrolled gallery in one NumPy operation instead of nested
        # Python loops. This matters when many cameras are sampling faces.
        vectors = []
        owners = []
        names = {}
        for person in self.people:
            names[person["id"]] = person["name"]
            for emb in person.get("embeddings", []):
                v = self.engine.normalize(emb)
                if v.size:
                    vectors.append(v)
                    owners.append(person["id"])
        self._gallery = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 512), dtype=np.float32)
        self._gallery_owner = owners
        self._gallery_names = names

    def process_enrollment_frame(self, frame) -> EnrollmentFrame:
        """Analyze one frame for free-form sweep-capture enrollment.

        No fixed target pose: this always reports bbox/pose/zone/lighting/
        quality when a face is visible, and only sets embedding/thumbnail
        once the frame is capture-worthy (single face, good lighting, good
        framing). The caller (EnrollDrawer) decides whether the angle is
        novel enough relative to what's already been captured.
        """
        faces = self.engine.get_faces(frame)
        if len(faces) == 0:
            return EnrollmentFrame(have_face=False, message="No face detected — center your face in frame.")
        if len(faces) > 1:
            return EnrollmentFrame(have_face=False, message="More than one face visible — only one person at a time.")

        face = faces[0]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        bbox = (x1c, y1c, max(0, x2c - x1c), max(0, y2c - y1c))
        landmarks = [(int(px), int(py)) for px, py in getattr(face, "kps", [])]

        pose = estimate_head_pose(getattr(face, "kps", []), frame.shape)
        lighting = analyze_lighting(frame, face.bbox)
        yaw, pitch, roll = pose if pose else (0.0, 0.0, 0.0)
        segment = ring_segment(pose)
        # Sanity check only (catches a garbled/degenerate landmark read) --
        # not a target the user has to hit, unlike the old guided flow.
        pose_ok = pose is not None and abs(roll) <= 30

        area_ratio = bbox[2] * bbox[3] / float(h * w) if h and w else 0.0
        det_score = float(getattr(face, "det_score", 1.0))
        # A typical laptop-webcam sitting distance often puts a face at only
        # ~4-6% of frame area, not the 8% originally assumed here (untested
        # against real hardware) -- loosened so normal seating distance
        # doesn't get silently rejected as "too far".
        quality = det_score * min(1.0, area_ratio / 0.045)
        quality_ok = quality >= 0.35

        result = EnrollmentFrame(
            have_face=True, bbox=bbox, landmarks=landmarks,
            yaw=yaw, pitch=pitch, roll=roll, ring_segment=segment,
            lighting_ok=lighting["ok"], lighting_label=lighting["label"],
            pose_ok=pose_ok, quality_ok=quality_ok,
        )
        # IMPORTANT: the face embedding comes directly from the detector's
        # own model output (face.embedding), not from pose estimation --
        # pose is only used to pick a ring segment for spreading captures
        # around. Gating capture on pose_ok here (as an earlier version did)
        # meant that if pose estimation ever failed on real camera frames --
        # for any reason, including one this sandbox's synthetic tests can't
        # reproduce -- every single frame would silently fail to capture
        # with NO visible error, since pose_ok isn't a banner-worthy gate.
        # Lighting and quality are the only things actually worth blocking
        # on here, and both already have visible banners.
        if lighting["ok"] and quality_ok and bbox[2] > 0 and bbox[3] > 0:
            result.embedding = self.engine.normalize(face.embedding)
            result.thumbnail = frame[y1c:y2c, x1c:x2c].copy()
        return result

    def process_frame(self, frame):
        results = []
        h, w = frame.shape[:2]
        gallery = self._gallery
        threshold = app_settings.get_recognition_threshold()
        for face in self.engine.get_faces(frame):
            # Surveillance cameras produce occasional false face detections
            # from leaves, branches and high-contrast textures. Require a
            # stronger detector score and a minimally usable face footprint
            # before sending an observation into temporal recognition.
            det_score = float(getattr(face, "det_score", 0.0))
            x1d, y1d, x2d, y2d = map(int, face.bbox)
            if det_score < 0.55 or (x2d - x1d) < 28 or (y2d - y1d) < 28:
                continue
            embedding = self.engine.normalize(face.embedding)
            best, score, pid = "UNKNOWN", 0.0, None
            if gallery.size:
                scores = gallery @ embedding
                best_index = int(np.argmax(scores))
                score = float(scores[best_index])
                pid = self._gallery_owner[best_index]
                best = self._gallery_names.get(pid, "UNKNOWN")
            if score < threshold:
                # Keep the strongest candidate attached to the observation even
                # when it is below the acceptance threshold. The surveillance
                # tracker needs this evidence across several frames before it
                # can decide whether the identity is real. The UI still shows
                # UNKNOWN until the track becomes stable.
                best = "UNKNOWN"
            x1, y1, x2, y2 = map(int, face.bbox)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            thumb = frame[y1c:y2c, x1c:x2c].copy() if x2c > x1c and y2c > y1c else None
            area_ratio = max(0, x2-x1) * max(0, y2-y1) / float(max(1, w*h))
            det = float(getattr(face, "det_score", 1.0))
            sharp = 0.0
            if thumb is not None and thumb.size:
                gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
                sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 180.0)
            size_quality = min(1.0, area_ratio / 0.035)
            # Quality deliberately does not reject side profiles; size, detector
            # confidence and sharpness decide whether an observation is useful.
            quality = max(0.0, min(1.0, det * 0.50 + size_quality * 0.30 + sharp * 0.20))
            pose = estimate_head_pose(getattr(face, "kps", []), frame.shape)
            lighting = analyze_lighting(frame, face.bbox)
            yaw, pitch, roll = pose if pose else (0.0, 0.0, 0.0)
            results.append(FaceResult(
                best, float(score), (x1, y1, x2 - x1, y2 - y1),
                [(int(x), int(y)) for x, y in getattr(face, "kps", [])], pid, thumb,
                yaw, pitch, roll, quality, embedding, None
            ))
        return results


class RecognitionWorker(QThread):
    results_ready = Signal(object)
    engine_ready = Signal(str)
    error = Signal(str)
    enrollment_ready = Signal(object)  # emits an EnrollmentFrame
    enrollment_error = Signal(str)     # a frame crashed during enrollment analysis -- distinct
                                        # from the general `error` signal so EnrollDrawer can show
                                        # it persistently instead of it getting lost as a one-frame
                                        # flash on the video panel while the drawer sits frozen

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._latest_frame = None
        self._stop = False
        self._latest_mode = "recognition"
        self._latest_batch = None
        self._refresh_requested = False
        self._generation = 0
        self.engine = None

    def submit(self, frame, mode="recognition"):
        with self._condition:
            self._latest_frame = frame.copy()
            self._latest_mode = mode
            self._condition.notify()

    def submit_batch(self, frames):
        with self._condition:
            self._latest_batch = {k: v.copy() for k, v in frames.items()}
            self._latest_frame = None
            self._latest_mode = "batch"
            self._condition.notify()

    def request_refresh(self):
        with self._condition:
            self._generation += 1
            self._refresh_requested = True
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stop = True
            self._condition.notify()
        self.wait(5000)

    def run(self):
        try:
            self.engine = RealEngine()
            self.engine_ready.emit(self.engine.engine.active_backend)
        except Exception as exc:
            self.error.emit(str(exc))
            return

        while True:
            with self._condition:
                while self._latest_frame is None and self._latest_batch is None and not self._stop and not self._refresh_requested:
                    self._condition.wait(0.5)
                if self._refresh_requested:
                    self._refresh_requested = False
                    if self.engine:
                        self.engine.refresh()
                    if self._latest_frame is None and self._latest_batch is None and not self._stop:
                        continue
                if self._stop:
                    return
                frame = self._latest_frame
                batch = self._latest_batch
                mode = self._latest_mode
                generation = self._generation
                self._latest_frame = None
                self._latest_batch = None

            try:
                if mode == "enrollment":
                    result = self.engine.process_enrollment_frame(frame)
                    self.enrollment_ready.emit(result)
                elif mode == "batch":
                    results = {index: self.engine.process_frame(image) for index, image in batch.items()}
                    self.results_ready.emit((generation, results))
                else:
                    faces = self.engine.process_frame(frame)
                    self.results_ready.emit((generation, faces))
            except Exception as exc:
                if mode == "enrollment":
                    self.enrollment_error.emit(str(exc))
                else:
                    self.error.emit(str(exc))


class EnrollDrawer(QFrame):
    """Guided enrollment panel docked beside the live camera view.

    Face-ID-style free-form capture: the user just moves their head around
    in a rough circle (up, right, down, left...) while the system rapidly
    captures distinct angles in the background, gated only by "is there one
    well-lit, close-enough face right now" — no fixed pose target to hit and
    hold. Coverage is tracked across a 12-position capture ring (like a clock
    face), capped per position, so angles spread out on their own without
    dictating which direction to face at any given moment.
    """

    finished = Signal(bool, str)

    TARGET_TOTAL = 20        # the real goal -- system keeps capturing until this many
                              # distinct angles are captured, full stop, no early settling
    MANUAL_FINISH_MIN = 3    # floor for the user's own "SKIP REST / END HERE" button --
                              # matches enroll_person()'s own minimum, since below that the
                              # save would just fail anyway. No automatic early-finish exists
                              # anymore; stopping short of 20 is entirely the user's call.
    MAX_PER_SEGMENT = 2      # cap captures kept from any single ring position
    MAX_CENTERED = TARGET_TOTAL  # no extra cap when pose is unavailable/dead-center;
                                  # the embedding-novelty check below already guards
                                  # diversity independently of ring classification
    HARD_CAP_TIME = 60.0     # seconds; a true last-resort safety net only (e.g. the user
                              # walked away) -- not a normal-path trigger anymore
    DUPLICATE_SIM_THRESHOLD = 0.985   # reject a capture too similar to one already kept
    SUBMIT_INTERVAL = 0.09   # ~11 fps of enrollment analysis

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setObjectName("enrollDrawer")
        self.setFixedWidth(360)

        self.embeddings = []
        self.thumbnails = []
        self.segment_counts = {}   # ring segment (int) or "CENTER" -> count kept
        self.covered_segments = set()
        self.active = False
        self.name = ""
        self.profile_thumbnail = None
        self.start_time = 0.0
        self._last_submit = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(14)

        title = QLabel("ENROLL PERSON")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        sub = QLabel("Slowly move your head in a circle — up, right, down, left — while "
                      "staying in frame. Captures happen automatically; there's no specific "
                      "direction to hit.")
        sub.setObjectName("pageSubtitle")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self.form_card = QFrame()
        self.form_card.setObjectName("pageCard")
        form = QFormLayout(self.form_card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(10)
        self.name_field = QLineEdit()
        self.name_field.setPlaceholderText("e.g. Lewis")
        self.name_field.setObjectName("input")
        form.addRow("Person name", self.name_field)
        root.addWidget(self.form_card)

        self.name_label = QLabel("")
        self.name_label.setObjectName("pageSubtitle")
        self.name_label.hide()
        root.addWidget(self.name_label)

        self.instruction = QLabel("Enter a name, then start the sweep capture.")
        self.instruction.setObjectName("pageSubtitle")
        self.instruction.setWordWrap(True)
        root.addWidget(self.instruction)

        self.progress = QProgressBar()
        self.progress.setRange(0, self.TARGET_TOTAL)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        self.state_label = QLabel("READY")
        self.state_label.setObjectName("captureCountdown")
        self.state_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.state_label)

        btn_row = QHBoxLayout()
        self.start_button = QPushButton("START SWEEP CAPTURE")
        self.start_button.setObjectName("primaryBtn")
        self.start_button.clicked.connect(self.start)
        btn_row.addWidget(self.start_button)

        self.finish_button = QPushButton("SKIP REST / END HERE")
        self.finish_button.setObjectName("primaryBtn")
        self.finish_button.clicked.connect(self.finish_now)
        self.finish_button.hide()
        self.finish_button.setEnabled(False)
        btn_row.addWidget(self.finish_button)

        self.cancel_button = QPushButton("CANCEL")
        self.cancel_button.setObjectName("dangerBtn")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.hide()
        btn_row.addWidget(self.cancel_button)
        root.addLayout(btn_row)
        root.addStretch()

    # ---- lifecycle --------------------------------------------------

    def start(self):
        name = self.name_field.text().strip()
        if not name:
            self.instruction.setText("Enter the person's name first.")
            return
        self.name = name
        self.embeddings = []
        self.thumbnails = []
        self.segment_counts = {}
        self.covered_segments = set()
        self.profile_thumbnail = None
        self.active = True
        self.start_time = time.monotonic()
        self._last_submit = 0.0

        self.form_card.hide()
        self.name_label.setText(f"ENROLLING: {name.upper()}")
        self.name_label.show()
        self.start_button.hide()
        self.cancel_button.show()
        self.finish_button.show()
        self.finish_button.setEnabled(False)
        self.instruction.setText("Slowly move your head in a circle.")
        self.state_label.setText("CAPTURING…")
        self.progress.setValue(0)
        self.parent_window.video_panel.set_enrollment_thumbnails([])

    def cancel(self):
        self.active = False
        self.embeddings = []
        self.thumbnails = []
        self.segment_counts = {}
        self.covered_segments = set()

        self.form_card.show()
        self.name_label.hide()
        self.start_button.show()
        self.cancel_button.hide()
        self.finish_button.hide()
        self.progress.setValue(0)
        self.state_label.setText("READY")
        self.instruction.setText("Setup cancelled. You can start again when ready.")
        self.parent_window.video_panel.clear_enrollment_overlay()

    def finish_now(self):
        """The only way to stop before 20 angles -- entirely the user's
        call, once at least enough have been captured for a valid profile."""
        if not self.active or len(self.embeddings) < self.MANUAL_FINISH_MIN:
            return
        self._finish()

    # ---- frame loop ---------------------------------------------------

    def tick(self, frame):
        """Called every camera tick while active; throttles submission to
        the worker (which keeps only the latest frame)."""
        if not self.active or frame is None:
            return
        now = time.monotonic()
        if now - self._last_submit < self.SUBMIT_INTERVAL:
            return
        self._last_submit = now
        self.parent_window.worker.submit(frame, mode="enrollment")

    def on_error(self, message: str):
        """A frame crashed during enrollment analysis. Shown as a persistent,
        impossible-to-miss banner (not a one-frame flash) so a real failure
        is never confused with 'quietly waiting for a better angle'."""
        if not self.active:
            return
        video = self.parent_window.video_panel
        total = len(self.embeddings)
        self.state_label.setText("ERROR")
        self.instruction.setText(f"Something went wrong analyzing the camera frame: {message}")
        video.update_enrollment_overlay(
            state="error", bbox=None, landmarks=None,
            guidance="CAPTURE ERROR", detail=message,
            count=total, target=self.TARGET_TOTAL, zones=self.covered_segments)

    def on_frame_result(self, result: EnrollmentFrame):
        if not self.active:
            return
        video = self.parent_window.video_panel
        elapsed = time.monotonic() - self.start_time
        total = len(self.embeddings)

        if not result.have_face:
            video.update_enrollment_overlay(
                state="searching", bbox=None, landmarks=None,
                guidance="NO FACE DETECTED", detail=result.message,
                count=total, target=self.TARGET_TOTAL, zones=self.covered_segments)
            self.state_label.setText("NO FACE DETECTED")
            self._maybe_finish(elapsed)
            return

        if not result.lighting_ok:
            video.update_enrollment_overlay(
                state="issue", bbox=result.bbox, landmarks=result.landmarks,
                guidance=result.lighting_label, detail="Move to better, more even lighting.",
                count=total, target=self.TARGET_TOTAL, zones=self.covered_segments)
            self.state_label.setText(result.lighting_label)
            self._maybe_finish(elapsed)
            return

        if not result.quality_ok:
            video.update_enrollment_overlay(
                state="issue", bbox=result.bbox, landmarks=result.landmarks,
                guidance="MOVE CLOSER", detail="Face is too small or unclear from here.",
                count=total, target=self.TARGET_TOTAL, zones=self.covered_segments)
            self.state_label.setText("MOVE CLOSER")
            self._maybe_finish(elapsed)
            return

        # Good frame: try to accept it as a new, sufficiently distinct angle.
        # No "hold still" wait and no fixed target — just capture whenever
        # conditions are good and the angle adds something new.
        if result.embedding is not None:
            key = result.ring_segment if result.ring_segment is not None else "CENTER"
            cap = self.MAX_CENTERED if key == "CENTER" else self.MAX_PER_SEGMENT
            under_cap = self.segment_counts.get(key, 0) < cap
            novel = True
            if self.embeddings:
                similarity = max(float(np.dot(result.embedding, old)) for old in self.embeddings)
                novel = similarity <= self.DUPLICATE_SIM_THRESHOLD
            if under_cap and novel:
                self.embeddings.append(result.embedding)
                self.segment_counts[key] = self.segment_counts.get(key, 0) + 1
                if key != "CENTER":
                    self.covered_segments.add(key)
                if result.thumbnail is not None:
                    if not self.thumbnails:
                        self.profile_thumbnail = result.thumbnail.copy()
                    self.thumbnails.append(result.thumbnail.copy())
                    video.set_enrollment_thumbnails(self.thumbnails[-10:])  # keep the strip compact
                video.flash_capture()

        total = len(self.embeddings)
        self.progress.setValue(min(total, self.TARGET_TOTAL))
        self.finish_button.setEnabled(total >= self.MANUAL_FINISH_MIN)
        self.state_label.setText(f"{total} / {self.TARGET_TOTAL} CAPTURED")
        video.update_enrollment_overlay(
            state="capturing", bbox=result.bbox, landmarks=result.landmarks,
            guidance="", detail="",
            count=total, target=self.TARGET_TOTAL, zones=self.covered_segments)
        self._maybe_finish(elapsed)

    def _maybe_finish(self, elapsed):
        # The only automatic finish is reaching the true target. Stopping
        # short of that is now exclusively the user's call via "SKIP REST /
        # END HERE" -- except for HARD_CAP_TIME, a true last-resort safety
        # net (e.g. the user walked away mid-capture) rather than a normal
        # expected path.
        total = len(self.embeddings)
        if total >= self.TARGET_TOTAL:
            self._finish()
        elif elapsed >= self.HARD_CAP_TIME:
            self._finish()  # finish with whatever we have; enroll_person enforces its own floor

    def _finish(self):
        if not self.active:
            return  # idempotency guard: never re-run enroll_person / _people_changed twice
        self.active = False
        thumbnail = self.profile_thumbnail if self.profile_thumbnail is not None else (
            self.thumbnails[-1] if self.thumbnails else None)
        ok, message = enroll_person(self.name, self.embeddings, thumbnail)

        self.form_card.show()
        self.name_label.hide()
        self.start_button.show()
        self.cancel_button.hide()
        self.finish_button.hide()
        self.parent_window.video_panel.clear_enrollment_overlay()

        if ok:
            self.name_field.clear()
            self.parent_window._people_changed()
            self.instruction.setText(message)
            self.state_label.setText("DONE")
            self.finished.emit(True, message)
        else:
            self.instruction.setText(message)
            self.state_label.setText("FAILED")
            self.finished.emit(False, message)


class LogsPage(QWidget):
    """Persistent surveillance history with compact evidence thumbnails."""
    def __init__(self, parent):
        super().__init__(); self.parent_window = parent
        root = QVBoxLayout(self); root.setContentsMargins(50, 45, 50, 40); root.setSpacing(14)
        top = QHBoxLayout()
        title = QLabel("LOGS"); title.setObjectName("pageTitle"); top.addWidget(title)
        top.addStretch()
        self.clear_btn = QPushButton("CLEAR LOGS"); self.clear_btn.setObjectName("dangerBtn"); self.clear_btn.clicked.connect(self.clear_logs); top.addWidget(self.clear_btn)
        root.addLayout(top)
        self.sub = QLabel("Surveillance detections will appear here."); self.sub.setObjectName("pageSubtitle"); root.addWidget(self.sub)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.holder = QWidget(); self.list_layout = QVBoxLayout(self.holder); self.list_layout.setSpacing(8); self.list_layout.setContentsMargins(0, 4, 8, 4); self.scroll.setWidget(self.holder); root.addWidget(self.scroll, 1)
        self.refresh_timer = QTimer(self); self.refresh_timer.timeout.connect(self.refresh); self.refresh_timer.start(2000)

    def refresh(self):
        logs = database.get_detection_logs(100)
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.sub.setText(f"{len(logs)} recent detection{'s' if len(logs) != 1 else ''}")
        for entry in logs:
            card = QFrame(); card.setObjectName("personTile")
            row = QHBoxLayout(card); row.setContentsMargins(10, 8, 10, 8); row.setSpacing(12)
            image = QLabel(); image.setFixedSize(72, 72); image.setAlignment(Qt.AlignCenter); image.setObjectName("personAvatar")
            if entry["snapshot"]:
                pix = QPixmap(); pix.loadFromData(entry["snapshot"], "JPG")
                image.setPixmap(pix.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else: image.setText("NO\nIMAGE")
            row.addWidget(image)
            info = QVBoxLayout()
            name = QLabel(str(entry["name"]).upper()); name.setObjectName("tileName"); info.addWidget(name)
            conf = f"{entry['confidence'] * 100:.1f}%"
            info.addWidget(QLabel(f"Camera {entry['camera']}  •  confidence {conf}"))
            info.addWidget(QLabel(str(entry["timestamp"])))
            row.addLayout(info, 1)
            self.list_layout.addWidget(card)
        self.list_layout.addStretch()

    def clear_logs(self):
        if QMessageBox.question(self, "Clear logs", "Delete all surveillance detection history?") == QMessageBox.Yes:
            database.clear_detection_logs(); self.refresh()


class PlaceholderPage(QWidget):
    def __init__(self, title, subtitle):
        super().__init__()
        l = QVBoxLayout(self); l.setContentsMargins(50, 50, 50, 50)
        h = QLabel(title); h.setObjectName("pageTitle"); s = QLabel(subtitle); s.setObjectName("pageSubtitle")
        l.addWidget(h); l.addWidget(s); l.addStretch()


class PeoplePage(QWidget):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(); self.parent_window = parent
        root = QVBoxLayout(self); root.setContentsMargins(50, 45, 50, 40); root.setSpacing(16)
        t = QLabel("PEOPLE"); t.setObjectName("pageTitle"); root.addWidget(t)
        self.sub = QLabel(); self.sub.setObjectName("pageSubtitle"); root.addWidget(self.sub)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.holder = QWidget(); self.grid = QGridLayout(self.holder); self.grid.setSpacing(14); self.scroll.setWidget(self.holder); root.addWidget(self.scroll)

    def refresh(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        people = database.get_people_records()
        self.sub.setText(f"{len(people)} enrolled {'person' if len(people) == 1 else 'people'}")
        for i, person in enumerate(people):
            name, embeddings = person["name"], person["embeddings"]
            card = QFrame(); card.setObjectName("personTile"); l = QVBoxLayout(card)
            avatar = QLabel(); avatar.setFixedSize(128, 128); avatar.setAlignment(Qt.AlignCenter)
            if person["thumbnail"]:
                pix = QPixmap(); pix.loadFromData(person["thumbnail"], "JPG")
                avatar.setPixmap(pix.scaled(128, 128, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            else:
                avatar.setText("NO\nPHOTO"); avatar.setObjectName("personAvatar")
            l.addWidget(avatar, 0, Qt.AlignCenter)
            n = QLabel(name.upper()); n.setObjectName("tileName"); l.addWidget(n)
            d = QLabel(f"{len(embeddings)} face angles stored"); d.setObjectName("tileMeta"); l.addWidget(d)
            b = QPushButton("REMOVE"); b.setObjectName("dangerBtn"); b.clicked.connect(lambda _, x=name: self.remove(x)); l.addWidget(b)
            self.grid.addWidget(card, i // 3, i % 3)

    def remove(self, name):
        if QMessageBox.question(self, "Remove person", f"Remove {name} from the database?") == QMessageBox.Yes:
            database.delete_person(name); self.parent_window._people_changed(); self.refresh(); self.changed.emit()


class SettingsPage(QWidget):
    """Organized application settings, grouped by subsystem."""

    def __init__(self, parent):
        super().__init__(); self.parent_window = parent
        root = QVBoxLayout(self); root.setContentsMargins(50, 45, 50, 40); root.setSpacing(16)
        t = QLabel("SETTINGS"); t.setObjectName("pageTitle"); root.addWidget(t)
        s = QLabel("Configure cameras, display behavior, and face recognition."); s.setObjectName("pageSubtitle"); root.addWidget(s)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("settingsTabs")
        root.addWidget(self.tabs, 1)

        self._build_camera_tab()
        self._build_recognition_tab()
        self._build_system_tab()

        self.status = QLabel(""); self.status.setObjectName("infoLabel"); root.addWidget(self.status)

        self.refresh_btn.clicked.connect(self.refresh)
        self.apply_btn.clicked.connect(self.apply)
        self.rtsp_btn.clicked.connect(self.apply_rtsp)
        self.all_rtsp_btn.clicked.connect(self.apply_all_rtsp)
        self.apply_selection_btn.clicked.connect(self.apply_enabled_cameras)
        self.enable_all_btn.clicked.connect(lambda: self._set_all_camera_checks(True))
        self.disable_all_btn.clicked.connect(lambda: self._set_all_camera_checks(False))
        self.camera_count.valueChanged.connect(self._rebuild_camera_checks)
        self.source_mode.currentIndexChanged.connect(self._update_camera_mode_visibility)
        self.aspect_combo.currentTextChanged.connect(self._apply_aspect)
        self._rebuild_camera_checks()
        self._update_camera_mode_visibility()
        self.refresh()

    def _build_camera_tab(self):
        page = QWidget(); root = QVBoxLayout(page)
        root.setContentsMargins(8, 20, 8, 8); root.setSpacing(14)

        title = QLabel("CAMERA SETTINGS"); title.setObjectName("sectionTitle"); root.addWidget(title)
        sub = QLabel("Choose how Lewiscrypt receives video. Only settings relevant to the selected source are shown.")
        sub.setObjectName("pageSubtitle"); sub.setWordWrap(True); root.addWidget(sub)

        source_card = QFrame(); source_card.setObjectName("pageCard")
        source_form = QFormLayout(source_card); source_form.setContentsMargins(24, 24, 24, 24); source_form.setSpacing(14)
        self.source_mode = QComboBox(); self.source_mode.addItems(["Local camera", "RTSP / DVR"])
        saved_source = app_settings.get_camera_source()
        self.source_mode.setCurrentIndex(1 if saved_source.lower().startswith("rtsp://") else 0)
        source_form.addRow("Video source", self.source_mode)
        root.addWidget(source_card)

        # Local-camera controls
        self.local_card = QFrame(); self.local_card.setObjectName("pageCard")
        local_form = QFormLayout(self.local_card); local_form.setContentsMargins(24, 24, 24, 24); local_form.setSpacing(14)
        self.combo = QComboBox(); self.combo.setObjectName("cameraCombo")
        local_form.addRow("Local camera", self.combo)
        root.addWidget(self.local_card)

        # RTSP/DVR controls
        self.rtsp_card = QFrame(); self.rtsp_card.setObjectName("pageCard")
        rtsp_form = QFormLayout(self.rtsp_card); rtsp_form.setContentsMargins(24, 24, 24, 24); rtsp_form.setSpacing(14)
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://user:password@DVR_IP:554/Streaming/channels/102")
        self.rtsp_url.setText(saved_source if saved_source.lower().startswith("rtsp://") else "")
        rtsp_form.addRow("RTSP stream", self.rtsp_url)
        self.camera_count = QSpinBox(); self.camera_count.setRange(1, 16)
        self.camera_count.setValue(max(1, min(16, len(app_settings.get_rtsp_streams()) or 1)))
        rtsp_form.addRow("Configured cameras", self.camera_count)

        self.enabled_card = QFrame(); self.enabled_card.setObjectName("pageCard")
        enabled_layout = QVBoxLayout(self.enabled_card); enabled_layout.setContentsMargins(18,18,18,18); enabled_layout.setSpacing(10)
        enabled_title = QLabel("CAMERA SELECTION"); enabled_title.setObjectName("sectionTitle"); enabled_layout.addWidget(enabled_title)
        enabled_hint = QLabel("Disable broken or unused DVR channels. Disabled cameras are never opened, retried, displayed, or sent to recognition.")
        enabled_hint.setObjectName("infoLabel"); enabled_hint.setWordWrap(True); enabled_layout.addWidget(enabled_hint)
        self.camera_checks_widget = QWidget(); self.camera_checks_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.camera_checks_widget.setMinimumHeight(124)
        self.camera_checks_grid = QGridLayout(self.camera_checks_widget)
        self.camera_checks_grid.setContentsMargins(0, 0, 0, 0); self.camera_checks_grid.setHorizontalSpacing(24); self.camera_checks_grid.setVerticalSpacing(6)
        for col in range(4): self.camera_checks_grid.setColumnStretch(col, 1)
        for row in range(4): self.camera_checks_grid.setRowMinimumHeight(row, 26)
        enabled_layout.addWidget(self.camera_checks_widget)
        selection_row = QHBoxLayout()
        self.enable_all_btn = QPushButton("ENABLE ALL"); self.disable_all_btn = QPushButton("DISABLE ALL")
        self.apply_selection_btn = QPushButton("APPLY CAMERA SELECTION"); self.apply_selection_btn.setObjectName("primaryBtn")
        selection_row.addWidget(self.enable_all_btn); selection_row.addWidget(self.disable_all_btn); selection_row.addStretch(); selection_row.addWidget(self.apply_selection_btn)
        enabled_layout.addLayout(selection_row)
        self.rtsp_enabled_holder = self.enabled_card
        root.addWidget(self.enabled_card)

        display_card = QFrame(); display_card.setObjectName("pageCard")
        display_form = QFormLayout(display_card); display_form.setContentsMargins(24,24,24,24); display_form.setSpacing(14)
        self.aspect_combo = QComboBox(); self.aspect_combo.addItems(["Auto", "16:9", "4:3"])
        saved_aspect = app_settings.get_display_aspect_ratio()
        self.aspect_combo.setCurrentText(saved_aspect if saved_aspect in ("Auto", "16:9", "4:3") else "16:9")
        display_form.addRow("Display aspect ratio", self.aspect_combo)
        aspect_hint = QLabel("Display only — recognition continues using the original camera frame.")
        aspect_hint.setObjectName("infoLabel"); aspect_hint.setWordWrap(True); display_form.addRow("", aspect_hint)
        root.addWidget(display_card)

        row = QHBoxLayout()
        self.refresh_btn = QPushButton("SCAN CAMERAS"); self.refresh_btn.setObjectName("primaryBtn")
        self.apply_btn = QPushButton("USE LOCAL CAMERA"); self.apply_btn.setObjectName("primaryBtn")
        self.rtsp_btn = QPushButton("USE RTSP"); self.rtsp_btn.setObjectName("primaryBtn")
        self.all_rtsp_btn = QPushButton("GENERATE DVR CAMERAS"); self.all_rtsp_btn.setObjectName("primaryBtn")
        row.addWidget(self.refresh_btn); row.addWidget(self.apply_btn); row.addWidget(self.rtsp_btn); row.addWidget(self.all_rtsp_btn); row.addStretch(); root.addLayout(row)
        root.addStretch()
        self.tabs.addTab(page, "CAMERA")

    def _update_camera_mode_visibility(self):
        rtsp = self.source_mode.currentIndex() == 1
        self.local_card.setVisible(not rtsp)
        self.rtsp_card.setVisible(rtsp)
        self.rtsp_btn.setVisible(rtsp)
        self.all_rtsp_btn.setVisible(rtsp)
        self.refresh_btn.setVisible(not rtsp)
        self.apply_btn.setVisible(not rtsp)
        # Per-camera enable/disable only becomes useful once there are at
        # least three DVR cameras. One or two cameras remain simple.
        self.rtsp_enabled_holder.setVisible(rtsp and self.camera_count.value() >= 3)
        self.camera_checks_widget.setVisible(self.camera_count.value() >= 3)
        self.enabled_card.setProperty("active", rtsp and self.camera_count.value() >= 3)
        self.enabled_card.style().unpolish(self.enabled_card); self.enabled_card.style().polish(self.enabled_card)
    def _build_recognition_tab(self):
        page = QWidget(); root = QVBoxLayout(page); root.setContentsMargins(8, 20, 8, 8); root.setSpacing(14)
        title = QLabel("RECOGNITION SETTINGS"); title.setObjectName("sectionTitle"); root.addWidget(title)
        sub = QLabel(
            "Control how closely a live face must match a stored profile before Lewiscrypt identifies it. "
            "Higher values are stricter; lower values are more permissive."
        )
        sub.setObjectName("pageSubtitle"); sub.setWordWrap(True); root.addWidget(sub)

        card = QFrame(); card.setObjectName("pageCard")
        form = QFormLayout(card); form.setContentsMargins(24,24,24,24); form.setSpacing(14)
        slider_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(30, 95)
        self.threshold_slider.setValue(int(round(app_settings.get_recognition_threshold() * 100)))
        self.threshold_value = QLabel(f"{self.threshold_slider.value()}%")
        self.threshold_value.setFixedWidth(48)
        slider_row.addWidget(self.threshold_slider, 1); slider_row.addWidget(self.threshold_value)
        form.addRow("Identification threshold", slider_row)
        root.addWidget(card)
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_value.setText(f"{v}%"))
        self.threshold_slider.sliderReleased.connect(self._apply_threshold)
        root.addStretch()
        self.tabs.addTab(page, "RECOGNITION")

    def _build_system_tab(self):
        page = QWidget(); root = QVBoxLayout(page); root.setContentsMargins(8, 20, 8, 8); root.setSpacing(14)
        title = QLabel("SYSTEM SETTINGS"); title.setObjectName("sectionTitle"); root.addWidget(title)
        sub = QLabel("Reserved for application-wide options as Lewiscrypt grows.")
        sub.setObjectName("pageSubtitle"); root.addWidget(sub)
        card = QFrame(); card.setObjectName("pageCard"); form = QFormLayout(card); form.setContentsMargins(24,24,24,24); form.setSpacing(14)
        info = QLabel("More system, storage, performance, security, and alert settings will live here.")
        info.setWordWrap(True); info.setObjectName("infoLabel"); form.addRow("Status", info)
        root.addWidget(card); root.addStretch()
        self.tabs.addTab(page, "SYSTEM")

    def _rebuild_camera_checks(self):
        while self.camera_checks_grid.count():
            item = self.camera_checks_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.camera_checks = []
        count = self.camera_count.value()
        saved = app_settings.get_rtsp_enabled_indices()
        streams = app_settings.get_rtsp_streams()
        # An absent/empty selection means all configured channels are enabled.
        enabled_set = set(saved) if saved else set(range(count))
        for i in range(count):
            cb = QCheckBox(f"Camera {i + 1:02d}")
            cb.setMinimumHeight(24)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cb.setChecked(i in enabled_set)
            self.camera_checks.append(cb)
            self.camera_checks_grid.addWidget(cb, i // 4, i % 4)
        configured = len(streams)
        if configured and configured < count:
            for i in range(configured, count):
                self.camera_checks[i].setEnabled(False)
                self.camera_checks[i].setChecked(False)
        if hasattr(self, "rtsp_enabled_holder"):
            self.rtsp_enabled_holder.setVisible(self.source_mode.currentIndex() == 1 and count >= 3)

    def _set_all_camera_checks(self, enabled):
        for cb in getattr(self, "camera_checks", []):
            if cb.isEnabled():
                cb.setChecked(enabled)

    def _selected_rtsp_indices(self):
        return [i for i, cb in enumerate(getattr(self, "camera_checks", [])) if cb.isEnabled() and cb.isChecked()]

    def apply_enabled_cameras(self):
        streams = app_settings.get_rtsp_streams()
        if not streams:
            self.status.setText("No RTSP camera streams are configured yet.")
            return
        selected = [i for i in self._selected_rtsp_indices() if i < len(streams)]
        if not selected:
            self.status.setText("Select at least one RTSP camera.")
            return
        app_settings.set_rtsp_enabled_indices(selected)
        active = [streams[i] for i in selected]
        if self.parent_window.start_multi_camera(active):
            self.status.setText(f"{len(active)} camera(s) enabled. Disabled channels are not opened or processed.")

    def _apply_threshold(self):
        pct = self.threshold_slider.value()
        app_settings.set_recognition_threshold(pct / 100.0)
        self.status.setText(f"Recognition threshold set to {pct}%.")

    def _apply_aspect(self, value):
        app_settings.set_display_aspect_ratio(value)
        ratio = None if value == "Auto" else (16 / 9 if value == "16:9" else 4 / 3)
        if self.parent_window.multi_mode:
            for panel in self.parent_window.camera_grid.panels.values():
                panel.set_display_aspect_ratio(ratio)
        else:
            self.parent_window.video_panel.set_display_aspect_ratio(ratio)
        self.status.setText(f"Display aspect ratio set to {value}.")

    def apply_all_rtsp(self):
        template = self.rtsp_url.text().strip()
        if not template.lower().startswith("rtsp://"):
            self.status.setText("Enter one working Hikvision RTSP URL first.")
            return
        match = re.search(r"(/Streaming/channels/)(\d+)(?=$|[/?#])", template, re.I)
        if not match:
            self.status.setText("Could not detect the Hikvision channel number in that RTSP URL.")
            return
        original = int(match.group(2))
        stream = original % 100
        if stream not in (1, 2):
            stream = 2
        urls = []
        for channel in range(1, self.camera_count.value() + 1):
            code = channel * 100 + stream
            urls.append(template[:match.start(2)] + str(code) + template[match.end(2):])
        app_settings.set_rtsp_streams(urls)
        app_settings.set_rtsp_enabled_indices(list(range(len(urls))))
        app_settings.set_camera_source(urls[0])
        self._rebuild_camera_checks()
        if self.parent_window.start_multi_camera(urls):
            self.status.setText(f"{len(urls)} RTSP camera feeds started. Uncheck any broken channels, then apply the selection.")

    def refresh(self):
        self.combo.clear()
        indices = Camera.list_cameras()
        for index in indices:
            self.combo.addItem(f"Camera {index}" + (" — built-in / first camera" if index == 0 else " — external/USB candidate"), index)
        self.status.setText(f"{len(indices)} local camera(s) available. RTSP streams can be entered in the Camera tab.")

    def apply(self):
        if self.combo.currentData() is None:
            self.status.setText("No local camera is available.")
            return
        index = int(self.combo.currentData())
        if self.parent_window.switch_camera(index):
            app_settings.set_camera_source(str(index))
            self.status.setText(f"Camera {index} is now active.")

    def apply_rtsp(self):
        url = self.rtsp_url.text().strip()
        if not url.lower().startswith("rtsp://"):
            self.status.setText("Enter a valid RTSP URL starting with rtsp://")
            return
        if self.parent_window.switch_camera(url):
            app_settings.set_camera_source(url)
            self.status.setText("RTSP camera is now active.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Lewiscrypt HUB"); self.resize(1500, 900)
        self.camera = None; self.camera_index = 0; self.latest_frame = None
        self.stream_readers = {}; self.multi_mode = False; self._multi_last_submit = 0.0
        self._focused_camera = None
        self._known_people = {}; self._camera_faces = {}; self._camera_seq = {}; self._frames = 0; self._fps_t0 = time.time(); self._last_submit = 0.0
        self._recognition_cursor = 0
        self._last_wall_recognition = 0.0
        self._trackers = {}
        self._logged_tracks = {}
        self._build_ui()
        self.worker = RecognitionWorker(self)
        self.worker.engine_ready.connect(lambda text: self.sidebar.set_status(engine=text))
        self.worker.results_ready.connect(self._on_results)
        self.worker.error.connect(self._on_worker_error)
        self.worker.enrollment_ready.connect(self.enroll_drawer.on_frame_result)
        self.worker.enrollment_error.connect(self.enroll_drawer.on_error)
        self.worker.start()
        saved_streams = app_settings.get_rtsp_streams()
        enabled_indices = app_settings.get_rtsp_enabled_indices()
        if saved_streams:
            # Empty selection is legacy/default: keep all configured streams.
            if enabled_indices:
                saved_streams = [s for i, s in enumerate(saved_streams) if i in enabled_indices]
            if len(saved_streams) > 1:
                self.start_multi_camera(saved_streams)
            elif len(saved_streams) == 1:
                self._start_camera(saved_streams[0])
            else:
                self._start_camera(0)
        else:
            saved_source = app_settings.get_camera_source()
            source = saved_source if saved_source.lower().startswith("rtsp://") else int(saved_source or "0")
            self._start_camera(source)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(15)

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c); outer = QVBoxLayout(c); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(TopBar("LEWISCRYPT HUB"))
        body = QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)
        self.sidebar = Sidebar(); body.addWidget(self.sidebar)
        self.stack = QStackedWidget()

        # Merged Live/Enroll page: the camera view is persistent, and the
        # enrollment drawer docks beside it instead of replacing it — no
        # more switching pages to see yourself while enrolling.
        self.live = QWidget()
        live_layout = QHBoxLayout(self.live)
        live_layout.setContentsMargins(18, 18, 18, 18)
        live_layout.setSpacing(18)
        self.camera_grid = CameraGrid()
        self.camera_grid.camera_double_clicked.connect(self._focus_camera)
        self.camera_grid.set_camera_count(1)
        self.video_panel = self.camera_grid.panel(0)
        live_layout.addWidget(self.camera_grid, 1)
        self.enroll_drawer = EnrollDrawer(self)
        self.enroll_drawer.hide()
        live_layout.addWidget(self.enroll_drawer, 0)

        self.people = PeoplePage(self); self.settings = SettingsPage(self); self.logs = LogsPage(self)
        for p in [self.live, self.people, self.settings, self.logs]: self.stack.addWidget(p)
        body.addWidget(self.stack, 1); bw = QWidget(); bw.setLayout(body); outer.addWidget(bw, 1)
        self.nav = BottomNav(); outer.addWidget(self.nav)
        for key in ("live", "enroll", "people", "settings", "logs"):
            self.nav.buttons[key].clicked.connect(lambda _, k=key: self._on_nav(k))
        self.nav.buttons["exit"].clicked.connect(self.close)
        self.sidebar.set_status(camera="Inactive", engine="Loading GPU…", database="Connected", fps=0)

    def _on_nav(self, key):
        if key in ("live", "enroll"):
            self.stack.setCurrentWidget(self.live)
            self.enroll_drawer.setVisible(key == "enroll")
            return
        # Leaving the camera view for People/Settings/Logs: restore the sidebar
        # and camera wall before changing pages.
        self._exit_camera_focus()
        # Leaving the camera view for People/Settings/Logs: close the drawer
        # unless a guided capture is actively in progress (it keeps running
        # in the background so the user doesn't lose progress).
        if not self.enroll_drawer.active:
            self.enroll_drawer.setVisible(False)
        pages = {"people": self.people, "settings": self.settings, "logs": self.logs}
        if key in pages:
            self.stack.setCurrentWidget(pages[key])
            if key == "people":
                self.people.refresh()

    def start_multi_camera(self, sources):
        sources = [str(s).strip() for s in sources if str(s).strip()]
        if not sources:
            return False
        self.stop_multi_camera()
        if self.camera:
            self.camera.release(); self.camera = None
        self.multi_mode = True
        self._focused_camera = None
        self._camera_faces.clear()
        self._camera_seq.clear()
        self._logged_tracks.clear()
        self._recognition_cursor = 0
        self.sidebar.show()
        self.camera_grid.set_camera_count(len(sources))
        self.video_panel = self.camera_grid.panel(0)
        saved_aspect = app_settings.get_display_aspect_ratio()
        ratio = None if saved_aspect == "Auto" else (16 / 9 if saved_aspect == "16:9" else 4 / 3)
        for panel in self.camera_grid.panels.values():
            panel.set_display_aspect_ratio(ratio)
        for i, source in enumerate(sources):
            self.stream_readers[i] = StreamReader(source).start()
        self.camera_index = sources[0]
        self.sidebar.set_status(camera=f"{len(sources)} ACTIVE")
        return True

    def stop_multi_camera(self):
        for reader in self.stream_readers.values():
            reader.stop()
        self.stream_readers.clear()
        self._trackers.clear()
        self._camera_faces.clear()
        self.multi_mode = False
        self._exit_camera_focus()

    def _start_camera(self, source):
        self.stop_multi_camera()
        try:
            self.camera = Camera(source)
            self.camera_index = source
            saved_aspect = app_settings.get_display_aspect_ratio()
            ratio = None if saved_aspect == "Auto" else (16 / 9 if saved_aspect == "16:9" else 4 / 3)
            self.video_panel.set_display_aspect_ratio(ratio)
            self.sidebar.set_status(camera="Active")
        except Exception as exc:
            self.camera = None
            self.sidebar.set_status(camera="Unavailable")
            self.video_panel.set_message("CAMERA ERROR", str(exc))

    def switch_camera(self, source):
        if source == self.camera_index and self.camera is not None:
            return True
        try:
            new_camera = Camera(source)
        except Exception as exc:
            self.video_panel.set_message("CAMERA ERROR", str(exc)); return False
        if self.camera: self.camera.release()
        self.camera = new_camera; self.camera_index = source
        saved_aspect = app_settings.get_display_aspect_ratio()
        ratio = None if saved_aspect == "Auto" else (16 / 9 if saved_aspect == "16:9" else 4 / 3)
        self.video_panel.set_display_aspect_ratio(ratio)
        self.sidebar.set_status(camera="Active")
        return True

    def _focus_camera(self, index):
        if not self.multi_mode:
            return
        if index not in self.camera_grid.panels:
            return
        if getattr(self, "_focused_camera", None) == index:
            self._exit_camera_focus()
            return
        self.camera_grid.set_focus(index)
        # Keep the bottom navigation visible; only the sidebar collapses.
        self.sidebar.hide()
        self._focused_camera = index

    def _exit_camera_focus(self):
        if getattr(self, "_focused_camera", None) is None:
            return
        self.camera_grid.set_focus(None)
        self._focused_camera = None
        self.sidebar.show()
        # Recalculate the body before the next paint so the restored wall does
        # not retain the focused-camera geometry and push the bottom nav away.
        self.sidebar.updateGeometry()
        self.camera_grid.updateGeometry()
        self.live.updateGeometry()
        if self.live.layout():
            self.live.layout().invalidate()
            self.live.layout().activate()
        self.centralWidget().layout().invalidate()
        self.centralWidget().layout().activate()
        self.nav.raise_()
        self.nav.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and getattr(self, "_focused_camera", None) is not None:
            self._exit_camera_focus()
            event.accept()
            return
        super().keyPressEvent(event)

    def _tick(self):
        if self.multi_mode:
            display_frames = {}
            for index, reader in self.stream_readers.items():
                frame, seq = reader.latest()
                panel = self.camera_grid.panel(index)
                if frame is not None:
                    display_frames[index] = frame
                    # Only convert/draw a new frame once. The stream reader
                    # continues decoding independently in the background.
                    if self._camera_seq.get(index) != seq:
                        self._camera_seq[index] = seq
                        if panel:
                            panel.clear_message()
                            panel.update_frame(frame)
                elif panel:
                    if reader.error:
                        panel.set_message("CAMERA OFFLINE", reader.error)
                    else:
                        panel.set_message("CONNECTING", "Waiting for RTSP frames…")

            # Recognition is deliberately decoupled from display. In wall
            # mode sample only a small round-robin group; the focused camera
            # gets priority and can be analyzed much more frequently.
            now = time.monotonic()
            if display_frames:
                if self._focused_camera is not None and self._focused_camera in display_frames:
                    if now - self._last_wall_recognition >= 0.20:
                        self._last_wall_recognition = now
                        self.worker.submit_batch({self._focused_camera: display_frames[self._focused_camera]})
                elif now - self._last_wall_recognition >= 0.16:
                    active = sorted(display_frames)
                    if active:
                        # One camera per recognition job keeps the expensive AI
                        # path from blocking the video wall. Cameras are sampled
                        # round-robin, so every active channel gets attention.
                        index = active[self._recognition_cursor % len(active)]
                        self._recognition_cursor = (self._recognition_cursor + 1) % len(active)
                        self._last_wall_recognition = now
                        self.worker.submit_batch({index: display_frames[index]})
            self._frames += 1
        else:
            if not self.camera: return
            frame = self.camera.get_frame()
            if frame is None: return
            self.latest_frame = frame.copy()
            self.video_panel.update_frame(frame)
            if self.enroll_drawer.active:
                self.enroll_drawer.tick(frame)
            else:
                now = time.monotonic()
                if now - self._last_submit >= 0.10:
                    self._last_submit = now
                    self.worker.submit(frame)
            self._frames += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 0.5:
            self.sidebar.set_status(fps=self._frames / elapsed)
            self._frames = 0; self._fps_t0 = time.time()

    def _on_results(self, payload):
        generation, results = payload
        if generation != self.worker._generation:
            return
        if self.multi_mode and isinstance(results, dict):
            visible = []
            threshold = app_settings.get_recognition_threshold()
            for index, faces in results.items():
                tracker = self._trackers.setdefault(index, SurveillanceTracker())
                panel = self.camera_grid.panel(index)
                frame, _ = self.stream_readers[index].latest() if index in self.stream_readers else (None, 0)
                shape = frame.shape if frame is not None else (720, 1280, 3)
                stable = tracker.update(faces, shape, threshold)
                self._camera_faces[index] = stable
                self._log_stable_detections(index, stable)
                if panel:
                    panel.clear_message(); panel.update_faces(stable); panel.set_quality("FACE" if stable else "SCANNING")
                visible.extend(stable)
            # Only currently visible/still-tracked identities reach the sidebar.
            self._update_people(visible)
            return
        faces = results
        self._log_stable_detections(self.camera_index, faces)
        self.video_panel.clear_message(); self.video_panel.update_faces(faces); self.video_panel.set_quality("GOOD" if faces else "NO FACE")
        self._update_people(faces)

    def _on_worker_error(self, message):
        self.sidebar.set_status(engine="GPU ERROR")
        self.video_panel.set_message("GPU RECOGNITION ERROR", message)

    def _log_stable_detections(self, camera, faces):
        """Write one evidence image per track/appearance, not every sampled frame."""
        now = time.monotonic()
        for face in faces or []:
            track_id = getattr(face, "track_id", None)
            key = (str(camera), track_id) if track_id is not None else (str(camera), f"single:{face.person_id}:{face.name}")
            # A new track can legitimately represent a later appearance.
            if key in self._logged_tracks and now - self._logged_tracks[key] < 4.0:
                continue
            self._logged_tracks[key] = now
            snapshot = None
            thumb = getattr(face, "thumbnail", None)
            if thumb is not None and hasattr(thumb, "size") and thumb.size:
                try:
                    ok, encoded = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
                    if ok: snapshot = encoded.tobytes()
                except Exception:
                    snapshot = None
            database.add_detection_log(camera, face.name, face.confidence, snapshot)
        # Keep the in-memory dedupe map bounded.
        if len(self._logged_tracks) > 500:
            self._logged_tracks = dict(list(self._logged_tracks.items())[-250:])

    def _update_people(self, faces):
        # Recognition is a live state, not a history list. A frame with no
        # faces must clear the previous result immediately.
        self._known_people.clear()
        for f in faces:
            key = f.person_id if f.person_id is not None else f"unknown:{id(f)}"
            thumbnail = None
            if f.person_id is not None:
                for person in database.get_people_records():
                    if person["id"] == f.person_id and person["thumbnail"]:
                        thumbnail = person["thumbnail"]
                        break
            self._known_people[key] = {
                "name": f.name.upper(), "match": f.confidence * 100,
                "unknown": f.name.upper() == "UNKNOWN", "thumbnail": thumbnail
            }
        self.sidebar.update_people(sorted(self._known_people.values(), key=lambda p: (p["unknown"], -p["match"]))[:8])

    def _people_changed(self):
        self.worker.request_refresh(); self._known_people.clear(); self.sidebar.refresh_people_from_db(); self.people.refresh()

    def closeEvent(self, event):
        if hasattr(self, "timer"): self.timer.stop()
        self.stop_multi_camera()
        if self.camera: self.camera.release()
        if hasattr(self, "worker"): self.worker.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    qss = Path(__file__).with_name("style.qss")
    app.setStyleSheet(qss.read_text(encoding="utf-8"))
    try:
        w = MainWindow(); w.show(); sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Lewiscrypt HUB failed to start", str(e)); raise


if __name__ == "__main__":
    main()
