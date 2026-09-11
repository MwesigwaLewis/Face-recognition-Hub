"""Lewiscrypt HUB desktop application.

GPU recognition runs in a worker thread so camera/UI rendering stays responsive.
Enrollment uses a guided multi-angle capture flow rather than one instant snapshot.
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QMessageBox, QLabel, QPushButton, QLineEdit, QFormLayout,
    QFrame, QScrollArea, QGridLayout, QComboBox, QProgressBar, QSlider
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
        for face in self.engine.get_faces(frame):
            best, score, pid = "UNKNOWN", 0.0, None
            for person in self.people:
                for saved_embedding in person["embeddings"]:
                    s = self.engine.compare_faces(face.embedding, saved_embedding)
                    if s > score:
                        best, score, pid = person["name"], s, person["id"]
            if score < app_settings.get_recognition_threshold():
                best = "UNKNOWN"
                pid = None
            x1, y1, x2, y2 = map(int, face.bbox)
            thumb = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy() if x2 > x1 and y2 > y1 else None
            pose = estimate_head_pose(getattr(face, "kps", []), frame.shape)
            lighting = analyze_lighting(frame, face.bbox)
            quality = float(getattr(face, "det_score", 1.0))
            if thumb is not None:
                area_ratio = max(0, x2-x1) * max(0, y2-y1) / float(frame.shape[0] * frame.shape[1])
                quality *= min(1.0, area_ratio / 0.08)
            yaw, pitch, roll = pose if pose else (0.0, 0.0, 0.0)
            results.append(FaceResult(
                best, float(score), (x1, y1, x2 - x1, y2 - y1),
                [(int(x), int(y)) for x, y in getattr(face, "kps", [])], pid, thumb,
                yaw, pitch, roll, quality
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
        self._refresh_requested = False
        self._generation = 0
        self.engine = None

    def submit(self, frame, mode="recognition"):
        with self._condition:
            self._latest_frame = frame.copy()
            self._latest_mode = mode
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
                while self._latest_frame is None and not self._stop and not self._refresh_requested:
                    self._condition.wait(0.5)
                if self._refresh_requested:
                    self._refresh_requested = False
                    if self.engine:
                        self.engine.refresh()
                    if self._latest_frame is None and not self._stop:
                        continue
                if self._stop:
                    return
                frame = self._latest_frame
                mode = self._latest_mode
                generation = self._generation
                self._latest_frame = None

            try:
                if mode == "enrollment":
                    result = self.engine.process_enrollment_frame(frame)
                    self.enrollment_ready.emit(result)
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

    TARGET_TOTAL = 20        # stop once this many distinct angles are captured
    MIN_TOTAL = 12           # floor before auto-finish-on-timeout or manual finish is allowed --
                              # raised from 8: that let the time-based auto-finish below settle
                              # for well under half the target the moment 6s elapsed, quitting
                              # long before genuine full-ring coverage
    MAX_PER_SEGMENT = 2      # cap captures kept from any single ring position
    MAX_CENTERED = TARGET_TOTAL  # no extra cap when pose is unavailable/dead-center;
                                  # the embedding-novelty check below already guards
                                  # diversity independently of ring classification
    TIME_BUDGET = 12.0       # seconds; auto-finish once reached IF MIN_TOTAL is met -- doubled
                              # from 6.0s, which is an unrealistically short window for a real
                              # person to physically sweep through most of a 12-position ring
    HARD_CAP_TIME = 25.0     # seconds; finish regardless, with whatever was captured
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

        self.finish_button = QPushButton("FINISH NOW")
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
        """Lets the user stop early once enough angles are in the bag,
        instead of waiting out the full time budget."""
        if not self.active or len(self.embeddings) < self.MIN_TOTAL:
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
        self.finish_button.setEnabled(total >= self.MIN_TOTAL)
        self.state_label.setText(f"{total} / {self.TARGET_TOTAL} CAPTURED")
        video.update_enrollment_overlay(
            state="capturing", bbox=result.bbox, landmarks=result.landmarks,
            guidance="", detail="",
            count=total, target=self.TARGET_TOTAL, zones=self.covered_segments)
        self._maybe_finish(elapsed)

    def _maybe_finish(self, elapsed):
        total = len(self.embeddings)
        if total >= self.TARGET_TOTAL:
            self._finish()
        elif elapsed >= self.TIME_BUDGET and total >= self.MIN_TOTAL:
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


class PlaceholderPage(QWidget):
    def __init__(self, title, subtitle):
        super().__init__()
        l = QVBoxLayout(self)
        l.setContentsMargins(50, 50, 50, 50)
        h = QLabel(title); h.setObjectName("pageTitle")
        s = QLabel(subtitle); s.setObjectName("pageSubtitle")
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
    camera_changed = Signal(int)

    def __init__(self, parent):
        super().__init__(); self.parent_window = parent
        root = QVBoxLayout(self); root.setContentsMargins(50, 45, 50, 40); root.setSpacing(16)
        t = QLabel("SETTINGS"); t.setObjectName("pageTitle"); root.addWidget(t)
        s = QLabel("Select the built-in webcam or any connected USB/external camera."); s.setObjectName("pageSubtitle"); root.addWidget(s)
        card = QFrame(); card.setObjectName("pageCard"); form = QFormLayout(card); form.setContentsMargins(24,24,24,24); form.setSpacing(14)
        self.combo = QComboBox(); self.combo.setObjectName("cameraCombo")
        form.addRow("Camera", self.combo)
        root.addWidget(card)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("SCAN CAMERAS"); self.refresh_btn.setObjectName("primaryBtn")
        self.apply_btn = QPushButton("USE CAMERA"); self.apply_btn.setObjectName("primaryBtn")
        row.addWidget(self.refresh_btn); row.addWidget(self.apply_btn); row.addStretch(); root.addLayout(row)
        self.status = QLabel("Scanning cameras…"); self.status.setObjectName("infoLabel"); root.addWidget(self.status)

        recog_title = QLabel("RECOGNITION"); recog_title.setObjectName("pageTitle"); root.addWidget(recog_title)
        recog_sub = QLabel(
            "How closely a live camera face must match a stored profile to be identified. "
            "Higher = stricter (fewer wrong matches, more people shown as UNKNOWN). "
            "Lower = looser (fewer UNKNOWNs, more risk of mismatches)."
        )
        recog_sub.setObjectName("pageSubtitle"); recog_sub.setWordWrap(True); root.addWidget(recog_sub)

        recog_card = QFrame(); recog_card.setObjectName("pageCard")
        recog_form = QFormLayout(recog_card); recog_form.setContentsMargins(24, 24, 24, 24); recog_form.setSpacing(14)
        slider_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(30, 95)
        self.threshold_slider.setValue(int(round(app_settings.get_recognition_threshold() * 100)))
        self.threshold_value = QLabel(f"{self.threshold_slider.value()}%")
        self.threshold_value.setFixedWidth(48)
        slider_row.addWidget(self.threshold_slider, 1)
        slider_row.addWidget(self.threshold_value)
        recog_form.addRow("Identification threshold", slider_row)
        root.addWidget(recog_card)
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_value.setText(f"{v}%"))
        self.threshold_slider.sliderReleased.connect(self._apply_threshold)

        root.addStretch()
        self.refresh_btn.clicked.connect(self.refresh)
        self.apply_btn.clicked.connect(self.apply)
        self.refresh()

    def _apply_threshold(self):
        pct = self.threshold_slider.value()
        app_settings.set_recognition_threshold(pct / 100.0)
        self.status.setText(f"Identification threshold set to {pct}% — applies to new recognitions immediately.")

    def refresh(self):
        self.combo.clear()
        indices = Camera.list_cameras()
        for index in indices:
            self.combo.addItem(f"Camera {index}" + (" — built-in / first camera" if index == 0 else " — external/USB candidate"), index)
        self.status.setText(f"{len(indices)} camera(s) available. External cameras appear here when connected.")

    def apply(self):
        if self.combo.currentData() is None:
            self.status.setText("No camera is available.")
            return
        index = int(self.combo.currentData())
        if self.parent_window.switch_camera(index):
            self.status.setText(f"Camera {index} is now active.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Lewiscrypt HUB"); self.resize(1500, 900)
        self.camera = None; self.camera_index = 0; self.latest_frame = None
        self._known_people = {}; self._frames = 0; self._fps_t0 = time.time(); self._last_submit = 0.0
        self._build_ui()
        self.worker = RecognitionWorker(self)
        self.worker.engine_ready.connect(lambda text: self.sidebar.set_status(engine=text))
        self.worker.results_ready.connect(self._on_results)
        self.worker.error.connect(self._on_worker_error)
        self.worker.enrollment_ready.connect(self.enroll_drawer.on_frame_result)
        self.worker.enrollment_error.connect(self.enroll_drawer.on_error)
        self.worker.start()
        self._start_camera(0)
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
        self.video_panel = VideoPanel()
        live_layout.addWidget(self.video_panel, 1)
        self.enroll_drawer = EnrollDrawer(self)
        self.enroll_drawer.hide()
        live_layout.addWidget(self.enroll_drawer, 0)

        self.people = PeoplePage(self); self.settings = SettingsPage(self); self.logs = PlaceholderPage("LOGS", "Recognition activity will appear here.")
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

    def _start_camera(self, index):
        try:
            self.camera = Camera(index)
            self.camera_index = index
            self.sidebar.set_status(camera="Active")
        except Exception as exc:
            self.camera = None
            self.sidebar.set_status(camera="Unavailable")
            self.video_panel.set_message("CAMERA ERROR", str(exc))

    def switch_camera(self, index):
        if index == self.camera_index and self.camera is not None:
            return True
        try:
            new_camera = Camera(index)
        except Exception as exc:
            self.video_panel.set_message("CAMERA ERROR", str(exc)); return False
        if self.camera: self.camera.release()
        self.camera = new_camera; self.camera_index = index; self.sidebar.set_status(camera="Active")
        return True

    def _tick(self):
        if not self.camera: return
        frame = self.camera.get_frame()
        if frame is None: return
        self.latest_frame = frame.copy()
        self.video_panel.update_frame(frame)

        # Enrollment owns the worker while a guided capture is active.
        # IMPORTANT: never submit the same frame as a normal recognition job
        # after tick() has submitted it as an enrollment job; the worker keeps
        # only the latest job, so doing both would overwrite enrollment.
        if self.enroll_drawer.active:
            self.enroll_drawer.tick(frame)
        else:
            # Submit only the latest frame at ~10 FPS. Camera rendering can
            # remain smooth while expensive face inference runs independently.
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
        generation, faces = payload
        if generation != self.worker._generation:
            return
        self.video_panel.clear_message()
        self.video_panel.update_faces(faces)
        self.video_panel.set_quality("GOOD" if faces else "NO FACE")
        self._update_people(faces)

    def _on_worker_error(self, message):
        self.sidebar.set_status(engine="GPU ERROR")
        self.video_panel.set_message("GPU RECOGNITION ERROR", message)

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
