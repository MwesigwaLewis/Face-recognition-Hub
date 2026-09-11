"""Live video panel with sci-fi HUD overlay (face box, landmarks, ID card)."""

import time
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout

GREEN = QColor("#3CFF7A")
GREEN_DIM = QColor(60, 255, 122, 120)
CYAN = QColor("#37E6FF")
AMBER = QColor("#FFB020")
PANEL_BG = QColor(10, 16, 14, 210)


class VideoPanel(QWidget):
    """Displays the camera frame and draws recognition overlays on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._faces = []
        self._frame_size = (1, 1)
        self._quality = "GOOD"
        self._message = None
        self._scan_wave = []
        self._t0 = time.time()
        # Rich enrollment overlay state, or None when not enrolling. Set via
        # update_enrollment_overlay(); replaces the old fixed-target box with
        # a live tracking box that follows the detected face.
        self._enroll_overlay = None
        self._enroll_thumbnails = []  # captured-angle QPixmaps, shown as a strip
        self._capture_flash_t = 0.0
        self.setMinimumSize(480, 360)

    # ---- public API -----------------------------------------------------

    def update_frame(self, frame_bgr: np.ndarray):
        """frame_bgr: OpenCV BGR numpy array."""
        h, w = frame_bgr.shape[:2]
        self._frame_size = (w, h)
        rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def update_faces(self, faces):
        self._faces = faces
        self.update()

    def set_quality(self, quality: str):
        self._quality = quality
        self.update()

    def set_message(self, title, detail=""):
        self._message = (title, detail)
        self.update()

    def clear_message(self):
        if self._message is not None:
            self._message = None
            self.update()

    def update_enrollment_overlay(self, state, bbox, landmarks, guidance, detail="", count=0, target=0, zones=None):
        """Called every enrollment frame result to draw a live tracking box.

        state: "searching" (no face), "issue" (lighting/quality problem),
               or "capturing" (conditions are good, frames are being kept).
        bbox: (x, y, w, h) in frame coordinates, or None if no face.
        zones: set of covered coarse-zone labels, for the coverage grid.
        """
        self._enroll_overlay = {
            "state": state,
            "bbox": bbox,
            "landmarks": landmarks or [],
            "guidance": guidance,
            "detail": detail,
            "count": count,
            "target": target,
            "zones": zones or set(),
        }
        self.update()

    def clear_enrollment_overlay(self):
        self._enroll_overlay = None
        self._enroll_thumbnails = []
        self.update()

    def set_enrollment_thumbnails(self, thumbnails_bgr):
        """thumbnails_bgr: list of OpenCV BGR numpy crops, one per captured angle."""
        pixes = []
        for thumb in thumbnails_bgr:
            if thumb is None or thumb.size == 0:
                continue
            h, w = thumb.shape[:2]
            rgb = thumb[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pixes.append(QPixmap.fromImage(qimg))
        self._enroll_thumbnails = pixes
        self.update()

    def flash_capture(self):
        """Brief green flash on the tracking box when an angle is captured."""
        self._capture_flash_t = time.time()
        self.update()

    # ---- painting ---------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#050807"))

        if self._pixmap is None:
            painter.setPen(QColor("#3a4a44"))
            painter.setFont(QFont("Consolas", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "NO SIGNAL")
            return

        target_rect, scale, offset = self._fit_rect()
        painter.drawPixmap(target_rect, self._pixmap, self._pixmap.rect())

        if self._message:
            painter.setPen(QColor("#FF6B6B")); painter.setFont(QFont("Consolas", 14, QFont.Bold))
            painter.drawText(self.rect().adjusted(30, 50, -30, -50), Qt.AlignCenter, self._message[0])

        # frame corner brackets
        self._draw_corner_brackets(painter, self.rect())

        if self._enroll_overlay is not None:
            self._draw_enrollment_overlay(painter, scale, offset)
        else:
            # top-left status text
            painter.setFont(QFont("Consolas", 12, QFont.Bold))
            if self._faces:
                painter.setPen(GREEN)
                painter.drawText(24, 34, "[ FACE DETECTED ]")
            else:
                painter.setPen(QColor("#6b7a74"))
                painter.drawText(24, 34, "[ SCANNING... ]")

            for face in self._faces:
                self._draw_face(painter, face, scale, offset)

            # bottom-left analyzing panel
            self._draw_analysis_panel(painter)

        painter.end()

    def _fit_rect(self):
        """Compute the target rect for drawing the pixmap letterboxed to fit,
        and return (rect, scale, offset) to map frame coords -> widget coords."""
        w, h = self._frame_size
        pw, ph = self.width(), self.height()
        if w == 0 or h == 0:
            return QRectF(0, 0, pw, ph), 1.0, (0, 0)
        frame_ratio = w / h
        panel_ratio = pw / ph
        if frame_ratio > panel_ratio:
            draw_w = pw
            draw_h = pw / frame_ratio
        else:
            draw_h = ph
            draw_w = ph * frame_ratio
        x = (pw - draw_w) / 2
        y = (ph - draw_h) / 2
        scale = draw_w / w
        return QRectF(x, y, draw_w, draw_h), scale, (x, y)

    def _map_pt(self, x, y, scale, offset):
        return QPointF(offset[0] + x * scale, offset[1] + y * scale)

    def _draw_face(self, painter, face, scale, offset):
        x, y, w, h = face.box
        p1 = self._map_pt(x, y, scale, offset)
        p2 = self._map_pt(x + w, y + h, scale, offset)
        rect = QRectF(p1, p2)

        color = GREEN if face.name.upper() != "UNKNOWN" else AMBER
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        # bracket-style corners instead of a plain rectangle
        L = min(rect.width(), rect.height()) * 0.18
        corners = [
            (rect.topLeft(), (1, 0), (0, 1)),
            (rect.topRight(), (-1, 0), (0, 1)),
            (rect.bottomLeft(), (1, 0), (0, -1)),
            (rect.bottomRight(), (-1, 0), (0, -1)),
        ]
        for pt, dx, dy in corners:
            painter.drawLine(pt, QPointF(pt.x() + dx[0] * L, pt.y() + dx[1] * L))
            painter.drawLine(pt, QPointF(pt.x() + dy[0] * L, pt.y() + dy[1] * L))

        # faint full rect
        faint = QPen(GREEN_DIM if color == GREEN else QColor(255, 176, 32, 90), 1)
        painter.setPen(faint)
        painter.drawRect(rect)

        # landmarks
        if face.landmarks:
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            for lx, ly in face.landmarks:
                lp = self._map_pt(lx, ly, scale, offset)
                painter.drawEllipse(lp, 1.6, 1.6)

        # identity card to the right of the face (or left if too close to edge)
        self._draw_id_card(painter, face, rect)

    def _draw_id_card(self, painter, face, face_rect):
        card_w, card_h = 250, 190
        gap = 24
        x = face_rect.right() + gap
        if x + card_w > self.width() - 12:
            x = face_rect.left() - gap - card_w
        y = face_rect.top() + max(0, (face_rect.height() - card_h) / 2)
        y = min(max(12, y), self.height() - card_h - 12)
        card = QRectF(x, y, card_w, card_h)

        # connector line
        anchor = QPointF(face_rect.right() if x > face_rect.right() else face_rect.left(),
                          face_rect.center().y())
        near_edge = QPointF(card.left() if x > face_rect.right() else card.right(),
                             card.center().y())
        painter.setPen(QPen(GREEN, 2))
        painter.drawLine(anchor, near_edge)
        painter.setBrush(GREEN)
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(anchor.x() - 4 if x <= face_rect.right() else anchor.x() - 4,
                                 anchor.y() - 4, 8, 8))

        # card background
        painter.setPen(QPen(QColor("#1e2b26"), 1))
        painter.setBrush(PANEL_BG)
        painter.drawRoundedRect(card, 4, 4)

        recognized = face.name.upper() != "UNKNOWN"
        title_color = GREEN if recognized else AMBER
        painter.setPen(title_color)
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(QRectF(card.x() + 16, card.y() + 14, card_w - 32, 20),
                          Qt.AlignLeft, "IDENTITY CONFIRMED" if recognized else "UNVERIFIED FACE")

        painter.setPen(QColor("#eafff0") if recognized else QColor("#ffe8c2"))
        painter.setFont(QFont("Consolas", 20, QFont.Bold))
        painter.drawText(QRectF(card.x() + 16, card.y() + 34, card_w - 32, 34),
                          Qt.AlignLeft, face.name.upper())

        painter.setFont(QFont("Consolas", 11))
        painter.setPen(QColor("#c9d6d0"))
        painter.drawText(QRectF(card.x() + 16, card.y() + 72, card_w - 32, 20),
                          Qt.AlignLeft, f"MATCH: {face.confidence * 100:.1f}%")

        # confidence bar
        bar_rect = QRectF(card.x() + 16, card.y() + 96, card_w - 32, 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#16211c"))
        painter.drawRect(bar_rect)
        filled = QRectF(bar_rect.x(), bar_rect.y(), bar_rect.width() * min(face.confidence, 1.0), bar_rect.height())
        painter.setBrush(title_color)
        painter.drawRect(filled)

        painter.setPen(QColor("#93a49c"))
        painter.setFont(QFont("Consolas", 9))
        now = time.strftime("%I:%M:%S %p")
        date = time.strftime("%m/%d/%Y")
        lines = [
            f"ID:   {face.person_id if face.person_id is not None else '-'}",
            f"TIME: {now}",
            f"DATE: {date}",
        ]
        for i, line in enumerate(lines):
            painter.drawText(QRectF(card.x() + 16, card.y() + 122 + i * 18, card_w - 32, 16),
                              Qt.AlignLeft, line)

    _ENROLL_STATE_COLORS = {
        "searching": QColor("#6b7a74"),
        "issue": AMBER,
        "error": QColor("#FF6B6B"),
        "capturing": GREEN,
    }
    NUM_RING_SEGMENTS = 12

    def _draw_enrollment_overlay(self, painter, scale, offset):
        """Face-ID-style capture: a ring around the face fills in as the
        user moves their head around. No fixed target or dictated direction.
        Blocking conditions (no face / bad lighting / too far / error) get a
        big, impossible-to-miss banner instead of small text, since a quiet
        failure is indistinguishable from 'just waiting for a good angle'."""
        info = self._enroll_overlay
        state = info["state"]
        bbox = info["bbox"]
        count = info.get("count", 0)
        target = info.get("target", 0)

        painter.setPen(CYAN)
        painter.setFont(QFont("Consolas", 11, QFont.Bold))
        header = f"{count} / {target} ANGLES CAPTURED" if target else "SWEEP CAPTURE"
        painter.drawText(QRectF(0, 12, self.width(), 20), Qt.AlignCenter, header)

        blocking = state in ("searching", "issue", "error")
        if blocking:
            self._draw_blocking_banner(painter, state, info.get("guidance", ""), info.get("detail", ""))

        if bbox is None:
            if self._enroll_thumbnails:
                self._draw_thumbnail_strip(painter)
            return

        x, y, w, h = bbox
        p1 = self._map_pt(x, y, scale, offset)
        p2 = self._map_pt(x + w, y + h, scale, offset)
        rect = QRectF(p1, p2)

        flashing = (time.time() - self._capture_flash_t) < 0.25
        color = GREEN if flashing else self._ENROLL_STATE_COLORS.get(state, CYAN)

        # Face-ID-style capture ring, centered on the face, filling in per
        # covered ring segment rather than a fixed rectangular target box.
        cx, cy = rect.center().x(), rect.center().y()
        radius = max(rect.width(), rect.height()) * 0.72
        self._draw_capture_ring(painter, QPointF(cx, cy), radius, info.get("zones", set()), flashing)

        # Soft face outline so it's clear what's being tracked, without the
        # old hard rectangular target box.
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect.adjusted(-6, -6, 6, 6))

        if self._enroll_thumbnails:
            self._draw_thumbnail_strip(painter)

    def _draw_blocking_banner(self, painter, state, guidance, detail):
        """A solid, full-width bar across the top of the frame — deliberately
        hard to miss, unlike small text under a box. Persists for as long as
        the blocking condition does (the caller re-sends it every frame)."""
        color = self._ENROLL_STATE_COLORS.get(state, AMBER)
        bar_h = 56
        bar = QRectF(0, 40, self.width(), bar_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 210))
        painter.drawRect(bar)

        painter.setPen(QColor("#0a0f0c"))
        painter.setFont(QFont("Consolas", 15, QFont.Bold))
        painter.drawText(bar.adjusted(16, 4, -16, -20), Qt.AlignCenter, guidance or "BLOCKED")
        if detail:
            painter.setFont(QFont("Consolas", 10))
            painter.drawText(bar.adjusted(16, 26, -16, -4), Qt.AlignCenter, detail[:80])

    def _draw_capture_ring(self, painter, center, radius, covered_segments, flashing):
        """12 arc segments around the face (like a clock), filling green as
        that direction gets captured -- the core Face-ID-style visual."""
        pen_width = max(6, int(radius * 0.09))
        span_deg = 360.0 / self.NUM_RING_SEGMENTS
        gap_deg = 5.0
        rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        for i in range(self.NUM_RING_SEGMENTS):
            # Segment i center sits at (90 - i*span) degrees in Qt's arc
            # convention (0 = 3 o'clock, positive = counter-clockwise),
            # matching core.face_pose.ring_segment's clock-face layout.
            center_deg = 90 - i * span_deg
            start_deg = center_deg - (span_deg - gap_deg) / 2.0
            covered = i in covered_segments
            color = GREEN if covered else QColor("#3a4a42")
            painter.setPen(QPen(color, pen_width, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(rect, int(start_deg * 16), int((span_deg - gap_deg) * 16))

    def _draw_thumbnail_strip(self, painter):
        """Row of captured-angle thumbnails along the bottom, so progress is
        visible without leaving the camera view."""
        size = 46
        gap = 8
        n = len(self._enroll_thumbnails)
        total_w = n * size + max(0, n - 1) * gap
        x0 = (self.width() - total_w) / 2
        y0 = self.height() - size - 18
        for i, pix in enumerate(self._enroll_thumbnails):
            x = x0 + i * (size + gap)
            target = QRectF(x, y0, size, size)
            painter.setPen(QPen(GREEN, 2))
            painter.setBrush(PANEL_BG)
            painter.drawRoundedRect(target, 6, 6)
            inner = target.adjusted(3, 3, -3, -3)
            scaled = pix.scaled(int(inner.width()), int(inner.height()),
                                 Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            painter.drawPixmap(inner, scaled, QRectF(0, 0, scaled.width(), scaled.height()))

    def _draw_corner_brackets(self, painter, rect, size=22, margin=10):
        pen = QPen(QColor("#6b7a74"), 2)
        painter.setPen(pen)
        r = rect.adjusted(margin, margin, -margin, -margin)
        corners = [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for pt, (dx, dy) in zip(corners, dirs):
            painter.drawLine(pt, QPointF(pt.x() + dx * size, pt.y()))
            painter.drawLine(pt, QPointF(pt.x(), pt.y() + dy * size))

    def _draw_analysis_panel(self, painter):
        w, h = 210, 100
        x, y = 24, self.height() - h - 24
        rect = QRectF(x, y, w, h)
        painter.setPen(QPen(QColor("#1e2b26"), 1))
        painter.setBrush(PANEL_BG)
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(GREEN if self._faces else QColor("#6b7a74"))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(QRectF(x + 12, y + 8, w - 24, 18), Qt.AlignLeft,
                          "ANALYZING..." if self._faces else "IDLE")

        # fake waveform
        t = time.time() - self._t0
        pts = []
        n = 24
        graph_rect = QRectF(x + 12, y + 30, w - 24, 40)
        for i in range(n):
            gx = graph_rect.x() + graph_rect.width() * i / (n - 1)
            noise = np.sin(t * 3 + i * 0.7) * 0.5 + np.sin(t * 7 + i) * 0.3
            gy = graph_rect.y() + graph_rect.height() / 2 - noise * graph_rect.height() / 2.4
            pts.append(QPointF(gx, gy))
        painter.setPen(QPen(GREEN, 1.5))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])
        painter.setPen(Qt.NoPen)
        painter.setBrush(GREEN)
        for p in pts[::4]:
            painter.drawEllipse(p, 2, 2)

        painter.setPen(QColor("#93a49c"))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(QRectF(x + 12, y + h - 20, w - 24, 16), Qt.AlignLeft,
                          f"QUALITY: {self._quality}")
