"""Left sidebar: LIVE indicator, system status, recognized people list."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QProgressBar, QSizePolicy, QScrollArea
)


def _round_pixmap(pix: QPixmap, size: int) -> QPixmap:
    scaled = pix.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    out = QPixmap(size, size)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPixmap(size, size)
    path.fill(Qt.transparent)
    p.setBrush(Qt.black)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, size, size, 6, 6)
    p.setCompositionMode(QPainter.CompositionMode_SourceIn)
    p.drawPixmap(0, 0, scaled)
    p.end()
    return out


class StatusRow(QWidget):
    def __init__(self, icon_text, label, value, value_color="#3CFF7A"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        icon = QLabel(icon_text)
        icon.setFixedWidth(20)
        name = QLabel(label)
        name.setObjectName("statusLabel")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {value_color}; font-weight: 600;")
        layout.addWidget(icon)
        layout.addWidget(name)
        layout.addStretch()
        layout.addWidget(self.value_label)

    def set_value(self, value, color=None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color}; font-weight: 600;")


class PersonCard(QFrame):
    def __init__(self, name, match_pct, thumbnail: QPixmap = None, unknown=False):
        super().__init__()
        self.setObjectName("personCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        avatar = QLabel()
        avatar.setFixedSize(48, 48)
        if isinstance(thumbnail, (bytes, bytearray)):
            pix = QPixmap()
            pix.loadFromData(bytes(thumbnail), "JPG")
            thumbnail = pix
        if thumbnail is not None and not thumbnail.isNull():
            avatar.setPixmap(_round_pixmap(thumbnail, 48))
        else:
            avatar.setStyleSheet("background:#1a2420; border-radius:6px;")
        layout.addWidget(avatar)

        col = QVBoxLayout()
        col.setSpacing(4)
        name_lbl = QLabel(name)
        name_lbl.setObjectName("personName")
        match_lbl = QLabel(f"Match: {match_pct:.1f}%")
        color = "#FFB020" if unknown else "#93a49c"
        match_lbl.setStyleSheet(f"color:{color}; font-size:12px;")

        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setValue(int(match_pct))
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        bar_color = "#FFB020" if unknown else "#3CFF7A"
        bar.setStyleSheet(f"""
            QProgressBar {{ background:#16211c; border-radius:3px; }}
            QProgressBar::chunk {{ background:{bar_color}; border-radius:3px; }}
        """)

        col.addWidget(name_lbl)
        col.addWidget(match_lbl)
        col.addWidget(bar)
        layout.addLayout(col)


class Sidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setObjectName("sidebar")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # LIVE indicator
        live_row = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet("color:#3CFF7A; font-size:14px;")
        live_lbl = QLabel("LIVE RECOGNITION")
        live_lbl.setObjectName("liveLabel")
        live_row.addWidget(dot)
        live_row.addWidget(live_lbl)
        live_row.addStretch()
        root.addLayout(live_row)

        # System status card
        status_card = QFrame()
        status_card.setObjectName("card")
        sc_layout = QVBoxLayout(status_card)
        sc_layout.setContentsMargins(14, 14, 14, 14)
        title = QLabel("SYSTEM STATUS")
        title.setObjectName("sectionTitle")
        sc_layout.addWidget(title)

        self.row_camera = StatusRow("🎥", "Camera:", "Inactive", "#6b7a74")
        self.row_engine = StatusRow("⚙", "Engine:", "Not Loaded", "#6b7a74")
        self.row_db = StatusRow("🗄", "Database:", "—", "#6b7a74")
        self.row_fps = StatusRow("〰", "FPS:", "0.0", "#3CFF7A")
        for r in (self.row_camera, self.row_engine, self.row_db, self.row_fps):
            sc_layout.addWidget(r)
        root.addWidget(status_card)

        # Recognized people
        people_title = QLabel("RECOGNIZED PEOPLE")
        people_title.setObjectName("sectionTitle")
        root.addWidget(people_title)

        self.people_container = QVBoxLayout()
        self.people_container.setSpacing(10)
        people_holder = QWidget()
        people_holder.setLayout(self.people_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(people_holder)
        root.addWidget(scroll, 1)

        self._cards = []

    # ---- public API -------------------------------------------------

    def set_status(self, camera=None, engine=None, database=None, fps=None):
        if camera is not None:
            self.row_camera.set_value(camera, "#3CFF7A" if camera.lower() == "active" else "#FFB020")
        if engine is not None:
            self.row_engine.set_value(engine, "#3CFF7A" if engine.lower() == "loaded" else "#FFB020")
        if database is not None:
            self.row_db.set_value(database, "#3CFF7A" if database.lower() == "connected" else "#FFB020")
        if fps is not None:
            self.row_fps.set_value(f"{fps:.1f}")

    def update_people(self, people: list):
        """people: list of dicts {name, match, thumbnail(QPixmap or None), unknown(bool)}"""
        while self.people_container.count():
            item = self.people_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for p in people:
            card = PersonCard(p["name"], p["match"], p.get("thumbnail"), p.get("unknown", False))
            self.people_container.addWidget(card)
        self.people_container.addStretch()
    def refresh_people_from_db(self):
        """Refresh the sidebar's enrolled-person summary from the database."""
        try:
            from core import database
            people = database.get_people()
            self.update_people([{"name": name.upper(), "match": 100.0, "unknown": False, "thumbnail": None} for name, _ in people[:8]])
        except Exception:
            pass

