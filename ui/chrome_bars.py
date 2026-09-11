"""Top title bar and bottom navigation bar."""

import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFrame


class TopBar(QWidget):
    def __init__(self, title="Face Recognition Hub", parent=None):
        super().__init__(parent)
        self.setObjectName("topBar")
        self.setFixedHeight(46)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        lock = QLabel("🔒")
        title_lbl = QLabel(title)
        title_lbl.setObjectName("appTitle")
        layout.addWidget(lock)
        layout.addWidget(title_lbl)
        layout.addStretch()

        self.time_lbl = QLabel()
        self.date_lbl = QLabel()
        self.time_lbl.setObjectName("clockLabel")
        self.date_lbl.setObjectName("dateLabel")
        layout.addWidget(self.time_lbl)
        layout.addSpacing(10)
        layout.addWidget(self.date_lbl)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()

    def _tick(self):
        self.time_lbl.setText(time.strftime("%I:%M:%S %p"))
        self.date_lbl.setText(time.strftime("%m/%d/%Y"))


class BottomNav(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottomNav")
        self.setFixedHeight(56)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.buttons = {}
        items = [
            ("live", "🎥  LIVE"),
            ("enroll", "➕  ENROLL"),
            ("people", "👥  PEOPLE"),
            ("settings", "⚙  SETTINGS"),
            ("logs", "📄  LOGS"),
            ("exit", "⏻  EXIT"),
        ]
        for key, label in items:
            btn = QPushButton(label)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setFlat(True)
            layout.addWidget(btn, 1)
            self.buttons[key] = btn

        self.buttons["live"].setChecked(True)
        self.buttons["live"].setProperty("active", True)

        for key, btn in self.buttons.items():
            btn.clicked.connect(lambda checked, k=key: self._on_click(k))

    def _on_click(self, key):
        for k, b in self.buttons.items():
            b.setChecked(k == key)
