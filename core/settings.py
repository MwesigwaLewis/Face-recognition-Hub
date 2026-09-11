"""Small persisted app settings, e.g. the live-recognition identification
threshold. Backed by a plain JSON file next to the face database so it
survives restarts without needing a schema migration.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS_FILE = str(BASE_DIR / "data" / "settings.json")

DEFAULTS = {
    # Minimum cosine similarity for a live camera face to be identified as a
    # known person rather than reported as UNKNOWN. Higher = stricter match
    # (fewer false accepts, more false rejects); lower = looser match.
    "recognition_threshold": 0.65,
}

_lock = threading.RLock()
_cache = None


def _load():
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        data = dict(DEFAULTS)
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r") as f:
                    data.update(json.load(f))
        except Exception:
            pass  # corrupt/missing settings file -> fall back to defaults
        _cache = data
        return _cache


def _save():
    with _lock:
        try:
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(_cache, f, indent=2)
        except Exception:
            pass  # best-effort persistence; in-memory value still applies


def get_recognition_threshold() -> float:
    with _lock:
        return float(_load().get("recognition_threshold", DEFAULTS["recognition_threshold"]))


def set_recognition_threshold(value: float) -> float:
    """Clamped to a sane range so the UI can't accidentally disable matching
    entirely (0.0) or make it impossible to ever match (1.0)."""
    value = max(0.30, min(0.95, float(value)))
    with _lock:
        _load()["recognition_threshold"] = value
        _save()
    return value
