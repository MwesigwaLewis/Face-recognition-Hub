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
    "camera_source": "0",
    # Display-only aspect ratio. Recognition always uses the original frame.
    "display_aspect_ratio": "16:9",
    "rtsp_streams": [],
    # Zero-based indices of enabled RTSP camera entries. Empty means all
    # configured streams for backwards compatibility; the UI saves an
    # explicit list once the user changes the selection.
    "rtsp_enabled_indices": [],
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


def get_camera_source() -> str:
    with _lock:
        return str(_load().get("camera_source", DEFAULTS["camera_source"]))


def set_camera_source(value: str) -> str:
    value = str(value).strip()
    with _lock:
        _load()["camera_source"] = value
        _save()
    return value


def get_display_aspect_ratio() -> str:
    with _lock:
        return str(_load().get("display_aspect_ratio", DEFAULTS["display_aspect_ratio"]))


def set_display_aspect_ratio(value: str) -> str:
    value = str(value).strip()
    if value not in ("Auto", "16:9", "4:3"):
        value = DEFAULTS["display_aspect_ratio"]
    with _lock:
        _load()["display_aspect_ratio"] = value
        _save()
    return value


def get_rtsp_streams() -> list[str]:
    with _lock:
        value = _load().get("rtsp_streams", DEFAULTS["rtsp_streams"])
        return [str(x).strip() for x in value if str(x).strip().lower().startswith("rtsp://")]



def get_rtsp_enabled_indices() -> list[int]:
    with _lock:
        raw = _load().get("rtsp_enabled_indices", DEFAULTS["rtsp_enabled_indices"])
        if raw is None:
            return []
        result = []
        for value in raw if isinstance(raw, list) else []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index >= 0 and index not in result:
                result.append(index)
        return result


def set_rtsp_enabled_indices(values) -> list[int]:
    cleaned = []
    for value in values or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index >= 0 and index not in cleaned:
            cleaned.append(index)
    with _lock:
        _load()["rtsp_enabled_indices"] = cleaned
        _save()
    return cleaned

def set_rtsp_streams(values) -> list[str]:
    cleaned = []
    for value in values or []:
        value = str(value).strip()
        if value.lower().startswith("rtsp://") and value not in cleaned:
            cleaned.append(value)
    with _lock:
        _load()["rtsp_streams"] = cleaned
        _save()
    return cleaned
