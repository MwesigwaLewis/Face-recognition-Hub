"""Temporal surveillance tracking and multi-frame identity stabilization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ui.recognition_interface import FaceResult


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _center_distance(a, b, frame_shape):
    h, w = frame_shape[:2]
    ac = (a[0] + a[2] / 2, a[1] + a[3] / 2)
    bc = (b[0] + b[2] / 2, b[1] + b[3] / 2)
    scale = max(1.0, float((w * w + h * h) ** 0.5))
    return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5 / scale


@dataclass
class Track:
    track_id: int
    box: tuple
    last_box: tuple
    last_seen: float
    hits: int = 0
    missed: int = 0
    name: str = "UNKNOWN"
    person_id: Optional[int] = None
    confidence: float = 0.0
    quality: float = 0.0
    evidence: dict = field(default_factory=dict)
    best_score: dict = field(default_factory=dict)
    best_face: Optional[FaceResult] = None

    def predict(self):
        # Small constant-velocity prediction makes brief detector misses less
        # visually jumpy without pretending a person is tracked indefinitely.
        x, y, w, h = self.box
        px, py, pw, ph = self.last_box
        dx, dy = x - px, y - py
        return (int(x + dx), int(y + dy), int(w), int(h))


class SurveillanceTracker:
    """Per-camera short-term tracker with temporal identity voting."""

    def __init__(self, max_missed=4, ttl=1.2):
        self.max_missed = max_missed
        self.ttl = ttl
        self.tracks = {}
        self._next_id = 1

    def reset(self):
        self.tracks.clear()

    def update(self, detections, frame_shape, threshold):
        now = time.monotonic()
        detections = list(detections or [])
        unmatched = set(range(len(detections)))
        matched_tracks = set()

        # Greedy association is sufficient for the small number of faces per
        # surveillance frame and avoids another heavyweight dependency.
        candidates = []
        for tid, track in self.tracks.items():
            predicted = track.predict()
            for i, det in enumerate(detections):
                score = max(_iou(predicted, det.box), 1.0 - _center_distance(predicted, det.box, frame_shape))
                if _iou(predicted, det.box) >= 0.10 or _center_distance(predicted, det.box, frame_shape) <= 0.18:
                    candidates.append((score, tid, i))
        candidates.sort(reverse=True)

        for _, tid, i in candidates:
            if tid in matched_tracks or i not in unmatched:
                continue
            track = self.tracks[tid]
            det = detections[i]
            track.last_box = track.box
            track.box = det.box
            track.last_seen = now
            track.hits += 1
            track.missed = 0
            matched_tracks.add(tid)
            unmatched.remove(i)
            self._add_evidence(track, det, threshold)

        for i in unmatched:
            det = detections[i]
            track = Track(
                self._next_id, det.box, det.box, now,
                hits=1, quality=det.quality, best_face=det,
            )
            self._next_id += 1
            self.tracks[track.track_id] = track
            self._add_evidence(track, det, threshold)

        for tid, track in list(self.tracks.items()):
            if tid not in matched_tracks and not any(i in unmatched and detections[i].box == track.box for i in unmatched):
                track.missed += 1
            if now - track.last_seen > self.ttl or track.missed > self.max_missed:
                del self.tracks[tid]

        return self._visible_results(now, threshold)

    def _add_evidence(self, track, det, threshold):
        track.quality = max(track.quality * 0.65, float(det.quality))
        if det.person_id is None or det.confidence <= 0:
            return
        # Good frames count more heavily, but even a small/less-perfect face
        # can contribute if it is detected reliably.
        weight = max(0.15, min(1.0, float(det.quality)))
        arr = track.evidence.setdefault(det.person_id, [])
        arr.append((float(det.confidence), weight))
        if len(arr) > 8:
            del arr[:-8]
        track.best_score[det.person_id] = max(track.best_score.get(det.person_id, 0.0), float(det.confidence))
        if track.best_face is None or det.quality > track.best_face.quality:
            track.best_face = det

        # Stable identity: repeated evidence from the same enrolled person,
        # or one exceptionally strong high-quality observation.
        best_pid = None
        best_value = -1.0
        for pid, values in track.evidence.items():
            scores = np.array([v[0] for v in values], dtype=np.float32)
            weights = np.array([v[1] for v in values], dtype=np.float32)
            weighted = float((scores * weights).sum() / max(weights.sum(), 1e-6))
            top = float(scores.max())
            consistency = float(np.mean(scores >= threshold - 0.035))
            value = weighted + min(0.06, max(0, len(values) - 1) * 0.012) * consistency
            if value > best_value:
                best_value, best_pid = value, pid
            if top >= threshold + 0.08 and values[-1][1] >= 0.45:
                best_value, best_pid = max(best_value, top), pid

        if best_pid is not None:
            values = track.evidence[best_pid]
            scores = np.array([v[0] for v in values], dtype=np.float32)
            weights = np.array([v[1] for v in values], dtype=np.float32)
            weighted = float((scores * weights).sum() / max(weights.sum(), 1e-6))
            if weighted >= threshold - 0.015 or (scores.max() >= threshold + 0.08 and len(values) >= 1):
                track.person_id = best_pid
                track.name = det.name if det.person_id == best_pid else track.name
                track.confidence = max(weighted, float(scores.max()) if len(values) >= 2 else weighted)

    def _visible_results(self, now, threshold):
        results = []
        for track in self.tracks.values():
            face = track.best_face
            if face is None:
                continue
            # A one-frame detector hit is not enough for a surveillance
            # identity overlay. This suppresses transient false positives
            # while allowing real people to become visible after persistence.
            if track.hits < 2:
                continue
            # Don't show stale boxes forever. Brief misses are retained to
            # bridge detector gaps, while identity remains attached to track.
            box = track.box if track.missed == 0 else track.predict()
            if track.person_id is not None:
                confidence = min(1.0, max(track.confidence, face.confidence))
                name = track.name
                pid = track.person_id
            else:
                confidence = face.confidence
                name = "UNKNOWN"
                pid = None
            results.append(FaceResult(
                name, float(confidence), box,
                face.landmarks, pid, face.thumbnail,
                face.yaw, face.pitch, face.roll, face.quality,
                face.embedding, track.track_id,
            ))
        return results
