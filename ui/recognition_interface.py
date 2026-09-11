"""Recognition interfaces and the safe fallback engine used by the UI."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FaceResult:
    """Normalized recognition result consumed by the UI."""

    name: str
    confidence: float
    box: tuple
    landmarks: List[tuple] = field(default_factory=list)
    person_id: Optional[int] = None
    thumbnail: Optional[object] = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    quality: float = 0.0
    embedding: Optional[object] = None
    track_id: Optional[int] = None


class RecognitionEngine:
    """Small interface implemented by real and fallback recognition engines."""

    def process_frame(self, frame) -> List[FaceResult]:
        raise NotImplementedError

    def refresh(self):
        """Refresh any cached recognition data. Optional for simple engines."""
        return None


class DemoEngine(RecognitionEngine):
    """Dependency-free fallback so the application can still open.

    If InsightFace cannot be loaded, the UI uses this engine rather than
    crashing during startup. It intentionally returns no recognized faces;
    the real engine is used automatically when its dependencies are healthy.
    """

    def __init__(self, reason: str = ""):
        self.reason = reason

    def process_frame(self, frame) -> List[FaceResult]:
        return []

    def refresh(self):
        return None
