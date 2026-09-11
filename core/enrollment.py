"""Face-ID-style enrollment and identity-safe profile updates."""

from __future__ import annotations

import numpy as np

from .face_engine import FaceEngine
from . import database

POSES = [
    "LOOK STRAIGHT", "TURN LEFT", "TURN RIGHT", "LOOK UP",
    "LOOK DOWN", "SLIGHT LEFT", "SLIGHT RIGHT",
]

# Re-enrollment must prove that the new face belongs to the existing profile.
# This is intentionally at least as strict as the normal recognition threshold.
PROFILE_UPDATE_THRESHOLD = 0.68


def _best_similarity(new_embeddings, old_embeddings):
    best = -1.0
    for new_embedding in new_embeddings:
        for old_embedding in old_embeddings:
            score = float(np.dot(
                FaceEngine.normalize(new_embedding),
                FaceEngine.normalize(old_embedding),
            ))
            best = max(best, score)
    return best


def enroll_person(name, embeddings, thumbnail=None):
    """Create a profile or safely replace an existing profile.

    Existing profiles are never overwritten solely because the name matches.
    A new enrollment must contain at least one embedding that is sufficiently
    similar to an embedding already stored for that person. This allows old
    one-angle profiles to be upgraded to the new multi-angle format while
    preventing an unrelated face from taking over the profile.
    """
    name = name.strip()
    if not name:
        return False, "Enter a name first."

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) < 3:
        return False, "Capture at least 3 good face angles."
    arr = np.asarray([FaceEngine.normalize(x) for x in arr], dtype=np.float32)

    existing = database.get_person_record(name)
    if existing is not None:
        old_embeddings = existing["embeddings"]
        best = _best_similarity(arr, old_embeddings)
        if best < PROFILE_UPDATE_THRESHOLD:
            return False, (
                "PROFILE UPDATE REJECTED — the new face does not match the "
                f"existing profile strongly enough ({best:.0%} < "
                f"{PROFILE_UPDATE_THRESHOLD:.0%})."
            )

        database.update_person(name, arr, thumbnail)
        return True, (
            f"Face profile upgraded with {len(arr)} angles — "
            f"existing identity verified ({best:.0%} match)."
        )

    database.add_person(name, arr, thumbnail)
    return True, f"New face profile enrolled with {len(arr)} angles."
