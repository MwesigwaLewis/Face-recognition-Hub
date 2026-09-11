"""Lightweight face pose and image-quality analysis for guided enrollment."""
from __future__ import annotations

import math
import cv2
import numpy as np


# Approximate 3-D facial landmarks in millimetres. The exact scale is irrelevant
# to solvePnP; the relative geometry is what matters.
#
# IMPORTANT: InsightFace's 5-point landmarks are ordered by the SUBJECT's own
# anatomical left/right (kps[0]=their left eye, kps[1]=their right eye), not
# by screen position. In an unmirrored frontal camera shot, the subject's own
# left eye lands on the image's RIGHT side (larger pixel x) — the mirror
# opposite of what "left" suggests. The X signs below are chosen so that,
# under a neutral/identity pose, kps[0] (their left eye) projects to a larger
# image-x than kps[1] (their right eye), matching that real geometry. Getting
# this backwards (as an earlier version of this file did) doesn't just make
# yaw noisy — it makes the recovered roll sit near +-180 degrees instead of
# near 0 for a level head, which silently makes every guided-enrollment pose
# impossible to ever satisfy, regardless of how the user positions their head.
_MODEL_POINTS = np.array([
    [ 30.0,  35.0,  30.0],   # left eye  (subject's own left)
    [-30.0,  35.0,  30.0],   # right eye (subject's own right)
    [  0.0,   0.0,   0.0],   # nose
    [ 24.0, -35.0,  20.0],   # left mouth corner
    [-24.0, -35.0,  20.0],   # right mouth corner
], dtype=np.float64)


def estimate_head_pose(kps, image_shape):
    """Return (yaw, pitch, roll) in degrees, or None if unavailable.

    Sign convention (verified against POSE_RANGES below via a synthetic
    rotate-and-reproject test, not just reasoned about on paper):
      yaw   negative = subject turned toward their own left, positive = right
      pitch negative = subject looking up, positive = looking down
      roll  ~0 for a level head; magnitude only, sign doesn't matter
    """
    if kps is None or len(kps) < 5:
        return None
    pts = np.asarray(kps[:5], dtype=np.float64)
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return None
    h, w = image_shape[:2]
    focal = float(max(w, h))
    camera = np.array([
        [focal, 0, w / 2.0],
        [0, focal, h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        # SOLVEPNP_ITERATIVE (the previous flag) needs >=6 points for its
        # internal DLT initialization when no extrinsic guess is supplied —
        # with only 5 landmarks it always threw here, silently, every call.
        # EPNP works with as few as 4 points, so it actually runs.
        ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, pts, camera, dist, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        # Refine the EPNP estimate with a Gauss-Newton pass for a steadier,
        # less noisy fit; this can only improve on the EPNP starting point.
        ok2, rvec2, tvec2 = cv2.solvePnP(
            _MODEL_POINTS, pts, camera, dist,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok2:
            rvec = rvec2
        rot, _ = cv2.Rodrigues(rvec)
        # Manually derived from the rotation matrix rather than
        # cv2.decomposeProjectionMatrix, whose Euler-angle output has known
        # quadrant ambiguities and didn't match this camera/model setup.
        yaw = math.degrees(math.atan2(rot[2, 0], math.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2)))
        pitch = -math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
        roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        return yaw, pitch, roll
    except Exception:
        # A genuinely degenerate point set (e.g. near-collinear landmarks
        # from a bad detection). Rather than fall back to a crude 2-point
        # heuristic that can't be reliably calibrated and previously caused
        # confidently-wrong directions, report "unavailable" so the UI asks
        # the user to reposition instead of guiding them the wrong way.
        return None


def analyze_lighting(image, bbox):
    """Return a compact lighting assessment for the detected face crop."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {"ok": False, "label": "NO FACE", "brightness": 0.0, "balance": 0.0}
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    std = float(np.std(gray))
    mid = gray.shape[1] // 2
    left = float(np.mean(gray[:, :max(1, mid)]))
    right = float(np.mean(gray[:, mid:]))
    balance = abs(left - right)

    # These are intentionally broad: a webcam should not need studio lighting.
    too_dark = brightness < 45
    too_bright = brightness > 215
    badly_uneven = balance > 65
    low_detail = std < 18
    ok = not (too_dark or too_bright or badly_uneven or low_detail)
    if too_dark:
        label = "TOO DARK"
    elif too_bright:
        label = "TOO BRIGHT"
    elif badly_uneven:
        label = "UNEVENTEN LIGHT"  # kept short for the HUD
    elif low_detail:
        label = "LOW CONTRAST"
    else:
        label = "GOOD LIGHT"
    return {"ok": ok, "label": label, "brightness": brightness, "balance": balance}


MAX_ROLL_DEG = 15

# (yaw_min, yaw_max, pitch_min, pitch_max) target windows, in degrees, for
# each guided-enrollment pose. Shared by pose_matches() and pose_guidance()
# so the "is this pose correct" check and the "which way to adjust" feedback
# never drift out of sync with each other.
POSE_RANGES = {
    "LOOK STRAIGHT": (-10, 10, -10, 10),
    "TURN LEFT": (-38, -12, -12, 12),
    "TURN RIGHT": (12, 38, -12, 12),
    "LOOK UP": (-12, 12, -30, -8),
    "LOOK DOWN": (-12, 12, 8, 30),
    "SLIGHT LEFT": (-25, -7, -12, 12),
    "SLIGHT RIGHT": (7, 25, -12, 12),
}


def pose_matches(pose_name, pose):
    return pose_guidance(pose_name, pose)["ok"]


def pose_guidance(pose_name, pose):
    """Real-time directional feedback for guided enrollment.

    Rather than a binary "in position / not in position" result, this tells
    the caller *which way* the person should move right now to reach the
    target pose window, so the UI can draw a live arrow/prompt instead of
    requiring the head to land in one fixed screen position.

    Returns a dict:
      ok:   bool, True once yaw/pitch/roll are all inside the target window
      text: short human-readable instruction (e.g. "TURN LEFT")
      dx:   -1 (turn head left), 0 (yaw fine), or 1 (turn head right)
      dy:   -1 (look up), 0 (pitch fine), or 1 (look down)
    """
    if pose is None:
        return {"ok": False, "text": "FACE POSE UNAVAILABLE", "dx": 0, "dy": 0}

    yaw, pitch, roll = pose
    yr = POSE_RANGES.get(pose_name)
    if yr is None:
        return {"ok": False, "text": "UNKNOWN POSE TARGET", "dx": 0, "dy": 0}
    ymin, ymax, pmin, pmax = yr

    dx = 0
    if yaw < ymin:
        dx = 1  # yaw too low -> turn head right to raise it
    elif yaw > ymax:
        dx = -1  # yaw too high -> turn head left to lower it

    dy = 0
    if pitch < pmin:
        dy = 1  # pitch too low -> look down to raise it
    elif pitch > pmax:
        dy = -1  # pitch too high -> look up to lower it

    roll_bad = abs(roll) > MAX_ROLL_DEG

    if dx == 0 and dy == 0 and not roll_bad:
        return {"ok": True, "text": "HOLD STILL", "dx": 0, "dy": 0}

    parts = []
    if dx == 1:
        parts.append("turn right")
    elif dx == -1:
        parts.append("turn left")
    if dy == 1:
        parts.append("look down")
    elif dy == -1:
        parts.append("look up")
    if roll_bad:
        parts.append("level your head")

    text = " & ".join(parts).upper() if parts else "ADJUST"
    return {"ok": False, "text": text, "dx": dx, "dy": dy}


# --- Free-form sweep capture -------------------------------------------
#
# Coarse 3x3 (yaw x pitch) zone labels, used to encourage spread across
# angles during guided enrollment WITHOUT requiring the user to hit an
# exact target pose. Unlike POSE_RANGES above (kept for reference / other
# potential uses), these zones are wide and just classify "roughly which
# broad direction is the head currently facing" so the capture flow can
# track coverage and cap how many angles it keeps per zone.
YAW_ZONES = (
    ("LEFT",   lambda yaw: yaw <= -10),
    ("CENTER", lambda yaw: -10 < yaw < 10),
    ("RIGHT",  lambda yaw: yaw >= 10),
)
PITCH_ZONES = (
    ("UP",     lambda pitch: pitch <= -8),
    ("CENTER", lambda pitch: -8 < pitch < 8),
    ("DOWN",   lambda pitch: pitch >= 8),
)
ALL_ZONES = tuple(f"{y}-{p}" for y, _ in YAW_ZONES for p, _ in PITCH_ZONES)


def pose_zone(pose):
    """Return a coarse zone label like 'LEFT-UP' for a measured pose, or
    None if pose is unavailable."""
    if pose is None:
        return None
    yaw, pitch, _roll = pose
    yaw_label = next(label for label, test in YAW_ZONES if test(yaw))
    pitch_label = next(label for label, test in PITCH_ZONES if test(pitch))
    return f"{yaw_label}-{pitch_label}"


# --- Face-ID-style capture ring -----------------------------------------
#
# Buckets a measured pose into one of 12 positions around a clock-like ring
# (segment 0 = 12 o'clock/looking up, increasing clockwise), so the capture
# UI can show a literal ring that fills in as the user moves their head
# around -- closer to Apple Face ID's enrollment motion than a static grid.
NUM_RING_SEGMENTS = 12
RING_YAW_SCALE = 35.0     # degrees of yaw treated as "full sweep" to the ring edge
RING_PITCH_SCALE = 28.0   # degrees of pitch treated as "full sweep" to the ring edge
RING_DEADZONE = 0.22      # fraction of full-scale magnitude treated as "centered" (no segment)


def ring_segment(pose):
    """Return an int 0..NUM_RING_SEGMENTS-1 for where this pose sits on the
    capture ring, or None if the pose is unavailable or near dead-center."""
    if pose is None:
        return None
    yaw, pitch, _roll = pose
    x = yaw / RING_YAW_SCALE          # positive = turned right
    y = -pitch / RING_PITCH_SCALE     # positive = looking up
    mag = math.hypot(x, y)
    if mag < RING_DEADZONE:
        return None
    angle = math.degrees(math.atan2(x, y))  # 0 = up (12 o'clock), clockwise positive
    return int(round(angle / (360.0 / NUM_RING_SEGMENTS))) % NUM_RING_SEGMENTS
