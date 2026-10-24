"""Deterministic facial-landmark geometry for FaceFlag.

Landmarks are plain 2D points in image coordinates (y increases downwards),
supplied by an on-device face mesh. This module never sees a photo, a frame,
or a video. It only sees points. All thresholds in the service layer are
APP_DEFINED and UNVALIDATED: they are screening splits chosen for this
proof of concept, not clinical cut-offs, and no sensitivity or specificity
is claimed for them.
"""
from __future__ import annotations

import math


def dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def mid(a: dict, b: dict) -> dict:
    return {"x": (a["x"] + b["x"]) / 2.0, "y": (a["y"] + b["y"]) / 2.0}


def eye_center(frame: dict, side: str) -> dict:
    return mid(frame[f"eye_{side}_upper"], frame[f"eye_{side}_lower"])


def intercanthal(frame: dict) -> float:
    """Scale unit: distance between eye centres at rest. Guards division."""
    d = dist(eye_center(frame, "left"), eye_center(frame, "right"))
    return d if d > 1e-6 else 1.0


def weaker_stronger(left: float, right: float) -> tuple[float, float, str]:
    """Return (weaker, stronger, weak_side) for two excursion magnitudes."""
    if left <= right:
        return left, right, "left"
    return right, left, "right"


def ratio(weaker: float, stronger: float) -> float:
    if stronger <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, weaker / stronger))
