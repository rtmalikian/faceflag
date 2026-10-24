"""Synthetic landmark fixtures with KNOWN asymmetry ratios.

These are not photos and not patients. They exist so the geometry and
zone logic can be tested against ground truth: a fixture built with a
brow ratio of 0.50 must measure 0.50. Synthetic accuracy proves pipeline
correctness only; it says nothing about real selfies.
"""
from __future__ import annotations

SCALE = 10.0  # intercanthal distance in fixture units


def _pt(x: float, y: float) -> dict:
    return {"x": x, "y": y, "confidence": 1.0}


def _frame(points: dict, quality: dict | None = None) -> dict:
    q = {"lighting": 1.0, "head_rotation_deg": 0.0, "landmark_confidence": 1.0,
         "movement_detected": True}
    if quality:
        q.update(quality)
    return {"points": points, "quality": q}


def _base() -> dict:
    # Rest face, symmetric, eyes centred at (40,50) and (60,50).
    return {
        "brow_left_inner": _pt(38, 35), "brow_left_outer": _pt(30, 36),
        "brow_right_inner": _pt(62, 35), "brow_right_outer": _pt(70, 36),
        "eye_left_upper": _pt(40, 47), "eye_left_lower": _pt(40, 53),
        "eye_right_upper": _pt(60, 47), "eye_right_lower": _pt(60, 53),
        "nose_tip": _pt(50, 62), "nose_left": _pt(46, 60), "nose_right": _pt(54, 60),
        "mouth_left": _pt(40, 72), "mouth_right": _pt(60, 72),
        "mouth_upper": _pt(50, 70), "mouth_lower": _pt(50, 74),
        "chin": _pt(50, 82), "cheek_left": _pt(34, 62), "cheek_right": _pt(66, 62),
    }


def make_fixture(weak_side: str = "left", brow_ratio: float = 1.0,
                 smile_ratio: float = 1.0, eye_weak_completeness: float = 0.95,
                 quality: dict | None = None) -> dict:
    """Build rest/brow/close/smile frames. Strong side brow excursion is
    1.0 scale units (10 px), strong smile excursion 1.2 scale units."""
    rest = _base()
    brow = _base()
    strong_brow = 1.0 * SCALE
    weak_brow = brow_ratio * strong_brow
    for side, exc in (("left", strong_brow), ("right", strong_brow)):
        if side == weak_side:
            exc = weak_brow
        brow[f"brow_{side}_inner"]["y"] -= exc
        brow[f"brow_{side}_outer"]["y"] -= exc
    close = _base()
    for side in ("left", "right"):
        comp = eye_weak_completeness if side == weak_side else 0.95
        rest_h = 6.0
        close_h = rest_h * (1.0 - comp)
        mid_y = 50.0
        close[f"eye_{side}_upper"]["y"] = mid_y - close_h / 2
        close[f"eye_{side}_lower"]["y"] = mid_y + close_h / 2
    smile = _base()
    strong_smile = 1.2 * SCALE
    weak_smile = smile_ratio * strong_smile
    for side, exc in (("left", strong_smile), ("right", strong_smile)):
        if side == weak_side:
            exc = weak_smile
        smile[f"mouth_{side}"]["x"] += exc if side == "right" else -exc
        smile[f"mouth_{side}"]["y"] -= exc * 0.3
    return {
        "rest": _frame(rest, quality),
        "brow_raise": _frame(brow, quality),
        "eye_close": _frame(close, quality),
        "smile": _frame(smile, quality),
    }


def demo_cases() -> dict[str, dict]:
    return {
        "stroke_pattern": make_fixture(weak_side="left", brow_ratio=0.95, smile_ratio=0.40),
        "peripheral_like": make_fixture(weak_side="left", brow_ratio=0.40, smile_ratio=0.50,
                                        eye_weak_completeness=0.40),
        "indeterminate": make_fixture(weak_side="left", brow_ratio=0.78, smile_ratio=0.78),
        "symmetric": make_fixture(weak_side="left", brow_ratio=1.0, smile_ratio=1.0),
        "poor_quality": make_fixture(quality={"lighting": 0.2, "landmark_confidence": 0.4}),
    }
