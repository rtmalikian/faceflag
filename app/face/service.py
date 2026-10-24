"""FaceFlag analysis service: deterministic ratios -> three-zone pattern.

APP_DEFINED_UNVALIDATED thresholds (screening splits for this proof of
concept only, never clinical cut-offs; sensitivity/specificity UNKNOWN):
  brow_ratio <= 0.70  -> forehead movement markedly asymmetric
  brow_ratio >= 0.85  -> forehead movement relatively preserved
  smile_ratio <= 0.70 -> lower-face movement markedly asymmetric
  smile_ratio <= 0.80 -> lower-face movement reduced
  eye completeness (weak side) < 0.80 -> eye does not close fully

Zone rules (see README/EVIDENCE.md for the audit basis and its limits):
  * quality gate failure -> indeterminate
  * bilateral weakness (both sides barely move) -> indeterminate: L/R
    ratios cannot describe bilateral disease, and bilateral causes
    (Lyme, sarcoidosis, Guillain-Barre) break the ratio logic entirely
  * brow preserved (>=0.85) + smile markedly asymmetric (<=0.70)
    -> stroke-pattern (lower-face-predominant)
  * brow markedly asymmetric (<=0.70) + smile reduced (<=0.80)
    -> peripheral-like pattern (whole side)
  * anything else -> indeterminate
Puff-cheeks is collected but NEVER changes the zone (audit verdict:
single case report, not a validated discriminator).

Context (sudden onset, limb weakness, speech change) is NOT scored by a
model and is never inferred from the face: any of those answers forces
emergency action regardless of the zone. The face mesh cannot test arm,
leg, balance, vision, or speech, and no copy claims it does.
"""
from __future__ import annotations

from app.face.geometry import dist, eye_center, intercanthal, ratio, weaker_stronger

BROW_WEAK_MAX = 0.70
BROW_PRESERVED_MIN = 0.85
SMILE_WEAK_MAX = 0.70
SMILE_REDUCED_MAX = 0.80
EYE_INCOMPLETE_MAX = 0.80
BILATERAL_EXCURSION_MAX = 0.30
QUALITY_CONFIDENCE_MIN = 0.60
QUALITY_LIGHTING_MIN = 0.40
QUALITY_ROTATION_MAX_DEG = 25.0

REQUIRED_FRAMES = ("rest", "brow_raise", "eye_close", "smile")


def _pts(frame: dict) -> dict:
    return frame["points"]


def _brow_y(frame: dict, side: str) -> float:
    pts = _pts(frame)
    return (pts[f"brow_{side}_inner"]["y"] + pts[f"brow_{side}_outer"]["y"]) / 2.0


def _eye_height(frame: dict, side: str) -> float:
    pts = _pts(frame)
    return abs(pts[f"eye_{side}_upper"]["y"] - pts[f"eye_{side}_lower"]["y"])


def _mouth_corner(frame: dict, side: str) -> dict:
    return _pts(frame)[f"mouth_{side}"]


def quality_gate(frames: dict) -> list[str]:
    problems: list[str] = []
    for name in REQUIRED_FRAMES:
        frame = frames.get(name)
        if frame is None:
            problems.append(f"missing {name} frame")
            continue
        q = frame.get("quality", {}) or {}
        if q.get("landmark_confidence", 1.0) < QUALITY_CONFIDENCE_MIN:
            problems.append(f"{name}: low landmark confidence")
        if q.get("lighting", 1.0) < QUALITY_LIGHTING_MIN:
            problems.append(f"{name}: poor lighting")
        if abs(q.get("head_rotation_deg", 0.0)) > QUALITY_ROTATION_MAX_DEG:
            problems.append(f"{name}: head rotation too large")
        if not q.get("movement_detected", True):
            problems.append(f"{name}: movement not detected")
    return problems


def measure(frames: dict) -> dict:
    rest, brow, close, smile = (frames[k] for k in REQUIRED_FRAMES)
    scale = intercanthal(_pts(rest))
    brow_exc = {}
    for side in ("left", "right"):
        brow_exc[side] = max(0.0, (_brow_y(rest, side) - _brow_y(brow, side)) / scale)
    bw, bs, brow_weak_side = weaker_stronger(brow_exc["left"], brow_exc["right"])
    brow_ratio = ratio(bw, bs)
    eye_comp = {}
    for side in ("left", "right"):
        rest_h = _eye_height(rest, side)
        close_h = _eye_height(close, side)
        eye_comp[side] = max(0.0, min(1.0, 1.0 - (close_h / rest_h if rest_h > 1e-9 else 1.0)))
    ew, _es, eye_weak_side = weaker_stronger(eye_comp["left"], eye_comp["right"])
    smile_exc = {}
    for side in ("left", "right"):
        smile_exc[side] = dist(_mouth_corner(rest, side), _mouth_corner(smile, side)) / scale
    sw, _ss, smile_weak_side = weaker_stronger(smile_exc["left"], smile_exc["right"])
    smile_ratio = ratio(sw, _ss)
    rest_mouth_asym = abs(_mouth_corner(rest, "left")["y"] - _mouth_corner(rest, "right")["y"]) / scale
    nasolabial = abs(
        dist(_pts(rest)["nose_tip"], _mouth_corner(rest, "left"))
        - dist(_pts(rest)["nose_tip"], _mouth_corner(rest, "right"))
    ) / scale
    return {
        "scale_units": round(scale, 2),
        "brow_excursion_left": round(brow_exc["left"], 3),
        "brow_excursion_right": round(brow_exc["right"], 3),
        "brow_ratio": round(brow_ratio, 3),
        "brow_weak_side": brow_weak_side,
        "eye_completeness_left": round(eye_comp["left"], 3),
        "eye_completeness_right": round(eye_comp["right"], 3),
        "eye_weak_completeness": round(ew, 3),
        "eye_weak_side": eye_weak_side,
        "smile_excursion_left": round(smile_exc["left"], 3),
        "smile_excursion_right": round(smile_exc["right"], 3),
        "smile_ratio": round(smile_ratio, 3),
        "smile_weak_side": smile_weak_side,
        "rest_mouth_asymmetry": round(rest_mouth_asym, 3),
        "nasolabial_proxy": round(nasolabial, 3),
    }


def classify(metrics: dict, quality_problems: list[str], context: dict) -> dict:
    if quality_problems:
        return {"zone": "indeterminate", "reason": "quality_gate_failed",
                "detail": "; ".join(quality_problems)}
    bilateral = (metrics["brow_excursion_left"] < BILATERAL_EXCURSION_MAX
                 and metrics["brow_excursion_right"] < BILATERAL_EXCURSION_MAX
                 and metrics["smile_excursion_left"] < BILATERAL_EXCURSION_MAX
                 and metrics["smile_excursion_right"] < BILATERAL_EXCURSION_MAX)
    if bilateral:
        return {"zone": "indeterminate", "reason": "bilateral_weakness",
                "detail": "Both sides barely move, so left/right ratios cannot describe the pattern."}
    if (metrics["brow_ratio"] >= BROW_PRESERVED_MIN
            and metrics["smile_ratio"] <= SMILE_WEAK_MAX):
        return {"zone": "stroke_pattern", "reason": "forehead_preserved_lower_face_predominant",
                "detail": "Forehead movement is relatively preserved while lower-face movement is markedly asymmetric."}
    if (metrics["brow_ratio"] <= BROW_WEAK_MAX
            and metrics["smile_ratio"] <= SMILE_REDUCED_MAX):
        return {"zone": "peripheral_like_pattern", "reason": "whole_side_involved",
                "detail": "Forehead and lower-face movement are both reduced on the same side."}
    return {"zone": "indeterminate", "reason": "pattern_not_clear",
            "detail": "The measured pattern does not match either classic pattern clearly."}


def decide_action(zone: str, context: dict) -> str:
    if context.get("sudden_onset") or context.get("arm_or_leg_weakness") or context.get("speech_change"):
        return "emergency_now"
    if zone in ("stroke_pattern", "indeterminate"):
        return "emergency_now"
    return "prompt_in_person"
