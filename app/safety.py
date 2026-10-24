"""Safety language for FaceFlag: PHI redaction, banned-language lint, and
the required-language blocks that every output must carry.

Banned phrases are narrow, exact verdict/reassurance strings. Note what is
deliberately NOT banned: "does not diagnose", "does not prove a stroke",
and "can still miss a stroke" are required honest language, not verdicts.
Tests scan every generated output (not README prose quoting banned
examples) against this list, and fail if any phrase appears.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("faceflag.safety")

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\d)(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)")),
    ("NAME", re.compile(r"\b(?:Patient(?: Name)?|Pt\.?|Name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})")),
]

BANNED_PHRASES = [
    "ruled out",
    "rule out a stroke",
    "you do not have a stroke",
    "you do not have bell",
    "you have a stroke",
    "you have bell's palsy",
    "you have bells palsy",
    "just bell's palsy",
    "just bells palsy",
    "all clear",
    "you are safe",
    "safe to wait",
    "not a stroke",
    "no stroke",
    "confirmed stroke",
    "confirmed bell",
    "diagnosed with",
    "you are having a stroke",
    "definitely a stroke",
    "definitely bell",
    "be-fast checked",
    "befast checked",
]

REQUIRED_FRAGMENTS = [
    "does not diagnose",
    "sensitivity and specificity of this pipeline are unknown",
    "can still miss a stroke",
    "does not prove a stroke",
    "seek emergency care now",
]


def redact_phi(text: str) -> tuple[str, dict[str, int]]:
    report: dict[str, int] = {}
    redacted = text or ""
    for kind, pattern in _PATTERNS:
        redacted, n = pattern.subn(f"[{kind}]", redacted)
        if n:
            report[kind] = report.get(kind, 0) + n
    return redacted, report


def find_banned_language(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in BANNED_PHRASES if p in low]


def missing_required_language(text: str) -> list[str]:
    low = (text or "").lower()
    return [f for f in REQUIRED_FRAGMENTS if f not in low]


def render_result(metrics: dict, zone_info: dict, action: str, context: dict) -> str:
    """Build the user-facing result. Pattern language only, with every
    required block. This text is also what the gateway phrases from
    metrics JSON; the gateway never measures and never sees frames."""
    zone = zone_info["zone"]
    if zone == "stroke_pattern":
        pattern = ("The movements measured are more consistent with a "
                   "lower-face-predominant pattern: forehead movement is "
                   "relatively preserved while lower-face movement is "
                   "markedly asymmetric.")
    elif zone == "peripheral_like_pattern":
        pattern = ("The movements measured are more consistent with a "
                   "whole-side pattern: forehead and lower-face movement "
                   "are both reduced on the same side.")
    else:
        pattern = ("The movements measured do not match either classic "
                   "pattern clearly, or could not be measured reliably. "
                   "This result is indeterminate.")
    numbers = (
        f"Measured numbers (app-defined screening values, not clinical "
        f"cut-offs): brow ratio {metrics['brow_ratio']:.2f} "
        f"(weak side: {metrics['brow_weak_side']}), smile ratio "
        f"{metrics['smile_ratio']:.2f} "
        f"(weak side: {metrics['smile_weak_side']}), eye closure "
        f"completeness on the weaker side "
        f"{metrics['eye_weak_completeness']:.2f}, rest mouth asymmetry "
        f"{metrics['rest_mouth_asymmetry']:.2f}, nasolabial proxy "
        f"{metrics['nasolabial_proxy']:.2f}. "
        f"Pattern reason: {zone_info['reason']}. {zone_info['detail']}"
    )
    if action == "emergency_now":
        action_text = ("Action: seek emergency care now. Go to the nearest "
                       "emergency room or dial 911 immediately, and tell "
                       "them the time symptoms started if you know it.")
    else:
        action_text = ("Action: seek prompt in-person assessment. This "
                       "pattern still needs a clinician to examine you in "
                       "person, and it does not exclude a stroke. If the "
                       "eye will not close, tell the clinician promptly, "
                       "and if symptoms worsen or become sudden, seek "
                       "emergency care now.")
    if context.get("sudden_onset") or context.get("arm_or_leg_weakness") or context.get("speech_change"):
        action_text += (" The answers about sudden onset, limb weakness, or "
                        "speech change also point to emergency care, "
                        "whatever the face measurements show.")
    body = (
        f"{pattern} {numbers}\n\n"
        f"This tool does not diagnose any condition, and it does not "
        f"exclude any condition. The sensitivity and specificity of "
        f"this pipeline are unknown: it has not been tested on labelled "
        f"clinical cases, and no accuracy percentage is claimed.\n\n"
        f"False negatives are possible: a result that looks like a "
        f"whole-side pattern can still miss a stroke. Strokes can be "
        f"early, mild, or cause no facial weakness at all, and some "
        f"strokes weaken the forehead too. Forehead pattern is a "
        f"tendency, not a rule.\n\n"
        f"False positives are possible: an abnormal pattern does not "
        f"prove a stroke. Other conditions, an old facial weakness, "
        f"head tilt, lighting, and camera angle can all produce it.\n\n"
        f"Scope limit: this app measures the face only. It does not "
        f"check balance, eyes, arm or leg strength, or speech, so it "
        f"cannot check for strokes that show up outside the face.\n\n"
        f"{action_text}"
    )
    banned = find_banned_language(body)
    missing = missing_required_language(body)
    assert not banned, f"banned language in output template: {banned}"
    assert not missing, f"missing required language in output: {missing}"
    return body
