"""FaceFlag tests: geometry, zones, safety language, gateway, API, eval."""
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import gateway, safety
from app.face import service, synth
from app.main import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parents[1]


def _assess(case, context=None, tier="frugal"):
    frames = synth.demo_cases()[case]
    ctx = {"sudden_onset": False, "arm_or_leg_weakness": False, "speech_change": False}
    if context:
        ctx.update(context)
    problems = service.quality_gate(frames)
    metrics = service.measure(frames)
    zone = service.classify(metrics, problems, ctx)
    action = service.decide_action(zone["zone"], ctx)
    return metrics, zone, action, safety.render_result(metrics, zone, action, ctx)


# --- Geometry / zone logic -------------------------------------------------

def test_brow_ratio_measures_known_fixture():
    metrics, _, _, _ = _assess("stroke_pattern")
    assert metrics["brow_ratio"] == pytest.approx(0.95, abs=0.02)
    assert metrics["smile_ratio"] == pytest.approx(0.40, abs=0.02)


def test_peripheral_fixture_ratios():
    metrics, _, _, _ = _assess("peripheral_like")
    assert metrics["brow_ratio"] == pytest.approx(0.40, abs=0.02)
    assert metrics["smile_ratio"] == pytest.approx(0.50, abs=0.02)
    assert metrics["eye_weak_completeness"] == pytest.approx(0.40, abs=0.02)


@pytest.mark.parametrize("case,zone", [
    ("stroke_pattern", "stroke_pattern"),
    ("peripheral_like", "peripheral_like_pattern"),
    ("indeterminate", "indeterminate"),
    ("symmetric", "indeterminate"),
    ("poor_quality", "indeterminate"),
])
def test_zone_cases(case, zone):
    _, z, _, _ = _assess(case)
    assert z["zone"] == zone


def test_quality_gate_failure_reason():
    _, z, _, _ = _assess("poor_quality")
    assert z["reason"] == "quality_gate_failed"
    assert z["zone"] == "indeterminate"


def test_bilateral_weakness_goes_indeterminate():
    frames = synth.make_fixture(weak_side="left", brow_ratio=1.0, smile_ratio=1.0)
    # Shrink all excursions: rebuild with tiny strong-side movement by
    # scaling brow/smiler displacement is not exposed, so emulate bilateral
    # weakness directly via a fixture whose excursions are all near zero:
    for frame in frames.values():
        for pt in frame["points"].values():
            pt["y"] = 50.0
            pt["x"] = 50.0
    # Degenerate all-identical points: brow and smile excursions are 0.
    problems = service.quality_gate(frames)
    metrics = service.measure(frames)
    zone = service.classify(metrics, problems, {})
    assert zone["zone"] == "indeterminate"
    assert zone["reason"] == "bilateral_weakness"


def test_puff_cheeks_never_changes_zone():
    frames = synth.demo_cases()["stroke_pattern"]
    frames2 = dict(frames)
    frames2["puff_cheeks"] = frames["smile"]
    m1 = service.measure(frames)
    m2 = service.measure({k: frames2[k] for k in service.REQUIRED_FRAMES})
    assert m1 == m2


@pytest.mark.parametrize("ctx", [
    {"sudden_onset": True},
    {"arm_or_leg_weakness": True},
    {"speech_change": True},
])
def test_context_always_escalates(ctx):
    _, _, action, _ = _assess("peripheral_like", context=ctx)
    assert action == "emergency_now"


def test_indeterminate_and_stroke_always_emergency():
    for case in ("stroke_pattern", "indeterminate", "symmetric", "poor_quality"):
        _, _, action, _ = _assess(case)
        assert action == "emergency_now"


def test_peripheral_without_context_is_prompt_in_person_not_emergency():
    _, _, action, _ = _assess("peripheral_like")
    assert action == "prompt_in_person"


# --- Safety language --------------------------------------------------------

@pytest.mark.parametrize("case", ["stroke_pattern", "peripheral_like", "indeterminate", "symmetric", "poor_quality"])
def test_no_banned_language_in_any_output(case):
    _, _, _, text = _assess(case)
    assert safety.find_banned_language(text) == []


@pytest.mark.parametrize("case", ["stroke_pattern", "peripheral_like", "indeterminate", "symmetric", "poor_quality"])
def test_required_language_present_in_every_output(case):
    _, _, _, text = _assess(case)
    assert safety.missing_required_language(text) == []


def test_output_states_unknown_accuracy_and_both_error_directions():
    _, _, _, text = _assess("peripheral_like")
    low = text.lower()
    assert "false negatives are possible" in low
    assert "false positives are possible" in low
    assert "tendency, not a rule" in low
    assert "does not check balance" in low


def test_banned_scan_catches_verdict_language():
    assert safety.find_banned_language("This ruled out a stroke, you are safe.") != []
    assert safety.find_banned_language("You have a stroke.") != []


def test_phi_redaction():
    redacted, report = safety.redact_phi("Call me at 661-648-0204, patient@example.com")
    assert "661" not in redacted
    assert report


# --- Gateway ----------------------------------------------------------------

def test_gateway_rejects_image_parts():
    with pytest.raises(Exception):
        gateway.assert_metrics_only([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])


def test_gateway_rejects_image_keys_in_text_json():
    with pytest.raises(Exception):
        gateway.assert_metrics_only([{"role": "user", "content": '{"image": "abc123"}'}])


def test_gateway_models_and_tiers():
    assert gateway.list_models()["data"][0]["id"] == "faceflag-frugal"
    assert gateway.resolve_tier("faceflag-full") == "full"
    assert gateway.resolve_tier(None, "full") == "full"
    assert gateway.resolve_tier() == "frugal"


def test_gateway_ledger_records_no_frame_content():
    gateway.reset_ledger()
    gateway.dispatch_chat([{"role": "user", "content": "brow_ratio=0.40, smile_ratio=0.50"}])
    assert gateway.CALL_LOG[0]["kind"] == "chat"
    assert "frames" not in str(gateway.CALL_LOG)


def test_embeddings_deterministic():
    a = gateway.dispatch_embeddings("brow_ratio=0.40")
    b = gateway.dispatch_embeddings("brow_ratio=0.40")
    assert a["data"][0]["embedding"] == b["data"][0]["embedding"]


# --- API --------------------------------------------------------------------

def test_healthz():
    assert client.get("/healthz").status_code == 200


def test_demo_assess_all_cases():
    for case in synth.demo_cases():
        r = client.post(f"/api/demo/assess?case={case}", headers={"X-Tier": "frugal"})
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]
        assert safety.find_banned_language(body["summary"]) == []
        assert safety.missing_required_language(body["summary"]) == []


def test_demo_full_tier_verifier():
    r = client.post("/api/demo/assess?case=stroke_pattern", headers={"X-Tier": "full"})
    assert r.json()["verifier_ok"] is True


def test_demo_unknown_case_rejected():
    assert client.post("/api/demo/assess?case=nope").status_code == 400


def test_assess_rejects_photo_payload_shape():
    payload = {
        "patient_id": "demo",
        "rest": {"points": {}, "quality": {}},
        "brow_raise": {"points": {}, "quality": {}},
        "eye_close": {"points": {}, "quality": {}},
        "smile": {"points": {}, "quality": {}},
        "photo": "base64data",
    }
    assert client.post("/api/assess", json=payload).status_code == 422


def test_ui_serves_disclaimer():
    html = client.get("/").text
    assert "NOT CLINICALLY TESTED" in html
    assert "dial 911" in html


def test_trend_endpoint():
    client.post("/api/demo/assess?case=stroke_pattern")
    r = client.get("/api/trend")
    assert r.status_code == 200
    assert r.json()["assessments"]


# --- Repo-level checks -------------------------------------------------------

def test_readme_disclaimer_is_first():
    text = (ROOT / "README.md").read_text()
    assert text.lstrip().startswith("> ## ⚠️ PROOF OF CONCEPT")


def test_readme_and_evidence_never_borrow_fastai_numbers():
    for path in (ROOT / "README.md", ROOT / "EVIDENCE.md"):
        t = path.read_text()
        assert "99.42" not in t, f"FAST.AI number leaked into {path.name}"
        assert "93.67" not in t


def test_evidence_file_covers_audit_findings():
    t = (ROOT / "EVIDENCE.md").read_text()
    for phrase in ("tendency, not a rule", "pontine", "splenial",
                   "sensitivity and specificity are unknown",
                   "does not check balance", "no diagnostic-accuracy study"):
        assert phrase.lower() in t.lower(), phrase


def test_eval_thresholds_are_labelled_app_defined():
    t = (ROOT / "README.md").read_text()
    assert "APP_DEFINED_UNVALIDATED" in t
