"""FaceFlag — FastAPI application.

Routes:
  /v1/*    Embedded OpenAI-compatible gateway (metrics JSON only).
           Tier by model alias (faceflag-frugal | full) or X-Tier header.
           The gateway never sees frames: they are not an input type.
  /api/*   Product API: landmark assessments, trend, synthetic demos.
  /        Static single-page demo UI.
  /healthz Health check.

Storage is in-memory in v0.1.0. Landmarks are the only input. A video /
frame / photo payload shape is rejected (422) before any processing, and
the gateway rejects image/video content parts (400).
"""
from __future__ import annotations

import logging
import re

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles

from app import gateway, safety
from app.face import models, service, synth

logger = logging.getLogger("faceflag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="FaceFlag",
    description="Guided selfie-sequence facial-movement pattern triage from "
                "landmark ratios only. Screening information for clinician / "
                "emergency review. Never a diagnosis, never a rule-out.",
    version="0.1.0",
)

ASSESSMENTS: list[dict] = []


def _frames(req) -> dict:
    out = {}
    for name in ("rest", "brow_raise", "eye_close", "smile"):
        frame = getattr(req, name)
        out[name] = {"points": {k: v.model_dump() for k, v in frame.points.items()},
                     "quality": frame.quality.model_dump()}
    return out


def _analyse(patient_id: str, frames: dict, context: dict, tier: str) -> dict:
    problems = service.quality_gate(frames)
    metrics = service.measure(frames)
    zone_info = service.classify(metrics, problems, context)
    action = service.decide_action(zone_info["zone"], context)
    text = safety.render_result(metrics, zone_info, action, context)
    # Gateway call: metrics JSON only, recorded in the ledger. The full
    # tier verifier checks the rendered summary's numbers against the
    # metrics block; it adds no new vision.
    gateway.dispatch_chat(
        [{"role": "user", "content": (
            f"FaceFlag metrics: brow_ratio={metrics['brow_ratio']:.2f}, "
            f"smile_ratio={metrics['smile_ratio']:.2f}, "
            f"zone={zone_info['zone']}")}], tier=tier)
    verifier_ok = True
    if tier == "full":
        verifier_ok = (f"{metrics['brow_ratio']:.2f}" in text
                       and f"{metrics['smile_ratio']:.2f}" in text
                       and ("seek emergency care now" in text
                            or "seek prompt in-person assessment" in text))
    record = {
        "patient_id": patient_id, "metrics": metrics, "zone": zone_info["zone"],
        "zone_reason": zone_info["reason"], "zone_detail": zone_info["detail"],
        "quality_problems": problems, "action": action, "tier": tier,
        "verifier_ok": verifier_ok, "summary": text,
        "context": {"sudden_onset": context.get("sudden_onset", False),
                    "arm_or_leg_weakness": context.get("arm_or_leg_weakness", False),
                    "speech_change": context.get("speech_change", False),
                    "last_known_well": context.get("last_known_well")},
    }
    ASSESSMENTS.append(record)
    return record


@app.get("/v1/models")
def v1_models():
    return gateway.list_models()


@app.post("/v1/chat/completions")
def v1_chat_completions(req: models.ChatRequest, x_tier: str | None = Header(default=None)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    tier = gateway.resolve_tier(req.model, x_tier)
    return gateway.dispatch_chat(req.messages, model=req.model,
                                 temperature=req.temperature, max_tokens=req.max_tokens,
                                 response_format=req.response_format, tier=tier)


@app.post("/v1/embeddings")
def v1_embeddings(req: models.EmbeddingsRequest):
    return gateway.dispatch_embeddings(req.input, model=req.model)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": gateway.get_backend(),
            "tier_default": gateway.resolve_tier(), "assessments": len(ASSESSMENTS)}


@app.post("/api/assess")
def api_assess(req: models.AssessmentRequest, x_tier: str | None = Header(default=None)):
    patient_id, _ = safety.redact_phi(req.patient_id)
    tier = gateway.resolve_tier(None, x_tier)
    return _analyse(patient_id, _frames(req), req.context.model_dump(), tier)


@app.post("/api/demo/assess")
def api_demo_assess(case: str = "stroke_pattern", x_tier: str | None = Header(default=None)):
    cases = synth.demo_cases()
    if case not in cases:
        raise HTTPException(status_code=400, detail=f"unknown demo case: {case}")
    tier = gateway.resolve_tier(None, x_tier)
    return _analyse("demo", cases[case], {"sudden_onset": False,
                                          "arm_or_leg_weakness": False,
                                          "speech_change": False}, tier)


@app.get("/api/trend")
def api_trend():
    return {"assessments": [
        {"zone": a["zone"], "action": a["action"], "tier": a["tier"],
         "brow_ratio": a["metrics"]["brow_ratio"],
         "smile_ratio": a["metrics"]["smile_ratio"]} for a in ASSESSMENTS]}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
