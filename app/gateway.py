"""Embedded OpenAI-compatible gateway for FaceFlag.

Same skeleton as pose-not-pixels / pill-tray / wound-delta. The gateway
takes METRICS JSON ONLY. Any message content part of type image_url /
video_url, or any part carrying an image / video / frame key, is rejected
with HTTP 400 before any backend is contacted. The call ledger records
kinds, tiers, backends and token estimates only: never message contents,
never landmarks, never frames. Tests scan the ledger for image/video
keys and must find none.

Landmark extraction is IDENTICAL in both tiers by design. Frugal phrases
a short summary; full buys richer longitudinal context plus a second-pass
check that the summary's numbers match the metrics JSON. It buys no new
vision, and no tier ever sees a photo.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from fastapi import HTTPException

STUB_MODEL_ID = "faceflag-stub-1"
FRUGAL_MODEL_ID = "faceflag-frugal"
FULL_MODEL_ID = "faceflag-full"

TIERS: dict[str, dict] = {
    "frugal": {"model_alias": FRUGAL_MODEL_ID, "verifier": False,
               "summary_max_tokens": 220, "longitudinal_context": "latest_only"},
    "full": {"model_alias": FULL_MODEL_ID, "verifier": True,
             "summary_max_tokens": 500, "longitudinal_context": "all_previous"},
}

_FORBIDDEN_PART_TYPES = {"image_url", "video_url", "image", "video", "frame"}
_FORBIDDEN_KEYS = {"image_url", "video_url", "video", "image", "frames",
                   "frame_data", "pixels", "base64"}

CALL_LEDGER: dict[str, int] = {"chat": 0, "embeddings": 0}
CALL_LOG: list[dict] = []


def reset_ledger() -> None:
    CALL_LEDGER["chat"] = 0
    CALL_LEDGER["embeddings"] = 0
    CALL_LOG.clear()


def get_backend() -> str:
    return (os.environ.get("AI_BACKEND", "stub") or "stub").strip().lower()


def resolve_tier(model: str | None = None, x_tier: str | None = None) -> str:
    if x_tier and x_tier.strip().lower() in TIERS:
        return x_tier.strip().lower()
    if model:
        m = model.strip().lower()
        if m.endswith("-full") or m == FULL_MODEL_ID:
            return "full"
        if m.endswith("-frugal") or m == FRUGAL_MODEL_ID:
            return "frugal"
    env = (os.environ.get("FACEFLAG_TIER", "frugal") or "frugal").strip().lower()
    return env if env in TIERS else "frugal"


def tier_profile(tier: str) -> dict:
    return dict(TIERS[tier])


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def assert_metrics_only(messages: list[dict]) -> None:
    for m in messages or []:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in _FORBIDDEN_PART_TYPES:
                    raise HTTPException(status_code=400, detail="gateway takes metrics JSON only; image/video parts rejected")
                if _FORBIDDEN_KEYS & set(part.keys()):
                    raise HTTPException(status_code=400, detail="gateway takes metrics JSON only; image/video keys rejected")
        elif isinstance(content, str):
            low = content.lower()
            for key in _FORBIDDEN_KEYS:
                if f'"{key}"' in low:
                    raise HTTPException(status_code=400, detail="gateway takes metrics JSON only; image/video payload rejected")


def _extract_metrics_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages or []:
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def dispatch_chat(messages: list[dict], model: str | None = None,
                  temperature: float = 0.2, max_tokens: int = 2000,
                  response_format: dict | None = None, tier: str = "frugal") -> dict:
    assert_metrics_only(messages)
    text = _extract_metrics_text(messages)
    CALL_LEDGER["chat"] += 1
    CALL_LOG.append({"kind": "chat", "tier": tier, "backend": get_backend(),
                     "tokens_est": estimate_text_tokens(text)})
    # Stub backend: phrases the metrics JSON deterministically. A real
    # local/upstream text model would phrase the same JSON; it would
    # never measure, and it never sees frames. The full tier verifier
    # (applied by the service layer) checks the summary's numbers
    # against the metrics block.
    summary = ("FaceFlag gateway summary (metrics JSON only): "
               + text[:600].replace("\n", " "))
    return {
        "id": "chatcmpl-faceflag-stub",
        "object": "chat.completion",
        "model": model or TIERS[tier]["model_alias"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": summary},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": estimate_text_tokens(text),
                  "completion_tokens": estimate_text_tokens(summary),
                  "total_tokens": estimate_text_tokens(text) + estimate_text_tokens(summary)},
    }


def dispatch_embeddings(text_input, model: str | None = None) -> dict:
    items = text_input if isinstance(text_input, list) else [text_input]
    CALL_LEDGER["embeddings"] += len(items)
    data = []
    for i, item in enumerate(items):
        digest = hashlib.sha256(str(item).encode()).digest()
        vector = [((b - 128) / 128.0) for b in digest[:16]]
        data.append({"object": "embedding", "index": i, "embedding": vector})
    return {"object": "list", "data": data, "model": model or STUB_MODEL_ID,
            "usage": {"prompt_tokens": sum(estimate_text_tokens(str(x)) for x in items),
                      "total_tokens": sum(estimate_text_tokens(str(x)) for x in items)}}


def list_models() -> dict:
    return {"object": "list", "data": [
        {"id": FRUGAL_MODEL_ID, "object": "model"},
        {"id": FULL_MODEL_ID, "object": "model"},
        {"id": STUB_MODEL_ID, "object": "model"},
    ]}
