# ARCHITECTURE.md — FaceFlag

## Layers

- `app/main.py` — FastAPI routes. Product API (`/api/*`) and the embedded
  gateway (`/v1/*`) share one process. Storage is in-memory in v0.1.0.
- `app/face/geometry.py` — pure landmark geometry: distances, eye centres,
  intercanthal scale, weaker/stronger sides, ratios.
- `app/face/service.py` — quality gates, measurement, three-zone
  classification, action decision. All zone splits are
  APP_DEFINED_UNVALIDATED and labelled as such in code comments.
- `app/face/synth.py` — synthetic landmark fixtures with known ratios.
- `app/face/models.py` — Pydantic models. Extra fields are forbidden, so
  a photo/frame/video payload shape is rejected (422) before processing.
- `app/gateway.py` — embedded OpenAI-compatible gateway. Metrics JSON
  only; image/video parts and keys rejected (400). Ledger records kinds,
  tiers, backends and token estimates only.
- `app/safety.py` — PHI redaction, banned-language list, required-language
  list, and the result renderer. Rendering asserts both scans pass.

## What v0.1.0 does NOT ship

Live on-device face-mesh extraction. The product API accepts landmark
JSON, and the demo UI runs synthetic landmark cases. A real deployment
integrates a face-mesh library (MediaPipe Face Mesh or equivalent) on the
device, extracts the point set defined in `app/face/models.py`, deletes
raw frames by default, and posts only landmarks. Until that integration
lands, do not describe FaceFlag as a camera app: it is a landmark-ratio
engine with a documented camera path.

## Tier design

Frugal and full share the landmark pipeline, the ratios, and the zone
logic. Full adds longitudinal context and a verifier that checks the
rendered summary contains the formatted brow and smile ratios and an
action phrase. Neither tier changes a measurement, and neither tier sees
a frame.
