> ## ⚠️ PROOF OF CONCEPT — NOT CLINICALLY TESTED
>
> **This tool has not been tested clinically and is a proof-of-concept for
> research and software demonstration only. It is not a medical device, it
> does not diagnose, treat, or rule out any condition, and its outputs may
> be wrong — including false positives and false negatives.**
>
> **Do not use it to make decisions about your health.** For any medical issue or concern, seek a licensed healthcare professional. **In an emergency, go to
> your nearest emergency room or dial 911 immediately.**

---

# FaceFlag

**Open-source facial-movement pattern triage from a guided selfie sequence.
Rest, raise eyebrows, close eyes tight, big smile: landmarks in, ratios out,
three zones, no photo ever reaching a model. It measures a pattern for a
clinician or emergency team to review. It never diagnoses, and it never says
a stroke is ruled out.**

## The measurable failure, and what moves it

| Metric | Current workflow | With FaceFlag |
| --- | --- | --- |
| Telling a peripheral pattern from a central pattern at home | A person googles "face drooping", guesses, or waits | Guided sequence with measured brow and smile ratios, a rest baseline, and a quality gate. Eval harness on synthetic landmark sets with known ratios: **brow ratio error within 0.02, smile ratio error within 0.02** |
| False reassurance from a single photo | "It looks like Bell's, I'll wait" | Three zones, never a binary call. Indeterminate always escalates. Every output states both error directions and that sensitivity/specificity are unknown |
| A face-only check pretending to cover a stroke | FAST-style apps that imply a full check | Scope stated in every output: this app measures the face only. It does not check balance, eyes, arm or leg strength, or speech |

The eval harness in `tests/test_faceflag.py` fails the build if a zone case
is wrong, if banned language appears in any generated output, if required
language (limitations, false-negative warning, false-positive explanation,
emergency instruction) is missing from any output template, or if an
escalation-path case does not route to emergency guidance. Current results,
run on the shipped synthetic fixtures: **44 tests passed** (count verified
at build time in the test run; see the test file for the full list).

Honest note on those numbers: the fixtures are landmark sets built by code
in this repo (`app/face/synth.py`) with known ground-truth ratios. They
prove the geometry and zone logic are internally correct. They do not prove
anything about real selfies, real skin tones, glasses, facial hair, lighting,
or real stroke and Bell's patients. Real-world accuracy is unproven, and
that validation set is roadmap issue #1.

## How it works

```
Guided sequence ──► on-device face mesh (integration path, see below)
  rest, brow raise, eye close, smile
        │
        ▼
Landmarks only ──► quality gates (lighting, head rotation, landmark
  (no photo)        confidence, movement detected) ──► fail? ──►
                    indeterminate, seek emergency care
        │
        ▼
Deterministic ratios, in code:
  brow ratio (forehead involvement), smile ratio (lower-face droop),
  eye-closure completeness, rest asymmetry, nasolabial proxy
        │
        ▼
Three zones, APP_DEFINED_UNVALIDATED splits:
  peripheral-like pattern  = forehead AND lower face reduced, same side
  stroke-pattern           = forehead relatively preserved, lower face
                             markedly asymmetric
  indeterminate            = quality fail, bilateral weakness, or a
                             pattern that matches neither classic pattern
        │
        ▼
Gateway phrases the metrics JSON (never measures, never sees frames)
        │
        ▼
Result: pattern language + numbers + both error directions + action
```

Design rules that do the safety work:

1. **The model never measures.** Every number comes from deterministic code
   in `app/face/geometry.py` and `app/face/service.py`. The gateway receives
   metrics JSON only. Image and video payload shapes are rejected (400 on
   the gateway, 422 on the product API).
2. **Forehead pattern is a tendency, not a rule.** Upper-face muscles get
   input from both sides of the brain, so a stroke above the pons usually
   spares the forehead relatively, while a peripheral lesion affects the
   whole side. Usually is the ceiling. Brainstem (pontine) strokes affect
   the whole side by anatomy, and rare strokes above the pons can weaken
   the forehead too (documented splenial infarct mimic; see EVIDENCE.md).
   That is why no pattern result ever excludes stroke.
3. **Context overrides the face.** Sudden onset, limb weakness, or speech
   change force emergency action regardless of the zone. These answers come
   from questions, not from the face mesh, which cannot test them.
4. **Puff-cheeks never changes the zone.** It is collected for
   completeness only; it is not a validated discriminator.
5. **Bilateral weakness goes indeterminate.** Left/right ratios cannot
   describe bilateral disease, and bilateral causes break the ratio logic.
6. **Quality gates fail to escalation, never to reassurance.** A bad
   capture is reported as "could not be measured reliably", with emergency
   guidance, not as a low-quality guess.

## Two tiers, one gateway

Tier is a routing profile on the same gateway, chosen by the `model` name
(`faceflag-frugal` / `faceflag-full`), an `X-Tier` header, or one env var.
Same endpoints, same JSON schema, same UI, same eval harness. Landmark
extraction and ratios are identical in both tiers by design.

|  | Frugal (default) | Full / commercial |
| --- | --- | --- |
| Vision pass | Landmarks only, on-device; gateway gets ~100 tokens of metrics JSON | Same landmarks, same ratios; no new vision |
| What the spend buys | The pattern result itself | Richer longitudinal context plus a second-pass check that the summary's numbers match the metrics block |
| Verification | Safety language scans in CI | Plus the number verifier on every summary |

## The architecture: the product ships its own AI gateway

FaceFlag serves OpenAI-compatible endpoints (`/v1/models`,
`/v1/chat/completions`, `/v1/embeddings`) from inside the app, and every AI
feature calls those endpoints. Nothing talks to a vendor API directly. One
environment variable picks the backend:

- `stub` (default): deterministic, rule-based. No model, no keys. This is
  what CI and the demo run.
- `local`: an OpenAI-compatible server on your own hardware (Ollama,
  llama.cpp, vLLM) with a small text model. Metrics JSON never has to
  leave the home or clinic. This is the deployment the product is built for.
- `upstream`: any OpenAI-compatible provider, for teams that accept that
  trade-off.

This gateway is the shared platform skeleton across this weekly product
series. Week 3's WoundDelta shipped the vision tiers, Week 4's PillTray
reused them for reconciliation, Week 5's Pose-Not-Pixels added the
landmark-input pattern that FaceFlag follows, and Week 6 keeps the skeleton
and swaps in the facial-movement domain module. See docs/ARCHITECTURE.md.

## Quickstart

```
git clone https://github.com/rtmalikian/faceflag
cd faceflag
cp .env.example .env
docker compose up --build
# open http://localhost:8000
```

Or without Docker:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
```

Try a synthetic demo in the UI (the same fixtures CI measures), or:

```
curl -X POST "http://localhost:8000/api/demo/assess?case=stroke_pattern" \
  -H "X-Tier: frugal"
```

## API sketch

| Route | What it does |
| --- | --- |
| `GET /v1/models` | Lists `faceflag-frugal`, `faceflag-full`, and the stub model |
| `POST /v1/chat/completions` | OpenAI-shaped chat, metrics JSON only, tier by model alias or `X-Tier` header |
| `POST /v1/embeddings` | OpenAI-shaped embeddings (metrics summaries, longitudinal use) |
| `POST /api/assess` | Landmark assessment: rest, brow_raise, eye_close, smile frames + context |
| `POST /api/demo/assess?case=...` | Synthetic landmark case with known ground truth, assessed live |
| `GET /api/trend` | Zone and ratio series for assessments in this session |

## Tests

```
.venv/bin/python -m pytest -q
```

The suite covers ratio geometry on known fixtures, all zone cases,
quality gates, bilateral handling, context escalation, banned-language
and required-language scans on every demo output, gateway rejection of
image/video payloads, the API end to end, the PHI redaction gate, and the
repo-level checks (disclaimer first in README, no borrowed accuracy
numbers, EVIDENCE.md coverage, thresholds labelled app-defined).

## Roadmap

Issues #1 to #5 are open as collaboration magnets: a labelled real-world
validation set in undifferentiated acute facial weakness, a live on-device
face-mesh extraction path, persistence plus auth, threshold and normative
data from healthy-volunteer selfies, and a prospective triage-outcomes
study. Contributions welcome, especially from neurologists, emergency
physicians, and facial-palsy clinicians willing to say where the zone
splits are wrong.

## License

MIT. Built by Raphael Malikian (https://github.com/rtmalikian) as Week 6
of a weekly open-source clinical AI series.
