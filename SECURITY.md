# SECURITY.md — FaceFlag

## The limits, plainly

- FaceFlag is a proof of concept. It has not been tested clinically, it is
  not a medical device, and it does not diagnose, treat, or rule out any
  condition. Its sensitivity and specificity are unknown.
- A selfie can never rule out a stroke. Some strokes weaken the forehead
  too (splenial and pontine mimics are documented), some strokes cause no
  facial weakness at all, and mild strokes can be subtle. A whole-side
  pattern result must still lead to prompt in-person assessment.
- FaceFlag measures the face only. It does not check balance, eyes, arm
  or leg strength, or speech. Strokes that show up outside the face are
  outside its scope entirely.
- Normal faces are asymmetric. A single ratio threshold will flag healthy
  people, which is why the app uses a rest baseline, capture-quality
  gates, and an indeterminate zone that always escalates.
- Bilateral facial weakness breaks left/right ratio logic. Bilateral
  cases go to indeterminate and escalate; they are never assigned a
  pattern zone by ratio.
- Puff-cheeks never changes the zone. It is collected for completeness
  only; it is not a validated discriminator.
- No threshold in this repo is a validated cut-off. All zone splits are
  APP_DEFINED_UNVALIDATED screening splits for this proof of concept.
- No treatment recommendation is ever given. Steroid timing, eye care
  beyond "seek prompt in-person review if the eye will not close", and
  any medication decision are clinician decisions, not app outputs.
- Regulatory status: none claimed. No FDA clearance, CE marking, or
  medical-device validation is implied anywhere in this repo.

## Data handling

- v0.1.0 accepts landmark point sets only. There is no photo, frame, or
  video input type, and the gateway rejects image/video payloads.
  A real deployment must extract landmarks on-device and delete raw
  frames by default; only landmarks and derived metrics are kept.
- Storage is in-memory in v0.1.0. There is no authentication. Do not
  deploy this version with real patient data.
- Patient labels pass through a PHI redaction gate before storage.
  Logs carry kinds and counts only, never values, landmarks, or text.

## Reporting a safety problem

If you find output language that diagnoses, reassures, or rules out a
stroke, or a code path that assigns a pattern zone from a failed-quality
capture, open an issue with the exact output text and the demo case.
Those are treated as safety bugs, not cosmetic bugs.
