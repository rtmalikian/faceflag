"""Request/response models for FaceFlag.

Frames are landmark point sets only. There is deliberately no field that
can carry a photo, frame, or video: Pydantic forbids extra fields, so a
payload shaped like an image is rejected (422) before any processing.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

POINT_NAMES = (
    "brow_left_inner", "brow_left_outer", "brow_right_inner", "brow_right_outer",
    "eye_left_upper", "eye_left_lower", "eye_right_upper", "eye_right_lower",
    "nose_tip", "nose_left", "nose_right",
    "mouth_left", "mouth_right", "mouth_upper", "mouth_lower",
    "chin", "cheek_left", "cheek_right",
)


class Point(BaseModel):
    x: float
    y: float
    confidence: float = 1.0

    model_config = {"extra": "forbid"}


class Quality(BaseModel):
    lighting: float = 1.0
    head_rotation_deg: float = 0.0
    landmark_confidence: float = 1.0
    movement_detected: bool = True

    model_config = {"extra": "forbid"}


class Frame(BaseModel):
    points: dict[str, Point]
    quality: Quality = Field(default_factory=Quality)

    model_config = {"extra": "forbid"}


class Context(BaseModel):
    sudden_onset: bool = False
    last_known_well: str | None = None
    arm_or_leg_weakness: bool = False
    speech_change: bool = False
    gradual_onset: bool = False

    model_config = {"extra": "forbid"}


class AssessmentRequest(BaseModel):
    patient_id: str = "anonymous"
    rest: Frame
    brow_raise: Frame
    eye_close: Frame
    smile: Frame
    puff_cheeks: Frame | None = None
    context: Context = Field(default_factory=Context)

    model_config = {"extra": "forbid"}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 2000
    response_format: dict | None = None

    model_config = {"extra": "forbid"}


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str] = ""

    model_config = {"extra": "forbid"}
