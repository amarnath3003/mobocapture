from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "0.1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProvenanceClass(str, Enum):
    """Who or what supplied a value.

    This is intentionally independent of :class:`EpistemicStatus`. For
    example, an offline processor can produce either a deterministic value or
    a hypothesis.
    """

    MEASURED = "measured"
    DEVICE_ESTIMATED = "device_estimated"
    OFFLINE_ESTIMATED = "offline_estimated"
    HUMAN_ANNOTATED = "human_annotated"
    HUMAN_VERIFIED = "human_verified"
    SYNTHETIC = "synthetic"


class EpistemicStatus(str, Enum):
    """How strongly a value is supported by available evidence."""

    DETERMINISTIC = "deterministic"
    ESTIMATE = "estimate"
    HYPOTHESIS = "hypothesis"
    CONDITIONAL = "conditional"


class InputAsset(StrictModel):
    original_name: str
    stored_path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class VideoMetadata(StrictModel):
    codec_name: str | None = None
    codec_long_name: str | None = None
    pixel_format: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    display_rotation_degrees: int = 0
    nominal_frame_rate: float | None = Field(default=None, gt=0)
    average_frame_rate: str | None = None
    time_base: str
    duration_ns: int | None = Field(default=None, ge=0)
    probed_frame_count: int = Field(ge=0)
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None


class SessionManifest(StrictModel):
    session_id: str
    schema_version: str = SCHEMA_VERSION
    created_at_utc: datetime = Field(default_factory=utc_now)
    status: Literal["processing", "complete", "complete_with_unavailable", "failed"]
    device_class: Literal["imported_rgb_video"] = "imported_rgb_video"
    capabilities: dict[str, bool] = Field(default_factory=lambda: {"rgb": True})
    coordinate_convention: str = "image: x-right, y-down, pixel units"
    input: InputAsset
    video: VideoMetadata | None = None
    requested_modules: list[str] = Field(default_factory=list)
    resolved_processors: list[str] = Field(default_factory=list)
    unavailable_processors: list[str] = Field(default_factory=list)


class RequestedModulesManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    profile: str | None = None
    modules: list[str]
    options: dict[str, Any] = Field(default_factory=dict)


class ResolvedProcessor(StrictModel):
    processor_id: str
    version: str
    status: Literal["ready", "unavailable"]
    required_by_modules: list[str]
    requires: list[str]
    produces: list[str]
    code_license: str
    model_license: str | None = None
    unavailable_reason: str | None = None


class ResolvedProcessorsManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    requested_modules: list[str]
    resolved_modules: list[str]
    processors: list[ResolvedProcessor]


class ProcessorOutput(StrictModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class ProcessorRunManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    processor_id: str
    processor_version: str
    run_id: str
    status: Literal["running", "complete", "failed"]
    started_at_utc: datetime
    completed_at_utc: datetime | None = None
    input_sha256: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[ProcessorOutput] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RunReport(StrictModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    status: Literal["complete", "complete_with_unavailable", "failed"]
    completed_processors: list[str]
    unavailable_processors: list[ResolvedProcessor]
    warnings: list[str] = Field(default_factory=list)

