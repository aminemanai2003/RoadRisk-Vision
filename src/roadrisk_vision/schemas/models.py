"""Pydantic models shared by the CLI, dashboard and exported artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WarningState(StrEnum):
    NORMAL = "normal"
    MONITOR = "monitor"
    SLOW_DOWN = "slow_down"
    BRAKE_RISK = "brake_risk"
    LANE_DEPARTURE = "lane_departure"
    HAZARD = "hazard"


class EventType(StrEnum):
    OBJECT_IN_PATH = "object_in_path"
    RAPID_CLOSING = "rapid_closing"
    COLLISION_RISK = "collision_risk"
    LANE_DEPARTURE = "lane_departure"


class Severity(IntEnum):
    INFORMATIONAL = 1
    CAUTION = 2
    WARNING = 3
    CRITICAL = 4


class NullReason(StrEnum):
    NOT_CALIBRATED = "not_calibrated"
    INCOMPATIBLE_CALIBRATION = "incompatible_calibration"
    UNSTABLE_ESTIMATE = "unstable_estimate"
    MISSING_TELEMETRY = "missing_telemetry"
    LOW_CONFIDENCE = "low_confidence"
    NOT_CLOSING = "not_closing"


class BoundingBox(BaseModel):
    """XYXY floating-point pixels in normalized output coordinates."""

    model_config = ConfigDict(extra="forbid")
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    track_id: int | None = None
    distance_m: float | None = None
    distance_null_reason: NullReason | None = None
    relative_speed_mps: float | None = None
    ttc_s: float | None = None
    ttc_null_reason: NullReason | None = None
    in_path: bool = False


class FramePacket(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    frame_index: int = Field(ge=0)
    video_time_ms: int = Field(ge=0)
    image: np.ndarray


class FramePerception(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    frame_index: int
    video_time_ms: int
    detections: list[Detection] = Field(default_factory=list)
    drivable_mask: np.ndarray | None = None
    lane_mask: np.ndarray | None = None


class TelemetryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_time_ms: int
    latitude: float | None = None
    longitude: float | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None
    accuracy_m: float | None = None


class RiskEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    event_id: str
    run_id: str
    event_type: EventType
    warning_state: WarningState
    severity: Severity
    start_time_ms: int
    end_time_ms: int
    track_ids: list[int] = Field(default_factory=list)
    min_ttc_s: float | None = None
    min_distance_m: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class TripSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    run_id: str
    duration_s: float = 0
    valid_distance_km: float | None = None
    gps_coverage_ratio: float = 0
    event_counts: dict[str, int] = Field(default_factory=dict)
    risk_events_per_hour: float = 0
    risk_events_per_100_km: float | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    run_id: str
    status: RunStatus = RunStatus.RUNNING
    source_basename: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    package_version: str
    normalized_fps: float | None = None
    frame_count: int | None = None
    device: str
    backend: str
    include_location: bool = False
    models: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
