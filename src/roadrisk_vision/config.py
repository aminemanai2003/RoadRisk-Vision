"""Validated application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hybridnets_weights: Path = Path("models/hybridnets.pth")
    yolox_weights: Path = Path("models/yolox_s.pth")
    hybridnets_size: tuple[int, int] = (640, 384)
    yolox_size: tuple[int, int] = (640, 640)
    confidence: float = Field(0.35, ge=0, le=1)
    nms_iou: float = Field(0.45, ge=0, le=1)
    classes: tuple[str, ...] = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    )


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monitor_ttc_s: float = Field(6.0, gt=0)
    slow_down_ttc_s: float = Field(4.0, gt=0)
    brake_risk_ttc_s: float = Field(2.5, gt=0)
    min_event_duration_ms: int = Field(350, ge=0)
    lane_departure_margin: float = Field(0.12, ge=0, le=0.5)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    device: Literal["auto", "cuda", "cpu"] = "auto"
    output_root: Path = Path("runs")
    normalized_fps_cap: int = Field(30, ge=1, le=60)
    backend: Literal["hybrid", "mock"] = "hybrid"
    include_location: bool = False
    models: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    @classmethod
    def from_yaml(cls, path: Path | None) -> AppConfig:
        if path is None:
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
