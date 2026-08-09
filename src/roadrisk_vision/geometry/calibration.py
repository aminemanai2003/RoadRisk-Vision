"""Road-plane calibration with explicit compatibility and null semantics."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from roadrisk_vision.schemas import Detection, NullReason


class CalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    camera_id: str
    lens_id: str = "default"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    orientation_deg: int = 0
    crop_id: str = "full-frame"
    mount_height_m: float = Field(gt=0)
    reference_width_m: float = Field(gt=0)
    reference_length_m: float = Field(gt=0)
    image_corners_xy: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    image_to_ground: list[list[float]]
    residual_pct: float = Field(ge=0)

    @classmethod
    def create(
        cls,
        *,
        camera_id: str,
        width: int,
        height: int,
        mount_height_m: float,
        reference_width_m: float,
        reference_length_m: float,
        corners: list[tuple[float, float]],
        lens_id: str = "default",
        orientation_deg: int = 0,
        crop_id: str = "full-frame",
    ) -> CalibrationProfile:
        if len(corners) != 4:
            raise ValueError(
                "Exactly four corners are required: near-left, near-right, far-right, far-left"
            )
        source = np.asarray(corners, dtype=np.float32)
        destination = np.asarray(
            [
                [0.0, 0.0],
                [reference_width_m, 0.0],
                [reference_width_m, reference_length_m],
                [0.0, reference_length_m],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Calibration points produce a degenerate road plane")
        return cls(
            camera_id=camera_id,
            lens_id=lens_id,
            source_width=width,
            source_height=height,
            orientation_deg=orientation_deg,
            crop_id=crop_id,
            mount_height_m=mount_height_m,
            reference_width_m=reference_width_m,
            reference_length_m=reference_length_m,
            image_corners_xy=corners,
            image_to_ground=matrix.tolist(),
            residual_pct=0.0,
        )

    @classmethod
    def load(cls, path: Path) -> CalibrationProfile:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def compatibility(self, width: int, height: int) -> tuple[bool, str | None]:
        if self.residual_pct > 5:
            return False, "calibration residual exceeds 5%"
        source_ratio = self.source_width / self.source_height
        ratio = width / height
        if abs(source_ratio - ratio) / source_ratio > 0.01:
            return False, "video aspect ratio/crop does not match calibration"
        return True, None

    def project(self, point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
        scale_x = self.source_width / width
        scale_y = self.source_height / height
        adjusted = np.asarray([[[point[0] * scale_x, point[1] * scale_y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(
            adjusted,
            np.asarray(self.image_to_ground, dtype=np.float64),
        )[0, 0]
        return float(projected[0]), float(projected[1])


class GeometryEstimator:
    def __init__(self, calibration: CalibrationProfile | None, history_size: int = 8) -> None:
        self.calibration = calibration
        self.history: dict[int, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(
        self,
        detections: list[Detection],
        video_time_ms: int,
        frame_size: tuple[int, int],
        drivable_mask: np.ndarray | None,
    ) -> list[Detection]:
        width, height = frame_size
        compatible = (
            self.calibration.compatibility(width, height)
            if self.calibration
            else (False, None)
        )
        for detection in detections:
            contact_x, contact_y = detection.bbox.bottom_center
            ix = min(max(round(contact_x), 0), width - 1)
            iy = min(max(round(contact_y), 0), height - 1)
            detection.in_path = bool(drivable_mask is not None and drivable_mask[iy, ix] > 0)
            if self.calibration is None:
                detection.distance_null_reason = NullReason.NOT_CALIBRATED
                detection.ttc_null_reason = NullReason.NOT_CALIBRATED
                continue
            if not compatible[0]:
                detection.distance_null_reason = NullReason.INCOMPATIBLE_CALIBRATION
                detection.ttc_null_reason = NullReason.INCOMPATIBLE_CALIBRATION
                continue
            if not detection.in_path:
                detection.distance_null_reason = NullReason.LOW_CONFIDENCE
                detection.ttc_null_reason = NullReason.LOW_CONFIDENCE
                continue
            _lateral, longitudinal = self.calibration.project((contact_x, contact_y), width, height)
            if not np.isfinite(longitudinal) or longitudinal < 0:
                detection.distance_null_reason = NullReason.UNSTABLE_ESTIMATE
                detection.ttc_null_reason = NullReason.UNSTABLE_ESTIMATE
                continue
            detection.distance_m = longitudinal
            if detection.track_id is None:
                detection.ttc_null_reason = NullReason.UNSTABLE_ESTIMATE
                continue
            samples = self.history[detection.track_id]
            samples.append((video_time_ms, longitudinal))
            if len(samples) < 3 or samples[-1][0] == samples[0][0]:
                detection.ttc_null_reason = NullReason.UNSTABLE_ESTIMATE
                continue
            elapsed = (samples[-1][0] - samples[0][0]) / 1000
            relative_speed = (samples[-1][1] - samples[0][1]) / elapsed
            detection.relative_speed_mps = relative_speed
            if relative_speed >= -0.2:
                detection.ttc_null_reason = NullReason.NOT_CLOSING
                continue
            detection.ttc_s = longitudinal / -relative_speed
        return detections
