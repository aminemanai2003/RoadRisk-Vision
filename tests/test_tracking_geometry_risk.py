from __future__ import annotations

import numpy as np

from roadrisk_vision.config import RiskConfig
from roadrisk_vision.geometry import CalibrationProfile, GeometryEstimator
from roadrisk_vision.risk import RiskEngine
from roadrisk_vision.schemas import (
    BoundingBox,
    Detection,
    EventType,
    NullReason,
    WarningState,
)
from roadrisk_vision.tracking import IoUTracker


def detection(x: float = 40, track_id: int | None = None) -> Detection:
    return Detection(
        label="car",
        confidence=0.9,
        bbox=BoundingBox(x1=x, y1=40, x2=x + 20, y2=80),
        track_id=track_id,
    )


def test_iou_tracker_keeps_stable_per_run_id() -> None:
    tracker = IoUTracker()
    first = tracker.update([detection(40)], frame_index=0)[0]
    second = tracker.update([detection(42)], frame_index=1)[0]
    assert first.track_id == second.track_id == 1


def test_uncalibrated_geometry_suppresses_distance_and_ttc() -> None:
    item = detection(track_id=1)
    road = np.ones((100, 100), dtype=np.uint8)
    GeometryEstimator(None).update([item], 0, (100, 100), road)
    assert item.in_path is True
    assert item.distance_m is None
    assert item.distance_null_reason == NullReason.NOT_CALIBRATED
    assert item.ttc_null_reason == NullReason.NOT_CALIBRATED


def test_calibrated_geometry_only_emits_ttc_for_closing_track() -> None:
    profile = CalibrationProfile.create(
        camera_id="phone-main",
        width=100,
        height=100,
        mount_height_m=1.2,
        reference_width_m=4,
        reference_length_m=20,
        corners=[(10, 90), (90, 90), (70, 30), (30, 30)],
    )
    estimator = GeometryEstimator(profile)
    road = np.ones((100, 100), dtype=np.uint8)
    estimates = []
    for timestamp, bottom in ((0, 40), (1000, 50), (2000, 60)):
        item = Detection(
            label="car",
            confidence=0.9,
            bbox=BoundingBox(x1=40, y1=20, x2=60, y2=bottom),
            track_id=1,
        )
        estimator.update([item], timestamp, (100, 100), road)
        estimates.append(item)
    assert estimates[-1].distance_m is not None
    assert estimates[-1].relative_speed_mps is not None


def test_risk_engine_aggregates_a_stable_event_id() -> None:
    config = RiskConfig(min_event_duration_ms=100)
    engine = RiskEngine("run-1", config)
    item = detection(track_id=7)
    item.in_path = True
    assert engine.evaluate([item], 0).warning_state == WarningState.MONITOR
    engine.evaluate([item], 200)
    engine.evaluate([], 300)
    events = engine.finish()
    assert len(events) == 1
    assert events[0].event_type == EventType.OBJECT_IN_PATH
    assert events[0].track_ids == [7]


def test_collision_risk_uses_ttc_evidence() -> None:
    engine = RiskEngine("run-2", RiskConfig(min_event_duration_ms=0))
    item = detection(track_id=3)
    item.in_path = True
    item.ttc_s = 1.2
    risk = engine.evaluate([item], 0)
    assert risk.warning_state == WarningState.HAZARD
