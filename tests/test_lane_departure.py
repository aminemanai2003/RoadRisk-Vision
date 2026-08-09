from __future__ import annotations

import numpy as np

from roadrisk_vision.config import RiskConfig
from roadrisk_vision.geometry import estimate_lane_departure
from roadrisk_vision.risk import RiskEngine
from roadrisk_vision.schemas import EventType, WarningState


def lane_mask(left_x: int, right_x: int) -> np.ndarray:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[55:, left_x] = 1
    mask[55:, right_x] = 1
    return mask


def test_centered_lane_does_not_trigger_departure() -> None:
    assert estimate_lane_departure(lane_mask(55, 145)) is None


def test_low_confidence_lane_does_not_trigger_departure() -> None:
    mask = lane_mask(90, 190)
    mask[60:] = 0
    assert estimate_lane_departure(mask) is None


def test_boundary_clearance_produces_directional_evidence() -> None:
    evidence = estimate_lane_departure(lane_mask(90, 190))
    assert evidence is not None
    assert evidence.direction == "left"
    assert evidence.ego_clearance_ratio < 0.12
    assert evidence.confidence == 1


def test_lane_departure_requires_temporal_event_duration() -> None:
    evidence = estimate_lane_departure(lane_mask(90, 190))
    assert evidence is not None
    engine = RiskEngine("lane-run", RiskConfig(min_event_duration_ms=350))
    first = engine.evaluate([], 0, evidence.as_dict())
    assert first.warning_state == WarningState.LANE_DEPARTURE
    engine.evaluate([], 400, evidence.as_dict())
    engine.evaluate([], 500)
    events = engine.finish()
    assert len(events) == 1
    assert events[0].event_type == EventType.LANE_DEPARTURE
    assert events[0].evidence["direction"] == "left"
