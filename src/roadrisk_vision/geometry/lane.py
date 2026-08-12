"""Evidence-based ego-lane position estimation from a lane segmentation mask."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LaneDepartureEvidence:
    direction: str
    confidence: float
    ego_clearance_ratio: float
    lateral_offset_ratio: float
    left_boundary_x: float
    right_boundary_x: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def estimate_lane_departure(
    lane_mask: np.ndarray | None,
    *,
    boundary_margin_ratio: float = 0.12,
    roi_top_ratio: float = 0.55,
    minimum_confidence: float = 0.35,
) -> LaneDepartureEvidence | None:
    """Return evidence only when the camera/ego reference nears or crosses a lane boundary."""
    if lane_mask is None or lane_mask.ndim != 2 or not np.any(lane_mask):
        return None
    height, width = lane_mask.shape
    center_x = width / 2
    start_y = round(height * roi_top_ratio)
    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    for y in range(start_y, height):
        xs = np.flatnonzero(lane_mask[y] > 0)
        left = xs[xs < center_x]
        right = xs[xs > center_x]
        if left.size and right.size:
            left_points.append((float(y), float(left.max())))
            right_points.append((float(y), float(right.min())))
    available_rows = max(1, height - start_y)
    confidence = min(len(left_points), len(right_points)) / available_rows
    if confidence < minimum_confidence:
        return None
    left_array = np.asarray(left_points, dtype=np.float64)
    right_array = np.asarray(right_points, dtype=np.float64)
    left_fit = np.polyfit(left_array[:, 0], left_array[:, 1], deg=1)
    right_fit = np.polyfit(right_array[:, 0], right_array[:, 1], deg=1)
    evaluation_y = height - 1
    left_x = float(np.polyval(left_fit, evaluation_y))
    right_x = float(np.polyval(right_fit, evaluation_y))
    lane_width = right_x - left_x
    if lane_width < width * 0.15 or lane_width > width * 0.95:
        return None
    lane_center = (left_x + right_x) / 2
    lateral_offset = (center_x - lane_center) / (lane_width / 2)
    left_clearance = (center_x - left_x) / lane_width
    right_clearance = (right_x - center_x) / lane_width
    clearance = min(left_clearance, right_clearance)
    if clearance >= boundary_margin_ratio:
        return None
    direction = "left" if left_clearance <= right_clearance else "right"
    return LaneDepartureEvidence(
        direction=direction,
        confidence=confidence,
        ego_clearance_ratio=clearance,
        lateral_offset_ratio=lateral_offset,
        left_boundary_x=left_x,
        right_boundary_x=right_x,
    )
