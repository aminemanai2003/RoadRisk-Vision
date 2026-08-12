"""Testable helpers for the local Streamlit dashboard."""

from roadrisk_vision.dashboard.calibration import (
    CORNER_LABELS,
    annotate_calibration_frame,
    read_video_frame,
    scale_display_click,
    validate_corner_order,
)

__all__ = [
    "CORNER_LABELS",
    "annotate_calibration_frame",
    "read_video_frame",
    "scale_display_click",
    "validate_corner_order",
]
