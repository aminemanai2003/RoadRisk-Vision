from __future__ import annotations

import numpy as np

from roadrisk_vision.dashboard import (
    annotate_calibration_frame,
    scale_display_click,
    validate_corner_order,
)


def test_display_click_scales_to_source_resolution() -> None:
    assert scale_display_click(250, 100, original_width=2000, display_width=1000) == (500, 200)


def test_corner_order_accepts_convex_road_reference() -> None:
    corners = [(100, 900), (900, 900), (700, 300), (300, 300)]
    valid, message = validate_corner_order(corners, (1000, 1000))
    assert valid is True
    assert "residual" in message


def test_corner_order_rejects_crossed_or_reversed_points() -> None:
    valid, message = validate_corner_order(
        [(900, 900), (100, 900), (700, 300), (300, 300)],
        (1000, 1000),
    )
    assert valid is False
    assert "Left points" in message


def test_annotation_does_not_modify_source_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    output = annotate_calibration_frame(frame, [(10, 90), (90, 90)])
    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(output) > 0
