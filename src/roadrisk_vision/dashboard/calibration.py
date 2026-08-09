"""Pure calibration-frame helpers used by the Streamlit point selector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CORNER_LABELS = ("near-left", "near-right", "far-right", "far-left")


def read_video_frame(path: Path, frame_ms: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, frame_ms)
        success, frame = capture.read()
    finally:
        capture.release()
    if not success:
        raise ValueError(f"Could not decode a frame at {frame_ms} ms")
    return frame


def scale_display_click(
    x: float,
    y: float,
    *,
    original_width: int,
    display_width: int,
) -> tuple[float, float]:
    scale = original_width / display_width
    return x * scale, y * scale


def validate_corner_order(
    corners: list[tuple[float, float]],
    frame_size: tuple[int, int],
) -> tuple[bool, str]:
    if len(corners) != 4:
        return False, "Select four points in the displayed order."
    width, height = frame_size
    if any(x < 0 or x >= width or y < 0 or y >= height for x, y in corners):
        return False, "Every point must be inside the frame."
    near_left, near_right, far_right, far_left = corners
    if near_left[0] >= near_right[0] or far_left[0] >= far_right[0]:
        return False, "Left points must remain left of their matching right points."
    near_y = (near_left[1] + near_right[1]) / 2
    far_y = (far_left[1] + far_right[1]) / 2
    if near_y <= far_y:
        return False, "Near points must appear below far points in the image."
    contour = np.asarray(corners, dtype=np.float32)
    if not cv2.isContourConvex(contour.astype(np.int32)):
        return False, "The selected road rectangle must be convex and non-crossing."
    area_ratio = abs(cv2.contourArea(contour)) / (width * height)
    if area_ratio < 0.005:
        return False, "The selected road rectangle is too small for stable geometry."
    return True, "Reference geometry is valid. Exact four-point fit residual: 0%."


def annotate_calibration_frame(
    frame: np.ndarray,
    corners: list[tuple[float, float]],
) -> np.ndarray:
    output = frame.copy()
    if len(corners) > 1:
        points = np.asarray(corners, dtype=np.int32)
        cv2.polylines(output, [points], len(corners) == 4, (0, 220, 255), 3)
    for index, (x, y) in enumerate(corners):
        point = (round(x), round(y))
        cv2.circle(output, point, 8, (30, 30, 230), -1)
        cv2.putText(
            output,
            f"{index + 1} {CORNER_LABELS[index]}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output
