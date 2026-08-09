"""Deterministic, weight-free backend used by CI and contributor demos."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import numpy as np

from roadrisk_vision.config import ModelConfig
from roadrisk_vision.schemas import BoundingBox, Detection, FramePacket, FramePerception


class MockBackend:
    def __init__(self) -> None:
        self._opened = False

    def open(self, config: ModelConfig, device: str) -> Self:
        del config, device
        self._opened = True
        return self

    def infer(self, frame: FramePacket) -> FramePerception:
        if not self._opened:
            raise RuntimeError("MockBackend must be opened before inference")
        height, width = frame.image.shape[:2]
        approach = min(frame.frame_index * 0.6, width * 0.08)
        half_width = width * 0.07 + approach / 2
        box_height = height * 0.16 + approach / 2
        center_x = width / 2
        bottom = height * 0.78
        detection = Detection(
            label="car",
            confidence=0.93,
            bbox=BoundingBox(
                x1=center_x - half_width,
                y1=bottom - box_height,
                x2=center_x + half_width,
                y2=bottom,
            ),
        )
        road = np.zeros((height, width), dtype=np.uint8)
        road[:, int(width * 0.28) : int(width * 0.72)] = 1
        lanes = np.zeros_like(road)
        lanes[:, int(width * 0.32)] = 1
        lanes[:, int(width * 0.68)] = 1
        return FramePerception(
            frame_index=frame.frame_index,
            video_time_ms=frame.video_time_ms,
            detections=[detection],
            drivable_mask=road,
            lane_mask=lanes,
        )

    def metadata(self) -> dict[str, Any]:
        return {"name": "mock", "weights": None, "deterministic": True}

    def close(self) -> None:
        self._opened = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
