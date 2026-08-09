"""Sequential HybridNets + YOLOX perception for bounded GPU memory."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from roadrisk_vision.config import ModelConfig
from roadrisk_vision.perception.hybridnets import HybridNetsBackend
from roadrisk_vision.perception.yolox import YoloxBackend
from roadrisk_vision.schemas import FramePacket, FramePerception


class CompositeBackend:
    def __init__(self) -> None:
        self.road = HybridNetsBackend()
        self.objects = YoloxBackend()

    def open(self, config: ModelConfig, device: str) -> Self:
        self.road.open(config, device)
        try:
            self.objects.open(config, device)
        except Exception:
            self.road.close()
            raise
        return self

    def infer(self, frame: FramePacket) -> FramePerception:
        road = self.road.infer(frame)
        objects = self.objects.infer(frame)
        return FramePerception(
            frame_index=frame.frame_index,
            video_time_ms=frame.video_time_ms,
            detections=objects.detections,
            drivable_mask=road.drivable_mask,
            lane_mask=road.lane_mask,
        )

    def metadata(self) -> dict[str, Any]:
        return {"road": self.road.metadata(), "objects": self.objects.metadata()}

    def close(self) -> None:
        self.objects.close()
        self.road.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
