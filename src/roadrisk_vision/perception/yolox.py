"""Lazy adapter for an official YOLOX-S checkout and checkpoint."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from roadrisk_vision.config import ModelConfig
from roadrisk_vision.perception.base import PerceptionError
from roadrisk_vision.schemas import BoundingBox, Detection, FramePacket, FramePerception

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class YoloxBackend:
    def __init__(self) -> None:
        self._model: Any = None
        self._torch: Any = None
        self._preproc: Any = None
        self._postprocess: Any = None
        self._device = "cpu"
        self._config: ModelConfig | None = None

    def open(self, config: ModelConfig, device: str) -> Self:
        weights = config.yolox_weights
        if not weights.is_file():
            raise PerceptionError("MISSING_WEIGHTS", f"YOLOX weights not found: {weights}")
        try:
            import torch

            from roadrisk_vision.perception.yolox_model import (
                build_yolox_s,
                checkpoint_model,
                postprocess,
                preproc,
            )
        except ImportError as exc:
            raise PerceptionError(
                "MISSING_YOLOX",
                "Install RoadRisk Vision with the inference extra.",
            ) from exc
        wants_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        resolved = "cuda" if wants_cuda else "cpu"
        model = build_yolox_s()
        checkpoint = torch.load(weights, map_location=resolved, weights_only=False)
        model.load_state_dict(checkpoint_model(checkpoint))
        model.eval().to(resolved)
        self._model = model
        self._torch = torch
        self._preproc = preproc
        self._postprocess = postprocess
        self._device = resolved
        self._config = config
        return self

    def infer(self, frame: FramePacket) -> FramePerception:
        if self._model is None or self._config is None:
            raise RuntimeError("YoloxBackend must be opened before inference")
        input_size = (self._config.yolox_size[1], self._config.yolox_size[0])
        image, ratio = self._preproc(frame.image, input_size)
        tensor = self._torch.from_numpy(image).unsqueeze(0).float().to(self._device)
        try:
            with self._torch.inference_mode():
                output = self._model(tensor)
                output = self._postprocess(
                    output,
                    80,
                    self._config.confidence,
                    self._config.nms_iou,
                    class_agnostic=True,
                )[0]
        except self._torch.cuda.OutOfMemoryError as exc:
            raise PerceptionError("GPU_OOM", "YOLOX exhausted GPU memory") from exc
        detections: list[Detection] = []
        if output is not None:
            rows = output.detach().cpu().numpy()
            for row in rows:
                class_id = int(row[6])
                label = COCO_NAMES[class_id]
                if label not in self._config.classes:
                    continue
                x1, y1, x2, y2 = (row[:4] / ratio).tolist()
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(row[4] * row[5]),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return FramePerception(
            frame_index=frame.frame_index,
            video_time_ms=frame.video_time_ms,
            detections=detections,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "yolox-s",
            "weights": self._config.yolox_weights.name if self._config else None,
            "device": self._device,
            "input_size": list(self._config.yolox_size) if self._config else None,
        }

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._device == "cuda":
            self._torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
