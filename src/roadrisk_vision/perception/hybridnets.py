"""Lazy HybridNets D3 segmentation adapter around the preserved upstream code."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Self

import cv2
import numpy as np

from roadrisk_vision.config import ModelConfig
from roadrisk_vision.perception.base import PerceptionError
from roadrisk_vision.schemas import FramePacket, FramePerception


class HybridNetsBackend:
    def __init__(self, project_file: Path = Path("projects/bdd100k.yml")) -> None:
        self.project_file = project_file
        self._model: Any = None
        self._torch: Any = None
        self._params: Any = None
        self._device = "cpu"
        self._weights: Path | None = None

    def open(self, config: ModelConfig, device: str) -> Self:
        weights = config.hybridnets_weights
        if not weights.is_file():
            raise PerceptionError("MISSING_WEIGHTS", f"HybridNets weights not found: {weights}")
        try:
            import torch

            from backbone import HybridNetsBackbone
            from utils.constants import BINARY_MODE, MULTICLASS_MODE
            from utils.utils import Params
        except ImportError as exc:
            raise PerceptionError(
                "MISSING_INFERENCE_DEPENDENCY",
                "Install RoadRisk Vision with the inference extra.",
            ) from exc

        wants_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        resolved = "cuda" if wants_cuda else "cpu"
        checkpoint = torch.load(weights, map_location=resolved, weights_only=False)
        state = checkpoint.get("model", checkpoint)
        params = Params(str(self.project_file))
        seg_channels = state["segmentation_head.0.weight"].size(0)
        seg_mode = BINARY_MODE if seg_channels == 1 else MULTICLASS_MODE
        model = HybridNetsBackbone(
            compound_coef=3,
            num_classes=len(params.obj_list),
            ratios=eval(params.anchors_ratios, {"__builtins__": {}}, {}),
            scales=eval(params.anchors_scales, {"__builtins__": {}}, {}),
            seg_classes=len(params.seg_list),
            backbone_name=None,
            seg_mode=seg_mode,
        )
        model.load_state_dict(state)
        model.requires_grad_(False).eval().to(resolved)
        self._model = model
        self._torch = torch
        self._params = params
        self._device = resolved
        self._weights = weights
        return self

    def infer(self, frame: FramePacket) -> FramePerception:
        if self._model is None or self._torch is None:
            raise RuntimeError("HybridNetsBackend must be opened before inference")
        torch = self._torch
        image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        target_w, target_h = 640, 384
        resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        mean = np.asarray(self._params.mean, dtype=np.float32)
        std = np.asarray(self._params.std, dtype=np.float32)
        array = (array - mean) / std
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        try:
            with torch.inference_mode():
                outputs = self._model(tensor)
                _features, _regression, _classification, _anchors, segmentation = outputs
        except torch.cuda.OutOfMemoryError as exc:
            raise PerceptionError("GPU_OOM", "HybridNets exhausted GPU memory") from exc
        if segmentation.shape[1] == 1:
            mask = (segmentation[0, 0] >= 0).to(torch.uint8).cpu().numpy()
            road = mask
            lane = np.zeros_like(mask)
        else:
            classes = torch.argmax(segmentation, dim=1)[0].cpu().numpy().astype(np.uint8)
            road = (classes == 1).astype(np.uint8)
            lane = (classes == 2).astype(np.uint8)
        road = cv2.resize(road, (width, height), interpolation=cv2.INTER_NEAREST)
        lane = cv2.resize(lane, (width, height), interpolation=cv2.INTER_NEAREST)
        return FramePerception(
            frame_index=frame.frame_index,
            video_time_ms=frame.video_time_ms,
            drivable_mask=road,
            lane_mask=lane,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "hybridnets-d3",
            "weights": self._weights.name if self._weights else None,
            "device": self._device,
            "input_size": [640, 384],
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
