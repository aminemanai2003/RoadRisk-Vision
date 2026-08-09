"""ByteTrack integration and a deterministic weight-free CI tracker."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from roadrisk_vision.schemas import BoundingBox, Detection


def intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    track_id: int
    label: str
    bbox: BoundingBox
    last_frame: int


class IoUTracker:
    """Greedy deterministic tracker for tests and mock-backed demonstrations."""

    def __init__(self, match_iou: float = 0.3, max_age: int = 30) -> None:
        self.match_iou = match_iou
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        unmatched = set(self._tracks)
        for detection in detections:
            candidates = [
                (intersection_over_union(detection.bbox, track.bbox), track_id)
                for track_id, track in self._tracks.items()
                if track_id in unmatched and track.label == detection.label
            ]
            score, track_id = max(candidates, default=(0.0, -1))
            if score < self.match_iou:
                track_id = self._next_id
                self._next_id += 1
            else:
                unmatched.discard(track_id)
            detection.track_id = track_id
            self._tracks[track_id] = _Track(
                track_id=track_id,
                label=detection.label,
                bbox=detection.bbox,
                last_frame=frame_index,
            )
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return detections


class ByteTrackAdapter:
    """Adapter for the BYTETracker shipped with the official YOLOX package."""

    def __init__(self, frame_rate: int = 30) -> None:
        try:
            from yolox.tracker.byte_tracker import BYTETracker
        except ImportError as exc:
            raise RuntimeError("BYTETracker requires the official YOLOX package") from exc
        args = SimpleNamespace(track_thresh=0.35, track_buffer=30, match_thresh=0.8, mot20=False)
        self._tracker: Any = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        del frame_index
        if not detections:
            self._tracker.update(np.empty((0, 5), dtype=np.float32), (1, 1), (1, 1))
            return detections
        width = max(detection.bbox.x2 for detection in detections)
        height = max(detection.bbox.y2 for detection in detections)
        rows = np.asarray(
            [
                [
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                    detection.confidence,
                ]
                for detection in detections
            ],
            dtype=np.float32,
        )
        online = self._tracker.update(rows, (height, width), (height, width))
        for track in online:
            box = BoundingBox(
                x1=track.tlbr[0],
                y1=track.tlbr[1],
                x2=track.tlbr[2],
                y2=track.tlbr[3],
            )
            candidate = max(
                detections,
                key=lambda detection: intersection_over_union(box, detection.bbox),
            )
            if intersection_over_union(box, candidate.bbox) >= 0.3:
                candidate.track_id = int(track.track_id)
        return detections
