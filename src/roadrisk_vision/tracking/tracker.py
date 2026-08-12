"""ByteTrack integration and a deterministic weight-free CI tracker."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Local two-stage ByteTrack association for the supported road classes.

    High-confidence observations are associated first; unmatched tracks then
    get a second chance against low-confidence observations. This preserves
    ByteTrack's central recovery behavior without its optional C++ extension.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou: float = 0.3,
    ) -> None:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError("ByteTrack requires the inference extra") from exc
        self._assign = linear_sum_assignment
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_iou = match_iou
        self.max_age = max(frame_rate, 1)
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def _associate(
        self,
        track_ids: list[int],
        detections: list[Detection],
        frame_index: int,
    ) -> tuple[set[int], set[int]]:
        if not track_ids or not detections:
            return set(), set()
        costs = np.ones((len(track_ids), len(detections)), dtype=np.float32)
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for column, detection in enumerate(detections):
                if track.label == detection.label:
                    costs[row, column] = 1 - intersection_over_union(
                        track.bbox, detection.bbox
                    )
        rows, columns = self._assign(costs)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if 1 - float(costs[row, column]) < self.match_iou:
                continue
            track_id = track_ids[row]
            detection = detections[column]
            detection.track_id = track_id
            self._tracks[track_id] = _Track(
                track_id=track_id,
                label=detection.label,
                bbox=detection.bbox,
                last_frame=frame_index,
            )
            matched_tracks.add(track_id)
            matched_detections.add(column)
        return matched_tracks, matched_detections

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        active = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame <= self.max_age
        ]
        high = [item for item in detections if item.confidence >= self.high_threshold]
        low = [
            item
            for item in detections
            if self.low_threshold <= item.confidence < self.high_threshold
        ]
        matched_tracks, matched_high = self._associate(active, high, frame_index)
        remaining_tracks = [item for item in active if item not in matched_tracks]
        self._associate(remaining_tracks, low, frame_index)
        for index, detection in enumerate(high):
            if index in matched_high:
                continue
            track_id = self._next_id
            self._next_id += 1
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
        return [
            detection
            for detection in detections
            if detection.track_id is not None or detection.confidence >= self.high_threshold
        ]
