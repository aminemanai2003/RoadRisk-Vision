"""Deterministic GPX/CSV ingestion and bounded-gap interpolation."""

from __future__ import annotations

import csv
import math
from bisect import bisect_left
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from roadrisk_vision.schemas import TelemetryPoint


class TelemetryError(ValueError):
    pass


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in {None, ""} else None


def _deduplicate(points: Iterable[TelemetryPoint]) -> list[TelemetryPoint]:
    by_time: dict[int, TelemetryPoint] = {}
    for point in points:
        by_time[point.video_time_ms] = point
    return [by_time[key] for key in sorted(by_time)]


def _load_csv(path: Path, offset_ms: int) -> list[TelemetryPoint]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    if "video_time_ms" not in rows[0] and "timestamp" not in rows[0]:
        raise TelemetryError("CSV requires video_time_ms or an ISO-8601 timestamp column")
    origin: datetime | None = None
    points: list[TelemetryPoint] = []
    for row in rows:
        if row.get("video_time_ms") not in {None, ""}:
            time_ms = int(float(row["video_time_ms"]))
        else:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise TelemetryError("CSV timestamps must include a UTC offset")
            origin = origin or timestamp
            time_ms = round((timestamp - origin).total_seconds() * 1000)
        points.append(
            TelemetryPoint(
                video_time_ms=time_ms + offset_ms,
                latitude=_optional_float(row.get("latitude")),
                longitude=_optional_float(row.get("longitude")),
                speed_mps=_optional_float(row.get("speed_mps")),
                heading_deg=_optional_float(row.get("heading_deg")),
                accuracy_m=_optional_float(row.get("accuracy_m")),
            )
        )
    return _deduplicate(points)


def _load_gpx(path: Path, offset_ms: int) -> list[TelemetryPoint]:
    try:
        import gpxpy
    except ImportError as exc:
        raise TelemetryError("Install the telemetry extra to read GPX files") from exc
    gpx = gpxpy.parse(path.read_text(encoding="utf-8"))
    raw = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
    timed = [point for point in raw if point.time is not None]
    if not timed:
        raise TelemetryError("GPX contains no timed track points")
    origin = timed[0].time
    points = [
        TelemetryPoint(
            video_time_ms=round((point.time - origin).total_seconds() * 1000) + offset_ms,
            latitude=point.latitude,
            longitude=point.longitude,
            speed_mps=point.speed,
        )
        for point in timed
    ]
    return _deduplicate(points)


def load_telemetry(path: Path, offset_ms: int = 0) -> TelemetrySeries:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        points = _load_csv(path, offset_ms)
    elif suffix == ".gpx":
        points = _load_gpx(path, offset_ms)
    else:
        raise TelemetryError("Telemetry must be .csv or .gpx")
    return TelemetrySeries(points)


class TelemetrySeries:
    def __init__(self, points: list[TelemetryPoint], max_gap_ms: int = 5000) -> None:
        self.points = _deduplicate(points)
        self.times = [point.video_time_ms for point in self.points]
        self.max_gap_ms = max_gap_ms

    def at(self, video_time_ms: int) -> TelemetryPoint | None:
        if not self.points:
            return None
        index = bisect_left(self.times, video_time_ms)
        if index < len(self.points) and self.times[index] == video_time_ms:
            return self.points[index]
        if index == 0 or index == len(self.points):
            return None
        left, right = self.points[index - 1], self.points[index]
        gap = right.video_time_ms - left.video_time_ms
        if gap > self.max_gap_ms:
            return None
        fraction = (video_time_ms - left.video_time_ms) / gap

        def interpolate(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else a + (b - a) * fraction

        return TelemetryPoint(
            video_time_ms=video_time_ms,
            latitude=interpolate(left.latitude, right.latitude),
            longitude=interpolate(left.longitude, right.longitude),
            speed_mps=interpolate(left.speed_mps, right.speed_mps),
            heading_deg=interpolate(left.heading_deg, right.heading_deg),
            accuracy_m=interpolate(left.accuracy_m, right.accuracy_m),
        )

    def coverage(self, duration_ms: int, sample_ms: int = 1000) -> float:
        if duration_ms <= 0:
            return 0.0
        samples = range(0, duration_ms + 1, sample_ms)
        total = 0
        valid = 0
        for timestamp in samples:
            total += 1
            if self.at(timestamp) is not None:
                valid += 1
        return valid / total if total else 0.0

    def distance_km(self) -> float | None:
        """Sum valid consecutive GPS segments no wider than the interpolation limit."""
        distance_m = 0.0
        segment_count = 0
        for left, right in zip(self.points, self.points[1:], strict=False):
            if right.video_time_ms - left.video_time_ms > self.max_gap_ms:
                continue
            if None in {left.latitude, left.longitude, right.latitude, right.longitude}:
                continue
            lat1, lat2 = math.radians(left.latitude), math.radians(right.latitude)
            delta_lat = lat2 - lat1
            delta_lon = math.radians(right.longitude - left.longitude)
            value = (
                math.sin(delta_lat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            )
            distance_m += 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(value)))
            segment_count += 1
        return distance_m / 1000 if segment_count else None
