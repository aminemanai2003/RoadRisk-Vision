from __future__ import annotations

import csv

import pytest

from roadrisk_vision.io.telemetry import TelemetrySeries, load_telemetry
from roadrisk_vision.schemas import TelemetryPoint


def test_csv_sort_duplicate_and_interpolate(tmp_path) -> None:
    path = tmp_path / "telemetry.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_time_ms", "speed_mps"])
        writer.writeheader()
        writer.writerows(
            [
                {"video_time_ms": 2000, "speed_mps": 20},
                {"video_time_ms": 0, "speed_mps": 10},
                {"video_time_ms": 2000, "speed_mps": 30},
            ]
        )
    series = load_telemetry(path)
    assert [point.video_time_ms for point in series.points] == [0, 2000]
    assert series.at(1000).speed_mps == pytest.approx(20)


def test_large_gap_is_not_interpolated() -> None:
    series = TelemetrySeries(
        [
            TelemetryPoint(video_time_ms=0, speed_mps=10),
            TelemetryPoint(video_time_ms=6000, speed_mps=20),
        ]
    )
    assert series.at(3000) is None


def test_distance_uses_only_valid_gps_segments() -> None:
    series = TelemetrySeries(
        [
            TelemetryPoint(video_time_ms=0, latitude=0, longitude=0),
            TelemetryPoint(video_time_ms=1000, latitude=0, longitude=0.001),
        ]
    )
    assert series.distance_km() == pytest.approx(0.111, rel=0.02)
