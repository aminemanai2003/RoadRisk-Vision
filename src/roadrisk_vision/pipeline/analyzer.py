"""Bounded-memory phone-video to validated run-artifacts pipeline."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roadrisk_vision.config import AppConfig
from roadrisk_vision.geometry import (
    CalibrationProfile,
    GeometryEstimator,
    estimate_lane_departure,
)
from roadrisk_vision.io import FrameStream, RunStore, load_telemetry, normalize_video
from roadrisk_vision.perception import CompositeBackend, MockBackend, PerceptionBackend
from roadrisk_vision.rendering import FFmpegVideoWriter, render_frame
from roadrisk_vision.risk import RiskEngine
from roadrisk_vision.schemas import TripSummary
from roadrisk_vision.tracking import ByteTrackAdapter, IoUTracker

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisOptions:
    video: Path
    output: Path = Path("runs")
    telemetry: Path | None = None
    telemetry_offset_ms: int = 0
    calibration: Path | None = None
    config: Path | None = None
    device: str | None = None
    backend: str | None = None
    include_location: bool = False


def _summary(
    run_id: str,
    duration_ms: int,
    events: list[Any],
    telemetry: Any,
) -> TripSummary:
    duration_s = duration_ms / 1000
    counts = Counter(event.event_type.value for event in events)
    rate_hour = len(events) * 3600 / duration_s if duration_s else 0.0
    coverage = telemetry.coverage(duration_ms) if telemetry else 0.0
    distance_km = telemetry.distance_km() if telemetry else None
    rate_distance = None
    if distance_km is not None and distance_km >= 1 and coverage >= 0.8:
        rate_distance = len(events) * 100 / distance_km
    return TripSummary(
        run_id=run_id,
        duration_s=duration_s,
        valid_distance_km=distance_km,
        gps_coverage_ratio=coverage,
        event_counts=dict(counts),
        risk_events_per_hour=rate_hour,
        risk_events_per_100_km=rate_distance,
    )


def analyze_video(options: AnalysisOptions) -> Path:
    config = AppConfig.from_yaml(options.config)
    config.output_root = options.output
    if options.device is not None:
        config.device = options.device  # type: ignore[assignment]
    if options.backend is not None:
        config.backend = options.backend  # type: ignore[assignment]
    config.include_location = options.include_location
    store = RunStore(
        config.output_root,
        options.video,
        config.device,
        config.backend,
        config.include_location,
    )
    try:
        normalized_info = normalize_video(
            options.video,
            store.path("normalized.mp4"),
            config.normalized_fps_cap,
        )
        telemetry = (
            load_telemetry(options.telemetry, options.telemetry_offset_ms)
            if options.telemetry
            else None
        )
        calibration = CalibrationProfile.load(options.calibration) if options.calibration else None
        backend: PerceptionBackend
        tracker: Any
        if config.backend == "mock":
            backend = MockBackend()
            tracker = IoUTracker()
        else:
            backend = CompositeBackend()
            tracker = ByteTrackAdapter(frame_rate=round(normalized_info.fps))
        geometry = GeometryEstimator(calibration)
        risk_engine = RiskEngine(store.run_id, config.risk)
        timeline: list[dict[str, Any]] = []
        frame_count = 0
        with (
            backend.open(config.models, config.device),
            FrameStream(store.path("normalized.mp4")) as frames,
            FFmpegVideoWriter(store.path("annotated.mp4"), frames.size, frames.fps) as writer,
        ):
            for packet in frames:
                perception = backend.infer(packet)
                perception.detections = tracker.update(perception.detections, packet.frame_index)
                perception.detections = geometry.update(
                    perception.detections,
                    packet.video_time_ms,
                    frames.size,
                    perception.drivable_mask,
                )
                lane_evidence = estimate_lane_departure(
                    perception.lane_mask,
                    boundary_margin_ratio=config.risk.lane_departure_margin,
                )
                risk = risk_engine.evaluate(
                    perception.detections,
                    packet.video_time_ms,
                    lane_evidence.as_dict() if lane_evidence else None,
                )
                writer.write(render_frame(packet.image, perception, risk))
                timeline.append(
                    {
                        "frame_index": packet.frame_index,
                        "video_time_ms": packet.video_time_ms,
                        "warning_state": risk.warning_state.value,
                        "severity": int(risk.severity) if risk.severity else None,
                        "object_count": len(perception.detections),
                        "lane_departure": lane_evidence is not None,
                        "lane_confidence": lane_evidence.confidence if lane_evidence else None,
                    }
                )
                frame_count += 1
        events = risk_engine.finish()
        if config.include_location and telemetry is not None:
            for event in events:
                point = telemetry.at(event.start_time_ms)
                if point is not None and point.latitude is not None and point.longitude is not None:
                    event.evidence["location"] = {
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                    }
        duration_ms = timeline[-1]["video_time_ms"] if timeline else 0
        store.write_events(events)
        store.write_timeline(timeline)
        store.write_summary(_summary(store.run_id, duration_ms, events, telemetry))
        return store.complete(frame_count, normalized_info.fps, backend.metadata())
    except KeyboardInterrupt:
        store.fail("CANCELLED_BY_USER", "Analysis cancelled by user", cancelled=True)
        raise
    except Exception as exc:
        code = getattr(exc, "code", exc.__class__.__name__.upper())
        LOGGER.exception("Analysis failed with %s", code)
        failed_path = store.fail(str(code), str(exc))
        raise RuntimeError(f"Analysis failed; diagnostic artifacts: {failed_path}") from exc
