"""Deterministic risk rules and temporal event aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5

from roadrisk_vision.config import RiskConfig
from roadrisk_vision.schemas import (
    Detection,
    EventType,
    RiskEvent,
    Severity,
    WarningState,
)


@dataclass(frozen=True)
class RiskSignal:
    event_type: EventType
    warning_state: WarningState
    severity: Severity
    track_ids: tuple[int, ...]
    ttc_s: float | None = None
    distance_m: float | None = None


@dataclass(frozen=True)
class RiskFrame:
    video_time_ms: int
    warning_state: WarningState
    severity: Severity | None
    signals: tuple[RiskSignal, ...] = ()


@dataclass
class _OpenEvent:
    signal: RiskSignal
    start_ms: int
    end_ms: int
    min_ttc_s: float | None = None
    min_distance_m: float | None = None


@dataclass
class RiskEngine:
    run_id: str
    config: RiskConfig
    _open: dict[tuple[EventType, tuple[int, ...]], _OpenEvent] = field(default_factory=dict)
    events: list[RiskEvent] = field(default_factory=list)

    def evaluate(self, detections: list[Detection], video_time_ms: int) -> RiskFrame:
        signals: list[RiskSignal] = []
        for detection in detections:
            if not detection.in_path:
                continue
            track_ids = (detection.track_id,) if detection.track_id is not None else ()
            signal = RiskSignal(
                EventType.OBJECT_IN_PATH,
                WarningState.MONITOR,
                Severity.INFORMATIONAL,
                track_ids,
                detection.ttc_s,
                detection.distance_m,
            )
            if detection.ttc_s is not None and detection.ttc_s <= self.config.brake_risk_ttc_s:
                severity = Severity.CRITICAL if detection.ttc_s <= 1.5 else Severity.WARNING
                warning = (
                    WarningState.HAZARD
                    if severity == Severity.CRITICAL
                    else WarningState.BRAKE_RISK
                )
                signal = RiskSignal(
                    EventType.COLLISION_RISK,
                    warning,
                    severity,
                    track_ids,
                    detection.ttc_s,
                    detection.distance_m,
                )
            elif detection.ttc_s is not None and detection.ttc_s <= self.config.slow_down_ttc_s:
                signal = RiskSignal(
                    EventType.RAPID_CLOSING,
                    WarningState.SLOW_DOWN,
                    Severity.CAUTION,
                    track_ids,
                    detection.ttc_s,
                    detection.distance_m,
                )
            signals.append(signal)
        self._advance(signals, video_time_ms)
        if not signals:
            return RiskFrame(video_time_ms, WarningState.NORMAL, None)
        highest = max(signals, key=lambda item: int(item.severity))
        return RiskFrame(video_time_ms, highest.warning_state, highest.severity, tuple(signals))

    def _advance(self, signals: list[RiskSignal], timestamp: int) -> None:
        current = {(signal.event_type, signal.track_ids): signal for signal in signals}
        for key in list(self._open):
            if key not in current:
                self._close(key)
        for key, signal in current.items():
            opened = self._open.get(key)
            if opened is None:
                self._open[key] = _OpenEvent(
                    signal=signal,
                    start_ms=timestamp,
                    end_ms=timestamp,
                    min_ttc_s=signal.ttc_s,
                    min_distance_m=signal.distance_m,
                )
                continue
            opened.end_ms = timestamp
            if signal.ttc_s is not None:
                opened.min_ttc_s = min(opened.min_ttc_s or signal.ttc_s, signal.ttc_s)
            if signal.distance_m is not None:
                opened.min_distance_m = min(
                    opened.min_distance_m or signal.distance_m,
                    signal.distance_m,
                )

    def _close(self, key: tuple[EventType, tuple[int, ...]]) -> None:
        opened = self._open.pop(key)
        if opened.end_ms - opened.start_ms < self.config.min_event_duration_ms:
            return
        identity = (
            f"{self.run_id}/{opened.signal.event_type}/{opened.start_ms}/"
            f"{','.join(map(str, opened.signal.track_ids))}"
        )
        self.events.append(
            RiskEvent(
                event_id=str(uuid5(NAMESPACE_URL, identity)),
                run_id=self.run_id,
                event_type=opened.signal.event_type,
                warning_state=opened.signal.warning_state,
                severity=opened.signal.severity,
                start_time_ms=opened.start_ms,
                end_time_ms=opened.end_ms,
                track_ids=list(opened.signal.track_ids),
                min_ttc_s=opened.min_ttc_s,
                min_distance_m=opened.min_distance_m,
                evidence={"rule_version": 1},
            )
        )

    def finish(self) -> list[RiskEvent]:
        for key in list(self._open):
            self._close(key)
        return sorted(self.events, key=lambda event: (event.start_time_ms, event.event_id))
