from __future__ import annotations

import json
from pathlib import Path

import pytest

from roadrisk_vision.analytics import build_exposure_report, write_exposure_report
from roadrisk_vision.schemas import (
    EventType,
    RiskEvent,
    Severity,
    TripSummary,
    WarningState,
)


def write_run(
    path: Path,
    *,
    run_id: str,
    duration_s: float,
    distance_km: float | None,
    eligible: bool,
    events: list[RiskEvent],
) -> None:
    path.mkdir()
    summary = TripSummary(
        run_id=run_id,
        duration_s=duration_s,
        valid_distance_km=distance_km,
        gps_coverage_ratio=1 if eligible else 0,
        event_counts={},
        risk_events_per_hour=0,
        risk_events_per_100_km=0 if eligible else None,
    )
    (path / "trip_summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    (path / "events.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in events),
        encoding="utf-8",
    )


def event(run_id: str, event_id: str, ttc: float | None) -> RiskEvent:
    return RiskEvent(
        event_id=event_id,
        run_id=run_id,
        event_type=EventType.COLLISION_RISK,
        warning_state=WarningState.BRAKE_RISK,
        severity=Severity.WARNING,
        start_time_ms=0,
        end_time_ms=500,
        min_ttc_s=ttc,
    )


def test_exposure_report_uses_correct_denominators_and_no_claims(tmp_path) -> None:
    first = tmp_path / "run-one"
    second = tmp_path / "run-two"
    write_run(
        first,
        run_id="one",
        duration_s=3600,
        distance_km=10,
        eligible=True,
        events=[event("one", "a", 1.5), event("one", "b", None)],
    )
    write_run(
        second,
        run_id="two",
        duration_s=1800,
        distance_km=None,
        eligible=False,
        events=[event("two", "c", 3.0)],
    )
    report = build_exposure_report([first, second], low_ttc_definition_s=2)
    assert report.exposure_hours == pytest.approx(1.5)
    assert report.risk_event_count == 3
    assert report.risk_events_per_100_hours == pytest.approx(200)
    assert report.risk_events_per_100_km == pytest.approx(20)
    assert report.low_ttc_event_count == 1
    assert report.claims_modeled is False
    assert report.premiums_modeled is False


def test_report_writes_flat_csv_json_and_plain_language_markdown(tmp_path) -> None:
    run = tmp_path / "run"
    write_run(
        run,
        run_id="one",
        duration_s=3600,
        distance_km=None,
        eligible=False,
        events=[event("one", "a", 1.5)],
    )
    report = build_exposure_report([run])
    outputs = write_exposure_report(report, tmp_path / "report")
    assert set(outputs) == {"json", "csv", "markdown"}
    assert json.loads(outputs["json"].read_text(encoding="utf-8"))["claims_modeled"] is False
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "does not model" in markdown
    assert "near-miss" in markdown
