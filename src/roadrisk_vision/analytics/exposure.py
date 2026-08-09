"""Aggregate trip artifacts into actuarial-ready telematics exposure features."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from roadrisk_vision.schemas import RiskEvent, TripSummary


class ExposureReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_run_ids: list[str]
    trip_count: int
    distance_eligible_trip_count: int
    exposure_hours: float
    valid_distance_km: float
    risk_event_count: int
    event_counts: dict[str, int]
    severity_counts: dict[str, int]
    risk_events_per_100_hours: float | None
    risk_events_per_100_km: float | None
    low_ttc_definition_s: float
    low_ttc_event_count: int
    low_ttc_events_per_100_hours: float | None
    claims_modeled: bool = False
    premiums_modeled: bool = False


def _load_run(path: Path) -> tuple[TripSummary, list[RiskEvent]]:
    summary_path = path / "trip_summary.json"
    events_path = path / "events.jsonl"
    if not summary_path.is_file() or not events_path.is_file():
        raise ValueError(f"Not a completed RoadRisk run directory: {path}")
    summary = TripSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    events = [
        RiskEvent.model_validate_json(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(event.run_id != summary.run_id for event in events):
        raise ValueError(f"Event run_id mismatch in {path}")
    return summary, events


def build_exposure_report(
    run_directories: list[Path],
    *,
    low_ttc_definition_s: float = 2.0,
) -> ExposureReport:
    if not run_directories:
        raise ValueError("At least one completed run directory is required")
    if low_ttc_definition_s <= 0:
        raise ValueError("Low-TTC definition must be positive")
    summaries: list[TripSummary] = []
    all_events: list[RiskEvent] = []
    eligible_events = 0
    eligible_distance = 0.0
    for path in run_directories:
        summary, events = _load_run(path)
        summaries.append(summary)
        all_events.extend(events)
        if summary.risk_events_per_100_km is not None and summary.valid_distance_km is not None:
            eligible_distance += summary.valid_distance_km
            eligible_events += len(events)
    exposure_hours = sum(summary.duration_s for summary in summaries) / 3600
    event_counts = Counter(event.event_type.value for event in all_events)
    severity_counts = Counter(str(int(event.severity)) for event in all_events)
    low_ttc_count = sum(
        event.min_ttc_s is not None and event.min_ttc_s <= low_ttc_definition_s
        for event in all_events
    )
    return ExposureReport(
        source_run_ids=[summary.run_id for summary in summaries],
        trip_count=len(summaries),
        distance_eligible_trip_count=sum(
            summary.risk_events_per_100_km is not None for summary in summaries
        ),
        exposure_hours=exposure_hours,
        valid_distance_km=sum(summary.valid_distance_km or 0 for summary in summaries),
        risk_event_count=len(all_events),
        event_counts=dict(sorted(event_counts.items())),
        severity_counts=dict(sorted(severity_counts.items())),
        risk_events_per_100_hours=(
            len(all_events) * 100 / exposure_hours if exposure_hours else None
        ),
        risk_events_per_100_km=(
            eligible_events * 100 / eligible_distance if eligible_distance else None
        ),
        low_ttc_definition_s=low_ttc_definition_s,
        low_ttc_event_count=low_ttc_count,
        low_ttc_events_per_100_hours=(
            low_ttc_count * 100 / exposure_hours if exposure_hours else None
        ),
    )


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown(report: ExposureReport) -> str:
    distance_rate = (
        f"{report.risk_events_per_100_km:.2f}"
        if report.risk_events_per_100_km is not None
        else "not eligible"
    )
    return f"""# RoadRisk telematics exposure report

This report describes research risk events and exposure. It does not model
claims, claim severity, premiums, fault, or insurance eligibility.

| Metric | Value |
|---|---:|
| Trips | {report.trip_count} |
| Exposure hours | {report.exposure_hours:.3f} |
| Valid distance (km) | {report.valid_distance_km:.3f} |
| Risk events | {report.risk_event_count} |
| Risk events / 100 hours | {report.risk_events_per_100_hours or 0:.2f} |
| Risk events / 100 km | {distance_rate} |
| Low-TTC threshold | ≤ {report.low_ttc_definition_s:.2f} s |
| Low-TTC events | {report.low_ttc_event_count} |

Per-100-km exposure includes only trips meeting the RoadRisk distance eligibility
rule (≥80% GPS coverage and ≥1 km). A low-TTC event is not automatically a
near-miss or claim.
"""


def write_exposure_report(report: ExposureReport, output_base: Path) -> dict[str, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")
    markdown_path = output_base.with_suffix(".md")
    _atomic_text(json_path, report.model_dump_json(indent=2) + "\n")
    values = report.model_dump(mode="json")
    event_counts = values.pop("event_counts")
    severity_counts = values.pop("severity_counts")
    values["source_run_ids"] = ";".join(values["source_run_ids"])
    values.update({f"event_count_{key}": value for key, value in event_counts.items()})
    values.update({f"severity_count_{key}": value for key, value in severity_counts.items()})
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)
    temporary.replace(csv_path)
    _atomic_text(markdown_path, _markdown(report))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
