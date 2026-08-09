"""Minimal use of the research exposure-report API over completed local runs."""

from pathlib import Path

from roadrisk_vision.analytics import build_exposure_report, write_exposure_report

RUNS = [Path("runs/trip-one"), Path("runs/trip-two")]

report = build_exposure_report(RUNS, low_ttc_definition_s=2.0)
outputs = write_exposure_report(report, Path("reports/portfolio_exposure"))
for format_name, path in outputs.items():
    print(f"{format_name}: {path}")
