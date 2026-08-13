"""Export Pydantic artifact schemas as JSON Schema files."""

from __future__ import annotations

import json
from pathlib import Path

from roadrisk_vision.geometry.calibration import CalibrationProfile
from roadrisk_vision.schemas.models import (
    SCHEMA_VERSION,
    Detection,
    Manifest,
    RiskEvent,
    TripSummary,
)

ARTIFACT_SCHEMAS: dict[str, type] = {
    "manifest": Manifest,
    "detection": Detection,
    "risk_event": RiskEvent,
    "trip_summary": TripSummary,
    "calibration_profile": CalibrationProfile,
}


def export_schemas(output: Path) -> dict[str, Path]:
    """Write each artifact schema as a stable JSON Schema file.

    Returns a mapping of artifact name to the written path.
    """
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, model in sorted(ARTIFACT_SCHEMAS.items()):
        schema = model.model_json_schema()
        schema["title"] = name
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema.setdefault("properties", {}).setdefault(
            "schema_version", {}
        ).setdefault("const", SCHEMA_VERSION)
        path = output / f"{name}.schema.json"
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written[name] = path
    return written
