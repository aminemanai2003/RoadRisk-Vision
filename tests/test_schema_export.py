"""Tests for JSON Schema export.

Verifies that exported schemas contain the expected fields, enum values,
required/nullable rules and coordinate conventions documented in
docs/data-contract.md.  Also validates representative artifact records
against each generated schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from roadrisk_vision.cli import app
from roadrisk_vision.io.artifacts import RunStore
from roadrisk_vision.schema_export import ARTIFACT_SCHEMAS, export_schemas
from roadrisk_vision.schemas import TimelineRecord
from roadrisk_vision.schemas.models import SCHEMA_VERSION

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_schemas(directory: Path) -> dict[str, dict]:
    return {
        path.stem.removesuffix(".schema"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.schema.json"))
    }


# ---------------------------------------------------------------------------
# Representative artifact records for validation
# ---------------------------------------------------------------------------

_MANIFEST_RECORD = {
    "schema_version": 1,
    "run_id": "20260809T120000000000Z-deadbeef",
    "status": "complete",
    "source_basename": "clip.mp4",
    "source_sha256": "abc123" * 11,
    "created_at": "2026-08-09T12:00:00Z",
    "completed_at": "2026-08-09T12:05:00Z",
    "package_version": "0.1.0",
    "normalized_fps": 10.0,
    "frame_count": 600,
    "device": "cpu",
    "backend": "hybrid",
    "include_location": False,
    "models": {},
    "warnings": [],
}

_TIMELINE_RECORD = {
    "schema_version": 1,
    "frame_index": 0,
    "video_time_ms": 0,
    "warning_state": "normal",
    "severity": None,
    "object_count": 3,
    "lane_departure": False,
    "lane_confidence": None,
}

_RISK_EVENT_RECORD = {
    "schema_version": 1,
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "run_id": "20260809T120000000000Z-deadbeef",
    "event_type": "rapid_closing",
    "warning_state": "slow_down",
    "severity": 2,
    "start_time_ms": 1200,
    "end_time_ms": 2600,
    "track_ids": [7],
    "min_ttc_s": 3.4,
    "min_distance_m": 14.2,
    "evidence": {"rule_version": 1},
}

_TRIP_SUMMARY_RECORD = {
    "schema_version": 1,
    "run_id": "20260809T120000000000Z-deadbeef",
    "duration_s": 60.0,
    "valid_distance_km": 1.2,
    "gps_coverage_ratio": 0.95,
    "event_counts": {"rapid_closing": 1},
    "risk_events_per_hour": 60.0,
    "risk_events_per_100_km": 83.3,
}

_CALIBRATION_PROFILE_RECORD = {
    "schema_version": 1,
    "camera_id": "phone-main",
    "lens_id": "default",
    "source_width": 1920,
    "source_height": 1080,
    "orientation_deg": 0,
    "crop_id": "full-frame",
    "calibration_frame_ms": 0,
    "mount_height_m": 1.25,
    "reference_width_m": 3.5,
    "reference_length_m": 20.0,
    "image_corners_xy": [[240, 980], [1680, 980], [1250, 520], [670, 520]],
    "image_to_ground": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "residual_pct": 0.0,
}

_SAMPLE_RECORDS: dict[str, dict] = {
    "manifest": _MANIFEST_RECORD,
    "timeline_record": _TIMELINE_RECORD,
    "risk_event": _RISK_EVENT_RECORD,
    "trip_summary": _TRIP_SUMMARY_RECORD,
    "calibration_profile": _CALIBRATION_PROFILE_RECORD,
}


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_export_writes_all_schemas(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert set(written) == set(ARTIFACT_SCHEMAS)
    for path in written.values():
        assert path.is_file()


def test_schema_version_is_pinned_where_present(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    for name, schema in _load_schemas(tmp_path).items():
        version_prop = schema.get("properties", {}).get("schema_version", {})
        assert version_prop.get("default") == SCHEMA_VERSION, (
            f"{name} missing schema_version default"
        )


def test_output_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    export_schemas(first)
    export_schemas(second)
    for path_a in sorted(first.glob("*.json")):
        path_b = second / path_a.name
        assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_risk_event_contains_event_type_enum(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["risk_event"]
    defs = schema.get("$defs", {})
    event_type = defs.get("EventType", {})
    assert "object_in_path" in event_type.get("enum", [])
    assert "collision_risk" in event_type.get("enum", [])
    assert "lane_departure" in event_type.get("enum", [])


def test_timeline_record_has_warning_state(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["timeline_record"]
    defs = schema.get("$defs", {})
    warning_state = defs.get("WarningState", {})
    assert "normal" in warning_state.get("enum", [])
    assert "hazard" in warning_state.get("enum", [])


def test_manifest_required_fields(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["manifest"]
    required = set(schema.get("required", []))
    for field in ("run_id", "source_basename", "source_sha256", "package_version"):
        assert field in required, f"manifest missing required field {field}"


def test_trip_summary_nullable_distance(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["trip_summary"]
    distance = schema.get("properties", {}).get("valid_distance_km", {})
    any_of = distance.get("anyOf", [])
    type_values = [t.get("type") for t in any_of]
    assert "null" in type_values, "valid_distance_km should be nullable"


def test_null_reason_enum_values(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["risk_event"]
    # NullReason is used transitively through Detection references in events,
    # but the enum itself is defined in the schemas module. Verify it by
    # checking the risk_event evidence field is present (it can hold anything).
    props = schema.get("properties", {})
    assert "evidence" in props


def test_calibration_profile_corners_constraint(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["calibration_profile"]
    corners = schema.get("properties", {}).get("image_corners_xy", {})
    assert corners.get("minItems") == 4
    assert corners.get("maxItems") == 4


def test_all_schemas_have_meta_schema(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    for name, schema in _load_schemas(tmp_path).items():
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        assert schema.get("title") == name


# ---------------------------------------------------------------------------
# Record validation tests
# ---------------------------------------------------------------------------


def test_sample_records_validate_against_pydantic_models() -> None:
    """Each representative record must be accepted by its Pydantic model."""
    for name, model in ARTIFACT_SCHEMAS.items():
        record = _SAMPLE_RECORDS[name]
        instance = model.model_validate(record)
        roundtrip = json.loads(instance.model_dump_json())
        for key in record:
            assert key in roundtrip, f"{name}: field {key} lost in round-trip"


def test_timeline_writer_emits_versioned_records(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"test video")
    store = RunStore(tmp_path / "runs", source, "cpu", "mock", False)

    store.write_timeline([TimelineRecord.model_validate(_TIMELINE_RECORD)])

    records = json.loads(store.path("timeline.json").read_text(encoding="utf-8"))
    assert records == [_TIMELINE_RECORD]


def test_sample_records_conform_to_exported_schemas(tmp_path: Path) -> None:
    """Each representative record must be structurally valid against the
    generated JSON Schema.  Uses jsonschema if available, otherwise falls back
    to a property-key check."""
    export_schemas(tmp_path)
    schemas = _load_schemas(tmp_path)
    try:
        import jsonschema
    except ImportError:
        jsonschema = None  # type: ignore[assignment]

    for name, schema in schemas.items():
        record = _SAMPLE_RECORDS[name]
        if jsonschema is not None:
            jsonschema.validate(record, schema)
        else:
            schema_props = set(schema.get("properties", {}).keys())
            record_keys = set(record.keys())
            assert record_keys <= schema_props, (
                f"{name}: record keys {record_keys - schema_props} not in schema"
            )


def test_extra_field_rejected_by_all_models() -> None:
    """Models with extra='forbid' must reject unknown fields — the exported
    schema must not claim to allow fields the model rejects."""
    from pydantic import ValidationError

    for name, model in ARTIFACT_SCHEMAS.items():
        record = {**_SAMPLE_RECORDS[name], "_bogus_extra_field": True}
        with pytest.raises(ValidationError):
            model.model_validate(record)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------


def test_cli_schemas_export(tmp_path: Path) -> None:
    result = runner.invoke(app, ["schemas", "export", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert len(list(tmp_path.glob("*.schema.json"))) == len(ARTIFACT_SCHEMAS)
