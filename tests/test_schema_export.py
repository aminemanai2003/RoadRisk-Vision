"""Tests for JSON Schema export.

Verifies that exported schemas contain the expected fields, enum values,
required/nullable rules and coordinate conventions documented in
docs/data-contract.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from roadrisk_vision.cli import app
from roadrisk_vision.schema_export import ARTIFACT_SCHEMAS, export_schemas
from roadrisk_vision.schemas.models import SCHEMA_VERSION

runner = CliRunner()


def _load_schemas(directory: Path) -> dict[str, dict]:
    return {
        path.stem.removesuffix(".schema"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.schema.json"))
    }


def test_export_writes_all_schemas(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert set(written) == set(ARTIFACT_SCHEMAS)
    for path in written.values():
        assert path.is_file()


def test_schema_version_is_pinned(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    for name, schema in _load_schemas(tmp_path).items():
        version_prop = schema.get("properties", {}).get("schema_version", {})
        assert version_prop.get("const") == SCHEMA_VERSION, f"{name} missing schema_version const"


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


def test_detection_bbox_fields_present(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["detection"]
    defs = schema.get("$defs", {})
    bbox = defs.get("BoundingBox", {}).get("properties", {})
    for coordinate in ("x1", "y1", "x2", "y2"):
        assert coordinate in bbox, f"BoundingBox missing {coordinate}"


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
    schema = _load_schemas(tmp_path)["detection"]
    defs = schema.get("$defs", {})
    null_reason = defs.get("NullReason", {})
    values = null_reason.get("enum", [])
    for reason in ("not_calibrated", "incompatible_calibration", "not_closing"):
        assert reason in values, f"NullReason missing {reason}"


def test_calibration_profile_corners_constraint(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = _load_schemas(tmp_path)["calibration_profile"]
    corners = schema.get("properties", {}).get("image_corners_xy", {})
    assert corners.get("minItems") == 4
    assert corners.get("maxItems") == 4


def test_cli_schemas_export(tmp_path: Path) -> None:
    result = runner.invoke(app, ["schemas", "export", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert len(list(tmp_path.glob("*.schema.json"))) == len(ARTIFACT_SCHEMAS)


def test_all_schemas_have_meta_schema(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    for name, schema in _load_schemas(tmp_path).items():
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        assert schema.get("title") == name
