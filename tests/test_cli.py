"""Hermetic CLI command tests.

Every command is exercised through Typer's CliRunner so tests need no GPU,
FFmpeg, model weights, or network access.  Temporary directories and
monkeypatching replace all external state.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from roadrisk_vision.cli import app

runner = CliRunner()


# ── doctor ──────────────────────────────────────────────────────────────


def test_doctor_json_emits_valid_json() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert "version" in report
    assert "tools" in report


def test_doctor_human_output_mentions_python() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output


# ── inspect ─────────────────────────────────────────────────────────────


def test_inspect_prints_manifest_and_summary(tmp_path: Path) -> None:
    manifest = {"run_id": "test-run", "schema_version": 1}
    summary = {"total_frames": 42}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "trip_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = runner.invoke(app, ["inspect", str(tmp_path)])
    assert result.exit_code == 0
    assert "test-run" in result.output
    assert "42" in result.output


def test_inspect_empty_directory_succeeds(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path)])
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_inspect_nonexistent_directory_fails() -> None:
    result = runner.invoke(app, ["inspect", "/no/such/path"])
    assert result.exit_code != 0


# ── calibrate ───────────────────────────────────────────────────────────


def test_calibrate_missing_video_fails() -> None:
    result = runner.invoke(app, [
        "calibrate", "/no/such/video.mp4",
        "--frame-ms", "0",
        "--camera-id", "test",
        "--height-m", "1.2",
        "--reference-length-m", "20",
        "--reference-width-m", "3.5",
        "--corners", "10,90;90,90;70,30;30,30",
    ])
    assert result.exit_code != 0


def test_calibrate_invalid_corners_format_fails(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    result = runner.invoke(app, [
        "calibrate", str(video),
        "--frame-ms", "0",
        "--camera-id", "test",
        "--height-m", "1.2",
        "--reference-length-m", "20",
        "--reference-width-m", "3.5",
        "--corners", "not;valid;corners",
    ])
    assert result.exit_code != 0


def test_calibrate_wrong_corner_count_fails(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    result = runner.invoke(app, [
        "calibrate", str(video),
        "--frame-ms", "0",
        "--camera-id", "test",
        "--height-m", "1.2",
        "--reference-length-m", "20",
        "--reference-width-m", "3.5",
        "--corners", "10,90;90,90;70,30",
    ])
    assert result.exit_code != 0


# ── models download ────────────────────────────────────────────────────


def test_models_download_dry_run_lists_without_fetching(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "schema_version": 1,
        "models": [{
            "name": "Fake Model",
            "filename": "fake.pth",
            "source_repository": "https://example.com",
            "source_revision": "abc123",
            "release": "v0",
            "url": "https://example.com/fake.pth",
            "size_bytes": 100,
            "sha256": "0" * 64,
            "input_size": [640, 640],
        }],
    }), encoding="utf-8")
    result = runner.invoke(app, ["models", "download", "--dry-run", "--lock-file", str(lock)])
    assert result.exit_code == 0
    assert "Fake Model" in result.output
    assert not (tmp_path / "models" / "fake.pth").exists()


# ── models verify ───────────────────────────────────────────────────────


def test_models_verify_missing_weights_fails(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "schema_version": 1,
        "models": [{
            "name": "Missing",
            "filename": "missing.pth",
            "source_repository": "https://example.com",
            "source_revision": "abc123",
            "release": "v0",
            "url": "https://example.com/missing.pth",
            "size_bytes": 100,
            "sha256": "0" * 64,
            "input_size": [640, 640],
        }],
    }), encoding="utf-8")
    result = runner.invoke(app, ["models", "verify", "--lock-file", str(lock)])
    assert result.exit_code == 1
    assert "missing" in result.output.lower()


# ── analyze ─────────────────────────────────────────────────────────────


def test_analyze_missing_video_fails() -> None:
    result = runner.invoke(app, ["analyze", "/no/such/video.mp4"])
    assert result.exit_code != 0


# ── exposure-report ─────────────────────────────────────────────────────


def test_exposure_report_missing_run_fails() -> None:
    result = runner.invoke(app, ["exposure-report", "/no/such/run"])
    assert result.exit_code != 0


# ── no-args shows help ──────────────────────────────────────────────────


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output or "usage" in result.output.lower()


# ── version presence ────────────────────────────────────────────────────


def test_doctor_json_contains_version_string() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    report = json.loads(result.output)
    assert isinstance(report["version"], str)
    assert len(report["version"]) > 0
