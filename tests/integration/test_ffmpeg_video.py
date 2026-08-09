from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from roadrisk_vision.io import VideoError, normalize_video, probe_video

pytestmark = pytest.mark.integration


def _generator():
    path = Path(__file__).parents[2] / "scripts" / "generate_synthetic_fixtures.py"
    spec = importlib.util.spec_from_file_location("fixture_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load fixture generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_generated_media_exercises_rotation_vfr_corruption_and_privacy(tmp_path) -> None:
    fixture_dir = tmp_path / "fixtures"
    manifest = _generator().generate(fixture_dir)
    assert manifest["license"] == "CC0-1.0"
    assert probe_video(fixture_dir / "synthetic-road.mp4").codec == "h264"
    assert probe_video(fixture_dir / "zero-objects.mp4").duration_s == pytest.approx(1, rel=0.1)

    rotated = probe_video(fixture_dir / "rotation-90.mp4")
    assert rotated.rotation_deg in {90, 270}
    rotated_output = tmp_path / "rotated-normalized.mp4"
    rotated_normalized = normalize_video(fixture_dir / "rotation-90.mp4", rotated_output)
    assert rotated_normalized.width == 180
    assert rotated_normalized.height == 320
    assert rotated_normalized.rotation_deg == 0

    vfr_output = tmp_path / "vfr-normalized.mp4"
    vfr_normalized = normalize_video(fixture_dir / "variable-frame-rate.mp4", vfr_output)
    assert vfr_normalized.fps <= 30

    with pytest.raises(VideoError) as error:
        probe_video(fixture_dir / "corrupt.mp4")
    assert error.value.code in {"UNSUPPORTED_OR_CORRUPT_VIDEO", "NO_VIDEO_STREAM"}
