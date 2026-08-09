from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from roadrisk_vision.config import AppConfig
from roadrisk_vision.schemas import BoundingBox, FramePacket, Manifest


def test_default_configuration_is_privacy_first() -> None:
    config = AppConfig()
    assert config.include_location is False
    assert config.schema_version == 1
    assert config.models.confidence == pytest.approx(0.35)


def test_configuration_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"cloud_upload": True})


def test_bbox_bottom_center_and_frame_time_contract() -> None:
    box = BoundingBox(x1=10, y1=20, x2=30, y2=60)
    packet = FramePacket(frame_index=4, video_time_ms=133, image=np.zeros((2, 2, 3)))
    assert box.bottom_center == (20, 60)
    assert packet.video_time_ms == 133


def test_manifest_never_requires_an_absolute_source_path() -> None:
    manifest = Manifest(
        run_id="run",
        source_basename="trip.mp4",
        source_sha256="0" * 64,
        package_version="test",
        device="cpu",
        backend="mock",
    )
    assert "source_path" not in manifest.model_dump()
