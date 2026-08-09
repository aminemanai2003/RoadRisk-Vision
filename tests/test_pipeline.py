from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np

from roadrisk_vision.io.video import VideoInfo
from roadrisk_vision.pipeline import AnalysisOptions, analyze_video
from roadrisk_vision.schemas import FramePacket


class FakeFrames:
    size = (160, 90)
    fps = 10.0

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        for index in range(6):
            yield FramePacket(
                frame_index=index,
                video_time_ms=index * 100,
                image=np.zeros((90, 160, 3), dtype=np.uint8),
            )


class FakeWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        self.path = path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.path.write_bytes(b"fake-silent-video")

    def write(self, image: np.ndarray) -> None:
        assert image.shape == (90, 160, 3)


def test_mock_pipeline_creates_complete_atomic_run(tmp_path, monkeypatch) -> None:
    source = tmp_path / "trip.mp4"
    source.write_bytes(b"private-video-fixture")

    def fake_normalize(source_path: Path, destination: Path, fps_cap: int) -> VideoInfo:
        del fps_cap
        destination.write_bytes(source_path.read_bytes())
        return VideoInfo(source_path, "h264", "mp4", 160, 90, 10.0, 0.6, 6)

    monkeypatch.setattr("roadrisk_vision.pipeline.analyzer.normalize_video", fake_normalize)
    monkeypatch.setattr("roadrisk_vision.pipeline.analyzer.FrameStream", FakeFrames)
    monkeypatch.setattr("roadrisk_vision.pipeline.analyzer.FFmpegVideoWriter", FakeWriter)
    result = analyze_video(
        AnalysisOptions(video=source, output=tmp_path / "runs", backend="mock", device="cpu")
    )
    assert result.is_dir()
    assert not result.name.endswith(".partial")
    assert (result / "annotated.mp4").is_file()
    assert (result / "manifest.json").read_text(encoding="utf-8").find('"complete"') > 0
    assert (result / "events.jsonl").is_file()
    assert not (result / "normalized.mp4").exists()
