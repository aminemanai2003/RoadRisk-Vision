from __future__ import annotations

import json

from roadrisk_vision import benchmark
from roadrisk_vision.io.video import VideoInfo
from roadrisk_vision.pipeline import AnalysisOptions


def test_benchmark_calculates_gates_and_writes_provenance(tmp_path, monkeypatch) -> None:
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"synthetic")
    run = tmp_path / "runs" / "run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "frame_count": 300,
                "backend": "hybrid",
                "device": "cuda",
                "models": {"road": "hybridnets", "objects": "yolox-s"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        benchmark,
        "probe_video",
        lambda path: VideoInfo(path, "h264", "mp4", 1920, 1080, 30, 10, 300),
    )
    monkeypatch.setattr(benchmark, "analyze_video", lambda options: run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([0.0, 20.0]).__next__)
    result = benchmark.run_benchmark(
        AnalysisOptions(video=source, output=tmp_path / "runs", backend="hybrid", device="cpu"),
        fixture_license="CC0-1.0",
    )
    assert result.duration_multiple == 2
    assert result.gates["duration_at_most_5x"] is True
    assert result.gates["dual_model_backend"] is True
    assert result.environment.python
    outputs = benchmark.write_benchmark_result(result, tmp_path / "benchmark")
    assert outputs["json"].is_file()
    assert "Release gates" in outputs["markdown"].read_text(encoding="utf-8")
