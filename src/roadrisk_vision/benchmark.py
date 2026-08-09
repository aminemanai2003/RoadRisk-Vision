"""Reproducible local performance benchmark with release-gate evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, ConfigDict, Field

from roadrisk_vision.io import probe_video
from roadrisk_vision.pipeline import AnalysisOptions, analyze_video


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_commit: str
    fixture_basename: str
    fixture_license: str
    fixture_sha256: str
    resolution: str
    source_fps: float
    source_duration_s: float
    analyzed_frames: int
    backend: str
    device: str
    wall_time_s: float
    duration_multiple: float
    peak_process_ram_gb: float
    peak_vram_gb: float | None
    model_provenance: dict[str, Any]
    gates: dict[str, bool]
    passed: bool
    run_directory: str


class PeakMemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            processes = [process, *process.children(recursive=True)]
            total = 0
            for item in processes:
                try:
                    total += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_benchmark(
    options: AnalysisOptions,
    *,
    fixture_license: str,
) -> BenchmarkResult:
    source = probe_video(options.video)
    try:
        import torch

        cuda_active = options.device in {"auto", "cuda"} and torch.cuda.is_available()
        if cuda_active:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None  # type: ignore[assignment]
        cuda_active = False
    sampler = PeakMemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        run_directory = analyze_video(options)
    finally:
        wall_time = time.perf_counter() - started
        sampler.stop()
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    peak_vram = None
    if cuda_active and torch is not None:
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    multiple = wall_time / source.duration_s if source.duration_s else float("inf")
    ram_gb = sampler.peak_bytes / 1024**3
    gates = {
        "duration_at_most_5x": multiple <= 5,
        "ram_at_most_8gb": ram_gb <= 8,
        "vram_at_most_3_5gb": peak_vram is not None and peak_vram <= 3.5,
        "all_frames_processed": manifest["frame_count"] == source.frame_count,
        "cuda_used": manifest["device"] in {"auto", "cuda"} and cuda_active,
        "dual_model_backend": manifest["backend"] == "hybrid",
    }
    return BenchmarkResult(
        git_commit=_git_commit(),
        fixture_basename=options.video.name,
        fixture_license=fixture_license,
        fixture_sha256=_sha256(options.video),
        resolution=f"{source.width}x{source.height}",
        source_fps=source.fps,
        source_duration_s=source.duration_s,
        analyzed_frames=int(manifest["frame_count"]),
        backend=manifest["backend"],
        device=manifest["device"],
        wall_time_s=wall_time,
        duration_multiple=multiple,
        peak_process_ram_gb=ram_gb,
        peak_vram_gb=peak_vram,
        model_provenance=manifest["models"],
        gates=gates,
        passed=all(gates.values()),
        run_directory=run_directory.name,
    )


def write_benchmark_result(result: BenchmarkResult, output_base: Path) -> dict[str, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    markdown_path = output_base.with_suffix(".md")
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temporary.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    json_temporary.replace(json_path)
    gate_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result.gates.items()
    )
    markdown = f"""# RTX 2050 RoadRisk benchmark

| Field | Result |
|---|---:|
| Commit | `{result.git_commit}` |
| Fixture | `{result.fixture_basename}` ({result.fixture_license}) |
| Resolution / FPS | {result.resolution} / {result.source_fps:.2f} |
| Source duration | {result.source_duration_s:.2f} s |
| Wall time | {result.wall_time_s:.2f} s |
| Duration multiple | {result.duration_multiple:.2f}× |
| Peak process RAM | {result.peak_process_ram_gb:.2f} GB |
| Peak VRAM | {result.peak_vram_gb if result.peak_vram_gb is not None else 'unavailable'} GB |
| Frames | {result.analyzed_frames} |

## Release gates

| Gate | Status |
|---|---|
{gate_rows}

Overall: **{'PASS' if result.passed else 'FAIL'}**.

CUDA and H.264 output are tolerance-deterministic, not byte-identical. The JSON
sidecar contains the fixture hash, model provenance and exact numeric results.
"""
    markdown_temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_temporary.write_text(markdown, encoding="utf-8")
    markdown_temporary.replace(markdown_path)
    return {"json": json_path, "markdown": markdown_path}
