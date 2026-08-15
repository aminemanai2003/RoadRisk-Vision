"""Atomic, privacy-aware run directory and artifact writers."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from roadrisk_vision.schemas import Manifest, RiskEvent, RunStatus, TimelineRecord, TripSummary
from roadrisk_vision.version import __version__


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _json(model: BaseModel | dict[str, Any] | list[Any]) -> str:
    value = model.model_dump(mode="json") if isinstance(model, BaseModel) else model
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


class RunStore:
    def __init__(
        self,
        root: Path,
        source: Path,
        device: str,
        backend: str,
        include_location: bool,
    ) -> None:
        source_hash = sha256_file(source)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = f"{stamp}-{source_hash[:8]}"
        self.root = root
        self.partial = root / f"{self.run_id}.partial"
        self.final = root / self.run_id
        self.partial.mkdir(parents=True, exist_ok=False)
        self.manifest = Manifest(
            run_id=self.run_id,
            source_basename=source.name,
            source_sha256=source_hash,
            package_version=__version__,
            device=device,
            backend=backend,
            include_location=include_location,
        )
        self.write_manifest()

    def path(self, name: str) -> Path:
        return self.partial / name

    def write_manifest(self) -> None:
        _atomic_text(self.path("manifest.json"), _json(self.manifest))

    def write_events(self, events: list[RiskEvent]) -> None:
        records = "".join(event.model_dump_json() + "\n" for event in events)
        _atomic_text(self.path("events.jsonl"), records)

    def write_timeline(self, timeline: list[TimelineRecord]) -> None:
        records = [record.model_dump(mode="json") for record in timeline]
        _atomic_text(self.path("timeline.json"), _json(records))

    def write_summary(self, summary: TripSummary) -> None:
        _atomic_text(self.path("trip_summary.json"), _json(summary))
        temporary = self.path("trip_summary.csv.tmp")
        values = summary.model_dump(mode="json")
        counts = values.pop("event_counts")
        values.update({f"event_count_{key}": value for key, value in sorted(counts.items())})
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values))
            writer.writeheader()
            writer.writerow(values)
        temporary.replace(self.path("trip_summary.csv"))

    def complete(self, frame_count: int, fps: float, models: dict[str, Any]) -> Path:
        self.manifest.status = RunStatus.COMPLETE
        self.manifest.completed_at = datetime.now(UTC)
        self.manifest.frame_count = frame_count
        self.manifest.normalized_fps = fps
        self.manifest.models = models
        self.write_manifest()
        self.path("normalized.mp4").unlink(missing_ok=True)
        if self.final.exists():
            raise FileExistsError(f"Run directory already exists: {self.final}")
        self.partial.replace(self.final)
        return self.final

    def fail(self, code: str, message: str, cancelled: bool = False) -> Path:
        self.manifest.status = RunStatus.CANCELLED if cancelled else RunStatus.FAILED
        self.manifest.completed_at = datetime.now(UTC)
        self.manifest.error_code = code
        self.manifest.error_message = message
        self.write_manifest()
        for name in (
            "annotated.mp4",
            "annotated.mp4.tmp.mp4",
            "normalized.mp4",
            "normalized.mp4.tmp.mp4",
        ):
            self.path(name).unlink(missing_ok=True)
        failed = self.root / f"{self.run_id}.{'cancelled' if cancelled else 'failed'}"
        if failed.exists():
            raise FileExistsError(f"Diagnostic run directory already exists: {failed}")
        self.partial.replace(failed)
        return failed
