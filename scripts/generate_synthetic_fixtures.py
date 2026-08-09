"""Generate tiny, private-data-free CC0 video fixtures with FFmpeg."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg fixture command failed")


def generate(output: Path) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to generate synthetic fixtures")
    output.mkdir(parents=True, exist_ok=True)
    base = output / "synthetic-road.mp4"
    black = output / "zero-objects.mp4"
    rotated = output / "rotation-90.mp4"
    vfr = output / "variable-frame-rate.mp4"
    corrupt = output / "corrupt.mp4"
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(base),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=320x180:rate=30:duration=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(black),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(base),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(rotated),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=15:duration=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-fps_mode",
            "vfr",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(vfr),
        ]
    )
    corrupt.write_bytes(base.read_bytes()[:128])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "license": "CC0-1.0",
        "privacy": "fully synthetic; no people, identifiers, GPS, metadata, or audio",
        "files": [path.name for path in (base, black, rotated, vfr, corrupt)],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
