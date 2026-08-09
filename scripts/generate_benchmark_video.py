"""Generate a licensed 1080p/30 synthetic benchmark video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def generate(output: Path, duration_s: int = 10) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1920x1080:rate=30:duration={duration_s}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Benchmark fixture generation failed")
    sidecar = {
        "schema_version": 1,
        "license": "CC0-1.0",
        "generator": "FFmpeg testsrc2",
        "privacy": "fully synthetic; no audio, GPS, people, plates, or captured metadata",
        "resolution": "1920x1080",
        "fps": 30,
        "duration_s": duration_s,
    }
    output.with_suffix(".license.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration-s", type=int, default=10)
    args = parser.parse_args()
    generate(args.output, args.duration_s)


if __name__ == "__main__":
    main()
