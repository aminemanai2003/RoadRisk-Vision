"""Phone-video inspection, privacy normalization and bounded frame decoding."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2

from roadrisk_vision.schemas import FramePacket


class VideoError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    codec: str
    container: str
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int | None


def _require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise VideoError(
            "MISSING_FFMPEG",
            f"{name} was not found. Install FFmpeg and ensure it is available on PATH.",
        )
    return executable


def _rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    return float(Fraction(value))


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise VideoError("VIDEO_NOT_FOUND", f"Video not found: {path}")
    ffprobe = _require_tool("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VideoError("UNSUPPORTED_OR_CORRUPT_VIDEO", result.stderr.strip() or "ffprobe failed")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise VideoError("NO_VIDEO_STREAM", "No video stream was found")
    stream = streams[0]
    codec = str(stream.get("codec_name", "unknown"))
    container = str(payload.get("format", {}).get("format_name", "unknown"))
    if codec not in {"h264", "hevc"} or not {"mov", "mp4"}.intersection(container.split(",")):
        raise VideoError(
            "UNSUPPORTED_CODEC",
            f"Supported input is MP4/MOV with H.264 or H.265; found {container}/{codec}. "
            "Transcode with: ffmpeg -i INPUT -an -c:v libx264 -pix_fmt yuv420p output.mp4",
        )
    duration = stream.get("duration") or payload.get("format", {}).get("duration") or 0
    frame_count = stream.get("nb_frames")
    return VideoInfo(
        path=path,
        codec=codec,
        container=container,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=_rate(stream.get("avg_frame_rate")),
        duration_s=float(duration),
        frame_count=int(frame_count) if frame_count not in {None, "N/A"} else None,
    )


def normalize_video(source: Path, destination: Path, fps_cap: int = 30) -> VideoInfo:
    """Normalize rotation/VFR and remove audio. Destination is replaced atomically."""
    source_info = probe_video(source)
    ffmpeg = _require_tool("ffmpeg")
    fps = min(source_info.fps or float(fps_cap), float(fps_cap))
    temporary = destination.with_suffix(destination.suffix + ".tmp.mp4")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-an",
        "-vf",
        f"fps={fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise VideoError("NORMALIZATION_FAILED", result.stderr.strip() or "FFmpeg failed")
    temporary.replace(destination)
    return probe_video(destination)


class FrameStream:
    """Context-managed iterator that decodes one frame at a time."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> Self:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise VideoError("DECODER_OPEN_FAILED", f"OpenCV could not open {self.path}")
        self.capture = capture
        return self

    @property
    def fps(self) -> float:
        if self.capture is None:
            raise RuntimeError("FrameStream is not open")
        return float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def size(self) -> tuple[int, int]:
        if self.capture is None:
            raise RuntimeError("FrameStream is not open")
        return (
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __iter__(self) -> Iterator[FramePacket]:
        if self.capture is None:
            raise RuntimeError("FrameStream is not open")
        index = 0
        fps = self.fps
        while True:
            success, image = self.capture.read()
            if not success:
                break
            yield FramePacket(
                frame_index=index,
                video_time_ms=round(index * 1000 / fps),
                image=image,
            )
            index += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
