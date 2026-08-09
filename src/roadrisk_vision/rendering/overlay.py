"""Privacy-safe overlays and silent H.264 writer."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import cv2
import numpy as np

from roadrisk_vision.risk import RiskFrame
from roadrisk_vision.schemas import FramePerception, WarningState

WARNING_COLORS = {
    WarningState.NORMAL: (80, 180, 80),
    WarningState.MONITOR: (40, 200, 230),
    WarningState.SLOW_DOWN: (0, 165, 255),
    WarningState.BRAKE_RISK: (20, 20, 230),
    WarningState.LANE_DEPARTURE: (180, 80, 220),
    WarningState.HAZARD: (0, 0, 255),
}


def render_frame(image: np.ndarray, perception: FramePerception, risk: RiskFrame) -> np.ndarray:
    output = image.copy()
    if perception.drivable_mask is not None:
        overlay = np.zeros_like(output)
        overlay[perception.drivable_mask > 0] = (40, 120, 40)
        output = cv2.addWeighted(output, 1.0, overlay, 0.25, 0)
    if perception.lane_mask is not None:
        output[perception.lane_mask > 0] = (0, 220, 255)
    for detection in perception.detections:
        box = detection.bbox
        color = (0, 170, 255) if detection.in_path else (180, 180, 180)
        top_left = (round(box.x1), round(box.y1))
        bottom_right = (round(box.x2), round(box.y2))
        cv2.rectangle(output, top_left, bottom_right, color, 2)
        details = [f"#{detection.track_id or '-'} {detection.label} {detection.confidence:.0%}"]
        if detection.distance_m is not None:
            details.append(f"{detection.distance_m:.1f}m")
        if detection.ttc_s is not None:
            details.append(f"TTC {detection.ttc_s:.1f}s")
        cv2.putText(
            output,
            " | ".join(details),
            (round(box.x1), max(22, round(box.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    color = WARNING_COLORS[risk.warning_state]
    cv2.rectangle(output, (0, 0), (output.shape[1], 54), (20, 24, 30), -1)
    cv2.putText(
        output,
        f"RoadRisk Vision | {risk.warning_state.value.replace('_', ' ').upper()}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


class FFmpegVideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        self.path = path
        self.size = size
        self.fps = fps
        self.process: subprocess.Popen[bytes] | None = None
        self.stdin: IO[bytes] | None = None

    def open(self) -> Self:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required to encode annotated output")
        width, height = self.size
        temporary = self.path.with_suffix(self.path.suffix + ".tmp.mp4")
        command = [
            ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{self.fps:g}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.stdin = self.process.stdin
        return self

    def write(self, image: np.ndarray) -> None:
        if self.stdin is None:
            raise RuntimeError("Video writer is not open")
        self.stdin.write(image.tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        if self.stdin is not None:
            self.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr
            else ""
        )
        return_code = self.process.wait()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp.mp4")
        if return_code != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg encoder failed: {stderr.strip()}")
        temporary.replace(self.path)
        self.process = None
        self.stdin = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None and self.process is not None:
            if self.stdin is not None:
                self.stdin.close()
            self.process.kill()
            self.path.with_suffix(self.path.suffix + ".tmp.mp4").unlink(missing_ok=True)
            return
        self.close()
