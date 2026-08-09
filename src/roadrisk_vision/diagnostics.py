"""Read-only environment diagnostics with actionable platform guidance."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

import cv2

MINIMUM_FFMPEG = (5, 0)


@dataclass(frozen=True)
class ToolDiagnostic:
    name: str
    status: str
    path: str | None
    version: str | None
    minimum_version: str | None
    guidance: str | None


def _install_guidance(system: str) -> str:
    guidance = {
        "Windows": (
            "Install a tested FFmpeg build with `winget install --id Gyan.FFmpeg.Shared` "
            "then open a new terminal. https://www.gyan.dev/ffmpeg/builds/"
        ),
        "Linux": (
            "Install FFmpeg from your distribution, for example `sudo apt update && "
            "sudo apt install ffmpeg`. https://ffmpeg.org/download.html#build-linux"
        ),
        "Darwin": (
            "Install FFmpeg with `brew install ffmpeg`. "
            "https://ffmpeg.org/download.html#build-mac"
        ),
    }
    return guidance.get(system, "Install FFmpeg 5.0+ from https://ffmpeg.org/download.html")


def _parse_version(output: str) -> tuple[int, ...] | None:
    match = re.search(r"\bversion\s+(\d+(?:\.\d+){0,3})", output)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def diagnose_executable(name: str, *, system: str | None = None) -> ToolDiagnostic:
    system = system or platform.system()
    path = shutil.which(name)
    guidance = _install_guidance(system)
    if path is None:
        return ToolDiagnostic(name, "missing", None, None, "5.0", guidance)
    result = subprocess.run(
        [path, "-version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    output = result.stdout or result.stderr
    parsed = _parse_version(output)
    version = ".".join(map(str, parsed)) if parsed else None
    if result.returncode != 0 or parsed is None:
        return ToolDiagnostic(name, "unreadable", path, version, "5.0", guidance)
    if parsed < MINIMUM_FFMPEG:
        return ToolDiagnostic(name, "unsupported", path, version, "5.0", guidance)
    return ToolDiagnostic(name, "ready", path, version, "5.0", None)


def run_diagnostics() -> dict[str, Any]:
    tools = [diagnose_executable("ffmpeg"), diagnose_executable("ffprobe")]
    try:
        import torch

        torch_version: str | None = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_device = None
    return {
        "ready": all(tool.status == "ready" for tool in tools),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "tools": [asdict(tool) for tool in tools],
    }


def format_diagnostics(report: dict[str, Any]) -> str:
    lines = [
        "RoadRisk Vision environment",
        f"  Python: {report['python']}",
        f"  OpenCV: {report['opencv']}",
        f"  PyTorch: {report['torch'] or 'not installed'}",
        f"  CUDA: {report['cuda_device'] or 'not available'}",
    ]
    for tool in report["tools"]:
        detail = tool["version"] or tool["path"] or "not found"
        lines.append(f"  {tool['name']}: {tool['status']} ({detail})")
        if tool["guidance"]:
            lines.append(f"    {tool['guidance']}")
    readiness = (
        "Ready for video analysis."
        if report["ready"]
        else "Setup required before analysis."
    )
    lines.append(readiness)
    return "\n".join(lines)
