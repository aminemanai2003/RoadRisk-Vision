"""RoadRisk Vision command-line interface."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import cv2
import typer

from roadrisk_vision.geometry import CalibrationProfile
from roadrisk_vision.io import probe_video
from roadrisk_vision.pipeline import AnalysisOptions, analyze_video
from roadrisk_vision.version import __version__

app = typer.Typer(help="Offline, privacy-first dashcam risk analysis.", no_args_is_help=True)
models_app = typer.Typer(help="Explicit model weight management.")
app.add_typer(models_app, name="models")


def _corners(value: str) -> list[tuple[float, float]]:
    try:
        points = [tuple(float(part) for part in pair.split(",")) for pair in value.split(";")]
    except ValueError as exc:
        raise typer.BadParameter("Use x,y;x,y;x,y;x,y") from exc
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise typer.BadParameter("Exactly four x,y pairs are required")
    return [(point[0], point[1]) for point in points]


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("runs"),
    telemetry: Annotated[Path | None, typer.Option("--telemetry")] = None,
    telemetry_offset_ms: Annotated[int, typer.Option("--telemetry-offset-ms")] = 0,
    calibration: Annotated[Path | None, typer.Option("--calibration")] = None,
    device: Annotated[str, typer.Option("--device")] = "auto",
    backend: Annotated[str, typer.Option("--backend")] = "hybrid",
    config: Annotated[Path | None, typer.Option("--config")] = None,
    include_location: Annotated[bool, typer.Option("--include-location")] = False,
) -> None:
    """Analyze one completed phone recording on this PC."""
    result = analyze_video(
        AnalysisOptions(
            video=video,
            output=output,
            telemetry=telemetry,
            telemetry_offset_ms=telemetry_offset_ms,
            calibration=calibration,
            device=device,
            backend=backend,
            config=config,
            include_location=include_location,
        )
    )
    typer.secho(f"Completed: {result}", fg=typer.colors.GREEN)


@app.command()
def calibrate(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    frame_ms: Annotated[int, typer.Option("--frame-ms")],
    camera_id: Annotated[str, typer.Option("--camera-id")],
    height_m: Annotated[float, typer.Option("--height-m")],
    reference_length_m: Annotated[float, typer.Option("--reference-length-m")],
    reference_width_m: Annotated[float, typer.Option("--reference-width-m")],
    corners: Annotated[
        str,
        typer.Option(
            "--corners",
            help="near-left;near-right;far-right;far-left as x,y pairs",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("calibration.json"),
) -> None:
    """Create a road-plane profile from four measured image points."""
    del frame_ms
    info = probe_video(video)
    profile = CalibrationProfile.create(
        camera_id=camera_id,
        width=info.width,
        height=info.height,
        mount_height_m=height_m,
        reference_width_m=reference_width_m,
        reference_length_m=reference_length_m,
        corners=_corners(corners),
    )
    profile.save(output)
    typer.secho(f"Calibration saved: {output}", fg=typer.colors.GREEN)


@app.command(name="inspect")
def inspect_run(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Print a run's privacy-safe manifest and trip summary."""
    for name in ("manifest.json", "trip_summary.json"):
        path = run_directory / name
        if path.is_file():
            typer.echo(path.read_text(encoding="utf-8"))


@app.command()
def doctor() -> None:
    """Check local prerequisites without changing the system."""
    checks: dict[str, object] = {
        "version": __version__,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "opencv": cv2.__version__,
    }
    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        checks["torch"] = None
        checks["cuda_available"] = False
    typer.echo(json.dumps(checks, indent=2))


@models_app.command("download")
def download_models(
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    lock_file: Annotated[Path, typer.Option("--lock-file")] = Path("models.lock.json"),
) -> None:
    """Download exactly the explicitly locked weights."""
    from roadrisk_vision.models import download_locked_models

    download_locked_models(lock_file, Path("models"), dry_run=dry_run)


@models_app.command("verify")
def verify_models(
    lock_file: Annotated[Path, typer.Option("--lock-file")] = Path("models.lock.json"),
) -> None:
    """Verify downloaded weight size and SHA-256."""
    from roadrisk_vision.models import verify_locked_models

    failures = verify_locked_models(lock_file, Path("models"))
    if failures:
        for failure in failures:
            typer.secho(failure, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("All locked models verified.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
