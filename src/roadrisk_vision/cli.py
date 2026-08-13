"""RoadRisk Vision command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from roadrisk_vision.analytics import build_exposure_report, write_exposure_report
from roadrisk_vision.benchmark import run_benchmark, write_benchmark_result
from roadrisk_vision.diagnostics import format_diagnostics, run_diagnostics
from roadrisk_vision.geometry import CalibrationProfile
from roadrisk_vision.io import probe_video
from roadrisk_vision.pipeline import AnalysisOptions, analyze_video
from roadrisk_vision.version import __version__

app = typer.Typer(help="Offline, privacy-first dashcam risk analysis.", no_args_is_help=True)
models_app = typer.Typer(help="Explicit model weight management.")
app.add_typer(models_app, name="models")


def _bundled_asset(name: str, repository_path: Path) -> Path:
    """Resolve an asset from a source checkout or an installed wheel."""
    if repository_path.is_file():
        return repository_path
    bundled = Path(__file__).parent / name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(f"RoadRisk Vision asset is missing: {name}")


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
    info = probe_video(video)
    profile = CalibrationProfile.create(
        camera_id=camera_id,
        width=info.width,
        height=info.height,
        mount_height_m=height_m,
        reference_width_m=reference_width_m,
        reference_length_m=reference_length_m,
        corners=_corners(corners),
        calibration_frame_ms=frame_ms,
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
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON")] = False,
) -> None:
    """Check local prerequisites without changing the system."""
    report = run_diagnostics()
    report["version"] = __version__
    typer.echo(json.dumps(report, indent=2) if json_output else format_diagnostics(report))


@app.command()
def dashboard(
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8501,
    headless: Annotated[bool, typer.Option("--headless/--open-browser")] = False,
) -> None:
    """Launch the private local analysis and calibration dashboard."""
    app_path = _bundled_asset(
        "dashboard/app.py",
        Path(__file__).parents[2] / "apps" / "dashboard.py",
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        f"--server.headless={str(headless).lower()}",
        "--browser.gatherUsageStats=false",
    ]
    try:
        completed = subprocess.run(command, check=False)
    except ModuleNotFoundError as exc:  # pragma: no cover - subprocess owns import failure
        raise typer.BadParameter("Install RoadRisk Vision with the 'dashboard' extra") from exc
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@models_app.command("download")
def download_models(
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    lock_file: Annotated[Path | None, typer.Option("--lock-file")] = None,
) -> None:
    """Download exactly the explicitly locked weights."""
    from roadrisk_vision.models import download_locked_models

    resolved_lock = lock_file or _bundled_asset("models.lock.json", Path("models.lock.json"))
    download_locked_models(resolved_lock, Path("models"), dry_run=dry_run)


@models_app.command("verify")
def verify_models(
    lock_file: Annotated[Path | None, typer.Option("--lock-file")] = None,
) -> None:
    """Verify downloaded weight size and SHA-256."""
    from roadrisk_vision.models import verify_locked_models

    resolved_lock = lock_file or _bundled_asset("models.lock.json", Path("models.lock.json"))
    failures = verify_locked_models(resolved_lock, Path("models"))
    if failures:
        for failure in failures:
            typer.secho(failure, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("All locked models verified.", fg=typer.colors.GREEN)


@app.command("exposure-report")
def exposure_report(
    run_directories: Annotated[
        list[Path],
        typer.Argument(exists=True, file_okay=False, help="One or more completed run folders"),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("exposure_report"),
    low_ttc_s: Annotated[float, typer.Option("--low-ttc-s")] = 2.0,
) -> None:
    """Aggregate risk events into exposure features without modelling insurance claims."""
    report = build_exposure_report(run_directories, low_ttc_definition_s=low_ttc_s)
    outputs = write_exposure_report(report, output)
    for name, path in outputs.items():
        typer.secho(f"{name}: {path}", fg=typer.colors.GREEN)


@app.command()
def benchmark(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    fixture_license: Annotated[str, typer.Option("--fixture-license")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("benchmark"),
    run_output: Annotated[Path, typer.Option("--run-output")] = Path("runs"),
    device: Annotated[str, typer.Option("--device")] = "cuda",
    backend: Annotated[str, typer.Option("--backend")] = "hybrid",
) -> None:
    """Measure an analysis run and evaluate the published release gates."""
    result = run_benchmark(
        AnalysisOptions(
            video=video,
            output=run_output,
            device=device,
            backend=backend,
        ),
        fixture_license=fixture_license,
    )
    outputs = write_benchmark_result(result, output)
    for name, path in outputs.items():
        typer.secho(f"{name}: {path}", fg=typer.colors.GREEN)
    if not result.passed:
        raise typer.Exit(2)


schemas_app = typer.Typer(help="Artifact schema management.")
app.add_typer(schemas_app, name="schemas")


@schemas_app.command("export")
def export_schemas_cmd(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("schemas"),
) -> None:
    """Write current artifact schemas as JSON Schema files."""
    from roadrisk_vision.schema_export import export_schemas

    written = export_schemas(output)
    for name, path in written.items():
        typer.secho(f"{name}: {path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
