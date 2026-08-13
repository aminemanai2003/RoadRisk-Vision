"""Checkerboard lens calibration against synthetic, CPU-only imagery.

Every image used here is generated from a known camera model, so no road
footage or external dataset is needed or committed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from roadrisk_vision.cli import app
from roadrisk_vision.geometry import LensProfile

COLUMNS, ROWS = 9, 6
SQUARE_M = 0.025
SIZE = (640, 480)
CAMERA_MATRIX = np.array(
    [[800.0, 0.0, 322.0], [0.0, 800.0, 238.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
DISTORTION = np.array([-0.25, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)

# Tilts and offsets that keep the board in frame while varying pose enough to
# separate focal length from distance.
_POSES = [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.60)),
    ((0.30, 0.0, 0.0), (-0.02, 0.01, 0.62)),
    ((-0.30, 0.0, 0.0), (0.02, -0.01, 0.62)),
    ((0.0, 0.30, 0.0), (0.01, 0.02, 0.62)),
    ((0.0, -0.30, 0.0), (-0.01, -0.02, 0.62)),
    ((0.25, 0.25, 0.10), (0.0, 0.0, 0.66)),
    ((-0.25, 0.25, -0.10), (0.0, 0.0, 0.66)),
    ((0.25, -0.25, 0.15), (0.0, 0.0, 0.70)),
    ((-0.20, -0.20, -0.15), (0.0, 0.0, 0.58)),
]


def _board_texture(pixels_per_square: int = 40, margin_squares: int = 1) -> np.ndarray:
    """A flat checkerboard bitmap with a white quiet zone around it."""
    squares_x, squares_y = COLUMNS + 1, ROWS + 1
    margin = margin_squares * pixels_per_square
    texture = np.full(
        (squares_y * pixels_per_square + 2 * margin, squares_x * pixels_per_square + 2 * margin),
        255,
        dtype=np.uint8,
    )
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2:
                continue
            y = margin + row * pixels_per_square
            x = margin + column * pixels_per_square
            texture[y : y + pixels_per_square, x : x + pixels_per_square] = 0
    return texture


def _distortion_maps() -> tuple[np.ndarray, np.ndarray]:
    """Maps that pull an ideal pinhole image into a lens-distorted one.

    ``remap`` samples ``source[undistort(x)]`` for each distorted pixel ``x``,
    which is the inverse of the usual undistortion maps.
    """
    width, height = SIZE
    grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).astype(np.float32)
    source = cv2.undistortPoints(
        grid.reshape(-1, 1, 2), CAMERA_MATRIX, DISTORTION, P=CAMERA_MATRIX
    ).reshape(height, width, 2)
    return source[..., 0].copy(), source[..., 1].copy()


def _render_view(rvec: tuple[float, float, float], tvec: tuple[float, float, float]) -> np.ndarray:
    """Render one checkerboard still through the known camera model."""
    texture = _board_texture()
    margin_m = SQUARE_M  # one square of quiet zone, matching _board_texture
    left, top = -SQUARE_M - margin_m, -SQUARE_M - margin_m
    right, bottom = COLUMNS * SQUARE_M + margin_m, ROWS * SQUARE_M + margin_m
    paper = np.array(
        [[left, top, 0.0], [right, top, 0.0], [right, bottom, 0.0], [left, bottom, 0.0]],
        dtype=np.float64,
    )
    # Project with zero distortion first: the pinhole image, then bend it.
    projected, _ = cv2.projectPoints(
        paper,
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        CAMERA_MATRIX,
        np.zeros(5),
    )
    texture_height, texture_width = texture.shape
    corners = np.array(
        [
            [0.0, 0.0],
            [texture_width - 1, 0.0],
            [texture_width - 1, texture_height - 1],
            [0.0, texture_height - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, projected.reshape(4, 2).astype(np.float32))
    pinhole = cv2.warpPerspective(
        texture, homography, SIZE, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    map_x, map_y = _distortion_maps()
    return cv2.remap(
        pinhole, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def _write_views(directory: Path, count: int = len(_POSES)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (rvec, tvec) in enumerate(_POSES[:count]):
        cv2.imwrite(str(directory / f"view_{index:02d}.png"), _render_view(rvec, tvec))
    return directory


@pytest.fixture(scope="module")
def checkerboard_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_views(tmp_path_factory.mktemp("checkerboard"))


def _calibrate(directory: Path, **overrides: object) -> LensProfile:
    options: dict[str, object] = {
        "camera_id": "phone-main",
        "board_columns": COLUMNS,
        "board_rows": ROWS,
        "square_size_m": SQUARE_M,
    }
    options.update(overrides)
    return LensProfile.from_images(directory, **options)  # type: ignore[arg-type]


def test_recovers_known_intrinsics_and_records_the_profile(checkerboard_dir: Path) -> None:
    profile = _calibrate(checkerboard_dir, lens_id="wide")

    matrix = np.asarray(profile.camera_matrix)
    assert matrix[0, 0] == pytest.approx(CAMERA_MATRIX[0, 0], rel=0.05)
    assert matrix[1, 1] == pytest.approx(CAMERA_MATRIX[1, 1], rel=0.05)
    assert matrix[0, 2] == pytest.approx(CAMERA_MATRIX[0, 2], abs=15)
    assert matrix[1, 2] == pytest.approx(CAMERA_MATRIX[1, 2], abs=15)
    # Higher-order terms trade off against each other, so only k1's sign and
    # rough magnitude are meaningful here; straightening is checked below.
    assert profile.distortion_coefficients[0] == pytest.approx(DISTORTION[0], abs=0.08)

    assert profile.schema_version == 1
    assert (profile.camera_id, profile.lens_id) == ("phone-main", "wide")
    assert (profile.image_width, profile.image_height) == SIZE
    assert profile.image_count == len(_POSES)
    assert profile.residual_px < 1.0


def test_profile_survives_a_save_load_round_trip(checkerboard_dir: Path, tmp_path: Path) -> None:
    profile = _calibrate(checkerboard_dir)
    path = tmp_path / "lens.json"
    profile.save(path)
    assert LensProfile.load(path) == profile


def test_residual_above_the_threshold_is_rejected(checkerboard_dir: Path) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _calibrate(checkerboard_dir, max_residual_px=0.0)


def test_mixed_resolution_directory_is_rejected(checkerboard_dir: Path, tmp_path: Path) -> None:
    mixed = _write_views(tmp_path / "mixed")
    odd = cv2.resize(_render_view(*_POSES[0]), (320, 240))
    cv2.imwrite(str(mixed / "zz_half.png"), odd)
    with pytest.raises(ValueError, match="one resolution/orientation at a time"):
        _calibrate(mixed)


def test_too_few_detected_boards_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        _calibrate(_write_views(tmp_path / "sparse", count=2))


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No calibration images"):
        _calibrate(empty)


def test_compatibility_rejects_rotated_and_mismatched_frames(checkerboard_dir: Path) -> None:
    profile = _calibrate(checkerboard_dir)
    assert profile.compatibility(*SIZE) == (True, None)

    rotated_ok, rotated_reason = profile.compatibility(SIZE[1], SIZE[0])
    assert rotated_ok is False
    assert rotated_reason is not None and "orientation" in rotated_reason

    resized_ok, resized_reason = profile.compatibility(1920, 1080)
    assert resized_ok is False
    assert resized_reason is not None and "1920x1080" in resized_reason


def _row_bow_px(image: np.ndarray) -> float:
    """Worst deviation of a board row from a straight line, in pixels.

    Rows of checkerboard corners are collinear under an ideal pinhole camera,
    so this measures how much the lens model still bends them.
    """
    found, corners = cv2.findChessboardCornersSB(image, (COLUMNS, ROWS))
    assert found, "corners must be detectable to measure straightness"
    points = corners.reshape(ROWS, COLUMNS, 2)
    worst = 0.0
    for row in points:
        start, end = row[0], row[-1]
        direction = end - start
        length = float(np.linalg.norm(direction))
        offsets = row - start
        cross = direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
        deviation = np.abs(cross) / length
        worst = max(worst, float(deviation.max()))
    return worst


def test_undistort_straightens_the_board_and_guards_frame_size(checkerboard_dir: Path) -> None:
    profile = _calibrate(checkerboard_dir)
    # Pose 0 is head-on, so every residual bow comes from the lens, not perspective.
    distorted = _render_view(*_POSES[0])
    corrected = profile.undistort(distorted)
    assert corrected.shape[:2] == (SIZE[1], SIZE[0])
    assert _row_bow_px(corrected) < 0.25 * _row_bow_px(distorted)

    with pytest.raises(ValueError, match="Cannot undistort"):
        profile.undistort(np.zeros((240, 320), dtype=np.uint8))


def test_cli_writes_a_profile_and_reports_failure(checkerboard_dir: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "lens.json"
    arguments = [
        "calibrate-lens",
        str(checkerboard_dir),
        "--camera-id",
        "phone-main",
        "--board-columns",
        str(COLUMNS),
        "--board-rows",
        str(ROWS),
        "--square-size-m",
        str(SQUARE_M),
        "--output",
        str(output),
    ]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert LensProfile.load(output).camera_id == "phone-main"

    rejected = runner.invoke(app, [*arguments, "--max-residual-px", "0"])
    assert rejected.exit_code == 1
