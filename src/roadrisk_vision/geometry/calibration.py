"""Road-plane and lens calibration with explicit compatibility and null semantics."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from roadrisk_vision.schemas import Detection, NullReason

#: Root-mean-square reprojection error above which a lens profile is rejected.
MAX_LENS_RESIDUAL_PX = 1.0
#: Fewest usable views that still constrain the intrinsics; 15-20 is recommended.
MIN_LENS_IMAGES = 5

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class _JsonProfile(BaseModel):
    """Atomic JSON persistence shared by the calibration profiles."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)


class CalibrationProfile(_JsonProfile):
    schema_version: int = 1
    camera_id: str
    lens_id: str = "default"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    orientation_deg: int = 0
    crop_id: str = "full-frame"
    calibration_frame_ms: int = Field(0, ge=0)
    mount_height_m: float = Field(gt=0)
    reference_width_m: float = Field(gt=0)
    reference_length_m: float = Field(gt=0)
    image_corners_xy: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    image_to_ground: list[list[float]]
    residual_pct: float = Field(ge=0)

    @classmethod
    def create(
        cls,
        *,
        camera_id: str,
        width: int,
        height: int,
        mount_height_m: float,
        reference_width_m: float,
        reference_length_m: float,
        corners: list[tuple[float, float]],
        lens_id: str = "default",
        orientation_deg: int = 0,
        crop_id: str = "full-frame",
        calibration_frame_ms: int = 0,
    ) -> CalibrationProfile:
        if len(corners) != 4:
            raise ValueError(
                "Exactly four corners are required: near-left, near-right, far-right, far-left"
            )
        source = np.asarray(corners, dtype=np.float32)
        destination = np.asarray(
            [
                [0.0, 0.0],
                [reference_width_m, 0.0],
                [reference_width_m, reference_length_m],
                [0.0, reference_length_m],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Calibration points produce a degenerate road plane")
        return cls(
            camera_id=camera_id,
            lens_id=lens_id,
            source_width=width,
            source_height=height,
            orientation_deg=orientation_deg,
            crop_id=crop_id,
            calibration_frame_ms=calibration_frame_ms,
            mount_height_m=mount_height_m,
            reference_width_m=reference_width_m,
            reference_length_m=reference_length_m,
            image_corners_xy=corners,
            image_to_ground=matrix.tolist(),
            residual_pct=0.0,
        )

    def compatibility(self, width: int, height: int) -> tuple[bool, str | None]:
        if self.residual_pct > 5:
            return False, "calibration residual exceeds 5%"
        source_ratio = self.source_width / self.source_height
        ratio = width / height
        if abs(source_ratio - ratio) / source_ratio > 0.01:
            return False, "video aspect ratio/crop does not match calibration"
        return True, None

    def project(self, point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
        scale_x = self.source_width / width
        scale_y = self.source_height / height
        adjusted = np.asarray([[[point[0] * scale_x, point[1] * scale_y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(
            adjusted,
            np.asarray(self.image_to_ground, dtype=np.float64),
        )[0, 0]
        return float(projected[0]), float(projected[1])


class LensProfile(_JsonProfile):
    """Lens intrinsics estimated from checkerboard stills.

    Independent of :class:`CalibrationProfile`: the four-point road-plane
    workflow stays usable with or without one of these.
    """

    schema_version: int = 1
    camera_id: str
    lens_id: str = "default"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    orientation_deg: int = 0
    board_columns: int = Field(gt=1)
    board_rows: int = Field(gt=1)
    square_size_m: float = Field(gt=0)
    image_count: int = Field(gt=0)
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    residual_px: float = Field(ge=0)

    @classmethod
    def from_images(
        cls,
        directory: Path,
        *,
        camera_id: str,
        board_columns: int,
        board_rows: int,
        square_size_m: float,
        lens_id: str = "default",
        orientation_deg: int = 0,
        max_residual_px: float = MAX_LENS_RESIDUAL_PX,
    ) -> LensProfile:
        """Estimate intrinsics from a local directory of checkerboard stills.

        ``board_columns``/``board_rows`` count *inner* corners, so a printed
        10x7 square board is 9x6.
        """
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"No calibration images found in {directory}")

        pattern = (board_columns, board_rows)
        board = np.zeros((board_rows * board_columns, 3), dtype=np.float32)
        board[:, :2] = np.mgrid[0:board_columns, 0:board_rows].T.reshape(-1, 2)
        board *= square_size_m

        size: tuple[int, int] | None = None
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unreadable calibration image: {path.name}")
            height, width = image.shape[:2]
            if size is None:
                size = (width, height)
            elif (width, height) != size:
                # Mixed resolutions cannot share one intrinsic matrix.
                raise ValueError(
                    f"{path.name} is {width}x{height} but earlier images are "
                    f"{size[0]}x{size[1]}; calibrate one resolution/orientation at a time"
                )
            found, corners = cv2.findChessboardCornersSB(
                image,
                pattern,
                flags=cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                object_points.append(board)
                image_points.append(corners)

        assert size is not None
        if len(image_points) < MIN_LENS_IMAGES:
            raise ValueError(
                f"Detected the {board_columns}x{board_rows} board in only "
                f"{len(image_points)} of {len(paths)} images; at least "
                f"{MIN_LENS_IMAGES} are required"
            )

        residual, matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points, image_points, size, None, None
        )
        if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
            raise ValueError("Calibration did not converge to a finite lens model")
        if residual > max_residual_px:
            raise ValueError(
                f"Lens calibration residual {residual:.3f} px exceeds the "
                f"{max_residual_px:.3f} px threshold; recapture with a flatter board, "
                "sharper focus and more varied tilts"
            )
        return cls(
            camera_id=camera_id,
            lens_id=lens_id,
            image_width=size[0],
            image_height=size[1],
            orientation_deg=orientation_deg,
            board_columns=board_columns,
            board_rows=board_rows,
            square_size_m=square_size_m,
            image_count=len(image_points),
            camera_matrix=matrix.tolist(),
            distortion_coefficients=[float(value) for value in np.ravel(distortion)],
            residual_px=float(residual),
        )

    def compatibility(self, width: int, height: int) -> tuple[bool, str | None]:
        if self.residual_px > MAX_LENS_RESIDUAL_PX:
            return False, f"lens residual exceeds {MAX_LENS_RESIDUAL_PX} px"
        if (width, height) == (self.image_width, self.image_height):
            return True, None
        if (width, height) == (self.image_height, self.image_width):
            return False, "frame orientation is rotated relative to the lens profile"
        return False, (
            f"frame is {width}x{height} but the lens profile is "
            f"{self.image_width}x{self.image_height}"
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from a frame captured with this lens."""
        height, width = image.shape[:2]
        compatible, reason = self.compatibility(width, height)
        if not compatible:
            raise ValueError(f"Cannot undistort: {reason}")
        return cv2.undistort(
            image,
            np.asarray(self.camera_matrix, dtype=np.float64),
            np.asarray(self.distortion_coefficients, dtype=np.float64),
        )


class GeometryEstimator:
    def __init__(self, calibration: CalibrationProfile | None, history_size: int = 8) -> None:
        self.calibration = calibration
        self.history: dict[int, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(
        self,
        detections: list[Detection],
        video_time_ms: int,
        frame_size: tuple[int, int],
        drivable_mask: np.ndarray | None,
    ) -> list[Detection]:
        width, height = frame_size
        compatible = (
            self.calibration.compatibility(width, height)
            if self.calibration
            else (False, None)
        )
        for detection in detections:
            contact_x, contact_y = detection.bbox.bottom_center
            ix = min(max(round(contact_x), 0), width - 1)
            iy = min(max(round(contact_y), 0), height - 1)
            detection.in_path = bool(drivable_mask is not None and drivable_mask[iy, ix] > 0)
            if self.calibration is None:
                detection.distance_null_reason = NullReason.NOT_CALIBRATED
                detection.ttc_null_reason = NullReason.NOT_CALIBRATED
                continue
            if not compatible[0]:
                detection.distance_null_reason = NullReason.INCOMPATIBLE_CALIBRATION
                detection.ttc_null_reason = NullReason.INCOMPATIBLE_CALIBRATION
                continue
            if not detection.in_path:
                detection.distance_null_reason = NullReason.LOW_CONFIDENCE
                detection.ttc_null_reason = NullReason.LOW_CONFIDENCE
                continue
            _lateral, longitudinal = self.calibration.project((contact_x, contact_y), width, height)
            if not np.isfinite(longitudinal) or longitudinal < 0:
                detection.distance_null_reason = NullReason.UNSTABLE_ESTIMATE
                detection.ttc_null_reason = NullReason.UNSTABLE_ESTIMATE
                continue
            detection.distance_m = longitudinal
            if detection.track_id is None:
                detection.ttc_null_reason = NullReason.UNSTABLE_ESTIMATE
                continue
            samples = self.history[detection.track_id]
            samples.append((video_time_ms, longitudinal))
            if len(samples) < 3 or samples[-1][0] == samples[0][0]:
                detection.ttc_null_reason = NullReason.UNSTABLE_ESTIMATE
                continue
            elapsed = (samples[-1][0] - samples[0][0]) / 1000
            relative_speed = (samples[-1][1] - samples[0][1]) / elapsed
            detection.relative_speed_mps = relative_speed
            if relative_speed >= -0.2:
                detection.ttc_null_reason = NullReason.NOT_CLOSING
                continue
            detection.ttc_s = longitudinal / -relative_speed
        return detections
