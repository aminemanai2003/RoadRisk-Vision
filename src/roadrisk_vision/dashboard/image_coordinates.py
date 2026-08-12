"""Streamlit-compatible image coordinate component.

The frontend assets are supplied by ``streamlit-image-coordinates`` (MIT,
Copyright Zachary Blackwood). This wrapper avoids importing the package's
legacy Streamlit type alias, which was removed in Streamlit 1.51.
"""

from __future__ import annotations

import base64
import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit.components.v1 as components
from PIL import Image


def frontend_directory() -> Path:
    """Return the installed component frontend without importing its package."""
    spec = importlib.util.find_spec("streamlit_image_coordinates")
    if spec is None:
        raise RuntimeError(
            "streamlit-image-coordinates is required; install RoadRisk Vision "
            "with the 'dashboard' extra"
        )
    if spec.submodule_search_locations:
        package_directory = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        package_directory = Path(spec.origin).parent
    else:  # pragma: no cover - defensive guard for unusual import loaders
        raise RuntimeError("Cannot locate streamlit-image-coordinates frontend assets")
    frontend = package_directory / "frontend"
    if not (frontend / "index.html").is_file():
        raise RuntimeError(f"Component frontend assets are missing from {frontend}")
    return frontend


def _encode_png(source: np.ndarray) -> str:
    """Encode an image array as the data URL expected by the component."""
    buffer = BytesIO()
    Image.fromarray(source).save(buffer, format="PNG", compress_level=1)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


_component = components.declare_component(
    "roadrisk_image_coordinates",
    path=str(frontend_directory()),
)


def streamlit_image_coordinates(
    source: np.ndarray,
    *,
    width: int | None = None,
    key: str | None = None,
) -> Any:
    """Render an image and return the most recent click coordinates."""
    return _component(
        src=_encode_png(source),
        height=None,
        width=width,
        use_column_width=None,
        key=key,
        click_and_drag=False,
        cursor="crosshair",
    )
