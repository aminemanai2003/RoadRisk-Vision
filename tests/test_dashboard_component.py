from __future__ import annotations

import base64

import numpy as np

from roadrisk_vision.dashboard.image_coordinates import _encode_png, frontend_directory


def test_coordinate_component_frontend_is_installed() -> None:
    assert (frontend_directory() / "index.html").is_file()


def test_coordinate_component_encodes_png_data_url() -> None:
    source = np.zeros((4, 6, 3), dtype=np.uint8)
    data_url = _encode_png(source)
    prefix, payload = data_url.split(",", maxsplit=1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(payload).startswith(b"\x89PNG\r\n\x1a\n")
