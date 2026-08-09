from __future__ import annotations

import pytest

from roadrisk_vision.io import RunStore


def test_failed_run_never_overwrites_existing_diagnostics(tmp_path) -> None:
    source = tmp_path / "trip.mp4"
    source.write_bytes(b"fixture")
    store = RunStore(tmp_path / "runs", source, "cpu", "mock", False)
    collision = store.root / f"{store.run_id}.failed"
    collision.mkdir()
    sentinel = collision / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.fail("TEST", "expected")
    assert sentinel.read_text(encoding="utf-8") == "keep"
