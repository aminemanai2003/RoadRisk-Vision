from __future__ import annotations

from pathlib import Path

from roadrisk_vision.cli import _bundled_asset


def test_asset_resolver_prefers_repository_asset(tmp_path: Path) -> None:
    repository_asset = tmp_path / "asset.json"
    repository_asset.write_text("{}", encoding="utf-8")
    assert _bundled_asset("missing.json", repository_asset) == repository_asset


def test_dashboard_source_is_available() -> None:
    path = _bundled_asset(
        "dashboard/app.py",
        Path(__file__).parents[1] / "apps" / "dashboard.py",
    )
    assert path.name == "dashboard.py"
