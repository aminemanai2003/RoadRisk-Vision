from __future__ import annotations

import json

from roadrisk_vision.models import verify_locked_models


def test_model_verifier_reports_missing_without_network(tmp_path) -> None:
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "filename": "missing.pth",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert verify_locked_models(lock, tmp_path) == [f"missing: {tmp_path / 'missing.pth'}"]
