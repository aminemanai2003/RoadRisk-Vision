"""Explicit, checksummed model downloads; never called during analysis."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, cast


def _lock(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_models(lock_file: Path, model_dir: Path) -> list[str]:
    failures: list[str] = []
    for model in _lock(lock_file)["models"]:
        path = model_dir / model["filename"]
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif path.stat().st_size != model["size_bytes"]:
            failures.append(f"wrong size: {path}")
        elif _sha256(path) != model["sha256"]:
            failures.append(f"checksum mismatch: {path}")
    return failures


def download_locked_models(lock_file: Path, model_dir: Path, *, dry_run: bool = False) -> None:
    lock = _lock(lock_file)
    model_dir.mkdir(parents=True, exist_ok=True)
    for model in lock["models"]:
        destination = model_dir / model["filename"]
        print(f"{model['name']}: {model['url']} -> {destination}")
        if dry_run:
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        urllib.request.urlretrieve(model["url"], temporary)
        if temporary.stat().st_size != model["size_bytes"] or _sha256(temporary) != model["sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Integrity check failed for {model['name']}")
        temporary.replace(destination)
