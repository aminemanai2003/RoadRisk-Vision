from __future__ import annotations

from subprocess import CompletedProcess

from roadrisk_vision import diagnostics


def test_missing_ffmpeg_has_windows_install_guidance(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    result = diagnostics.diagnose_executable("ffmpeg", system="Windows")
    assert result.status == "missing"
    assert "winget install" in result.guidance


def test_old_ffmpeg_is_distinguished_from_missing(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "ffmpeg version 4.4.2", ""),
    )
    result = diagnostics.diagnose_executable("ffmpeg", system="Linux")
    assert result.status == "unsupported"
    assert result.version == "4.4.2"
    assert "apt install ffmpeg" in result.guidance


def test_supported_ffmpeg_is_ready(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "ffmpeg version 7.1.1", ""),
    )
    result = diagnostics.diagnose_executable("ffmpeg", system="Linux")
    assert result.status == "ready"
    assert result.guidance is None
