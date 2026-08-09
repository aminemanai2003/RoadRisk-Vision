"""Streaming perception contract."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, Self

from roadrisk_vision.config import ModelConfig
from roadrisk_vision.schemas import FramePacket, FramePerception


class PerceptionError(RuntimeError):
    """Actionable model or device failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PerceptionBackend(Protocol):
    def open(self, config: ModelConfig, device: str) -> Self: ...

    def infer(self, frame: FramePacket) -> FramePerception: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
