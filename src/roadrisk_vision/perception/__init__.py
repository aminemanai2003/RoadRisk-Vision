"""Perception backends with no import-time model downloads."""

from roadrisk_vision.perception.base import PerceptionBackend, PerceptionError
from roadrisk_vision.perception.composite import CompositeBackend
from roadrisk_vision.perception.mock import MockBackend

__all__ = ["CompositeBackend", "MockBackend", "PerceptionBackend", "PerceptionError"]
