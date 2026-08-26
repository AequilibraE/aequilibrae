"""Helpers for dependencies provided by the ``create`` extra."""

import importlib
from types import ModuleType


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency required for a feature is missing."""


_EXTRA_NAME = "create"


def require(package_name: str, *, feature=None) -> ModuleType:
    """Import an optional package with an actionable error when unavailable."""
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        feature_label = f" for {feature}" if feature else ""
        raise OptionalDependencyError(
            f"`{package_name}` is required{feature_label} but is not installed. "
            f"Install via `pip install aequilibrae[{_EXTRA_NAME}]` "
            f"or `pip install {package_name}`."
        ) from exc
