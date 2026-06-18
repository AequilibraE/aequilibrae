"""Utilities for handling optional third-party dependencies.

The new network importer (``aequilibrae.project.network.importer``) is built on
top of several optional packages (``osmnx``, ``pyrosm``, ``neatnet``,
``overturemaps``). They are grouped under the ``aequilibrae[create]`` extra.

This module centralises the ``ImportError`` handling so users get a single,
actionable message when an optional package is missing rather than a stack
trace from an inner module.
"""

import importlib
from types import ModuleType


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency required for a feature is missing."""


_EXTRA_NAME = "create"


def require(package_name: str, *, feature=None) -> ModuleType:
    """Import an optional package, raising ``OptionalDependencyError`` if missing.

    :Arguments:
        **package_name** (:obj:`str`): The importable package name (e.g. ``"osmnx"``).
        **feature** (:obj:`str`, *Optional*): A short label describing the feature
        that requires the package. Used in the error message.

    :Returns:
        The imported module.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        feature_label = f" for {feature}" if feature else ""
        raise OptionalDependencyError(
            f"`{package_name}` is required{feature_label} but is not installed. "
            f"Install via `pip install aequilibrae[{_EXTRA_NAME}]` "
            f"or `pip install {package_name}`."
        ) from exc
