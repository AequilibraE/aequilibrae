"""Exceptions raised by the network importer framework."""


class ImporterError(Exception):
    """Base class for errors raised by the network importer."""


class IRValidationError(ImporterError):
    """Raised when a ``RoutableNetwork`` fails its schema invariants."""


class SourceResolutionError(ImporterError):
    """Raised when a source/simplifier string name cannot be resolved."""
