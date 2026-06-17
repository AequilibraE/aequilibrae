"""``Source`` protocol + string-name registry."""

from __future__ import annotations

from typing import Callable, ClassVar, Protocol, runtime_checkable

from ..exceptions import SourceResolutionError
from ..download_cache import DownloadCache
from ..ir import RoutableNetwork


@runtime_checkable
class Source(Protocol):
    """Anything that can produce a ``RoutableNetwork``."""

    name: ClassVar[str]
    required_extras: ClassVar[tuple[str, ...]]

    def acquire(
        self,
        *,
        modes: tuple[str, ...],
        download_cache: DownloadCache,
    ) -> RoutableNetwork:  # pragma: no cover - protocol
        ...


SOURCES: dict[str, type] = {}


def register_source(cls: type) -> type:
    """Class decorator that registers a Source subclass under its ``name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a class attribute `name`")
    SOURCES[name] = cls
    return cls


def resolve_source(source: "Source | str", **kwargs) -> Source:
    """Resolve a Source instance from either an instance or its registered name."""
    if isinstance(source, str):
        if source not in SOURCES:
            available = sorted(SOURCES.keys())
            raise SourceResolutionError(
                f"Unknown source name: {source!r}. Available sources: {available}"
            )
        return SOURCES[source](**kwargs)
    return source
