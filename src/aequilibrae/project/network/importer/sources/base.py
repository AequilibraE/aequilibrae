"""``Source`` protocol + string-name registry."""

from typing import ClassVar, Protocol, runtime_checkable

from ..download_cache import DownloadCache
from ..exceptions import SourceResolutionError
from ..staged_network import StagedNetwork


@runtime_checkable
class Source(Protocol):
    """Anything that can produce a ``StagedNetwork``."""

    name: ClassVar[str]
    required_extras: ClassVar[tuple]

    def acquire(
        self,
        *,
        modes: tuple,
        download_cache: DownloadCache,
    ) -> StagedNetwork:
        ...


SOURCES: dict = {}


def register_source(cls: type) -> type:
    """Class decorator that registers a Source subclass under its ``name``."""
    SOURCES[cls.name] = cls
    return cls


def resolve_source(source, **kwargs) -> Source:
    """Resolve a Source instance from either an instance or its registered name."""
    if isinstance(source, str):
        if source not in SOURCES:
            available = sorted(SOURCES.keys())
            raise SourceResolutionError(
                f"Unknown source name: {source!r}. Available sources: {available}"
            )
        return SOURCES[source](**kwargs)
    return source
