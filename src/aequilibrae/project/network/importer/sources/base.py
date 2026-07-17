from typing import ClassVar, Protocol, runtime_checkable

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@runtime_checkable
class Source(Protocol):
    name: ClassVar[str]

    def acquire(self, *, modes: tuple, download_cache: DownloadCache) -> StagedNetwork: ...


def _sources() -> dict:
    from aequilibrae.project.network.importer.sources.osm.overpass import OSMOverpassSource
    from aequilibrae.project.network.importer.sources.osm.pbf import OSMPbfSource
    from aequilibrae.project.network.importer.sources.overture.cloud import OvertureCloudSource

    return {
        OSMOverpassSource.name: OSMOverpassSource,
        OSMPbfSource.name: OSMPbfSource,
        OvertureCloudSource.name: OvertureCloudSource,
    }


SOURCES = _sources()


def resolve_source(source, **kwargs) -> Source:
    if isinstance(source, str):
        sources = _sources()
        if source not in sources:
            available = sorted(sources.keys())
            raise SourceResolutionError(f"Unknown source name: {source!r}. Available sources: {available}")
        return sources[source](**kwargs)
    if kwargs:
        raise SourceResolutionError(
            f"Keyword arguments {sorted(kwargs)} only apply when the source is given by name; "
            "pass them to the source object's constructor instead"
        )
    return source
