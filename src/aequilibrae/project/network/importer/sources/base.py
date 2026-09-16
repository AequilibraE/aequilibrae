from functools import partial

from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.sources.osm.impl import acquire_overpass, acquire_pbf
from aequilibrae.project.network.importer.sources.overture.impl import acquire_cloud

SOURCES = {
    "osm-overpass": acquire_overpass,
    "osm-pbf": acquire_pbf,
    "overture-cloud": acquire_cloud,
}


def resolve_source(source, **kwargs) -> tuple:
    """Return ``(name, acquire)`` for a source given by name or as a duck-typed object."""
    if isinstance(source, str):
        if source not in SOURCES:
            raise SourceResolutionError(f"Unknown source name: {source!r}. Available sources: {sorted(SOURCES)}")
        return source, partial(SOURCES[source], **kwargs)
    if kwargs:
        raise SourceResolutionError(
            f"Keyword arguments {sorted(kwargs)} only apply when the source is given by name; "
            "pass them to the source object's constructor instead"
        )
    return source.name, source.acquire
