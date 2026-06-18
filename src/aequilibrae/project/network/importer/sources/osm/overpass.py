"""``OSMOverpassSource``: OSM via Overpass + osmnx."""

from typing import ClassVar

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.sources.base import register_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@register_source
class OSMOverpassSource:
    name: ClassVar[str] = "osm-overpass"
    required_extras: ClassVar[tuple] = ("osmnx",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from aequilibrae.project.network.importer.sources.osm.impl import acquire_overpass

        return acquire_overpass(modes=modes, download_cache=download_cache, **self.kwargs)
