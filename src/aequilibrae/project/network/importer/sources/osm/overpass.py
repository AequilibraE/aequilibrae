"""``OSMOverpassSource``: OSM via Overpass + osmnx."""

from typing import ClassVar

from ...download_cache import DownloadCache
from ...staged_network import StagedNetwork
from ..base import register_source


@register_source
class OSMOverpassSource:
    name: ClassVar[str] = "osm-overpass"
    required_extras: ClassVar[tuple] = ("osmnx",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from .impl import acquire_overpass

        return acquire_overpass(modes=modes, download_cache=download_cache, **self.kwargs)
