"""``OSMOverpassSource``: OSM via Overpass + osmnx (place name / bbox / polygon).

Implemented in PR 3. Placeholder until then.
"""

from __future__ import annotations

from typing import ClassVar

from ...download_cache import DownloadCache
from ...exceptions import ImporterError
from ...ir import RoutableNetwork
from ..base import register_source


@register_source
class OSMOverpassSource:
    name: ClassVar[str] = "osm-overpass"
    required_extras: ClassVar[tuple[str, ...]] = ("osmnx",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> RoutableNetwork:
        from .impl import acquire_overpass

        return acquire_overpass(modes=modes, download_cache=download_cache, **self.kwargs)
