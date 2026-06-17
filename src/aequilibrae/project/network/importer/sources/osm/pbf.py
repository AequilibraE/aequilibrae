"""``OSMPbfSource``: OSM via local .osm.pbf using pyrosm.

Implemented in PR 3. Placeholder until then.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ...download_cache import DownloadCache
from ...ir import RoutableNetwork
from ..base import register_source


@register_source
class OSMPbfSource:
    name: ClassVar[str] = "osm-pbf"
    required_extras: ClassVar[tuple[str, ...]] = ("pyrosm",)

    def __init__(self, *, pbf_path: str | Path, **kwargs):
        self.pbf_path = Path(pbf_path)
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> RoutableNetwork:
        from .impl import acquire_pbf

        return acquire_pbf(
            pbf_path=self.pbf_path, modes=modes, download_cache=download_cache, **self.kwargs
        )
