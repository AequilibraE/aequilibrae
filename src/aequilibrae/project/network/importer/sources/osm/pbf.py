"""``OSMPbfSource``: OSM via a local .osm.pbf file using pyrosm."""

from pathlib import Path
from typing import ClassVar

from ...download_cache import DownloadCache
from ...staged_network import StagedNetwork
from ..base import register_source


@register_source
class OSMPbfSource:
    name: ClassVar[str] = "osm-pbf"
    required_extras: ClassVar[tuple] = ("pyrosm",)

    def __init__(self, *, pbf_path, **kwargs):
        self.pbf_path = Path(pbf_path)
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from .impl import acquire_pbf

        return acquire_pbf(
            pbf_path=self.pbf_path, modes=modes, download_cache=download_cache, **self.kwargs
        )
