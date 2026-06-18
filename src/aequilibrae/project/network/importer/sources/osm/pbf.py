"""``OSMPbfSource``: OSM via a local .osm.pbf file using pyrosm."""

from pathlib import Path
from typing import ClassVar

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.sources.base import register_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@register_source
class OSMPbfSource:
    name: ClassVar[str] = "osm-pbf"
    required_extras: ClassVar[tuple] = ("pyrosm",)

    def __init__(self, *, pbf_path, **kwargs):
        self.pbf_path = Path(pbf_path)
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from aequilibrae.project.network.importer.sources.osm.impl import acquire_pbf

        return acquire_pbf(pbf_path=self.pbf_path, modes=modes, download_cache=download_cache, **self.kwargs)
