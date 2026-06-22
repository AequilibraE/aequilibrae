
from pathlib import Path
from typing import ClassVar

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.staged_network import StagedNetwork


class OSMPbfSource:
    name: ClassVar[str] = "osm-pbf"

    def __init__(self, *, pbf_path):
        self.pbf_path = Path(pbf_path)

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from aequilibrae.project.network.importer.sources.osm.impl import acquire_pbf

        return acquire_pbf(pbf_path=self.pbf_path, modes=modes, download_cache=download_cache)
