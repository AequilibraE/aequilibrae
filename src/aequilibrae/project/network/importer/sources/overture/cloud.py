
from typing import ClassVar

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.staged_network import StagedNetwork


class OvertureCloudSource:
    name: ClassVar[str] = "overture-cloud"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from aequilibrae.project.network.importer.sources.overture.impl import acquire_cloud

        return acquire_cloud(modes=modes, download_cache=download_cache, **self.kwargs)
