"""``OvertureCloudSource``: Overture Maps transportation theme via the official client."""

from typing import ClassVar

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.sources.base import register_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@register_source
class OvertureCloudSource:
    name: ClassVar[str] = "overture-cloud"
    required_extras: ClassVar[tuple] = ("overturemaps",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from aequilibrae.project.network.importer.sources.overture.impl import acquire_cloud

        return acquire_cloud(modes=modes, download_cache=download_cache, **self.kwargs)
