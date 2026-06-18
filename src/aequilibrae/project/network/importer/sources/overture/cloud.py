"""``OvertureCloudSource``: Overture Maps transportation theme via the official client."""

from typing import ClassVar

from ...download_cache import DownloadCache
from ...staged_network import StagedNetwork
from ..base import register_source


@register_source
class OvertureCloudSource:
    name: ClassVar[str] = "overture-cloud"
    required_extras: ClassVar[tuple] = ("overturemaps",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        from .impl import acquire_cloud

        return acquire_cloud(modes=modes, download_cache=download_cache, **self.kwargs)
