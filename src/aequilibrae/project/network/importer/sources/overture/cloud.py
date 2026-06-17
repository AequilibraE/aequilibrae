"""``OvertureCloudSource``: Overture Maps transportation theme via the official client.

Implemented in PR 5. Placeholder until then.
"""

from __future__ import annotations

from typing import ClassVar

from ...download_cache import DownloadCache
from ...ir import RoutableNetwork
from ..base import register_source


@register_source
class OvertureCloudSource:
    name: ClassVar[str] = "overture-cloud"
    required_extras: ClassVar[tuple[str, ...]] = ("overturemaps",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def acquire(self, *, modes, download_cache: DownloadCache) -> RoutableNetwork:
        from .impl import acquire_cloud

        return acquire_cloud(modes=modes, download_cache=download_cache, **self.kwargs)
