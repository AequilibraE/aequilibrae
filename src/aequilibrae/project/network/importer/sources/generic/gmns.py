"""``GMNSSource``: registered for the registry; delegates to the legacy
``GMNSBuilder`` which already lives in ``aequilibrae.project.network.gmns_builder``.

Note: rather than re-implement the GMNS parser inside the new framework, we
keep the well-tested legacy ``GMNSBuilder`` and use ``Network.create_from_gmns``
as the public entry point. ``GMNSSource`` exists in the registry so that
``import_network('gmns', ...)`` works for callers who prefer the unified API.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ...download_cache import DownloadCache
from ...exceptions import ImporterError
from ...ir import RoutableNetwork
from ..base import register_source


@register_source
class GMNSSource:
    name: ClassVar[str] = "gmns"
    required_extras: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        link_file_path: str | Path,
        node_file_path: str | Path,
        use_group_path: str | Path | None = None,
        geometry_path: str | Path | None = None,
        srid: int = 4326,
    ):
        self.link_file_path = Path(link_file_path)
        self.node_file_path = Path(node_file_path)
        self.use_group_path = Path(use_group_path) if use_group_path else None
        self.geometry_path = Path(geometry_path) if geometry_path else None
        self.srid = srid

    def acquire(self, *, modes, download_cache: DownloadCache) -> RoutableNetwork:
        raise ImporterError(
            "Acquiring a RoutableNetwork from a GMNS bundle via the new framework "
            "is not yet implemented. Use Network.create_from_gmns() for now."
        )
