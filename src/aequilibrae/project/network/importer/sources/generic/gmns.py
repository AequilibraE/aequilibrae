"""``GMNSSource``: registered for the registry; delegates to the legacy GMNSBuilder.

Rather than re-implement the GMNS parser inside the new framework, we keep
the well-tested legacy ``GMNSBuilder`` and use ``Network.create_from_gmns``
as the public entry point. ``GMNSSource`` exists in the registry so that
``import_network('gmns', ...)`` resolves successfully.
"""

from pathlib import Path
from typing import ClassVar

from ...download_cache import DownloadCache
from ...exceptions import ImporterError
from ...staged_network import StagedNetwork
from ..base import register_source


@register_source
class GMNSSource:
    name: ClassVar[str] = "gmns"
    required_extras: ClassVar[tuple] = ()

    def __init__(
        self,
        *,
        link_file_path,
        node_file_path,
        use_group_path=None,
        geometry_path=None,
        srid: int = 4326,
    ):
        self.link_file_path = Path(link_file_path)
        self.node_file_path = Path(node_file_path)
        self.use_group_path = Path(use_group_path) if use_group_path else None
        self.geometry_path = Path(geometry_path) if geometry_path else None
        self.srid = srid

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        raise ImporterError(
            "Acquiring a StagedNetwork from a GMNS bundle via the new framework "
            "is not yet implemented. Use Network.create_from_gmns() for now."
        )
