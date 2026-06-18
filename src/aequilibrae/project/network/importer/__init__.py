"""Pluggable network-acquisition framework.

See ``.kilo/plans/replace-osm-importer.md`` for the design document.
"""

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.importer import NetworkImporter
from aequilibrae.project.network.importer.simplifiers.base import (
    SIMPLIFIERS,
    Simplifier,
    register_simplifier,
)
from aequilibrae.project.network.importer.sources.base import (
    SOURCES,
    Source,
    register_source,
)
from aequilibrae.project.network.importer.staged_network import StagedNetwork

# Trigger source / simplifier registration via their decorators
from aequilibrae.project.network.importer.simplifiers import (  # noqa: F401
    neatnet_simplifier as _neatnet_simplifier,
)
from aequilibrae.project.network.importer.simplifiers import (  # noqa: F401
    osmnx_simplifier as _osmnx_simplifier,
)
from aequilibrae.project.network.importer.sources.generic import (  # noqa: F401
    file as _file_source,
)
from aequilibrae.project.network.importer.sources.generic import (  # noqa: F401
    geodataframe as _geodataframe_source,
)
from aequilibrae.project.network.importer.sources.generic import (  # noqa: F401
    gmns as _gmns_source,
)
from aequilibrae.project.network.importer.sources.osm import (  # noqa: F401
    overpass as _osm_overpass_source,
)
from aequilibrae.project.network.importer.sources.osm import (  # noqa: F401
    pbf as _osm_pbf_source,
)
from aequilibrae.project.network.importer.sources.overture import (  # noqa: F401
    cloud as _overture_cloud_source,
)

__all__ = [
    "StagedNetwork",
    "ImporterError",
    "NetworkImporter",
    "DownloadCache",
    "Source",
    "SOURCES",
    "register_source",
    "Simplifier",
    "SIMPLIFIERS",
    "register_simplifier",
]
