"""Pluggable network-acquisition framework.

See ``.kilo/plans/replace-osm-importer.md`` for the design document.
"""

from .download_cache import DownloadCache
from .exceptions import ImporterError
from .importer import NetworkImporter
from .simplifiers.base import SIMPLIFIERS, Simplifier, register_simplifier
from .sources.base import SOURCES, Source, register_source
from .staged_network import StagedNetwork

# Trigger source / simplifier registration via their decorators
from .simplifiers import neatnet_simplifier as _neatnet_simplifier  # noqa: F401
from .simplifiers import osmnx_simplifier as _osmnx_simplifier  # noqa: F401
from .sources.generic import file as _file_source  # noqa: F401
from .sources.generic import geodataframe as _geodataframe_source  # noqa: F401
from .sources.generic import gmns as _gmns_source  # noqa: F401
from .sources.osm import overpass as _osm_overpass_source  # noqa: F401
from .sources.osm import pbf as _osm_pbf_source  # noqa: F401
from .sources.overture import cloud as _overture_cloud_source  # noqa: F401

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
