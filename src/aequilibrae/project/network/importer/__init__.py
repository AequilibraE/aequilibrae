"""Pluggable network-acquisition framework.

See ``.kilo/plans/replace-osm-importer.md`` for the design document.
"""

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.importer import NetworkImporter
from aequilibrae.project.network.importer.simplifiers.base import SIMPLIFIERS, Simplifier, register_simplifier
from aequilibrae.project.network.importer.sources.base import SOURCES, Source, register_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork

# Trigger source / simplifier registration via their decorators
from aequilibrae.project.network.importer.simplifiers import neatnet_simplifier as _neatnet_simplifier  # noqa: F401
from aequilibrae.project.network.importer.simplifiers import osmnx_simplifier as _osmnx_simplifier  # noqa: F401
from aequilibrae.project.network.importer.sources.generic import file as _file_source  # noqa: F401
from aequilibrae.project.network.importer.sources.generic import geodataframe as _geodataframe_source  # noqa: F401
from aequilibrae.project.network.importer.sources.generic import gmns as _gmns_source  # noqa: F401
from aequilibrae.project.network.importer.sources.osm import overpass as _osm_overpass_source  # noqa: F401
from aequilibrae.project.network.importer.sources.osm import pbf as _osm_pbf_source  # noqa: F401
from aequilibrae.project.network.importer.sources.overture import cloud as _overture_cloud_source  # noqa: F401

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
