"""Adapter source: user supplies ``(nodes_gdf, links_gdf)`` directly.

Validates the supplied GDFs against the staged-network invariants, optionally
renames columns, reprojects to EPSG:4326, and emits the staged network.
Writes nothing to the download cache.
"""

import logging
from datetime import datetime, timezone
from typing import ClassVar

import geopandas as gpd

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import StagedNetworkValidationError
from aequilibrae.project.network.importer.sources.base import register_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork

logger = logging.getLogger(__name__)


@register_source
class GeoDataFrameSource:
    name: ClassVar[str] = "geodataframe"
    required_extras: ClassVar[tuple] = ()

    def __init__(
        self,
        *,
        nodes: gpd.GeoDataFrame,
        links: gpd.GeoDataFrame,
        crs=None,
        column_mapping=None,
    ):
        if nodes is None or links is None:
            raise ValueError("GeoDataFrameSource requires both `nodes` and `links` GeoDataFrames")
        self.nodes_input = nodes
        self.links_input = links
        self.crs = crs
        self.column_mapping = dict(column_mapping or {})

    def acquire(self, *, modes, download_cache: DownloadCache) -> StagedNetwork:
        nodes = self._prepare(self.nodes_input)
        links = self._prepare(self.links_input)

        if self.column_mapping:
            nodes = nodes.rename(columns=self.column_mapping)
            links = links.rename(columns=self.column_mapping)

        source_meta = {
            "source": "geodataframe",
            "backend": "user",
            "source_url": "<in-memory GeoDataFrames>",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        net = StagedNetwork(nodes=nodes, links=links, source_meta=source_meta)
        net.validate()
        return net

    def _prepare(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise StagedNetworkValidationError(
                f"GeoDataFrameSource expects a geopandas.GeoDataFrame; got {type(gdf).__name__}"
            )
        if gdf.crs is None and self.crs is not None:
            gdf = gdf.set_crs(self.crs)
        if gdf.crs is None:
            raise StagedNetworkValidationError(
                "GeoDataFrame has no CRS and no `crs` was supplied to GeoDataFrameSource"
            )
        return gdf.to_crs("EPSG:4326")
