"""``FileSource``: read links/nodes from disk via geopandas (PR 6 placeholder)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ...download_cache import DownloadCache
from ...ir import RoutableNetwork
from ..base import register_source
from .geodataframe import GeoDataFrameSource


@register_source
class FileSource:
    name: ClassVar[str] = "file"
    required_extras: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        links_path: str | Path,
        nodes_path: str | Path | None = None,
        layer_links: str | None = None,
        layer_nodes: str | None = None,
        column_mapping: dict | None = None,
    ):
        self.links_path = Path(links_path)
        self.nodes_path = Path(nodes_path) if nodes_path else None
        self.layer_links = layer_links
        self.layer_nodes = layer_nodes
        self.column_mapping = column_mapping or {}

    def acquire(self, *, modes, download_cache: DownloadCache) -> RoutableNetwork:
        import geopandas as gpd

        links = gpd.read_file(self.links_path, layer=self.layer_links) if self.layer_links \
            else gpd.read_file(self.links_path)
        if self.nodes_path is None:
            raise NotImplementedError(
                "FileSource currently requires both `links_path` and `nodes_path`. "
                "Auto-noding from links alone is a future enhancement."
            )
        nodes = gpd.read_file(self.nodes_path, layer=self.layer_nodes) if self.layer_nodes \
            else gpd.read_file(self.nodes_path)
        inner = GeoDataFrameSource(
            nodes=nodes, links=links, column_mapping=self.column_mapping
        )
        return inner.acquire(modes=modes, download_cache=download_cache)
