"""``project.network.exporter`` -- every way of getting a network *out of* a project."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aequilibrae.project.network.network import Network

logger = logging.getLogger(__name__)


class Exporter:
    """Network export entry points, reached as ``project.network.exporter``.

    .. code-block:: python

        >>> project.network.exporter.gmns(folder)          # doctest: +SKIP
        >>> project.network.exporter.geo_parquet(folder)   # doctest: +SKIP
    """

    def __init__(self, network: "Network"):
        self._network = network
        self._project = network.project

    def gmns(self, path: str) -> None:
        """Export the project network to GMNS format.

        :Arguments:
            **path** (:obj:`str`): Folder the GMNS tables are written to
        """
        from aequilibrae.project.network.gmns_exporter import GMNSExporter

        GMNSExporter(self._network, path).doWork()
        logger.info("Network exported successfully")

    def geo_parquet(self, path: str) -> tuple:
        """Export the links and nodes tables as GeoParquet.

        Writes ``links.parquet`` and ``nodes.parquet`` into ``path``, preserving
        geometry and every attribute column, so the network round-trips into any
        GeoParquet-aware tool (GeoPandas, DuckDB, QGIS).

        :Arguments:
            **path** (:obj:`str`): Folder the two files are written to. Created if missing

        :Returns:
            :obj:`tuple`: The ``(links_path, nodes_path)`` that were written
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        written = []
        for name, table in (("links", self._network.links), ("nodes", self._network.nodes)):
            gdf = table.data
            target = folder / f"{name}.parquet"
            gdf.to_parquet(target)
            logger.info(f"Exported {len(gdf)} {name} to {target}")
            written.append(target)

        return tuple(written)
