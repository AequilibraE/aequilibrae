"""Download Overture building footprints for neatnet's exclusion mask."""

import geopandas as gpd
import logging
from typing import Optional

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)


def fetch_building_footprints(
    net: StagedNetwork,
    download_cache: DownloadCache,
) -> Optional[gpd.GeoDataFrame]:
    """Download buildings covering the current staged network extent."""
    overturemaps = require("overturemaps", feature="building footprint download for neatnet")

    bbox = tuple(net.links.total_bounds)
    logger.info(f"Downloading building footprints from Overture Maps for bbox={bbox}")

    from aequilibrae.project.network.importer.sources.overture.impl import (
        get_latest_overture_version,
    )

    release = get_latest_overture_version()

    rbr = overturemaps.record_batch_reader("building", bbox=bbox, release=release)
    if rbr is None:
        logger.warning(
            "Overture Maps returned no building data for the requested bbox; proceeding without exclusion mask"
        )
        return None

    table = rbr.read_all()
    if table.num_rows == 0:
        logger.warning(
            "Overture Maps returned zero buildings for the requested bbox; proceeding without exclusion mask"
        )
        return None

    logger.info(f"Downloaded {table.num_rows} building footprints from Overture Maps (release={release})")
    buildings_gdf = _table_to_gdf(table)
    download_cache.write_geoparquet("buildings.parquet", buildings_gdf)
    logger.info(f"Building footprints GeoDataFrame: {len(buildings_gdf)} rows")
    return buildings_gdf


def _table_to_gdf(table) -> gpd.GeoDataFrame:
    from shapely import from_wkb

    df = table.to_pandas(use_threads=True)
    if "geometry" not in df.columns:
        raise ValueError("Overture buildings table must contain a geometry column")
    df["geometry"] = df["geometry"].apply(lambda v: from_wkb(v) if v is not None else None)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
