"""Download Overture building footprints for neatnet's exclusion mask."""

import geopandas as gpd
import logging
from dataclasses import dataclass
from typing import Optional

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)

# Maximum bounding-box span (degrees, max of width/height) for which an
# automatic building download is allowed. Larger extents would pull millions of
# polygons and can exhaust memory / hit rate limits, so we skip them.
_MAX_BUILDINGS_BBOX_SPAN_DEGREES = 0.5


@dataclass
class BuildingMaskResult:
    gdf: Optional[gpd.GeoDataFrame]
    status: str
    attempted: bool
    retries: int
    reason: str = ""

    def as_meta(self) -> dict:
        return {
            "building_mask_status": self.status,
            "building_mask_attempted": self.attempted,
            "building_mask_retries": self.retries,
            "building_mask_reason": self.reason,
        }


def fetch_building_footprints(
    net: StagedNetwork,
    download_cache: DownloadCache,
    *,
    max_bbox_span_degrees: float = _MAX_BUILDINGS_BBOX_SPAN_DEGREES,
) -> BuildingMaskResult:
    """Download Overture building footprints for neatnet's exclusion mask.

    Retries once; falls back to no mask on failure.
    """
    bbox = tuple(net.links.total_bounds)
    span = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))

    if span > max_bbox_span_degrees:
        logger.warning(
            "Skipping building-footprint download: bounding-box span %.3f deg exceeds the limit of %.3f deg. "
            "Downloading buildings over an area this large risks running out of memory.",
            span,
            max_bbox_span_degrees,
        )
        return BuildingMaskResult(gdf=None, status="skipped", attempted=False, retries=0, reason="bbox_guard")

    overturemaps = require("overturemaps", feature="building footprint download for neatnet")
    logger.info(f"Downloading building footprints from Overture Maps for bbox={bbox}")

    from aequilibrae.project.network.importer.sources.overture.impl import get_latest_overture_version, table_to_gdf

    last_reason = ""
    for attempt in range(2):
        try:
            release = get_latest_overture_version()
            rbr = overturemaps.record_batch_reader("building", bbox=bbox, release=release)
            if rbr is None:
                last_reason = "no_reader"
                continue
            table = rbr.read_all()
            if table.num_rows == 0:
                last_reason = "zero_rows"
                continue

            logger.info(f"Downloaded {table.num_rows} building footprints from Overture Maps (release={release})")
            buildings_gdf = table_to_gdf(table)
            download_cache.write_geoparquet("buildings.parquet", buildings_gdf)
            return BuildingMaskResult(gdf=buildings_gdf, status="downloaded", attempted=True, retries=attempt)
        except Exception as exc:
            last_reason = exc.__class__.__name__
            if attempt == 0:
                logger.warning("Building-footprint download failed on first attempt (%s); retrying once", exc)

    logger.warning(
        "Building-footprint download failed after retry (reason=%s); proceeding without an exclusion mask",
        last_reason or "unknown",
    )
    return BuildingMaskResult(gdf=None, status="fallback", attempted=True, retries=1, reason=last_reason or "empty")
