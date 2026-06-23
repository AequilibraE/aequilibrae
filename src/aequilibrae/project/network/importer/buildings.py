"""Download Overture building footprints for neatnet's exclusion mask."""

import geopandas as gpd
import logging
from dataclasses import dataclass
from time import perf_counter
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
    elapsed_s: float
    bbox_span_degrees: float
    cache_written: bool
    reason: str = ""

    def as_meta(self) -> dict:
        return {
            "building_mask_status": self.status,
            "building_mask_attempted": self.attempted,
            "building_mask_retries": self.retries,
            "building_mask_elapsed_s": round(self.elapsed_s, 6),
            "building_mask_bbox_span_degrees": round(self.bbox_span_degrees, 6),
            "building_mask_cache_written": self.cache_written,
            "building_mask_reason": self.reason,
        }


def fetch_building_footprints(
    net: StagedNetwork,
    download_cache: DownloadCache,
    *,
    enabled: bool = True,
    max_bbox_span_degrees: float = _MAX_BUILDINGS_BBOX_SPAN_DEGREES,
) -> BuildingMaskResult:
    """Optionally download buildings covering the current staged network extent.

    Buildings are enabled by default for neatnet because they can improve the
    exclusion mask and simplify around built form more cleanly. The import must
    still remain resilient, so the fetch is retried once and then falls back to
    "no mask" with a warning if Overture is unavailable or empty.
    """
    bbox = tuple(net.links.total_bounds)
    span = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    start = perf_counter()

    if not enabled:
        logger.info("Building-footprint download is disabled; proceeding without an exclusion mask")
        return BuildingMaskResult(
            gdf=None,
            status="disabled",
            attempted=False,
            retries=0,
            elapsed_s=perf_counter() - start,
            bbox_span_degrees=span,
            cache_written=False,
        )

    if span > max_bbox_span_degrees:
        logger.warning(
            "Skipping building-footprint download: bounding-box span %.3f deg exceeds the limit of %.3f deg. "
            "Downloading buildings over an area this large risks running out of memory.",
            span,
            max_bbox_span_degrees,
        )
        return BuildingMaskResult(
            gdf=None,
            status="skipped",
            attempted=False,
            retries=0,
            elapsed_s=perf_counter() - start,
            bbox_span_degrees=span,
            cache_written=False,
            reason="bbox_guard",
        )

    overturemaps = require("overturemaps", feature="building footprint download for neatnet")
    logger.info(f"Downloading building footprints from Overture Maps for bbox={bbox}")

    from aequilibrae.project.network.importer.sources.overture.impl import get_latest_overture_version

    release = get_latest_overture_version()

    last_reason = ""
    for attempt in range(2):
        try:
            rbr = overturemaps.record_batch_reader("building", bbox=bbox, release=release)
            if rbr is None:
                last_reason = "no_reader"
                continue
            table = rbr.read_all()
            if table.num_rows == 0:
                last_reason = "zero_rows"
                continue

            logger.info(f"Downloaded {table.num_rows} building footprints from Overture Maps (release={release})")
            buildings_gdf = _table_to_gdf(table)
            download_cache.write_geoparquet("buildings.parquet", buildings_gdf)
            logger.info(f"Building footprints GeoDataFrame: {len(buildings_gdf)} rows")
            return BuildingMaskResult(
                gdf=buildings_gdf,
                status="downloaded",
                attempted=True,
                retries=attempt,
                elapsed_s=perf_counter() - start,
                bbox_span_degrees=span,
                cache_written=True,
            )
        except Exception as exc:
            last_reason = exc.__class__.__name__
            if attempt == 0:
                logger.warning("Building-footprint download failed on first attempt (%s); retrying once", exc)
                continue
            logger.warning(
                "Building-footprint download failed after retry (%s); proceeding without an exclusion mask",
                exc,
            )
            return BuildingMaskResult(
                gdf=None,
                status="fallback",
                attempted=True,
                retries=1,
                elapsed_s=perf_counter() - start,
                bbox_span_degrees=span,
                cache_written=False,
                reason=last_reason,
            )

    logger.warning(
        "Overture Maps returned no usable building data for the requested bbox "
        "(reason=%s); proceeding without exclusion mask",
        last_reason or "unknown",
    )
    return BuildingMaskResult(
        gdf=None,
        status="fallback",
        attempted=True,
        retries=1,
        elapsed_s=perf_counter() - start,
        bbox_span_degrees=span,
        cache_written=False,
        reason=last_reason or "empty",
    )


def _table_to_gdf(table) -> gpd.GeoDataFrame:
    from shapely import from_wkb

    df = table.to_pandas(use_threads=True)
    if "geometry" not in df.columns:
        raise ValueError("Overture buildings table must contain a geometry column")
    df["geometry"] = df["geometry"].apply(lambda v: from_wkb(v) if v is not None else None)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
