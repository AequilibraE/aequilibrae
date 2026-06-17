"""Implementation backbone for Overture cloud source.

Always uses the official ``overturemaps`` Python client. The two non-cloud
backends (DuckDB, pyarrow.dataset) were dropped per plan §1.3 rule 7.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Sequence

import geopandas as gpd

from aequilibrae.utils.optional_dependency import require

from ...download_cache import DownloadCache
from ...exceptions import ImporterError
from ...ir import RoutableNetwork
from .schema_to_ir import build_ir_from_overture

logger = logging.getLogger(__name__)


def acquire_cloud(
    *,
    modes: Sequence[str],
    download_cache: DownloadCache,
    model_area,
    release: str | None = None,
) -> RoutableNetwork:
    overturemaps = require("overturemaps", feature="Overture cloud download")

    if model_area is None:
        raise ImporterError("OvertureCloudSource requires a `model_area` Polygon")

    bbox = tuple(model_area.bounds)

    # ---- Fetch raw tables. We use record_batch_reader so we can persist the
    # raw response as parquet before any conversion (plan §4.6).
    segments_table = _fetch_table(
        overturemaps, "segment", bbox, release=release
    )
    connectors_table = _fetch_table(
        overturemaps, "connector", bbox, release=release
    )

    download_cache.write_table("segments.parquet", segments_table)
    download_cache.write_table("connectors.parquet", connectors_table)
    download_cache.write_manifest({
        "source": "overture-cloud",
        "backend": "overturemaps",
        "release": release,
        "bbox": list(bbox),
        "modes": list(modes),
        "segments_rows": segments_table.num_rows,
        "connectors_rows": connectors_table.num_rows,
    })

    # ---- Build the IR
    segments_gdf = _table_to_gdf(segments_table)
    connectors_gdf = _table_to_gdf(connectors_table)
    source_meta = {
        "source": "overture",
        "backend": "cloud",
        "source_url": _overture_url(release),
        "release": release,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return build_ir_from_overture(
        connectors=connectors_gdf,
        segments=segments_gdf,
        modes=modes,
        source_meta=source_meta,
    )


def _fetch_table(overturemaps, theme_type: str, bbox, *, release: str | None):
    """Read an Overture theme as a ``pyarrow.Table`` via the official client."""
    rbr = overturemaps.record_batch_reader(theme_type, bbox=bbox)
    if rbr is None:
        raise ImporterError(
            f"overturemaps returned no record batch reader for type={theme_type!r}; "
            f"check connectivity and bbox={bbox}"
        )
    return rbr.read_all()


def _table_to_gdf(table) -> gpd.GeoDataFrame:
    """Convert a pyarrow Table with a WKB ``geometry`` column to a GeoDataFrame."""
    import pyarrow as pa  # noqa: F401  (ensured by base install)
    import pandas as pd
    from shapely import from_wkb

    df = table.to_pandas(use_threads=True)
    if "geometry" in df.columns:
        # overturemaps emits WKB-encoded geometries on the table
        df["geometry"] = df["geometry"].apply(
            lambda v: from_wkb(v) if v is not None else None
        )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _overture_url(release: str | None) -> str:
    base = "s3://overturemaps-us-west-2/release"
    if release:
        return f"{base}/{release}"
    return f"{base}/<latest via STAC>"
