import geopandas as gpd
import json
import logging
from datetime import datetime, timezone
from typing import Sequence
from urllib.request import urlopen

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.sources.overture.schema_to_staged import build_staged_from_overture
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)

_STAC_CATALOG_URL = "https://stac.overturemaps.org/catalog.json"


def acquire_cloud(
    *,
    modes: Sequence[str],
    download_cache: DownloadCache,
    model_area,
) -> StagedNetwork:
    overturemaps = require("overturemaps", feature="Overture cloud download")

    if model_area is None:
        raise ImporterError("OvertureCloudSource requires a `model_area` Polygon")

    bbox = tuple(model_area.bounds)
    release = get_latest_overture_version()

    segments_table = _fetch_table(overturemaps, "segment", bbox, release=release)
    connectors_table = _fetch_table(overturemaps, "connector", bbox, release=release)

    download_cache.write_parquet("segments.parquet", segments_table)
    download_cache.write_parquet("connectors.parquet", connectors_table)
    download_cache.write_manifest(
        {
            "source": "overture-cloud",
            "backend": "overturemaps",
            "release": release,
            "bbox": list(bbox),
            "modes": list(modes),
            "segments_rows": segments_table.num_rows,
            "connectors_rows": connectors_table.num_rows,
        }
    )

    segments_gdf = _table_to_gdf(segments_table)
    connectors_gdf = _table_to_gdf(connectors_table)
    source_meta = {
        "source": "overture",
        "backend": "cloud",
        "source_url": _overture_url(release),
        "release": release,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return build_staged_from_overture(
        connectors=connectors_gdf,
        segments=segments_gdf,
        modes=modes,
        source_meta=source_meta,
    )


def get_latest_overture_version() -> str:
    with urlopen(_STAC_CATALOG_URL, timeout=30) as response:
        catalog = json.load(response)
    latest = catalog.get("latest")
    if not latest:
        raise ImporterError("Overture STAC catalog does not advertise a latest release")
    return str(latest)


def _fetch_table(overturemaps, theme_type: str, bbox, *, release):
    rbr = overturemaps.record_batch_reader(theme_type, bbox=bbox, release=release)
    if rbr is None:
        raise ImporterError(
            f"overturemaps returned no record batch reader for type={theme_type!r}; check connectivity and bbox={bbox}"
        )
    return rbr.read_all()


def _table_to_gdf(table) -> gpd.GeoDataFrame:
    from shapely import from_wkb

    df = table.to_pandas(use_threads=True)
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(lambda v: from_wkb(v) if v is not None else None)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _overture_url(release) -> str:
    return f"s3://overturemaps-us-west-2/release/{release}"
