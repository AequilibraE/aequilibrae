"""Overture Maps source (cloud only).

The DuckDB and pyarrow-parquet backends were evaluated and dropped to keep the
API single-pathway. Power users wanting offline re-imports can construct a
``GeoDataFrameSource`` from the parquet payloads stored in
``<project>/downloaded data/overture-cloud/...``.
"""

from aequilibrae.project.network.importer.sources.overture.cloud import OvertureCloudSource

__all__ = ["OvertureCloudSource"]
