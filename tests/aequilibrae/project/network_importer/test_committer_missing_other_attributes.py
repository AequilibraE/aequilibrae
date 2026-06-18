"""When ``other_attributes`` is missing, the importer must fail with a clear message."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError


def test_missing_other_attributes_raises(empty_project):
    # Simulate an old project file by renaming other_attributes away. Drop the
    # network triggers first because they validate cross-table state on every
    # schema change.
    from aequilibrae.project.project_creation import remove_triggers

    with empty_project.db_connection_spatial as conn:
        remove_triggers(conn, "network")
        conn.execute("ALTER TABLE links RENAME COLUMN other_attributes TO _dropped_other")

    nodes = gpd.GeoDataFrame(
        {
            "node_id": [10000],
            "geometry": [Point(0, 0)],
            "modes": ["c"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1],
            "a_node": [10000],
            "b_node": [10000],
            "direction": [0],
            "modes": ["c"],
            "link_type": ["residential"],
            "distance": [111000.0],
            "geometry": [LineString([(0, 0), (0, 1)])],
        },
        crs="EPSG:4326",
    )

    with pytest.raises(ImporterError, match="other_attributes"):
        empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)
