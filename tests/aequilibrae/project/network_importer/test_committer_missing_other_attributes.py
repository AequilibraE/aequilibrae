import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.db_writer import LINK_JSON_DESCRIPTION, SpatialiteWriter
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.db_utils import list_columns
from aequilibrae.utils.sqlite_utils import without_triggers


def _network(**link_columns) -> StagedNetwork:
    nodes = gpd.GeoDataFrame(
        {"node_id": [100000], "geometry": [Point(0, 0)], "modes": ["c"]},
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1],
            "a_node": [100000],
            "b_node": [100000],
            "direction": [0],
            "modes": ["c"],
            "link_type": ["residential"],
            "distance": [111000.0],
            "geometry": [LineString([(0, 0), (0, 1)])],
            **link_columns,
        },
        crs="EPSG:4326",
    )
    return StagedNetwork(nodes=nodes, links=links)


def test_link_other_attributes_is_added_to_keep_attributes_without_a_field(empty_project):
    """Test that importing links with attributes that have no field adds other_attributes to keep them."""
    SpatialiteWriter(empty_project).write(_network(surface=["asphalt"]))

    with empty_project.db_connection as conn:
        payload = conn.execute("SELECT other_attributes FROM links_full WHERE link_id = 1").fetchone()[0]
    assert json.loads(payload) == {"surface": "asphalt"}
    assert "other_attributes" in empty_project.tables.links.columns
    assert empty_project.tables.links.fields.get_description("other_attributes") == LINK_JSON_DESCRIPTION


def test_link_other_attributes_is_not_added_without_attributes_to_keep(empty_project):
    """Test that importing links whose attributes all have fields of their own does not add other_attributes."""
    SpatialiteWriter(empty_project).write(_network(speed_ab=[50]))

    with empty_project.db_connection as conn:
        assert "other_attributes" not in list_columns(conn, "links_full")
        assert conn.execute("SELECT speed_ab FROM links_full WHERE link_id = 1").fetchone() == (50,)


def test_missing_node_other_attributes_raises(empty_project):
    """Test that importing into a project whose nodes have no other_attributes field asks for a new project."""
    with empty_project.db_connection as conn, without_triggers(conn):
        conn.execute("ALTER TABLE nodes RENAME COLUMN other_attributes TO _dropped_other")

    with pytest.raises(ImporterError, match="You must create a new empty project"):
        SpatialiteWriter(empty_project).write(_network())
