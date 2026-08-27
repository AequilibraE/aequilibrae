"""Tests for the pure-Python SpatiaLite function shim (no native extension involved)."""

import sqlite3

import pytest
import shapely

from aequilibrae.utils.gaia_geometry import gaia_to_shapely, make_point_blob
from aequilibrae.utils.spatialite_shim import (
    register_spatialite_functions,
    rewrite_spatialindex_sql,
    upgrade_legacy_spatialindex_triggers,
)


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    register_spatialite_functions(conn)
    conn.execute("SELECT InitSpatialMetaData()")
    yield conn
    conn.close()


@pytest.fixture
def spatial_table(conn):
    conn.execute("CREATE TABLE roads (road_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("SELECT AddGeometryColumn('roads', 'geometry', 4326, 'LINESTRING', 'XY', 1)")
    conn.execute("SELECT CreateSpatialIndex('roads', 'geometry')")
    conn.commit()  # so a trigger RAISE(ROLLBACK) cannot undo the table setup
    return conn


def test_scalar_functions(conn):
    c = conn.execute
    assert c("SELECT ST_X(MakePoint(1.5, -2.5, 4326))").fetchone()[0] == 1.5
    assert c("SELECT ST_Y(MakePoint(1.5, -2.5, 4326))").fetchone()[0] == -2.5
    assert c("SELECT Srid(MakePoint(1, 2, 4326))").fetchone()[0] == 4326
    assert c("SELECT AsText(GeomFromText('POINT(1 2)', 4326))").fetchone()[0] == "POINT(1 2)"

    wkb = shapely.LineString([(0, 0), (3, 4)]).wkb
    assert c("SELECT ST_AsBinary(GeomFromWKB(?, 4326))", (wkb,)).fetchone()[0] == wkb
    assert c("SELECT GLength(GeomFromWKB(?, 4326))", (wkb,)).fetchone()[0] == 5.0

    line = "GeomFromText('LINESTRING(0 0, 1 1, 2 0)', 4326)"
    assert gaia_to_shapely(c(f"SELECT StartPoint({line})").fetchone()[0]) == shapely.Point(0, 0)
    assert gaia_to_shapely(c(f"SELECT EndPoint({line})").fetchone()[0]) == shapely.Point(2, 0)

    moved = c(f"SELECT SetStartPoint({line}, MakePoint(9, 9, 4326))").fetchone()[0]
    assert list(gaia_to_shapely(moved).coords)[0] == (9, 9)

    multi = c("SELECT GeometryType(ST_Multi(GeomFromText('POLYGON((0 0,1 0,1 1,0 0))', 4326)))").fetchone()[0]
    assert multi == "MULTIPOLYGON"

    made = c("SELECT AsText(MakeLine(MakePoint(0, 0, 4326), MakePoint(1, 1, 4326)))").fetchone()[0]
    assert made == "LINESTRING(0 0, 1 1)"

    assert c("SELECT MbrMinX(GeomFromText('LINESTRING(0 5, 2 1)', 4326))").fetchone()[0] == 0
    assert c("SELECT MbrMaxY(GeomFromText('LINESTRING(0 5, 2 1)', 4326))").fetchone()[0] == 5

    # NULL propagation
    assert c("SELECT ST_X(NULL)").fetchone()[0] is None
    assert c("SELECT GeomFromWKB(NULL, 4326)").fetchone()[0] is None


def test_geodesic_length_in_metres(conn):
    # one degree of latitude on the WGS84 ellipsoid at the equator is ~110574 m
    d = conn.execute("SELECT GeodesicLength(GeomFromText('LINESTRING(0 0, 0 1)', 4326))").fetchone()[0]
    assert d == pytest.approx(110574.4, abs=1.0)


def test_geometry_constraint_triggers(spatial_table):
    line_wkb = shapely.LineString([(0, 0), (1, 1)]).wkb
    spatial_table.execute("INSERT INTO roads (road_id, geometry) VALUES (1, GeomFromWKB(?, 4326))", (line_wkb,))
    spatial_table.commit()

    with pytest.raises(sqlite3.IntegrityError, match="Geometry constraint"):
        spatial_table.execute(
            "INSERT INTO roads (road_id, geometry) VALUES (2, GeomFromWKB(?, 4326))", (shapely.Point(0, 0).wkb,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="Geometry constraint"):
        spatial_table.execute("INSERT INTO roads (road_id, geometry) VALUES (3, GeomFromWKB(?, 3857))", (line_wkb,))


def test_spatial_index_maintenance(spatial_table):
    c = spatial_table.execute
    c("INSERT INTO roads (road_id, geometry) VALUES (1, GeomFromText('LINESTRING(0 0, 2 2)', 4326))")
    c("INSERT INTO roads (road_id, geometry) VALUES (2, GeomFromText('LINESTRING(10 10, 12 12)', 4326))")

    rows = c("SELECT pkid, xmin, xmax FROM idx_roads_geometry ORDER BY pkid").fetchall()
    assert len(rows) == 2 and rows[0][1] <= 0 <= rows[0][2]

    hits = c("SELECT pkid FROM idx_roads_geometry WHERE xmin <= 1 AND xmax >= 1 AND ymin <= 1 AND ymax >= 1").fetchall()
    assert [r[0] for r in hits] == [1]

    c("UPDATE roads SET geometry = GeomFromText('LINESTRING(20 20, 22 22)', 4326) WHERE road_id = 1")
    hits = c("SELECT pkid FROM idx_roads_geometry WHERE xmin <= 1 AND xmax >= 1 AND ymin <= 1 AND ymax >= 1").fetchall()
    assert hits == []

    c("DELETE FROM roads WHERE road_id = 2")
    assert c("SELECT count(*) FROM idx_roads_geometry").fetchone()[0] == 1

    assert c("SELECT CheckSpatialIndex('roads', 'geometry')").fetchone()[0] == 1
    c("DELETE FROM idx_roads_geometry")  # corrupt the index
    assert c("SELECT CheckSpatialIndex('roads', 'geometry')").fetchone()[0] == 0
    assert c("SELECT RecoverSpatialIndex('roads', 'geometry')").fetchone()[0] == 1
    assert c("SELECT CheckSpatialIndex('roads', 'geometry')").fetchone()[0] == 1


def test_layer_statistics_and_extent(spatial_table):
    c = spatial_table.execute
    c("INSERT INTO roads (road_id, geometry) VALUES (1, GeomFromText('LINESTRING(0 0, 2 3)', 4326))")
    c("SELECT UpdateLayerStatistics('roads')")
    row = spatial_table.execute(
        "SELECT row_count, extent_min_x, extent_max_y FROM geometry_columns_statistics "
        "WHERE f_table_name = 'roads'"
    ).fetchone()
    assert row == (1, 0, 3)

    extent = gaia_to_shapely(c("SELECT GetLayerExtent('roads')").fetchone()[0])
    assert extent.bounds == (0, 0, 2, 3)

    c("SELECT InvalidateLayerStatistics('roads')")
    row = spatial_table.execute(
        "SELECT row_count FROM geometry_columns_statistics WHERE f_table_name = 'roads'"
    ).fetchone()
    assert row == (None,)


def test_rename_and_drop_table(spatial_table):
    c = spatial_table.execute
    c("INSERT INTO roads (road_id, geometry) VALUES (1, GeomFromText('LINESTRING(0 0, 2 3)', 4326))")
    c("SELECT RenameTable(NULL, 'roads', 'streets')")

    assert c("SELECT count(*) FROM streets").fetchone()[0] == 1
    assert c("SELECT count(*) FROM idx_streets_geometry").fetchone()[0] == 1
    assert c("SELECT f_table_name FROM geometry_columns").fetchone()[0] == "streets"

    # triggers must maintain the renamed index
    c("INSERT INTO streets (road_id, geometry) VALUES (2, GeomFromText('LINESTRING(5 5, 6 6)', 4326))")
    assert c("SELECT count(*) FROM idx_streets_geometry").fetchone()[0] == 2

    c("SELECT DropTable(NULL, 'streets')")
    remaining = {
        r[0] for r in c("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%streets%'").fetchall()
    }
    assert not remaining
    assert c("SELECT count(*) FROM geometry_columns").fetchone()[0] == 0


LEGACY_TRIGGER = """create trigger aequilibrae_new_link_a_node before insert on links
  when
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = StartPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
        search_frame = StartPoint(new.geometry)) OR
      nodes.node_id = new.a_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            StartPoint(new.geometry));
  END"""


def test_rewrite_spatialindex_sql():
    rewritten = rewrite_spatialindex_sql(LEGACY_TRIGGER)
    assert "SpatialIndex" not in rewritten
    assert '"idx_nodes_geometry"' in rewritten
    assert "MbrMaxX(StartPoint(new.geometry))" in rewritten
    # statements without the pattern are left untouched
    assert rewrite_spatialindex_sql("select 1 from nodes") is None


def test_upgrade_legacy_triggers_end_to_end(conn):
    conn.execute("CREATE TABLE nodes (node_id INTEGER PRIMARY KEY, a_node INTEGER)")
    conn.execute("SELECT AddGeometryColumn('nodes', 'geometry', 4326, 'POINT', 'XY')")
    conn.execute("SELECT CreateSpatialIndex('nodes', 'geometry')")
    conn.execute("CREATE TABLE links (link_id INTEGER PRIMARY KEY, a_node INTEGER)")
    conn.execute("SELECT AddGeometryColumn('links', 'geometry', 4326, 'LINESTRING', 'XY')")
    conn.execute(LEGACY_TRIGGER)

    assert upgrade_legacy_spatialindex_triggers(conn) == 1
    assert upgrade_legacy_spatialindex_triggers(conn) == 0  # idempotent

    # the upgraded trigger must actually work: inserting a link creates its a-node
    conn.execute("INSERT INTO links (link_id, geometry) VALUES (1, GeomFromText('LINESTRING(5 5, 6 6)', 4326))")
    node = conn.execute("SELECT node_id FROM nodes").fetchall()
    assert node == [(1,)]
    # and the node it created is byte-identical to a directly-built point
    blob = conn.execute("SELECT geometry FROM nodes").fetchone()[0]
    assert bytes(blob) == make_point_blob(5, 5, 4326)
