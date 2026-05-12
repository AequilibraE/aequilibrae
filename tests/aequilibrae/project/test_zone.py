import sqlite3
from math import sqrt
from random import randint

import shapely.wkb
from shapely.geometry import Point, MultiPolygon, LineString, MultiLineString
import pytest

from aequilibrae.utils.db_utils import read_and_close


def create_zones(project):
    zones = 5
    network = project.network
    nodes = network.nodes

    geo = network.convex_hull()
    zone_area = geo.area / zones
    zone_side = sqrt(2 * sqrt(3) * zone_area / 9)
    extent = network.extent()
    b = extent.bounds

    with read_and_close(project.path_to_file, spatial=True) as conn:
        sql = "select st_asbinary(HexagonalGrid(GeomFromWKB(?), ?, 0, GeomFromWKB(?)))"
        grid = conn.execute(sql, [extent.wkb, zone_side, Point(b[2], b[3]).wkb]).fetchone()[0]

    grid = shapely.wkb.loads(grid)
    grid = [p for p in grid.geoms if p.intersects(geo)]

    zoning = project.zoning
    for i, zone_geo in enumerate(grid):
        zone = zoning.new(i + 1)
        zone.geometry = zone_geo
        zone.save()
        node = nodes.get(i + 1)
        node.renumber(i + 10001)

    return project


def test_delete(nauru_example):
    project = create_zones(nauru_example)
    zones = project.zoning
    zone_downtown = zones.get(3)
    zone_downtown.delete()

    with pytest.raises(ValueError):
        _ = zones.get(3)


def test_save(nauru_example):
    project = create_zones(nauru_example)
    zones = project.zoning
    zn = zones.get(2)
    area = randint(0, 9999999999)
    zn.area = area
    zn.save()

    with read_and_close(project.path_to_file, spatial=True) as conn:
        cnt = conn.execute("Select area from Zones where zone_id=2").fetchone()[0]
        assert cnt == area, "Zone didn't save area properly"

        geo = Point(0, 0).buffer(1)
        zn.geometry = geo
        zn.save()
        wkb = conn.execute("Select asBinary(geometry) from Zones where zone_id=2").fetchone()[0]
        assert shapely.wkb.loads(wkb) == MultiPolygon([geo]), "Zone didn't save geometry properly"

        zn2 = zones.get(1)
        geo = MultiPolygon([Point(0, 0).buffer(1)])
        zn2.geometry = geo
        zn2.save()
        wkb = conn.execute("Select asBinary(geometry) from Zones where zone_id=1").fetchone()[0]
        assert shapely.wkb.loads(wkb) == geo, "Zone didn't save geometry properly"


def test_add_centroid(nauru_example):
    project = create_zones(nauru_example)
    zones = project.zoning
    nodes = project.network.nodes
    network = project.network
    zone1 = zones.get(1)
    tot = network.count_centroids()
    zone1.add_centroid(None)
    assert tot + 1 == network.count_centroids(), "Added less than it should've"

    tot = network.count_centroids()
    zone1.add_centroid(None)
    zone1.add_centroid(Point(0, 0))
    assert tot == network.count_centroids(), "Added more than should've"

    node1 = nodes.get(1)
    assert node1.geometry == zone1.geometry.centroid

    zone2 = zones.get(2)
    zone2.add_centroid(Point(0, 0))
    node2 = nodes.get(2)
    assert node2.geometry == Point(0, 0)

    point_that_should = zone1.geometry.centroid
    nd = network.nodes.get(1000)
    nd.geometry = point_that_should
    nd.save()
    zone1 = zones.get(1)
    with pytest.raises(sqlite3.IntegrityError):
        zone1.add_centroid(None, robust=False)
    zone1.add_centroid(None, robust=True)


def test_connect_mode(nauru_example):
    project = create_zones(nauru_example)
    zones = project.zoning
    zone1 = zones.get(1)
    zone1.add_centroid(None)

    with project.db_connection_spatial as conn:
        zone1.connect_mode(mode_id="c", conn=conn)
        cnt = conn.execute("Select count(*) from links where a_node=?", [1]).fetchone()[0]
        assert cnt != 0, "failed to add connectors"

        zone1.connect_mode(mode_id="t", conn=conn)
        sql = """Select count(*) from links where a_node=? and instr(modes,'t')>0"""
        cnt = conn.execute(sql, [1]).fetchone()[0]
        assert cnt != 0, "failed to add connectors for mode t"

        zone2 = zones.get(2)
        zone2.connect_mode(mode_id="c", conn=conn)


def test_disconnect_mode(nauru_example):
    project = create_zones(nauru_example)
    zones = project.zoning
    zone1 = zones.get(1)
    zone1.add_centroid(None)

    with project.db_connection_spatial as conn:
        zone1.connect_mode(mode_id="c", conn=conn)
        zone1.connect_mode(mode_id="w", conn=conn)
        tot = conn.execute("""select COUNT(*) from links where a_node=1""").fetchone()[0]
        conn.execute("""Update links set modes = modes || 'w' where instr(modes,'w')=0""")

    zone1.disconnect_mode("w")

    with project.db_connection_spatial as conn:
        cnt = conn.execute("""select COUNT(*) from links where a_node=1""").fetchone()[0]
        assert tot != cnt, "failed to delete links"
        cnt = conn.execute("""Select count(*) from links where a_node=1 and instr(modes,'w')>0""").fetchone()[0]
        assert cnt == 0, "Failed to remove mode from all connectors"


def test_get_closest_zone(sioux_falls_example):
    pt_in = Point(-96.7716, 43.6069)
    pt_out = Point(-96.7754, 43.5664)

    assert sioux_falls_example.zoning.get_closest_zone(pt_in) == 1
    assert sioux_falls_example.zoning.get_closest_zone(pt_out) == 3

    line_in = LineString([(-96.7473, 43.6046), (-96.7341, 43.6046)])
    line_out = LineString([(-96.7209, 43.6132), (-96.7033, 43.61316)])

    assert sioux_falls_example.zoning.get_closest_zone(line_in) == 1
    assert sioux_falls_example.zoning.get_closest_zone(line_out) == 2

    multi_line_in = MultiLineString(
        [((-96.7589, 43.5692), (-96.7531, 43.5807)), ((-96.7531, 43.5807), (-96.7504, 43.5704))]
    )
    multi_line_out = MultiLineString(
        [((-96.7716, 43.5769), (-96.7683, 43.5801)), ((-96.7683, 43.5801), (-96.7574, 43.5784))]
    )

    assert sioux_falls_example.zoning.get_closest_zone(multi_line_in) == 4
    assert sioux_falls_example.zoning.get_closest_zone(multi_line_out) == 3
