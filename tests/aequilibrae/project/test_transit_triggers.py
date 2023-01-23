from shapely.geometry import LineString
from aequilibrae.project import Project
from aequilibrae.transit import Transit
import pytest

from aequilibrae.project.database_connection import database_connection


class TestTransitTriggers:
    def test_link_insert(self, project: Project):
        Transit(project)

        conn = database_connection(db_type="transit")

        data = [10001001000, 3, 20000001, 5, 6, 0, LineString([[-23.59, -46.64], [-23.43, -46.50]]).wkb]
        conn.execute(
            """INSERT INTO route_links (pattern_id, seq, transit_link, from_stop, to_stop, distance, geometry)
                        VALUES(?, ?, ?, ?, ?, ?, GeomFromWKB(?, 4326));""",
            data,
        )
        conn.commit()

        distance = conn.execute("SELECT distance FROM route_links WHERE seq=3;").fetchone()[0]

        assert distance != 0

    def test_geometry_update(self, project: Project):
        Transit(project)

        conn = database_connection(db_type="transit")

        data = [10001001000, 3, 20000001, 5, 6, 0, LineString([[-23.59, -46.64], [-23.43, -46.50]]).wkb]
        conn.execute(
            """INSERT INTO route_links (pattern_id, seq, transit_link, from_stop, to_stop, distance, geometry)
                        VALUES(?, ?, ?, ?, ?, ?, GeomFromWKB(?, 4326));""",
            data,
        )
        conn.commit()

        conn.execute(
            "UPDATE route_links SET geometry=GeomFromWKB(?, 4326) WHERE seq=3;",
            [LineString([[-23.59, -46.64], [-23.01, -47.14]]).wkb],
        )
        conn.commit()

        distance = conn.execute("SELECT distance FROM route_links WHERE seq=3;").fetchone()[0]

        assert round(distance, 2) != 19815.63
