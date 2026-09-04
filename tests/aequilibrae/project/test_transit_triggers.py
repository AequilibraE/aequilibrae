from shapely.geometry import LineString, Point


_PATTERN_ID = 10001001000


def _add_route_link_parents(conn):
    """Insert the minimum parent records needed to exercise route-link triggers."""
    conn.execute("INSERT INTO agencies (agency_id, agency) VALUES (?, ?)", (1, "test agency"))
    conn.execute(
        "INSERT INTO routes (pattern_id, route_id, route, agency_id, route_type) VALUES (?, ?, ?, ?, ?)",
        (_PATTERN_ID, 1, "test route", 1, 0),
    )
    conn.executemany(
        "INSERT INTO stops (stop_id, stop, agency_id, route_type, geometry) VALUES (?, ?, ?, ?, GeomFromWKB(?, 4326))",
        (("5", "5", 1, 0, Point(0, 0).wkb), ("6", "6", 1, 0, Point(1, 1).wkb)),
    )


def test_link_insert(empty_project):
    empty_project.scenario.create_transit_database()

    with empty_project.transit_connection as conn:
        _add_route_link_parents(conn)
        data = [_PATTERN_ID, 3, 20000001, 5, 6, 0, LineString([[-23.59, -46.64], [-23.43, -46.50]]).wkb]
        conn.execute(
            """INSERT INTO route_links (pattern_id, seq, transit_link, from_stop, to_stop, distance, geometry)
                        VALUES(?, ?, ?, ?, ?, ?, GeomFromWKB(?, 4326));""",
            data,
        )
        conn.commit()

        distance = conn.execute("SELECT distance FROM route_links WHERE seq=3;").fetchone()[0]

    assert distance != 0


def test_geometry_update(empty_project):
    empty_project.scenario.create_transit_database()

    with empty_project.transit_connection as conn:
        _add_route_link_parents(conn)
        data = [_PATTERN_ID, 3, 20000001, 5, 6, 0, LineString([[-23.59, -46.64], [-23.43, -46.50]]).wkb]
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
