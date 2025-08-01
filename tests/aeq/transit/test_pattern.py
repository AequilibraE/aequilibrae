import pytest


@pytest.fixture(scope="function")
def pat(build_gtfs_project):
    gtfs_fldr = build_gtfs_project.project.project_base_path / "gtfs_coquimbo.zip"

    transit = build_gtfs_project.new_gtfs_builder(agency="Lisanco", file_path=gtfs_fldr, description="")
    transit.load_date("2016-04-13")

    patterns = transit.select_patterns
    yield list(patterns.values())[0]


# We put multiple tests in the same test to avoid long setup times
def test_pattern_complete(build_gtfs_project, pat):
    shp = pat.best_shape()

    # Tests that we get the stop-based shape when we build it and not map-match it
    assert shp != pat._stop_based_shape, "Returned the wrong shape"

    # Asserts that we dont have any errors
    assert pat.get_error() is None, "Resulted a map-matching error when should have returned none"

    # We map-match
    pat.map_match()

    # We save the pattern to the database
    with build_gtfs_project.project.transit_connection as transit_conn:
        pat.save_to_database(transit_conn)
        routes = transit_conn.execute("SELECT COUNT(*) FROM routes;").fetchone()[0]
        pattern_map = transit_conn.execute("SELECT COUNT(*) FROM pattern_mapping;").fetchone()[0]

    assert routes == 1
    assert pattern_map > 0
