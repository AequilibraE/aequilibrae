import pytest
import os


@pytest.fixture
def pat(create_path, create_gtfs_project):
    gtfs_fldr = os.path.join(create_path, "gtfs_coquimbo.zip")

    transit = create_gtfs_project.new_gtfs_builder(
        agency="LISERCO, LISANCO, LINCOSUR", file_path=gtfs_fldr, description=""
    )
    transit.load_date("2016-04-13")

    patterns = transit.select_patterns
    yield [x for x in patterns.values()][0]


def test_save_to_database(pat, transit_conn):
    pat.save_to_database(transit_conn)

    routes = transit_conn.execute("SELECT COUNT(*) FROM routes;").fetchone()[0]
    assert routes == 1


def test_best_shape(pat):
    shp = pat.best_shape()
    assert shp == pat._stop_based_shape, "Returned the wrong shape"


def test_get_error(pat):
    assert pat.get_error() is None, "Resulted a map-matching error when should have returned none"


def test_map_match(pat, transit_conn):
    pat.map_match()
    pat.save_to_database(transit_conn)

    pattern_map = transit_conn.execute("SELECT COUNT(*) FROM pattern_mapping;").fetchone()[0]
    assert pattern_map > 0
