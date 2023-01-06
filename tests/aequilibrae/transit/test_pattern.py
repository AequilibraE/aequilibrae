import pytest
import os
import shutil
from uuid import uuid4
from aequilibrae.utils.create_example import create_example
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit import Transit


@pytest.fixture
def path(tmp_path):
    return tmp_path / uuid4().hex


@pytest.fixture
def pat(path):
    prj = create_example(path, "coquimbo")

    if os.path.isfile(os.path.join(path, "public_transport.sqlite")):
        os.remove(os.path.join(path, "public_transport.sqlite"))

    prj.create_empty_transit()

    gtfs_fldr = os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/gtfs_coquimbo.zip")

    data = Transit(prj)
    transit = data.new_gtfs(agency="LISERCO, LISANCO, LINCOSUR", file_path=gtfs_fldr, description="")
    transit.load_date("2016-04-13")

    patterns = transit.select_patterns
    yield [x for x in patterns.values()][0]
    prj.close()


@pytest.fixture
def network(path):
    return database_connection(table_type="transit")


def test_save_to_database(pat, network):
    pat.save_to_database(network)

    routes = network.execute("SELECT COUNT(*) FROM routes;").fetchone()[0]
    assert routes == 1


def test_best_shape(pat):
    shp = pat.best_shape()
    assert shp == pat._stop_based_shape, "Returned the wrong shape"


def test_get_error(pat):
    assert pat.get_error() is None, "Resulted a map-matching error when should have returned none"


def test_map_match(pat, network):
    pat.map_match()
    pat.save_to_database(network)

    pattern_map = network.execute("SELECT COUNT(*) FROM pattern_mapping;").fetchone()[0]
    assert pattern_map == 207
