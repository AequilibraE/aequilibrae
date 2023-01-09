import os
import pytest
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


@pytest.fixture
def create_project(tmp_path):
    path = tmp_path / uuid4().hex
    prj = create_example(path, "coquimbo")

    if os.path.isfile(os.path.join(path, "public_transport.sqlite")):
        os.remove(os.path.join(path, "public_transport.sqlite"))

    yield prj
    prj.close()


def test_new_gtfs(create_project):
    data = Transit(create_project)
    transit = data.new_gtfs(
        agency="",
        file_path=os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/gtfs_coquimbo.zip"),
    )

    assert str(type(transit)) == "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>"


def test___create_transit_database(create_project):
    data = Transit(create_project)

    assert os.path.isfile(os.path.join(data.project_base_path, "public_transport.sqlite")) is True
