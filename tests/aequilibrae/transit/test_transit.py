import os
import pytest
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


def test_new_gtfs(project: Project):
    data = Transit(project)
    transit = data.new_gtfs(
        agency="",
        file_path=os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/gtfs_coquimbo.zip"),
    )

    assert str(type(transit)) == "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>"


def test__check_connection(tmp_path):
    path = tmp_path / uuid4().hex
    example = create_example(path)

    with pytest.raises(FileNotFoundError):
        Transit(example)


def test_create_empty_transit_exception(project: Project):
    with pytest.raises(FileExistsError):
        project.create_empty_transit()


def test_create_empty_transit(tmp_path):
    path = tmp_path / uuid4().hex
    example = create_example(path, "nauru")
    example.create_empty_transit()

    assert os.path.isfile(os.path.join(path, "public_transport.sqlite")) is True
