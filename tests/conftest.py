# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package without
# needing to import them (pytest will automatically discover them).

import faulthandler
import shutil
import zipfile
from pathlib import Path

import pytest

from aequilibrae import Project
from aequilibrae.project.project_creation import remove_triggers
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.spatialite_utils import ensure_spatialite_binaries
from tests.data import siouxfalls_project

faulthandler.enable()

DEFAULT_PROJECT = siouxfalls_project
ensure_spatialite_binaries()


@pytest.fixture(scope="session")
def cache_path(tmp_path_factory):
    return tmp_path_factory.mktemp("cache", numbered=True)


@pytest.fixture(scope="session")
def test_data_path():
    return Path(__file__).parent.parent / "tests" / "data"


@pytest.fixture(scope="function")
def omx_example(test_data_path, tmp_path):
    file = tmp_path / "test_omx.omx"
    shutil.copy(test_data_path / "test_omx.omx", file)
    return file


@pytest.fixture(scope="function")
def no_index_omx(test_data_path, tmp_path):
    file = tmp_path / "no_index.omx"
    shutil.copy(test_data_path / "no_index.omx", file)
    return file


def cached_model(model_name, cache_pth, test_folder) -> Project:
    source = cache_pth / model_name
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    return Project.from_path(test_folder)


@pytest.fixture(scope="session")
def cached_sioux_falls_example(cache_path):
    create_example(cache_path / "sioux_falls", "sioux_falls")


@pytest.fixture(scope="function")
def sioux_falls_example(cached_sioux_falls_example, cache_path, tmp_path) -> Project:
    project = cached_model("sioux_falls", cache_path, tmp_path)
    yield project
    project.close()


@pytest.fixture(scope="session")
def cached_nauru_example(cache_path):
    create_example(cache_path / "nauru", "nauru")


@pytest.fixture(scope="function")
def nauru_example(cached_nauru_example, cache_path, tmp_path) -> Project:
    project = cached_model("nauru", cache_path, tmp_path)
    yield project
    project.close()


@pytest.fixture(scope="session")
def cached_coquimbo_example(cache_path):
    create_example(cache_path / "coquimbo", "coquimbo")


@pytest.fixture(scope="function")
def coquimbo_example(cached_coquimbo_example, cache_path, tmp_path) -> Project:
    project = cached_model("coquimbo", cache_path, tmp_path)
    yield project
    project.close()


@pytest.fixture
def empty_project(tmp_path) -> Project:
    project = Project()
    project.new(tmp_path / "p")
    yield project
    project.close()


@pytest.fixture(scope="session")
def cached_empty_no_triggers_project(cache_path):
    create_example(cache_path / "empty_no_triggers", "empty_no_triggers")


@pytest.fixture
def empty_no_triggers_project(empty_project, tmp_path) -> Project:
    with empty_project.db_connection as conn:
        remove_triggers(conn, empty_project.logger, db_type="network")
        tables = ["nodes", "links"]
        for tbl in tables:
            conn.execute(f"DELETE FROM {tbl}")

    yield empty_project
    empty_project.close()


@pytest.fixture(scope="function")
def sioux_falls_test(test_data_path, tmp_path) -> Project:
    project = cached_model("SiouxFalls_project", test_data_path, tmp_path)
    yield project
    project.close()


@pytest.fixture(scope="function")
def no_triggers_test(test_data_path, tmp_path) -> Project:
    project = cached_model("no_triggers_project", test_data_path, tmp_path)
    yield project
    project.close()


@pytest.fixture(scope="session")
def cached_sioux_falls_single_class(test_data_path, cache_path):
    zipfile.ZipFile(test_data_path / "sioux_falls_single_class.zip").extractall(cache_path / "sioux_falls_single_class")


@pytest.fixture(scope="function")
def sioux_falls_single_class(cached_sioux_falls_single_class, cache_path, tmp_path) -> Project:
    project = cached_model("sioux_falls_single_class", cache_path, tmp_path)
    yield project
    project.close()


@pytest.fixture(scope="function")
def triangle_graph_blocking(test_data_path, tmp_path) -> Project:
    project = cached_model("blocking_triangle_graph_project", test_data_path, tmp_path)
    yield project
    project.close()


@pytest.fixture
def build_gtfs_project(coquimbo_example):
    prj = coquimbo_example

    (coquimbo_example.project_base_path / "public_transport.sqlite").unlink(True)
    data = Transit(prj)
    yield data
    prj.close()
