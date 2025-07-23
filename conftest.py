# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package without
# needing to import them (pytest will automatically discover them).

import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from shutil import copytree
from tempfile import gettempdir

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.spatialite_utils import ensure_spatialite_binaries
from tests.data import siouxfalls_project

DEFAULT_PROJECT = siouxfalls_project
ensure_spatialite_binaries()


@pytest.fixture(scope="session")
def centroids():
    return np.arange(27) + 1


@pytest.fixture(scope="session")
def cache_path(test_base):
    return test_base / "cache"


@pytest.fixture(scope="session")
def test_base():
    return Path(tempfile.gettempdir()) / "aequilibrae_testing"


@pytest.fixture(scope="function")
def test_folder(test_base):
    right_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = test_base / f"{right_now}--{uuid.uuid4().hex[:4]}"
    while dir.exists():
        dir = test_base / f"{right_now}--{uuid.uuid4().hex[:4]}"
    dir.mkdir(parents=True, exist_ok=True)
    return dir


@pytest.fixture(scope="session")
def test_data_path():
    return Path(__file__).parent / "tests/data"


@pytest.fixture(scope="function")
def omx_example(test_data_path, test_folder):
    test_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(test_data_path / "test_omx.omx", test_folder / "test_omx.omx")
    return test_folder / "test_omx.omx"


@pytest.fixture(scope="function")
def no_index_omx(test_data_path, test_folder):
    test_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(test_data_path / "no_index.omx", test_folder / "no_index.omx")
    return test_folder / "test_omx.omx"


@pytest.fixture(scope="function")
def sioux_falls_example(cache_path, test_folder) -> Project:
    source = cache_path / "sioux_falls"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def sioux_falls_test(test_data_path, test_folder) -> Project:
    source = test_data_path / "SiouxFalls_project"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def sioux_falls_single_class(cache_path, test_folder) -> Project:
    source = cache_path / "sioux_falls_single_class"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def triangle_graph_blocking(test_data_path, test_folder) -> Project:
    source = test_data_path / "blocking_triangle_graph_project"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def coquimbo_example(cache_path, test_folder):
    source = cache_path / "coquimbo"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


def project_factory_fixture(scope):
    @pytest.fixture(scope=scope)
    def create_project_fixture(tmp_path_factory):
        base_dir = tmp_path_factory.mktemp(f"projects_{scope}")
        projects = []

        def _create_project(name=None, source_dir=DEFAULT_PROJECT):
            proj_dir = base_dir / (name or uuid.uuid4().hex)
            copytree(source_dir, proj_dir)
            project = Project()
            project.open(str(proj_dir))
            projects.append(project)
            return project

        yield _create_project

        for project in projects:
            project.close()

    return create_project_fixture


create_project = project_factory_fixture(scope="function")
create_project_session = project_factory_fixture(scope="session")


@pytest.fixture
def create_empty_project(_empty_project, create_project):
    def _create_empty_project(name=None):
        return create_project(name=name, source_dir=_empty_project)

    return _create_empty_project


@pytest.fixture(scope="session")
def create_empty_project_session(_empty_project, create_project_session):
    def _create_empty_project(name=None):
        return create_project_session(name=name, source_dir=_empty_project)

    return _create_empty_project


# This fixture creates a default empty structure on disk that can be used as the
# source folder for creating temporary empty projects
@pytest.fixture(scope="session")
def _empty_project(tmp_path_factory):
    proj_dir = tmp_path_factory.mktemp("_empty_project") / uuid.uuid4().hex
    project = Project()
    project.new(str(proj_dir))
    return proj_dir


@pytest.fixture
def empty_project(create_empty_project):
    return create_empty_project()


@pytest.fixture
def create_path(tmp_path):
    return tmp_path / uuid.uuid4().hex


@pytest.fixture
def create_gtfs_project(create_path):
    prj = create_example(create_path, "coquimbo")

    if os.path.isfile(os.path.join(create_path, "public_transport.sqlite")):
        os.remove(os.path.join(create_path, "public_transport.sqlite"))

    data = Transit(prj)

    yield data
    prj.close()


@pytest.fixture
def transit_conn(create_gtfs_project):
    return database_connection("transit")


@pytest.fixture(autouse=True)
def doctest_fixtures(doctest_namespace, test_folder, tmp_path_factory):
    doctest_namespace["project_path"] = str(test_folder)
    doctest_namespace["my_folder_path"] = tmp_path_factory.mktemp(uuid.uuid4().hex)
    doctest_namespace["create_example"] = create_example
    doctest_namespace["Project"] = Project
    doctest_namespace["Transit"] = Transit
    doctest_namespace["AequilibraeMatrix"] = AequilibraeMatrix

    doctest_namespace["os"] = os
    doctest_namespace["pd"] = pd
    doctest_namespace["np"] = np
    doctest_namespace["Polygon"] = Polygon


def pytest_sessionstart(session):
    test_base = Path(tempfile.gettempdir()) / "aequilibrae_testing"
    tgt = test_base / "cache" / "sioux_falls"
    if not tgt.exists():
        create_example(tgt, "sioux_falls").close()

    tgt = test_base / "cache" / "coquimbo"
    if not tgt.exists():
        create_example(tgt, "coquimbo").close()

    tgt = test_base / "cache" / "sioux_falls_single_class"
    if not tgt.exists():
        zipfile.ZipFile(Path(__file__).parent / "data" / "sioux_falls_single_class.zip").extractall(tgt)

    right_now = datetime.now().strftime("%Y-%m-%d_%H")
    for item in test_base.glob("*"):
        if item.is_dir():
            try:
                if right_now not in item.name and "cache" not in item.name:
                    shutil.rmtree(item)
            except Exception as e:
                # Skip folders with non-matching name pattern
                logging.error(f"Couldn't delete dir {item}, reason: {e}")
