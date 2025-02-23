# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package without
# needing to import them (pytest will automatically discover them).

import os
import uuid
from pathlib import Path
from shutil import copytree

import pytest

import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.spatialite_utils import ensure_spatialite_binaries
from tempfile import gettempdir
from tests.data import siouxfalls_project

DEFAULT_PROJECT = siouxfalls_project
ensure_spatialite_binaries()


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
def project(create_empty_project):
    return create_empty_project()


@pytest.fixture
def create_path(tmp_path):
    return tmp_path / uuid.uuid4().hex


@pytest.fixture(scope="session")
def coquimbo_project():
    return create_example(Path(gettempdir()) / uuid.uuid4().hex, "coquimbo")


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
def doctest_fixtures(doctest_namespace, create_path, tmp_path_factory):
    doctest_namespace["project_path"] = str(create_path)
    doctest_namespace["coquimbo_project"] = coquimbo_project
    doctest_namespace["my_folder_path"] = tmp_path_factory.mktemp(uuid.uuid4().hex)
    doctest_namespace["create_example"] = create_example
    doctest_namespace["Project"] = Project
    doctest_namespace["Transit"] = Transit
    doctest_namespace["AequilibraeMatrix"] = AequilibraeMatrix

    doctest_namespace["os"] = os
    doctest_namespace["pd"] = pd
    doctest_namespace["np"] = np
    doctest_namespace["Polygon"] = Polygon
