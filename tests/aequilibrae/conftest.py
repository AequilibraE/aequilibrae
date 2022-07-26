# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package without
# needing to import them (pytest will automatically discover them).

import uuid
from shutil import copytree

import pytest

from aequilibrae import Project

from ..data import siouxfalls_project


@pytest.fixture
def create_project(create_project_session):
    projects = []

    def _create_project(name=None, source_dir=siouxfalls_project):
        project = create_project_session(name=name, source_dir=source_dir, teardown=False)
        projects.append(project)
        return project

    yield _create_project
    for project in projects:
        project.close()


@pytest.fixture
def create_empty_project(_empty_project, create_project):
    def _create_empty_project(name=None):
        return create_project(name=name, source_dir=_empty_project)

    return _create_empty_project


@pytest.fixture(scope="session")
def create_project_session(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("projects")
    projects = []

    def _create_project(name=None, source_dir=siouxfalls_project, teardown=True):
        proj_dir = base_dir / (name or uuid.uuid4().hex)
        copytree(source_dir, proj_dir)
        project = Project()
        project.open(str(proj_dir))
        if teardown:
            projects.append(project)
        return project

    yield _create_project

    for project in projects:
        project.close()


@pytest.fixture(scope="session")
def create_empty_project_session(_empty_project, create_project_session):
    def _create_empty_project(name=None):
        return create_project_session(name=name, source_dir=_empty_project)

    return _create_empty_project


@pytest.fixture(scope="session")
def _empty_project(tmp_path_factory):
    proj_dir = tmp_path_factory.mktemp("_empty_project") / uuid.uuid4().hex
    project = Project()
    project.new(str(proj_dir))
    return proj_dir


@pytest.fixture
def project(create_empty_project):
    return create_empty_project()
