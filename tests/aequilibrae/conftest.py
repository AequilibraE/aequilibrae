import uuid
from shutil import copytree

import pytest

from aequilibrae import Project

from ..data import siouxfalls_project


@pytest.fixture(scope="session")
def create_project(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("projects")
    projects = []

    def _create_project(name=None):
        proj_dir = base_dir / (name or uuid.uuid4().hex)
        copytree(siouxfalls_project, proj_dir)
        project = Project()
        project.open(str(proj_dir))
        projects.append(project)
        return project

    yield _create_project

    for project in projects:
        project.close()
