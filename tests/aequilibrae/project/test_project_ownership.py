"""Project factories, context lifetime and explicit configuration ownership."""

from copy import deepcopy

import pytest

from aequilibrae import Project
from aequilibrae.distribution import GravityApplication, GravityCalibration, Ipf
from aequilibrae.project import Parameters
from aequilibrae.project.tables import Tables


def test_create_and_open_contexts(tmp_path):
    path = tmp_path / "project"
    with Project.create(path) as project:
        assert project.project_base_path == path
        assert isinstance(project.tables, Tables)
        assert project.tables.connections is project.scenario.connections
        assert project.db_connection.is_open()
    assert not project.db_connection.is_open()

    with Project.open(path) as reopened:
        assert reopened.project_base_path == path
        assert reopened.db_connection.is_open()
        assert reopened.tables.count_links() == 0
    assert not reopened.db_connection.is_open()


def test_context_closes_after_exception(tmp_path):
    with pytest.raises(ValueError, match="operation failed"):
        with Project.create(tmp_path / "project") as project:
            raise ValueError("operation failed")
    assert not project.db_connection.is_open()


def test_context_closes_all_scenario_connections(tmp_path):
    with Project.create(tmp_path / "project") as project:
        root = project.root_scenario
        project.create_empty_scenario("alternative")
        project.use_scenario("alternative")
        alternative = project.scenario
        assert root.db_connection.is_open()
        assert alternative.db_connection.is_open()
    assert not root.db_connection.is_open()
    assert not alternative.db_connection.is_open()


def test_run_binds_the_owning_project(tmp_path):
    with Project.create(tmp_path / "first") as first, Project.create(tmp_path / "second") as second:
        (first.project_base_path / "run" / "__init__.py").write_text(
            "def owner(project, label):\n    return project.project_base_path, label\n"
        )
        first.parameters["run"] = {"owner": {"label": "first"}}
        assert first.run["owner"]() == (first.project_base_path, "first")
        assert first.run["owner"](label="override") == (first.project_base_path, "override")
        assert second.db_connection.is_open()


def test_default_parameters_are_independent_of_active_project(empty_project):
    defaults = deepcopy(Parameters().parameters)
    empty_project.parameters["system"]["cpus"] = 3
    assert Parameters().parameters == defaults
    assert empty_project.project_parameters.parameters["system"]["cpus"] == 3


@pytest.mark.parametrize(
    "procedure, kwargs", [(Ipf, {}), (GravityApplication, {}), (GravityCalibration, {"function": "EXPO"})]
)
def test_distribution_uses_explicit_project_parameters(empty_project, procedure, kwargs):
    empty_project.parameters["distribution"]["ipf"]["max iterations"] = 7
    empty_project.parameters["distribution"]["gravity"]["max iterations"] = 7
    assert procedure(project=empty_project, **kwargs).parameters["max iterations"] == 7


def test_parameter_defaults_are_independent_copies():
    first, second = Parameters(), Parameters()
    first.parameters["system"]["cpus"] = 3
    assert second.parameters == Parameters.load_default()


def test_writing_parameters_requires_a_path():
    with pytest.raises(ValueError, match="explicit path"):
        Parameters().write_back()
