import pytest

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def configuration(sioux_falls_example):
    project = sioux_falls_example
    demand = project.matrices.get_matrix("demand_omx")
    return {
        "network": project.network.build(modes=["c"]),
        "classes": [
            TrafficClass(name="car", mode="c", cost="free_flow_time", demand=demand, block_centroid_flows=False)
        ],
        "time_field": "free_flow_time",
        "capacity_field": "capacity",
        "vdf": bpr,
        "name_mapping": {"alpha": "b", "beta": "power"},
        "algorithm": "msa",
        "project": project,
    }


@pytest.mark.parametrize(
    "change, error",
    [
        ({"vdf": object()}, TypeError),
        ({"classes": []}, TypeError),
        ({"classes": [1, 2]}, TypeError),
        ({"algorithm": "invalid"}, ValueError),
        ({"time_field": "missing"}, ValueError),
        ({"capacity_field": "missing"}, ValueError),
        ({"name_mapping": {"b": "alpha"}}, ValueError),
    ],
)
def test_constructor_validates_configuration(configuration, change, error):
    with pytest.raises(error):
        TrafficAssignment(**(configuration | change))


def test_duplicate_class_names(configuration):
    configuration["classes"] *= 2
    with pytest.raises(ValueError, match="unique"):
        TrafficAssignment(**configuration)


def test_old_assignment_interface_is_removed(configuration):
    with pytest.raises(TypeError):
        TrafficAssignment()
    assignment = TrafficAssignment(**configuration)
    for method in (
        "set_classes",
        "add_class",
        "set_vdf",
        "set_algorithm",
        "set_time_field",
        "set_capacity_field",
        "set_cores",
        "add_preload",
        "set_bfw_conjugacy",
    ):
        assert not hasattr(assignment, method)
    assert assignment.classes == tuple(configuration["classes"])


def test_skim_after(configuration):
    assig = TrafficAssignment(**configuration)
    assig.execute(max_iterations=10, cores=1)
    skims = assig.skim_congested(["distance"], return_matrices=True)
    assert skims["car"].field_names == ("distance", "__congested_time__", "__assignment_cost__")
    skims = assig.skim_congested(return_matrices=True)
    assert skims["car"].field_names == ("__congested_time__", "__assignment_cost__")


def test_info_and_execution_options(configuration):
    assignment = TrafficAssignment(**configuration)
    assert assignment.info()["Classes"]["car"]["network mode"] == "c"
    assignment.execute(max_iterations=3, rgap=1e-3, cores=1)
    assert assignment.info()["Maximum iterations"] == 3
    assert assignment.info()["Target RGap"] == 1e-3
    assert assignment.cores == 1
    assert set(assignment.algorithms_available()) == {"all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"}
