import numpy as np
import pytest

from aequilibrae import Network, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def configuration(sioux_falls_example):
    project = sioux_falls_example
    demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])
    return {
        "network": project.network,
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
        ({"time_field": ""}, ValueError),
        ({"capacity_field": None}, ValueError),
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


def test_matrix_with_wrong_type(configuration):
    matrix = configuration["classes"][0].demand
    core = next(iter(matrix))
    matrix[core] = np.array(matrix[core], np.int32)
    with pytest.raises(TypeError):
        TrafficClass(name="car", mode="c", cost="free_flow_time", demand=matrix)


def test_algorithms_available(configuration):
    assignment = TrafficAssignment(**configuration)
    algs = assignment.algorithms_available()
    real = ["all-or-nothing", "msa", "frank-wolfe", "bfw", "cfw", "fw"]
    diff = [x for x in real if x not in algs]
    diff2 = [x for x in algs if x not in real]
    assert len(diff) + len(diff2) <= 0, "list of algorithms raised is wrong"


def test_steps_below_needed_to_terminate_reaches_algorithm(configuration):
    assignment = TrafficAssignment(**(configuration | {"algorithm": "bfw"}))
    assignment.execute(max_iterations=1, cores=1, steps_below_needed_to_terminate=3)
    assert assignment.assignment.steps_below_needed_to_terminate == 3


def test_info_and_execution_options(configuration):
    assignment = TrafficAssignment(**configuration)
    assert assignment.info()["Classes"]["car"]["network mode"] == "c"
    assignment.execute(max_iterations=3, rgap=1e-3, cores=1)
    assert assignment.info()["Maximum iterations"] == 3
    assert assignment.info()["Target RGap"] == 1e-3
    assert assignment.cores == 1


@pytest.mark.parametrize("algorithm", ["fw", "FW", "frank-wolfe"])
def test_frank_wolfe_alias(configuration, algorithm):
    assignment = TrafficAssignment(**(configuration | {"algorithm": algorithm}))
    assignment.execute(max_iterations=2, cores=1)
    assert assignment.algorithm == assignment.assignment.algorithm == "frank-wolfe"


def test_vdf_defaults_do_not_depend_on_matching_network_fields(configuration):
    configuration["network"] = Network(configuration["network"].links.data.assign(alpha=0.8, beta=2.0))
    assignment = TrafficAssignment(**(configuration | {"name_mapping": None}))
    assignment.execute(max_iterations=1, cores=1)
    np.testing.assert_array_equal(assignment.vdf_parameters["alpha"], 0.15)
    np.testing.assert_array_equal(assignment.vdf_parameters["beta"], 4.0)

    mapped = TrafficAssignment(**(configuration | {"name_mapping": {"alpha": "alpha", "beta": "beta"}}))
    mapped.execute(max_iterations=1, cores=1)
    np.testing.assert_array_equal(mapped.vdf_parameters["alpha"], 0.8)
    np.testing.assert_array_equal(mapped.vdf_parameters["beta"], 2.0)
