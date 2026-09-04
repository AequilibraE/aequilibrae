import re

import numpy as np
import pytest

from aequilibrae import VDF, Graph, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import DEFAULT_PRESET_SPECS, FUNCTION_MAP, VDFsManager


@pytest.fixture
def project(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    return sioux_falls_example


@pytest.fixture
def car_graph(project):
    graph: Graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)
    return graph


@pytest.fixture
def matrix(project):
    mat = project.matrices.get_matrix("demand_omx")
    mat.computational_view()
    return mat


@pytest.fixture
def traffic_class(car_graph, matrix):
    return TrafficClass("car", car_graph, matrix)


@pytest.fixture
def assignment(project):
    return TrafficAssignment(project)


@pytest.mark.parametrize(
    "vdf_name,name_mapping",
    [
        *[(k, {"alpha": "b", "beta": "power"}) for k in FUNCTION_MAP if (k != "akcelik" and k != "conical")],
        *[(k, {"alpha": 0.15, "beta": 4.0}) for k in FUNCTION_MAP if (k != "akcelik" and k != "conical")],
        ("akcelik", {"alpha": "b", "tau": "power", "length": "distance"}),
        ("akcelik", {"alpha": 0.25, "tau": 0.1 * 8.0, "length": "distance"}),
        ("akcelik", {"tau": 0.1 * 8.0, "length": "distance"}),
    ],
)
def test_set_vdf_with_parameters(
    assignment: TrafficAssignment,
    traffic_class: TrafficClass,
    vdf_name: str,
    name_mapping: dict[str, str | float],
):
    assignment.add_class(traffic_class)
    function, derivative = FUNCTION_MAP[vdf_name]
    vdf = VDF(vdf_name, function, DEFAULT_PRESET_SPECS[vdf_name], derivative)
    assignment.set_vdf(vdf, name_mapping=name_mapping)


@pytest.mark.parametrize(
    "vdf_name,name_mapping",
    [
        *[(k, {"alpha": -1, "beta": "power"}) for k in FUNCTION_MAP if (k != "akcelik" and k != "conical")],
        *[(k, {"alpha": -1, "beta": 4.0}) for k in FUNCTION_MAP if (k != "akcelik" and k != "conical")],
        ("akcelik", {"alpha": -1, "tau": "power", "length": "distance"}),
        ("akcelik", {"alpha": -1, "tau": 0.1 * 8.0, "length": "distance"}),
    ],
)
def test_check_bounds_of_vdfs_inclusive(
    assignment: TrafficAssignment,
    traffic_class: TrafficClass,
    vdf_name: str,
    name_mapping: dict[str, str | float],
):
    assignment.add_class(traffic_class)
    function, derivative = FUNCTION_MAP[vdf_name]
    vdf = VDF(vdf_name, function, DEFAULT_PRESET_SPECS[vdf_name], derivative)
    with pytest.raises(ValueError, match="At least one alpha is less than 0.0"):
        assignment.set_vdf(vdf, name_mapping=name_mapping)


def test_check_bounds_of_vdfs_exclusive(
    assignment: TrafficAssignment,
    traffic_class: TrafficClass,
):
    assignment.add_class(traffic_class)

    exclusive_bound_vdf = VDF(
        "ebv", lambda x: x, {"param": {"bounds": (0.0, 1.0), "inclusive_lower": False, "inclusive_upper": False}}
    )
    with pytest.raises(ValueError, match="At least one param is less than or equal to 0.0"):
        assignment.set_vdf(exclusive_bound_vdf, name_mapping={"param": -1})

    with pytest.raises(ValueError, match="At least one param is greater than or equal to 1.0"):
        assignment.set_vdf(exclusive_bound_vdf, name_mapping={"param": 2})


def test_make_preset_vdfs():
    vdfs = VDFsManager(add_preset_vdfs=True)
    for vdf_name in FUNCTION_MAP:
        assert isinstance(vdfs.get_vdf(vdf_name), VDF)

    with pytest.raises(ValueError):
        vdfs.get_vdf("fake_vdf")


def test_vdf_as_parsed_string():
    vdfs = VDFsManager(
        vdf_data_from_parameters={
            "default": "bpr_tyler",
            "bpr_tyler": {
                "function": "bpr",
                "spec": {
                    "alpha": {"fill_NA": 0.15, "bounds": [0, 10]},
                    "beta": 4,
                },
            },
            "quadratic": {
                "functional_form": "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)",
                "derivative_functional_form": "fftime * (2 * a * (link_flows/capacity) + b) / capacity",
                "spec": {
                    "a": {"fill_NA": 0.15, "bounds": [0, float("inf")]},
                    "b": {"fill_NA": 1.0, "bounds": [0, float("inf")]},
                },
            },
        }
    )
    quadratic_vdf: VDF = vdfs.get_vdf("quadratic")

    link_flows = np.full(3, 0.5)
    capacity = np.ones(3)
    free_flow_time = np.array([1.0, 2.0, 3.0])
    a, b = 1, 2

    output = np.zeros(3)
    quadratic_vdf.apply_vdf(
        output,
        link_flows,
        free_flow_time,
        capacity,
        1,
        a=a,
        b=b,
    )
    expected = free_flow_time * (a * (link_flows / capacity) ** 2 + b * (link_flows / capacity) + 1)
    np.testing.assert_array_equal(expected, output)

    derivative_output = np.zeros(3)
    quadratic_vdf.apply_derivative(
        derivative_output,
        link_flows,
        free_flow_time,
        capacity,
        1,
        a=a,
        b=b,
    )
    expected_derivative = free_flow_time * (2 * a * (link_flows / capacity) + b) / capacity
    np.testing.assert_array_equal(expected_derivative, derivative_output)


def test_finite_difference():
    vdfs = VDFsManager(
        vdf_data_from_parameters={
            "quadratic": {
                "functional_form": "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)",
                "spec": {
                    "a": {"fill_NA": 0.15, "bounds": [0, float("inf")]},
                    "b": {"fill_NA": 1.0, "bounds": [0, float("inf")]},
                },
            },
        }
    )
    quadratic_vdf: VDF = vdfs.get_vdf("quadratic")

    link_flows = np.full(3, 0.5)
    capacity = np.ones(3)
    free_flow_time = np.array([1.0, 2.0, 3.0])
    a, b = 1, 2
    derivative_output = np.zeros(3)

    quadratic_vdf.apply_derivative(
        derivative_output,
        link_flows,
        free_flow_time,
        capacity,
        1,
        a=a,
        b=b,
    )
    expected = free_flow_time * (2 * a * (link_flows / capacity) + b) / capacity
    np.testing.assert_allclose(expected, derivative_output)


def test_malformed_vdf_parameters():
    no_function_def_parameters = {
        "quadratic": {
            "wrongly_named_functional_form": "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)",
            "derivative_functional_form": "fftime * (2 * a * (link_flows/capacity) + b) / capacity",
            "spec": {
                "a": {
                    "fill_NA": 0.15,
                    "bounds": [0, float("inf")],
                },
                "b": {
                    "fill_NA": 1.0,
                    "bounds": [0, float("inf")],
                },
            },
        },
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            "VDF 'quadratic' must define either 'function' (a preset name) or 'functional_form' (a custom expression)."
        ),
    ):
        VDFsManager(vdf_data_from_parameters=no_function_def_parameters)

    no_preset_function = {
        "quadratic": {
            "function": "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)",
        },
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"VDF 'quadratic' references unknown preset function '{no_preset_function['quadratic']['function']}'. "
            f"Available presets are: {', '.join(FUNCTION_MAP.keys())}."
        ),
    ):
        VDFsManager(vdf_data_from_parameters=no_preset_function)


@pytest.mark.parametrize(
    "vdf_name, parameter_values, expected_convex",
    [
        ("bpr", {"alpha": 0.15, "beta": 4.0}, True),
        ("bpr2", {"alpha": 0.15, "beta": 4.0}, True),
        ("conical", {"alpha": 2.0, "beta": 1.5}, True),
        ("inrets", {"alpha": 0.9}, False),
        ("akcelik", {"alpha": 0.25, "tau": 0.8, "length": 1.0}, True),
    ],
)
def test_plot_vdf_and_check_valid_vdf(vdf_name, parameter_values, expected_convex):
    vdfs_preset = VDFsManager(add_preset_vdfs=True)
    vdf = vdfs_preset.get_vdf(vdf_name)

    num_points = 300
    link_attributes = {name: np.full(num_points, value, dtype=np.float64) for name, value in parameter_values.items()}

    valid_0_value, increasing_f_vals, nonnegative_derivative, convex = vdf.check_valid(num_points, link_attributes)
    assert valid_0_value and increasing_f_vals and nonnegative_derivative
    assert convex == expected_convex

    vdf.plot_vdf(num_points, link_attributes)
