import pytest

from aequilibrae.paths.vdf import FUNCTION_MAP, VDFsManager, DEFAULT_PRESET_SPECS
from aequilibrae import TrafficAssignment, TrafficClass, VDF

import numpy as np
import re


@pytest.fixture(scope="function")
def project(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    return sioux_falls_example


@pytest.fixture(scope="function")
def car_graph(project):
    graph: Graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)
    return graph


@pytest.fixture(scope="function")
def matrix(project):
    mat = project.matrices.get_matrix("demand_omx")
    mat.computational_view()
    return mat


@pytest.fixture(scope="function")
def assigclass(car_graph, matrix):
    return TrafficClass("car", car_graph, matrix)


@pytest.fixture(scope="function")
def assignment(project):
    return TrafficAssignment(project)


@pytest.mark.parametrize(
    "vdf_name,name_mapping",
    [
        *[
            (k, {"alpha": "b", "beta": "power", "capacity": "capacity"})
            for k in FUNCTION_MAP
            if (k != "akcelik" and k != "conical")
        ],
        *[
            (k, {"alpha": 0.15, "beta": 4.0, "capacity": "capacity"})
            for k in FUNCTION_MAP
            if (k != "akcelik" and k != "conical")
        ],
        ("akcelik", {"alpha": "b", "tau": "power", "length": "distance", "capacity": "capacity"}),
        ("akcelik", {"alpha": 0.25, "tau": 0.1 * 8.0, "length": "distance", "capacity": "capacity"}),
        ("akcelik", {"tau": 0.1 * 8.0, "length": "distance", "capacity": "capacity"}),
    ],
)
def test_set_vdf_with_parameters(
    assignment: TrafficAssignment,
    assigclass: TrafficClass,
    vdf_name: str,
    name_mapping: dict,
):
    assignment.add_class(assigclass)
    f, f_dash = FUNCTION_MAP[vdf_name]
    vdf = VDF(vdf_name, f, DEFAULT_PRESET_SPECS[vdf_name], f_dash)
    assignment.set_vdf(vdf, name_mapping=name_mapping)


@pytest.mark.parametrize(
    "vdf_name,name_mapping",
    [
        *[
            (k, {"alpha": -1, "beta": "power", "capacity": "capacity"})
            for k in FUNCTION_MAP
            if (k != "akcelik" and k != "conical")
        ],
        *[
            (k, {"alpha": -1, "beta": 4.0, "capacity": "capacity"})
            for k in FUNCTION_MAP
            if (k != "akcelik" and k != "conical")
        ],
        ("akcelik", {"alpha": -1, "tau": "power", "length": "distance", "capacity": "capacity"}),
        ("akcelik", {"alpha": -1, "tau": 0.1 * 8.0, "length": "distance", "capacity": "capacity"}),
    ],
)
def test_check_bounds_of_vdfs_inclusive(
    assignment: TrafficAssignment,
    assigclass: TrafficClass,
    vdf_name: str,
    name_mapping: dict,
):
    assignment.add_class(assigclass)
    f, f_dash = FUNCTION_MAP[vdf_name]
    vdf = VDF(vdf_name, f, DEFAULT_PRESET_SPECS[vdf_name], f_dash)
    with pytest.raises(ValueError, match="At least one alpha is less than 0.0"):
        assignment.set_vdf(vdf, name_mapping=name_mapping)


def test_check_bounds_of_vdfs_exclusive(
    assignment: TrafficAssignment,
    assigclass: TrafficClass,
    # vdf_name: str,
    # name_mapping: dict,
):
    assignment.add_class(assigclass)

    exclusive_bound_vdf = VDF(
        "ebv", lambda x: x, {"param": {"bounds": (0.0, 1.0), "inclusive_lower": False, "inclusive_upper": False}}
    )
    with pytest.raises(ValueError, match="At least one param is less than or equal to 0.0"):
        assignment.set_vdf(exclusive_bound_vdf, name_mapping={"param": -1})

    with pytest.raises(ValueError, match="At least one param is greater than or equal to 1.0"):
        assignment.set_vdf(exclusive_bound_vdf, name_mapping={"param": 2})


def test_make_preset_vdfs():
    vdfs_preset = VDFsManager(add_preset_vdfs=True)
    for f in FUNCTION_MAP:
        vdf = vdfs_preset.get_vdf(f)
        assert isinstance(vdf, VDF)
    with pytest.raises(ValueError):
        vdfs_preset.get_vdf("fake_vdf")


def test_vdf_as_parsed_string():
    """
    vdfs:
        default: "bpr_tyler"
        bpr_tyler:
            function: bpr
            spec:
            alpha:
                fillNA: 0.15
                bounds: [0, 10]
            beta: 4
            capacity:
                bounds: [0, .inf]
        quadratic:
            functional_form: "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)"
            derivative_functional_form: "fftime * (2 * a * (link_flows/capacity) + b) / capacity"
            spec:
            a:
                fillNA: 0.15
                bounds: [0, .inf]
            b:
                fillNA: 1.0
                bounds: [0, .inf]
            capacity:
                bounds: [0, .inf]
    """
    vdfs = VDFsManager(
        vdf_data_from_parameters={
            "default": "bpr_tyler",
            "bpr_tyler": {
                "function": "bpr",
                "spec": {
                    "alpha": {
                        "fill_NA": 0.15,
                        "bounds": [0, 10],
                    },
                    "beta": 4,
                    "capacity": {"bounds": [0, float("inf")]},
                },
            },
            "quadratic": {
                "functional_form": "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)",
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
                    "capacity": {"bounds": [0, float("inf")]},
                },
            },
        }
    )
    quadratic_vdf: VDF = vdfs.get_vdf("quadratic")
    out = np.zeros(3)
    fake_link_flows = np.array([0.5, 0.5, 0.5])
    fake_capacity = np.array([1.0, 1.0, 1.0])
    fake_free_flow_time = np.array([1.0, 2.0, 3.0])
    a = 1
    b = 2
    quadratic_vdf.apply_vdf(
        out,
        fake_link_flows,
        fake_free_flow_time,
        1,
        a=a,
        b=b,
        capacity=fake_capacity,
    )
    assert np.all(
        out
        == fake_free_flow_time
        * (a * (fake_link_flows / fake_capacity) ** 2 + b * (fake_link_flows / fake_capacity) + 1)
    )

    derivative_out = np.zeros(3)
    quadratic_vdf.apply_derivative(
        derivative_out,
        fake_link_flows,
        fake_free_flow_time,
        1,
        a=a,
        b=b,
        capacity=fake_capacity,
    )
    assert np.all(
        derivative_out == fake_free_flow_time * (2 * a * (fake_link_flows / fake_capacity) + b) / fake_capacity
    )


def test_finite_difference():
    """
    vdfs:
        default: "bpr_tyler"
        bpr_tyler:
            function: bpr
            spec:
            alpha:
                fillNA: 0.15
                bounds: [0, 10]
            beta: 4
            capacity:
                bounds: [0, .inf]
        quadratic:
            functional_form: "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)"
            derivative_functional_form: "fftime * (2 * a * (link_flows/capacity) + b) / capacity"
            spec:
            a:
                fillNA: 0.15
                bounds: [0, .inf]
            b:
                fillNA: 1.0
                bounds: [0, .inf]
            capacity:
                bounds: [0, .inf]
    """
    vdfs = VDFsManager(
        vdf_data_from_parameters={
            "quadratic": {
                "functional_form": "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)",
                "spec": {
                    "a": {
                        "fill_NA": 0.15,
                        "bounds": [0, float("inf")],
                    },
                    "b": {
                        "fill_NA": 1.0,
                        "bounds": [0, float("inf")],
                    },
                    "capacity": {"bounds": [0, float("inf")]},
                },
            },
        }
    )
    quadratic_vdf: VDF = vdfs.get_vdf("quadratic")
    fake_link_flows = np.array([0.5, 0.5, 0.5])
    fake_capacity = np.array([1.0, 1.0, 1.0])
    fake_free_flow_time = np.array([1.0, 2.0, 3.0])
    a = 1
    b = 2

    derivative_out = np.zeros(3)
    quadratic_vdf.apply_derivative(
        derivative_out,
        fake_link_flows,
        fake_free_flow_time,
        1,
        a=a,
        b=b,
        capacity=fake_capacity,
    )
    actual = fake_free_flow_time * (2 * a * (fake_link_flows / fake_capacity) + b) / fake_capacity
    np.testing.assert_allclose(actual, derivative_out)


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


def test_plot_vdf_and_check_valid_vdf(tmp_path):
    vdfs_preset = VDFsManager(add_preset_vdfs=True)

    bpr = vdfs_preset.get_vdf("bpr")
    num_points = 300
    alphas = np.ones(num_points, dtype=np.float64) * 0.15
    betas = np.ones(num_points, dtype=np.float64) * 4.0

    capacity = np.ones(num_points, dtype=np.float64)
    bpr_link_attributes = {"alpha": alphas, "beta": betas, "capacity": capacity}

    valid_0_value, increasing_f_vals, nonnegative_derivative, convex = bpr.check_valid(num_points, bpr_link_attributes)

    # still a error with the vdfs that the convex part is wrong - derivatives are set to 1 for 0 volume
    assert valid_0_value and increasing_f_vals and nonnegative_derivative and convex
    bpr.plot_vdf(tmp_path, num_points, bpr_link_attributes)

    bpr2 = vdfs_preset.get_vdf("bpr2")

    valid_0_value, increasing_f_vals, nonnegative_derivative, convex = bpr2.check_valid(num_points, bpr_link_attributes)
    assert valid_0_value and increasing_f_vals and nonnegative_derivative and convex
    bpr2.plot_vdf(tmp_path, num_points, {"alpha": alphas, "beta": betas, "capacity": capacity})

    conical_alphas = np.ones(num_points, dtype=np.float64) * 2.0
    conical_betas = np.ones(num_points, dtype=np.float64) * 1.5

    conical = vdfs_preset.get_vdf("conical")

    valid_0_value, increasing_f_vals, nonnegative_derivative, convex = bpr2.check_valid(
        num_points, {"alpha": conical_alphas, "beta": conical_betas, "capacity": capacity}
    )
    assert valid_0_value and increasing_f_vals and nonnegative_derivative and convex
    conical.plot_vdf(tmp_path, num_points, {"alpha": conical_alphas, "beta": conical_betas, "capacity": capacity})

    inrets_alphas = np.ones(num_points, dtype=np.float64) * 0.9

    inrets = vdfs_preset.get_vdf("inrets")
    valid_0_value, increasing_f_vals, nonnegative_derivative, convex = inrets.check_valid(
        num_points, {"alpha": inrets_alphas, "capacity": capacity}
    )
    assert valid_0_value and increasing_f_vals and nonnegative_derivative and not convex
    inrets.plot_vdf(tmp_path, num_points, {"alpha": inrets_alphas, "capacity": capacity})

    akcelik_alphas = np.ones(num_points, dtype=np.float64) * 0.25
    taus = np.ones(num_points, dtype=np.float64) * 0.8
    lengths = np.ones(num_points, dtype=np.float64)

    akcelik = vdfs_preset.get_vdf("akcelik")
    valid_0_value, increasing_f_vals, nonnegative_derivative, convex = akcelik.check_valid(
        num_points, {"alpha": akcelik_alphas, "tau": taus, "length": lengths, "capacity": capacity}
    )
    assert valid_0_value and increasing_f_vals and nonnegative_derivative and convex
    akcelik.plot_vdf(
        tmp_path,
        num_points,
        {"alpha": akcelik_alphas, "tau": taus, "length": lengths, "capacity": capacity},
    )
