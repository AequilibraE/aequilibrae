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
        *[(k, {"alpha": "b", "beta": "power"}) for k in FUNCTION_MAP if k != "akcelik"],
        *[(k, {"alpha": 0.15, "beta": 4.0}) for k in FUNCTION_MAP if k != "akcelik"],
        ("akcelik", {"alpha": "b", "tau": "power", "length": "distance"}),
        ("akcelik", {"alpha": 0.25, "tau": 0.1 * 8.0, "length": "distance"}),
        ("akcelik", {"tau": 0.1 * 8.0, "length": "distance"}),
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


def test_make_preset_vdfs():
    vdfs_preset = VDFsManager(add_preset_vdfs=True)
    for f in FUNCTION_MAP:
        vdf = vdfs_preset.get_vdf(f)
        assert isinstance(vdf, VDF)
    with pytest.raises(ValueError):
        vdfs_preset.get_vdf("fake_vdf")


def test_vdf_as_string():
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
        fake_capacity,
        fake_free_flow_time,
        1,
        a=a,
        b=b,
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
        fake_capacity,
        fake_free_flow_time,
        1,
        a=a,
        b=b,
    )
    assert np.all(
        derivative_out == fake_free_flow_time * (2 * a * (fake_link_flows / fake_capacity) + b) / fake_capacity
    )

    # out = np.zeros(3)
    #     quadratic_vdf.apply_vdf(
    #         out,
    #         np.array([0.5, 0.5, 0.5]),
    #         np.array([1.0, 1.0, 1.0]),
    #         np.array([1.0, 2.0, 3.0]),
    #         1,
    #         a=1,
    #         b=2,
    #     )


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
