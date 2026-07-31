import pytest

from aequilibrae.paths.vdf import FUNCTION_MAP, VDFsManager, DEFAULT_PRESET_SPECS
from aequilibrae import TrafficAssignment, TrafficClass, VDF


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
