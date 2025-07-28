import pytest

from aequilibrae.paths import TrafficClass, Graph
from aequilibrae.paths.results import AssignmentResults


@pytest.fixture
def tc(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    car_graph = sioux_falls_example.network.graphs["c"]  # type: Graph
    car_graph.set_graph("distance")
    car_graph.set_blocked_centroid_flows(False)
    matrix = sioux_falls_example.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    return TrafficClass(name="car", graph=car_graph, matrix=matrix)


def test_result_type(tc):
    assert isinstance(tc.results, AssignmentResults), "Results have the wrong type"
    assert isinstance(tc._aon_results, AssignmentResults), "Results have the wrong type"


def test_set_pce(tc):
    with pytest.raises(ValueError):
        tc.set_pce("not a number")
    tc.set_pce(1)
    tc.set_pce(3.9)


def test_set_vot(tc):
    assert tc.vot == 1.0
    tc.set_vot(4.5)
    assert tc.vot == 4.5


def test_set_fixed_cost(tc):
    assert tc.fc_multiplier == 1.0
    with pytest.raises(ValueError):
        tc.set_fixed_cost("Field_Does_Not_Exist", 2.5)
    assert tc.fc_multiplier == 1.0
    tc.set_fixed_cost("distance", 3.0)
    assert tc.fc_multiplier == 3.0
    assert tc.fixed_cost_field == "distance"
