import numpy as np
import pandas as pd
import pytest

from aequilibrae import AequilibraeMatrix, NetworkData, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def network():
    return NetworkData(
        pd.DataFrame(
            {
                "link_id": [1, 2, 3],
                "a_node": [1, 2, 1],
                "b_node": [2, 3, 3],
                "direction": [0, 0, 0],
                "modes": ["ch", "ch", "c"],
                "time": [1.0, 1.0, 10.0],
                "cost": [2.0, 2.0, 3.0],
                "capacity": [100.0, 100.0, 100.0],
                "toll": [0.0, 0.0, 10.0],
                "alpha": [0.15, 0.15, 0.15],
                "beta": [4.0, 4.0, 4.0],
            }
        ),
        centroids=np.array([1, 3]),
    )


@pytest.fixture
def demand():
    matrix = AequilibraeMatrix(index=[1, 3])
    matrix["demand"] = [[0.0, 10.0], [0.0, 0.0]]
    yield matrix
    matrix.close()


def assignment(network, classes, **kwargs):
    return TrafficAssignment(
        network=network,
        classes=classes,
        time_field="time",
        capacity_field="capacity",
        vdf=bpr,
        algorithm="all-or-nothing",
        **kwargs,
    )


def test_deferred_graphs_and_repeat_execution(network, demand, monkeypatch):
    traffic = TrafficClass(mode="c", cost="time", demand=demand, skimming=["time"])
    build_graph = network.build_graph
    calls = []

    def build(*args, **kwargs):
        calls.append(args)
        return build_graph(*args, **kwargs)

    monkeypatch.setattr(network, "build_graph", build)
    assig = assignment(network, [traffic])
    assert calls == []
    assert not hasattr(traffic, "graph")
    assert assig.assignment is None
    assert assig.info()["Classes"]["c"]["network mode"] == "c"
    result = assig.execute(max_iterations=1, rgap=1e-4, cores=1)
    assert len(calls) == 1
    assert not hasattr(traffic, "graph")
    np.testing.assert_allclose(assig.class_results[traffic.name].total_link_loads, [0, 10, 0, 10, 0, 0])
    np.testing.assert_allclose(assig.class_results[traffic.name].skims.matrices["time"][0, 1], 2)
    pd.testing.assert_frame_equal(result, assig.execute(cores=1))
    assert len(calls) == 2


def test_generalized_cost_and_fixed_cost(network, demand):
    traffic = TrafficClass(mode="c", cost="cost", demand=demand, skimming=["time", "cost"])
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
    # The direct link costs 3; the two-link path costs 4.
    np.testing.assert_allclose(assig.class_results[traffic.name].total_link_loads, [0, 0, 0, 0, 0, 10])
    np.testing.assert_allclose(assig.class_results[traffic.name].skims.matrices["cost"][0, 1], 3)
    np.testing.assert_allclose(assig.class_results[traffic.name].skims.matrices["time"][0, 1], 10)
    traffic = TrafficClass(mode="c", cost="cost", demand=demand, fixed_cost_field="toll", vot=2.0)
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
    np.testing.assert_allclose(assig.class_results[traffic.name].total_link_loads, [0, 10, 0, 10, 0, 0])


def test_multiple_modes_and_selected_loads_without_od(network, demand):
    car = TrafficClass(mode="c", cost="time", demand=demand)
    truck = TrafficClass(
        mode="h",
        cost="time",
        demand=demand,
        pce=2.0,
        select_link_sets={"screenline": [1]},
        select_link_od=False,
    )
    assig = assignment(network, [car, truck])
    assig.execute(cores=1)
    np.testing.assert_allclose(assig.total_flow, [0, 30, 0, 30, 0, 0])
    assert assig.class_results[truck.name].select_link_od is None
    assert "screenline" in assig.class_results[truck.name].select_link_loading


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"mode": "x"}, "Mode"),
        ({"cost": "missing"}, "not in network"),
        ({"skimming": ["missing"]}, "not in network"),
        ({"fixed_cost_field": "missing"}, "not in network"),
        ({"select_link_sets": {"bad": [99]}}, "Selected link"),
    ],
)
def test_invalid_configuration_fails_before_execution(network, demand, changes, match):
    traffic = TrafficClass(**dict({"mode": "c", "cost": "time", "demand": demand}, **changes))
    with pytest.raises(ValueError, match=match):
        assignment(network, [traffic])


def test_demand_centroid_order_and_invalid_nodes(network, demand):
    demand.index[:] = [3, 1]
    traffic = TrafficClass(mode="c", cost="time", demand=demand)
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
    np.testing.assert_array_equal(assig._class_states[0].graph.centroids, [3, 1])
    demand.index[:] = [1, 99]
    with pytest.raises(ValueError, match="centroids"):
        assignment(network, [traffic])


def test_preload_array_and_frame_are_equivalent(network, demand):
    traffic = TrafficClass(mode="c", cost="time", demand=demand)
    array = np.array([0, 5, 0, 0, 0, 0], dtype=float)
    frame = pd.DataFrame({"link_id": [1], "direction": [1], "preload": [5.0]})
    first = assignment(network, [traffic], preload=array)
    first.execute(cores=1)
    second = assignment(network, [TrafficClass(mode="c", cost="time", demand=demand)], preload=frame)
    second.execute(cores=1)
    np.testing.assert_allclose(first.assignment.preload, second.assignment.preload)
    assert list(frame.columns) == ["link_id", "direction", "preload"]


def test_project_build_and_independent_paths(sioux_falls_example):
    fields = ["free_flow_time", "capacity", "distance"]
    original = fields.copy()
    network = sioux_falls_example.network.build(fields=fields, modes=["c"])
    assert fields == original
    assert not hasattr(sioux_falls_example.network, "graphs")
    graph = network.build_graph(mode="c", cost="free_flow_time", skimming=["distance"], block_centroid_flows=False)
    path = graph.compute_path(1, 24)
    assert path.path is not None
    graph.network.loc[:, "distance"] = 999
    assert not (network.links.distance == 999).all()


@pytest.mark.parametrize("algorithm", ["msa", "bfw"])
def test_equilibrium_mapping_and_repeat_execution(sioux_falls_example, algorithm):
    project = sioux_falls_example
    demand = project.matrices.get_matrix("demand_omx")
    network = project.network.build(modes=["c"])
    traffic = TrafficClass(
        mode="c",
        cost="free_flow_time",
        demand=demand,
        skimming=["free_flow_time", "distance"],
        block_centroid_flows=False,
    )
    assig = TrafficAssignment(
        network=network,
        classes=[traffic],
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        name_mapping={"alpha": "b", "beta": "power"},
        algorithm=algorithm,
    )
    actual = assig.execute(max_iterations=5, rgap=1e-4, cores=1)
    skims = assig.class_results[traffic.name].skims
    expected = assig.execute(max_iterations=5, rgap=1e-4, cores=1)
    pd.testing.assert_frame_equal(actual, expected)
    for name in skims.matrices:
        np.testing.assert_allclose(skims.matrices[name], assig.class_results[traffic.name].skims.matrices[name])


def test_assignments_own_results_independently(network, demand):
    traffic = TrafficClass(mode="c", cost="time", demand=demand)
    first = assignment(network, [traffic])
    first.execute(cores=1)
    result = first.class_results["c"]
    original_loads = result.link_loads.copy()
    demand["demand"] *= 2
    second = assignment(network, [traffic])
    second.execute(cores=1)
    assert first.class_results["c"] is result
    assert second.class_results["c"] is not result
    np.testing.assert_allclose(result.link_loads, original_loads)
    np.testing.assert_allclose(second.class_results["c"].link_loads, 2 * original_loads)


@pytest.mark.parametrize("method", ["results", "report", "get_skim_results"])
def test_results_require_execution(network, demand, method):
    assig = assignment(network, [TrafficClass(mode="c", cost="time", demand=demand)])
    with pytest.raises(RuntimeError, match="Run the assignment"):
        getattr(assig, method)()
