import numpy as np
import pandas as pd
import pytest

from aequilibrae import AequilibraeMatrix, Graph, Network, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def network():
    return Network(
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
    return matrix


def assignment(network, classes, **kwargs):
    return TrafficAssignment(
        network=network, classes=classes, time_field="time", capacity_field="capacity",
        vdf=bpr, algorithm="all-or-nothing", **kwargs,
    )


def test_deferred_graph_and_repeat_execution(network, demand, monkeypatch):
    traffic = TrafficClass(mode="c", cost="time", demand=demand, skimming=["time"])
    build = Network.build_graph
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return build(*args, **kwargs)

    monkeypatch.setattr(Network, "build_graph", prepare)
    assig = assignment(network, [traffic])
    assert calls == []
    assert not hasattr(traffic, "graph")
    assert assig.assignment is None
    result = assig.execute(max_iterations=1, rgap=1e-4, cores=1)
    assert len(calls) == 1
    graph = assig._class_states[0].graph
    assert isinstance(graph, Graph)
    assert graph.compact
    assert not any(isinstance(value, pd.DataFrame) for value in vars(graph).values())
    assert not hasattr(graph, "network")
    assert not hasattr(graph, "compact_graph")
    np.testing.assert_allclose(assig.class_results[traffic.name].total_link_loads, [0, 10, 0, 10, 0, 0])
    np.testing.assert_allclose(assig.class_results[traffic.name].skims.matrices["time"][0, 1], 2)
    pd.testing.assert_frame_equal(result, assig.execute(cores=1))
    assert len(calls) == 2


def test_generalized_cost_and_fixed_cost(network, demand):
    traffic = TrafficClass(mode="c", cost="cost", demand=demand, skimming=["time", "cost"])
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
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
        mode="h", cost="time", demand=demand, pce=2.0,
        select_link_sets={"screenline": [1]}, select_link_od=False,
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
def test_invalid_network_configuration_fails_at_construction(network, demand, changes, match):
    traffic = TrafficClass(**dict({"mode": "c", "cost": "time", "demand": demand}, **changes))
    with pytest.raises(ValueError, match=match):
        assignment(network, [traffic])


@pytest.mark.parametrize(
    "turns, match",
    [
        (pd.DataFrame({"from_node": [1]}), "missing required columns"),
        (pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [3], "penalty": [-1.0]}), "Negative turn"),
        (pd.DataFrame({"from_node": [1], "via_node": [99], "to_node": [3], "penalty": [1.0]}), "missing directed legs"),
    ],
)
def test_invalid_turns_fail_at_construction(network, demand, turns, match):
    network.turn_restrictions = turns
    with pytest.raises(ValueError, match=match):
        assignment(network, [TrafficClass(mode="c", cost="time", demand=demand)])


def test_registered_mode_without_links_fails_at_construction(network, demand):
    network.modes = (*network.modes, "x")
    with pytest.raises(ValueError, match="No links are available"):
        assignment(network, [TrafficClass(mode="x", cost="time", demand=demand)])


def test_demand_centroid_order_and_invalid_nodes(network, demand):
    demand.index = [3, 1]
    traffic = TrafficClass(mode="c", cost="time", demand=demand)
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
    np.testing.assert_array_equal(assig._class_states[0].graph.mapping.centroids, [3, 1])
    demand.index = [1, 99]
    with pytest.raises(ValueError, match="centroids"):
        assignment(network, [traffic])
    with pytest.raises(ValueError, match="centroids"):
        assig.execute(cores=1)


def test_preload_array_and_frame_are_equivalent(network, demand):
    traffic = TrafficClass(mode="c", cost="time", demand=demand)
    array = np.array([0, 5, 0, 0, 0, 0], dtype=float)
    frame = pd.DataFrame({"link_id": [1], "direction": [1], "preload": [5.0]})
    first = assignment(network, [traffic], preload=array)
    first.execute(cores=1)
    second = assignment(network, [traffic], preload=frame)
    second.execute(cores=1)
    np.testing.assert_allclose(first.assignment.preload, second.assignment.preload)
    assert list(frame.columns) == ["link_id", "direction", "preload"]


def test_project_build_and_independent_paths(sioux_falls_example):
    project = sioux_falls_example
    fields = ["free_flow_time", "capacity", "distance"]
    original = fields.copy()
    network = project.network.build(fields=fields, modes=["c"])
    assert not hasattr(project.network, "graphs")
    graph = network.build_graph("c", "free_flow_time", skimming=["distance"], block_centroid_flows=False)
    assert fields == original
    assert graph.compute_path(1, 24).path is not None
    network.links.loc[:, "distance"] = 999
    assert not (project.network.links.data["distance"] == 999).all()
    project.close()
    assert graph.compute_path(1, 24).path is not None
    assert network.build_graph("c", "free_flow_time", block_centroid_flows=False).compute_path(1, 24).path is not None


@pytest.mark.parametrize("compact", [False, True])
def test_graph_topology_is_static_and_costs_can_close_links(network, compact):
    graph = network.build_graph("c", "time", compact=compact, block_centroid_flows=False, skimming=["time"])
    assert graph.compact == compact
    assert not any(isinstance(value, pd.DataFrame) for value in vars(graph).values())
    path = graph.compute_path(1, 3)
    np.testing.assert_array_equal(path.path_nodes, [1, 2, 3])
    np.testing.assert_allclose(path.milepost, [0, 1, 2])
    for method in ("prepare_graph", "exclude_links", "set_turn_restrictions", "set_skimming"):
        assert not hasattr(graph, method)
    costs = network.field_values("time").copy()
    costs[network.link_ids == 1] = np.inf
    graph.set_costs(costs)
    np.testing.assert_array_equal(graph.compute_path(1, 3).path_nodes, [1, 3])
    assert graph.compute_skims(cores=1).results.skims["time"][0, 1] == 10
    # Existing path results keep the costs of their own snapshot.
    np.testing.assert_allclose(path.milepost, [0, 1, 2])


@pytest.mark.parametrize("costs", [[1.0], [np.nan] * 6, [-1.0] * 6])
def test_invalid_cost_updates_leave_graph_unchanged(network, costs):
    graph = network.build_graph("c", "time", block_centroid_flows=False)
    original = graph.cost.copy()
    with pytest.raises(ValueError):
        graph.set_costs(costs)
    np.testing.assert_array_equal(graph.cost, original)


def test_route_choice_keeps_a_consistent_cost_snapshot(network):
    from aequilibrae.paths import RouteChoice

    graph = network.build_graph("c", "time", compact=True, block_centroid_flows=False)
    choice = RouteChoice(graph)
    routes = pd.DataFrame({"origin id": [1], "destination id": [3], "route set": [[1, 2]]})
    assert choice.recompute_psl(routes)["cost"].iloc[0] == 2.0
    graph.set_costs(network.field_values("cost"))
    assert choice.recompute_psl(routes)["cost"].iloc[0] == 2.0
    assert RouteChoice(graph).recompute_psl(routes)["cost"].iloc[0] == 4.0


def test_assignment_rejects_prepared_graphs(network, demand):
    graph = network.build_graph("c", "time")
    with pytest.raises(TypeError, match="network must be a Network"):
        assignment(graph, [TrafficClass(mode="c", cost="time", demand=demand)])


@pytest.mark.parametrize("compact", [False, True])
def test_mode_subset_preserves_isolated_centroids(network, compact):
    source = Network(network.links.assign(modes=["c", "h", "h"]), modes=["c"], centroids=[1, 3])
    with pytest.warns(UserWarning, match="centroids not present"):
        graph = source.copy().build_graph("c", "time", compact=compact)
    assert graph.compute_path(1, 3).path is None
    np.testing.assert_array_equal(graph.centroids, [1, 3])


@pytest.mark.parametrize("algorithm", ["msa", "bfw"])
def test_equilibrium_mapping_and_repeat_execution(sioux_falls_example, algorithm):
    project = sioux_falls_example
    demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])
    traffic = TrafficClass(
        mode="c", cost="free_flow_time", demand=demand,
        skimming=["free_flow_time", "distance"], block_centroid_flows=False,
    )
    assig = TrafficAssignment(
        network=project.network, classes=[traffic], time_field="free_flow_time", capacity_field="capacity",
        vdf=bpr, name_mapping={"alpha": "b", "beta": "power"}, algorithm=algorithm,
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


def test_same_mode_classes_share_graph_but_not_costs(network, demand):
    from dataclasses import replace

    first = TrafficClass(name="time", mode="c", cost="time", demand=demand, skimming=["time"])
    second = replace(first, name="cost", cost="cost", skimming=["time", "cost"])
    assig = assignment(network, [first, second])
    assig.execute(cores=1)
    a, b = assig._class_states
    assert a.graph is b.graph
    assert a.network is b.network
    assert a.congested_time is b.congested_time
    assert not a.congested_time.flags.writeable
    inputs_a, inputs_b = assig.assignment.inputs["time"], assig.assignment.inputs["cost"]
    assert np.shares_memory(inputs_a.routing.fs, inputs_b.routing.fs)
    assert not np.shares_memory(inputs_a.routing.costs, inputs_b.routing.costs)
    assert not np.shares_memory(inputs_a.time_skim, inputs_b.time_skim)
    np.testing.assert_allclose(a.results.total_link_loads, [0, 10, 0, 10, 0, 0])
    np.testing.assert_allclose(b.results.total_link_loads, [0, 0, 0, 0, 0, 10])


@pytest.mark.parametrize("changed", ["mode", "centroids", "blocking"])
def test_graph_sharing_respects_routing_policy(network, demand, changed):
    from types import SimpleNamespace

    base = SimpleNamespace(mode="c", demand=demand, block_centroid_flows=True)
    alternative = SimpleNamespace(**vars(base))
    if changed == "mode":
        alternative.mode = "h"
    elif changed == "centroids":
        alternative.demand = SimpleNamespace(index=np.array([3, 1]))
    else:
        alternative.block_centroid_flows = False
    assert network._assignment_graph(base) is not network._assignment_graph(alternative)


@pytest.mark.parametrize("fields", [["time"], ["time", "cost"]])
@pytest.mark.parametrize("turn_delay", [0.0, 0.5])
def test_time_skim_is_independent_of_requested_cost_skim(network, demand, fields, turn_delay):
    network.links["cost"] = [2.0, 2.0, 20.0]
    if turn_delay:
        network.turn_restrictions = pd.DataFrame(
            {"from_node": [1], "via_node": [2], "to_node": [3], "penalty": [turn_delay]}
        )
    traffic = TrafficClass(mode="c", cost="cost", demand=demand, skimming=fields)
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
    skims = assig.class_results["c"].skims
    assert set(skims.matrices) == set(fields)
    assert skims.matrices["time"][0, 1] == pytest.approx(2.0 + turn_delay)
    if "cost" in fields:
        assert skims.matrices["cost"][0, 1] == pytest.approx(4.0 + turn_delay)
    assert assig.class_results["c"].skims is skims
    assig.class_results["c"].reset()
    assert np.isinf(skims.skims).all()


def test_numeric_text_time_field_is_normalized(network, demand):
    network.links["time"] = network.links.time.astype(str)
    assig = assignment(network, [TrafficClass(mode="c", cost="cost", demand=demand, skimming=["time", "cost"])])
    assig.execute(cores=1)
    assert assig.class_results["c"].skims.matrices["time"][0, 1] == 10.0


def test_congested_skims_use_execution_attribute_snapshot(network, demand):
    assig = assignment(network, [TrafficClass(mode="c", cost="time", demand=demand)])
    assig.execute(cores=1)
    network.links["cost"] = 999.0
    skims = assig.skim_congested(["cost"], return_matrices=True)["c"]
    assert skims.matrices["cost"][0, 1] == 4.0


def test_selected_od_export_reindexes_class_centroids(network, demand, tmp_path):
    from types import SimpleNamespace
    import openmatrix as omx

    matrices = [demand]
    for index, trips in (([3, 1], 20.0), ([1, 2], 30.0)):
        matrix = AequilibraeMatrix(index=index)
        matrix["demand"] = [[0.0, trips], [0.0, 0.0]]
        matrices.append(matrix)
    classes = [
        TrafficClass(name=name, mode="c", cost="time", demand=matrix, select_link_sets={"screen": [1]})
        for name, matrix in zip(("forward", "reverse", "subset"), matrices, strict=True)
    ]
    assig = assignment(network, classes)
    assig.execute(cores=1)
    project = SimpleNamespace(matrices=SimpleNamespace(folder=tmp_path, get=lambda *args, **kwargs: None))
    assig.save_select_link_matrices("selected", project=project)
    with omx.open_file(str(tmp_path / "selected.omx"), "r") as output:
        assert output.mapping("main_index") == {1: 0, 2: 1, 3: 2}
        for name, origin, destination, trips in (
            ("forward", 0, 2, 10), ("reverse", 2, 0, 20), ("subset", 0, 1, 30),
        ):
            expected = np.zeros((3, 3))
            expected[origin, destination] = trips
            np.testing.assert_array_equal(output[f"screen_{name}_demand"][:], expected)


def test_directional_fields_and_preloads_share_network_order(demand):
    network = Network(pd.DataFrame({
        "link_id": [23, 9], "a_node": [1, 2], "b_node": [2, 3], "direction": [0, 0], "modes": ["c", "c"],
        "time_ab": [1.0, 2.0], "time_ba": [4.0, 5.0],
        "capacity_ab": [100.0, 200.0], "capacity_ba": [300.0, 400.0],
    }))
    demand["demand"][1, 0] = 20.0
    traffic = TrafficClass(mode="c", cost="time", demand=demand, skimming=["time"])
    preload = pd.DataFrame({"link_id": [23], "direction": [-1], "preload": [5.0]})
    assig = assignment(network, [traffic], name_mapping={"alpha": 0.0, "beta": 4.0}, preload=preload)
    results = assig.execute(cores=1)
    np.testing.assert_allclose(assig.class_results["c"].total_link_loads, [20, 10, 20, 10])
    assert assig.class_results["c"].skims.matrices["time"][0, 1] == 3.0
    assert assig.class_results["c"].skims.matrices["time"][1, 0] == 9.0
    np.testing.assert_allclose(results.loc[[9, 23], "Congested_Time_AB"], [2, 1])
    np.testing.assert_allclose(results.loc[[9, 23], "Congested_Time_BA"], [5, 4])
    np.testing.assert_allclose(results.loc[[9, 23], "Preload_BA"], [0, 5])
    np.testing.assert_allclose(results.loc[[9, 23], "PCE_BA"], [20, 25])


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("penalty", [0.5, None])
def test_public_and_assignment_graphs_share_routing_policy(network, demand, blocking, penalty):
    network.turn_restrictions = pd.DataFrame(
        {"from_node": [1], "via_node": [2], "to_node": [3], "penalty": [penalty], "modes": ["c"]}
    )
    traffic = TrafficClass(mode="c", cost="time", demand=demand, block_centroid_flows=blocking)
    public = network.build_graph("c", "time", block_centroid_flows=blocking)
    assig = assignment(network, [traffic])
    assig.execute(cores=1)
    path = public.compute_path(1, 3)
    mapping = assig.class_results["c"].state.mapping
    expected = np.zeros(mapping.link_count)
    for link, direction in zip(path.path, path.path_link_directions, strict=True):
        expected[(mapping.link_ids == link) & (mapping.directions == direction)] = 10.0
    np.testing.assert_array_equal(assig.class_results["c"].total_link_loads, expected)


def test_network_mutation_is_isolated_and_assignment_edits_are_revalidated(network, demand):
    assig = assignment(network, [TrafficClass(mode="c", cost="time", demand=demand)])
    network.links["capacity"] = 0.0
    assig.execute(cores=1)
    assig.network.links["capacity"] = 0.0
    with pytest.raises(ValueError, match="positive finite"):
        assig.execute(cores=1)


def test_project_network_loads_once_at_construction(sioux_falls_example, monkeypatch):
    project = sioux_falls_example
    demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])
    traffic = TrafficClass(mode="c", cost="free_flow_time", demand=demand, block_centroid_flows=False)
    source = project.network
    load = source._load_graph_data
    calls = []

    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return load(*args, **kwargs)

    monkeypatch.setattr(source, "_load_graph_data", read)
    assig = TrafficAssignment(
        network=source, classes=[traffic], time_field="free_flow_time", capacity_field="capacity",
        vdf=bpr, algorithm="all-or-nothing",
    )
    assert len(calls) == 1
    assert assig.assignment is None
    with project.db_connection as conn:
        conn.execute("UPDATE links SET free_flow_time=free_flow_time*2, distance=999")
    assig.execute(cores=1)
    assert len(calls) == 1
    original = assig.free_flow_tt.copy()
    skims = assig.skim_congested(["distance"], return_matrices=True)["c"].matrices["distance"].copy()
    project.close()
    assig.execute(cores=1)
    np.testing.assert_array_equal(assig.free_flow_tt, original)
    current_skims = assig.skim_congested(["distance"], return_matrices=True)["c"].matrices["distance"]
    np.testing.assert_array_equal(current_skims, skims)


@pytest.mark.parametrize("field", ["time_field", "capacity_field"])
def test_project_field_validation_at_construction(sioux_falls_example, field):
    project = sioux_falls_example
    demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])
    configuration = {"time_field": "free_flow_time", "capacity_field": "capacity"}
    configuration[field] = "missing"
    with pytest.raises(ValueError, match="not in network"):
        TrafficAssignment(
            network=project.network,
            classes=[TrafficClass(mode="c", cost="free_flow_time", demand=demand)],
            vdf=bpr, algorithm="all-or-nothing", **configuration,
        )


def test_in_memory_network_needs_no_preparation_or_modes_column(network, demand):
    source = Network(network.links.drop(columns="modes"))
    assig = assignment(source, [TrafficClass(mode="h", cost="time", demand=demand)])
    assig.execute(cores=1)
    np.testing.assert_array_equal(assig.class_results["h"].total_link_loads, [0, 10, 0, 10, 0, 0])
    assert not hasattr(source, "routing")
    assert "modes" not in source.links


def test_mode_excluded_links_cannot_bypass_turn_bans(network, demand):
    network.links = network.links.copy()
    network.links.loc[network.links.link_id == 3, ["a_node", "b_node", "modes"]] = [2, 4, "h"]
    network.turn_restrictions = pd.DataFrame(
        {"from_node": [1], "via_node": [2], "to_node": [3], "penalty": [None], "modes": ["c"]}
    )
    for compact in (False, True):
        assert network.build_graph("c", "time", compact=compact).compute_path(1, 3).path is None
    assig = assignment(network, [TrafficClass(mode="c", cost="time", demand=demand)])
    assig.execute(cores=1)
    assert assig.class_results["c"].unassigned_demand == 10
    assert not assig.class_results["c"].total_link_loads.any()
