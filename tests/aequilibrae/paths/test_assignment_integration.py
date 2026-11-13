"""Assignment translation, output ownership and optimizer units."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from aequilibrae.matrix.cython.dense_matrix import DenseMatrix
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.demand import assignment_demand
from aequilibrae.paths.cython.outputs import LoadingOutputs, SkimmingOutputs

from aequilibrae import Network, TrafficAssignment, TrafficClass
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import available_heaps
from aequilibrae.paths.assignment_context import AssignmentInputs
from aequilibrae.paths.vdf import bpr


def diamond(turn=False):
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 55, 24],
            "a_node": [10, 20, 10, 30],
            "b_node": [20, 40, 30, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0, 1.0, 2.0, 2.0],
            "distance": [3.0, 4.0, 5.0, 6.0],
            "toll": [0.0, 0.0, 0.0, 0.0],
            "capacity": [10.0] * 4,
        }
    )
    turns = None
    if turn:
        turns = pd.DataFrame(
            {
                "from_node": [10],
                "via_node": [20],
                "to_node": [40],
                "penalty": [0.5],
            }
        )
    return Network(links.assign(modes="c"), centroids=[10, 20, 30, 40], turn_restrictions=turns)


def matrix_for(network, demand=None):
    if demand is None:
        zones = len(network.centroids)
        demand = {
            "work": np.zeros((zones, zones)),
            "other": np.zeros((zones, zones)),
        }
        demand["work"][0, -1] = 8.0
        demand["other"][0, -1] = 12.0

    matrix = AequilibraeMatrix(index=network.centroids.copy())
    if isinstance(demand, dict):
        for name, values in demand.items():
            matrix[name] = values
    elif demand.ndim == 2:
        matrix["work"] = demand
    else:
        for index, name in enumerate(("work", "other")[: demand.shape[2]]):
            matrix[name] = demand[:, :, index]
    return matrix


def routing_inputs(network, matrix, time_field, selected_links, cores, *, block_centroid_flows=False, **kwargs):
    prepared = network.build_graph(
        "c",
        centroids=matrix.index,
        compact=True,
        remove_dead_ends=True,
        block_centroid_flows=block_centroid_flows,
    )
    return AssignmentInputs(
        prepared,
        matrix,
        time_field,
        selected_links,
        cores,
        field_values=network.field_values,
        skim_fields=kwargs.pop("skim_fields", ["time", "distance"]),
        **kwargs,
    )


def assignment_for(network, matrix=None, algorithm="all-or-nothing", pce=2.5, name_mapping=None, **options):
    traffic = TrafficClass(
        name="cars",
        mode="c",
        cost="time",
        demand=matrix or matrix_for(network),
        pce=pce,
        skimming=["time", "distance"],
        graph_options={"block_centroid_flows": False},
        select_link_sets={"upper": [(71, 1)], "either": [(71, 1), (55, 1)]},
        **options,
    )
    assignment = TrafficAssignment(
        network=network,
        classes=[traffic],
        time_field="time",
        capacity_field="capacity",
        vdf=bpr,
        name_mapping={"alpha": 0.15, "beta": 1.0} if name_mapping is None else name_mapping,
        algorithm=algorithm,
    )
    return assignment, traffic


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("tolls", [[np.nan] * 4, [np.nan, 2.0, 0.0, np.nan]])
def test_nan_fixed_costs_warn_and_assign_as_zero(caplog, turn, tolls):
    network = diamond(turn)
    network = replace(network, links=network.links.assign(toll=tolls))
    options = {"fixed_cost_field": "toll", "fixed_cost_multiplier": 2.0, "vot": 4.0}
    assignment, traffic = assignment_for(network, **options)
    assert "Cost field toll has NaN values. Converted to zero" in caplog.text
    reference_network = replace(network, links=network.links.assign(toll=network.links.toll.fillna(0.0)))
    reference, _ = assignment_for(reference_network, **options)
    for _ in range(2):
        assignment.execute(max_iterations=1, cores=1)
        reference.execute(max_iterations=1, cores=1)
        np.testing.assert_array_equal(assignment.results(), reference.results())
        np.testing.assert_array_equal(
            assignment.class_results[traffic.name].skims.skims,
            reference.class_results[traffic.name].skims.skims,
        )
        np.testing.assert_array_equal(
            assignment._class_states[0].fixed_cost, np.nan_to_num(assignment.network.field_values("toll")) / 2.0
        )
    np.testing.assert_array_equal(assignment.network.links.toll, tolls)


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("turn", [False, True])
def test_assignment_uses_selected_heap(heap, turn):
    assignment, traffic = assignment_for(diamond(turn), heap=heap)
    assignment.execute(max_iterations=8, rgap=0.0, cores=1, elementwise_cores=1)
    assert assignment.class_results[traffic.name].get_heaps() == available_heaps()
    assert assignment.class_results[traffic.name]._heap == assignment._class_states[0]._aon_results._heap == heap
    assert assignment.class_results[traffic.name].get_load_results().loc[71, "work_ab"] == 8.0


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cores", [1, 2])
def test_aon_uses_compact_owners_and_keeps_demand_units(turn, cores):
    network = diamond(turn)
    assignment, traffic = assignment_for(network)
    original = {name: values.copy() for name, values in traffic.demand.items()}
    assignment.execute(max_iterations=8, rgap=0.0, cores=cores, elementwise_cores=1)

    graph = assignment._class_states[0].graph
    results = assignment.class_results[traffic.name]
    assert isinstance(results.skims, SkimmingOutputs)
    assert isinstance(results.output.loading, LoadingOutputs)
    inputs = assignment.assignment.inputs[traffic.name]
    assert isinstance(inputs.routing, TurnBasedContext if turn else NodeBasedContext)
    assert results.output.loading.link_count == graph.num_links
    assert not np.shares_memory(inputs.costs, graph.cost)
    assert "time" in inputs.skimming.fields
    loads = results.get_load_results()
    for link in (71, 12):
        assert loads.loc[link, "work_ab"] == 8.0
        assert loads.loc[link, "other_ab"] == 12.0
    for link in (55, 24):
        assert loads.loc[link, "work_ab"] == 0.0
    np.testing.assert_allclose(results.link_loads, assignment._class_states[0]._aon_results.link_loads)
    np.testing.assert_allclose(assignment.assignment.total_flow, 2.5 * results.total_link_loads)
    np.testing.assert_array_equal(
        [matrix[0, 3] for matrix in results.select_link_od.matrices["upper"].values()], [8.0, 12.0]
    )
    np.testing.assert_array_equal(results.select_link_loading["either"], results.compact_link_loads)
    assert results.total_turn_penalty == (10.0 if turn else 0.0)
    assert assignment.assignment.fw_total_turn_cost == 2.5 * results.total_turn_penalty
    assert results.skims.matrices["time"][0, 3] == (2.5 if turn else 2.0)
    assert results.skims.matrices["distance"][0, 3] == 7.0
    for name, values in original.items():
        np.testing.assert_array_equal(traffic.demand[name], values)
    np.testing.assert_array_equal(graph.cost, np.zeros(graph.num_links))
    assert graph.skim_fields == []


@pytest.mark.parametrize("algorithm", ["msa", "frank-wolfe", "cfw", "bfw"])
def test_turn_cost_and_selected_outputs_follow_accepted_solution(algorithm):
    assignment, traffic = assignment_for(diamond(True), algorithm=algorithm)
    assignment.execute(max_iterations=12, rgap=0.0, cores=1, elementwise_cores=1)
    results = assignment.class_results[traffic.name]
    upper = results.get_load_results().loc[71, "work_tot"] + results.get_load_results().loc[71, "other_tot"]
    assert results.total_turn_penalty == pytest.approx(0.5 * upper)
    np.testing.assert_allclose(results.select_link_loading["either"], results.compact_link_loads)
    np.testing.assert_allclose(
        [matrix[0, 3] for matrix in results.select_link_od.matrices["either"].values()], [8.0, 12.0]
    )
    assert sum(matrix.sum() for matrix in results.select_link_od.matrices["upper"].values()) == pytest.approx(upper)

    optimizer = assignment.assignment
    current = traffic.pce * (
        np.dot(optimizer.congested_time + assignment._class_states[0].fixed_cost, results.total_link_loads)
        + results.total_turn_penalty
    )
    aon = traffic.pce * (
        np.dot(
            optimizer.congested_time + assignment._class_states[0].fixed_cost,
            assignment._class_states[0]._aon_results.total_link_loads,
        )
        + assignment._class_states[0]._aon_results.total_turn_penalty
    )
    assert current >= aon - 1e-10
    assert optimizer.rgap == pytest.approx(abs(current - aon) / current)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_demand_rejected_at_assignment_entry(bad):
    network = diamond()
    matrix = matrix_for(network)
    matrix["work"][0, 1] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        assignment_for(network, matrix)


def test_strided_demand_is_copied_once_without_reshaping_caller():
    network = diamond()
    source = np.arange(4 * 8, dtype=float).reshape(4, 8)
    view = {"work": source[:, ::2], "other": source[:, 1::2]}
    matrix = matrix_for(network, view)
    input_values = {name: values.copy() for name, values in matrix.items()}
    packed = assignment_demand(matrix, network.centroids)
    assert isinstance(packed, DenseMatrix)
    assert packed.field_names == tuple(matrix)
    for name, values in packed.to_matrix(network.centroids).items():
        assert not np.shares_memory(values, source)
        np.testing.assert_array_equal(values, input_values[name])

    matrix = matrix_for(network, source[:, ::2])
    packed = assignment_demand(matrix, network.centroids)
    assert (packed.origin_count, packed.destination_count, packed.field_count) == (4, 4, 1)
    assert matrix["work"].ndim == 2


@pytest.mark.parametrize("cores", [1, 2])
def test_unreachable_demand_is_reported_without_loading_or_counting_intrazonal(caplog, cores):
    network = diamond()
    matrix = matrix_for(network)
    matrix["work"][3, 0] = 3.0
    matrix["other"][3, 0] = 7.0
    matrix["work"][3, 3] = 100.0
    matrix["other"][3, 3] = 200.0
    assignment, traffic = assignment_for(network, matrix)
    assignment.execute(max_iterations=8, rgap=0.0, cores=cores, elementwise_cores=1)
    assert assignment.class_results[traffic.name].unassigned_demand == 10.0
    assert assignment._class_states[0]._aon_results.unassigned_demand == 10.0
    assert "10 demand could not be assigned" in caplog.text
    results = assignment.class_results[traffic.name]
    assert np.isinf(results.skims.matrices["time"][3, 0])
    assert results.skims.matrices["time"][3, 3] == 0.0
    assert all(
        matrix[3:4].nnz == 0 for matrices in results.select_link_od.matrices.values() for matrix in matrices.values()
    )
    assert results.total_link_loads.sum() == 40.0

    assignment.class_results[traffic.name].reset()
    assert assignment.class_results[traffic.name].unassigned_demand == 0.0
    assert assignment.class_results[traffic.name].total_turn_penalty == 0.0


def test_repeated_execution_clears_history_and_preserves_old_output_views():
    assignment, traffic = assignment_for(diamond(True), algorithm="bfw")
    assignment.execute(max_iterations=8, rgap=0.0, cores=1, elementwise_cores=1)
    first = assignment.class_results[traffic.name].output
    retained = first.loading.link_loads
    expected = retained.copy()
    first_report_length = len(assignment.assignment.convergence_report["iteration"])
    assignment.execute(max_iterations=8, rgap=0.0, cores=1, elementwise_cores=1)
    np.testing.assert_allclose(assignment.class_results[traffic.name].compact_link_loads, expected)
    np.testing.assert_array_equal(retained, expected)
    assert assignment.class_results[traffic.name].output is not first
    assert len(assignment.assignment.convergence_report["iteration"]) == first_report_length


def test_assignment_classes_build_independent_graphs_and_cost_buffers():
    network = diamond()
    configuration = TrafficClass(
        mode="c",
        cost="time",
        demand=matrix_for(network),
        graph_options={"block_centroid_flows": False},
    )
    prepared = network._assignment_graph(configuration)
    second_graph = network._assignment_graph(replace(configuration, name="trucks"))
    assert second_graph is not prepared
    assert second_graph.mapping is not prepared.mapping
    first, second = [
        AssignmentInputs(
            prepared if index == 0 else second_graph,
            configuration.demand,
            "time",
            {},
            1,
            field_values=network.field_values,
            skim_fields=["time", "distance"],
        )
        for index in range(2)
    ]
    assert first.mapping is prepared.mapping
    assert second.mapping is second_graph.mapping
    assert not np.shares_memory(first.routing.fs, second.routing.fs)
    assert not np.shares_memory(first.routing.heads, second.routing.heads)
    first.update_costs(np.ones(network.mapping.link_count), np.zeros(network.mapping.link_count))
    second.update_costs(np.full(network.mapping.link_count, 3.0), np.zeros(network.mapping.link_count))
    assert not np.shares_memory(first.costs, second.costs)
    assert first.time_skim is not second.time_skim
    assert "time" in first.skimming.fields
    np.testing.assert_array_equal(first.routing.costs, np.ones(prepared.mapping.link_count))
    np.testing.assert_array_equal(second.routing.costs, np.full(prepared.mapping.link_count, 3.0))


def test_prepared_driver_reuses_output_storage_and_replaces_unassigned_total():
    network = diamond(True)
    matrix = matrix_for(network)
    inputs = routing_inputs(network, matrix, "time", {"upper": [0]}, 2)
    state = inputs.make_state(1, 10000)
    output = state.output
    buffers = [output.loading.link_loads, output.skimming.skims]
    addresses = [buffer.ctypes.data for buffer in buffers]
    for cost in (1.0, 2.0, np.inf, 1.0):
        inputs.update_costs(np.full(network.mapping.link_count, cost), np.zeros(network.mapping.link_count))
        inputs.driver.run(output)
        state.update_totals()
        assert [
            output.loading.link_loads.ctypes.data,
            output.skimming.skims.ctypes.data,
        ] == addresses
        assert output.unassigned_demand == (20.0 if np.isinf(cost) else 0.0)
    output.reset()
    assert not np.any(buffers[0])
    assert np.isinf(buffers[1]).all()
    assert all(matrix.nnz == 0 for matrix in output.select_link.od.matrices["upper"].values())


def test_compressed_chain_and_removed_link_project_without_dummy_storage():
    links = pd.DataFrame(
        {
            "link_id": [9, 2, 7],
            "a_node": [10, 20, 20],
            "b_node": [20, 30, 40],
            "direction": [0, 0, 0],
            "time": [1.0, 2.0, 3.0],
        }
    )
    network = Network(links.assign(modes="c"), centroids=[10, 30])
    inputs = routing_inputs(network, matrix_for(network), "time", {}, 1, skim_fields=[])
    full = network.field_values("time")
    inputs.update_costs(full, np.zeros_like(full))
    output = inputs.driver.run(inputs.driver.make_outputs())
    projected = inputs.mapping.project_loads(output.loading.link_loads)
    mapping = inputs.mapping.network
    for row, (link_id, direction) in enumerate(zip(mapping.link_ids, mapping.directions, strict=True)):
        expected = [8.0, 12.0] if link_id in (9, 2) and direction == 1 else [0.0, 0.0]
        np.testing.assert_array_equal(projected[row], expected)
    assert output.loading.link_count == inputs.mapping.link_count
    assert inputs.mapping.link_count < inputs.mapping.network.link_count


@pytest.mark.parametrize("turn", [False, True])
def test_zero_demand_without_skims_leaves_no_optional_outputs(turn):
    network = diamond(turn)
    matrix = matrix_for(network, np.zeros((4, 4, 1)))
    inputs = routing_inputs(network, matrix, "time", {}, 2, skim_fields=[])
    output = inputs.driver.run(inputs.driver.make_outputs())
    assert inputs.driver.origins.size == 0
    assert output.skimming is None
    assert output.select_link is None
    assert output.turn_cost_total == output.unassigned_demand == 0.0
    assert not np.any(output.loading.link_loads)


def test_unassigned_total_is_copied_blended_and_reset_with_output_group():
    network = diamond()
    inputs = routing_inputs(network, matrix_for(network), "time", {}, 1)
    reached, missing, output = [inputs.driver.make_outputs() for _ in range(3)]
    inputs.update_costs(np.ones(4), np.zeros(4))
    inputs.driver.run(reached)
    inputs.update_costs(np.full(4, np.inf), np.zeros(4))
    inputs.driver.run(missing)
    assert reached.unassigned_demand == 0.0
    assert missing.unassigned_demand == 20.0
    output.copy_from(missing)
    assert output.unassigned_demand == 20.0
    output.blend_cfw(reached, missing, 0.25)
    assert output.unassigned_demand == 5.0
    output.blend_bfw(reached, missing, output, [0.2, 0.3, 0.5])
    assert output.unassigned_demand == 8.5
    output.blend_result(reached, output, 0.5)
    assert output.unassigned_demand == 4.25
    output.reset()
    assert output.unassigned_demand == 0.0


def test_classes_sharing_network_data_route_with_their_own_costs_and_pce():
    network = diamond()
    network = replace(network, links=network.links.assign(toll=np.where(network.links.link_id == 71, 10.0, 0.0)))
    cars = TrafficClass(
        name="cars",
        mode="c",
        cost="time",
        demand=matrix_for(network),
        pce=0.5,
        graph_options={"block_centroid_flows": False},
    )
    trucks = replace(cars, name="trucks", pce=3.0, vot=2.0, fixed_cost_field="toll")
    assignment = TrafficAssignment(
        network=network,
        classes=[cars, trucks],
        time_field="time",
        capacity_field="capacity",
        vdf=bpr,
        name_mapping={"alpha": 0.15, "beta": 1.0},
        algorithm="all-or-nothing",
    )
    assignment.execute(cores=2, elementwise_cores=1)
    car_results, truck_results = assignment.class_results["cars"], assignment.class_results["trucks"]
    assert car_results.get_load_results().loc[71, "work_ab"] == 8.0
    assert truck_results.get_load_results().loc[71, "work_ab"] == 0.0
    assert truck_results.get_load_results().loc[55, "work_ab"] == 8.0
    expected = 0.5 * car_results.total_link_loads + 3.0 * truck_results.total_link_loads
    np.testing.assert_allclose(assignment.assignment.total_flow, expected)
    assert assignment._class_states[1].fixed_cost.max() == 5.0


@pytest.mark.parametrize("turn", [False, True])
def test_translation_preserves_graph_centroid_blocking(turn):
    from .routing_helpers import search

    network = diamond(turn)
    inputs = routing_inputs(network, matrix_for(network), "time", {}, 1, block_centroid_flows=True)
    # The turn branch relies on the graph's generated connector bans,
    # rather than adding the routing context's blocked centroid prefix.
    expected = 0 if turn else len(network.centroids)
    assert inputs.routing.blocked_centroid_count == expected
    assert not search(inputs.routing, 0).reachable_to(3)


def test_core_settings_can_change_after_algorithm_selection():
    assignment, traffic = assignment_for(diamond())
    assignment.execute(max_iterations=8, rgap=0.0, cores=2, elementwise_cores=2, threading_threshold=-1)
    optimizer = assignment.assignment
    assert optimizer.cores == optimizer.elementwise_cores == 2
    assert optimizer.blend_options == {"cores": 2, "threading_threshold": -1}
    assert assignment.class_results[traffic.name].state.cores == 2


def test_reporting_snapshots_and_thread_changes_do_not_modify_output():
    assignment, traffic = assignment_for(diamond(True))
    assignment.execute(max_iterations=8, rgap=0.0, cores=1, elementwise_cores=1)
    output = assignment.class_results[traffic.name].output
    snapshot = assignment.class_results[traffic.name].link_loads
    compact = assignment.class_results[traffic.name].compact_link_loads
    skim = assignment.class_results[traffic.name].skims.matrices["time"]
    expected = snapshot.copy()
    assignment.class_results[traffic.name].set_cores(2)
    assert assignment.class_results[traffic.name].output is output
    np.testing.assert_array_equal(assignment.class_results[traffic.name].link_loads, expected)
    assert not snapshot.flags.writeable
    assignment.class_results[traffic.name].reset()
    assert not np.any(compact)
    assert np.isinf(skim).all()
    np.testing.assert_array_equal(snapshot, expected)


def test_exports_use_named_skims_and_every_selected_demand_column(sioux_falls_single_class):
    project = sioux_falls_single_class
    assignment, traffic = assignment_for(diamond(True))
    assignment.execute(max_iterations=8, rgap=0.0, cores=1, elementwise_cores=1)
    file_names = assignment.save_skims("new_skims", which_ones="all", project=project)
    assert file_names == ["new_skims_cars.omx"]
    assignment.save_select_link_matrices("new_selected", project=project)

    metadata = project.matrices.get_available_matrices_and_metadata()[file_names[0]]
    assert metadata["procedure"] == "Traffic Assignment"
    assert metadata["procedure_id"] == assignment.procedure_id
    assert metadata["timestamp"] == assignment.procedure_date
    skims = project.matrices.get_matrix(file_names[0])
    selected = project.matrices.get_matrix("new_selected.omx", sparse=True)
    try:
        for name, values in assignment.class_results[traffic.name].skims.matrices.items():
            np.testing.assert_array_equal(skims[f"{name}_blended"], values)
            np.testing.assert_array_equal(skims[f"{name}_final"], values)
        expected_names = {f"{name}_cars_{column}" for name in ("upper", "either") for column in ("work", "other")}
        assert set(selected) == expected_names
        results = assignment.class_results[traffic.name]
        for name, values in results.select_link_od.matrices.items():
            for column, matrix in values.items():
                np.testing.assert_array_equal(selected[f"{name}_cars_{column}"].toarray(), matrix.toarray())
    finally:
        skims.close()
        selected.close()


def test_congested_skims_use_final_costs_without_replacing_aon_or_graph_fields():
    assignment, traffic = assignment_for(diamond(True), algorithm="msa")
    assignment.execute(max_iterations=4, rgap=0.0, cores=1, elementwise_cores=1)
    graph = assignment._class_states[0].graph
    original_fields = graph.skim_fields.copy()
    original_skims = assignment._class_states[0]._aon_results.skims
    original_values = original_skims.skims.copy()
    output = assignment.skim_congested("distance", return_matrices=True)[traffic.name]
    assert isinstance(output, SkimmingOutputs)
    assert assignment._class_states[0].congested_skims is output
    assert assignment._class_states[0]._aon_results.skims is original_skims
    np.testing.assert_array_equal(original_skims.skims, original_values)
    assert graph.skim_fields == original_fields
    full_times = assignment._class_states[0].congested_time
    by_link = dict(zip(assignment.class_results[traffic.name].state.mapping.network.link_ids, full_times, strict=True))
    shortest = min(by_link[71] + by_link[12] + 0.5, by_link[55] + by_link[24])
    assert output.matrices["__assignment_cost__"][0, 3] == pytest.approx(shortest)
    assert output.matrices["__congested_time__"][0, 3] == pytest.approx(shortest)


@pytest.mark.parametrize("turn", [False, True])
def test_assignment_time_and_cost_skims_have_separate_meanings(turn):
    from .routing_helpers import search

    network = diamond(turn)
    inputs = routing_inputs(
        network,
        matrix_for(network),
        "time",
        {},
        1,
        skim_fields=["time", "distance", "assignment_cost"],
        cost_name="assignment_cost",
    )
    full = network.field_values("time")
    inputs.update_costs(full, np.full(inputs.mapping.network.link_count, 0.25))
    raw = inputs.driver.run(inputs.driver.make_outputs()).skimming
    output = inputs.report_skims(raw)
    assert output.matrices["time"][0, 3] == (2.5 if turn else 2.0)
    assert output.matrices["assignment_cost"][0, 3] == (3.0 if turn else 2.5)
    for origin in range(len(network.centroids)):
        results = search(inputs.routing, origin)
        for destination in range(len(network.centroids)):
            assert output.matrices["assignment_cost"][origin, destination] == results.path_cost_to(destination)


@pytest.mark.parametrize("turn", [False, True])
def test_congested_time_excludes_tolls_but_keeps_turn_delay(turn):
    network = diamond(turn)
    network = replace(network, links=network.links.assign(toll=np.where(network.links.link_id == 71, 2.0, 0.0)))
    assignment, traffic = assignment_for(
        network, fixed_cost_field="toll", vot=2.0, name_mapping={"alpha": 0.0, "beta": 1.0}
    )
    assignment.execute(max_iterations=8, rgap=0.0, cores=1, elementwise_cores=1)

    output = assignment.skim_congested("distance", return_matrices=True)[traffic.name]
    matrices = output.matrices
    time = matrices["__congested_time__"]
    cost = matrices["__assignment_cost__"]

    assert set(matrices) == {"distance", "__congested_time__", "__assignment_cost__"}
    assert time[0, 3] == (2.5 if turn else 2.0)
    assert cost[0, 3] == time[0, 3] + 1.0
    assert matrices["distance"][0, 3] == 7.0
    assert cost[0, 0] == time[0, 0] == 0.0
    assert np.isinf(time[3, 0])
    assert np.isinf(cost[3, 0])
    assert not time.flags.writeable


def test_congested_skim_reporting_does_not_add_turn_delay_twice():
    network = diamond(True)
    inputs = routing_inputs(
        network,
        matrix_for(network),
        "time",
        {},
        1,
        skim_fields=["time", "distance", "assignment_cost"],
        cost_name="assignment_cost",
    )
    full = network.field_values("time")
    inputs.update_costs(full, np.zeros_like(full))
    raw = inputs.driver.run(inputs.driver.make_outputs()).skimming
    original = raw.skims.copy()
    first = inputs.report_skims(raw)
    second = inputs.report_skims(raw)

    assert raw.matrices["time"][0, 3] == 2.0
    assert first.matrices["time"][0, 3] == 2.5
    np.testing.assert_array_equal(first.skims, second.skims)
    np.testing.assert_array_equal(raw.skims, original)
    raw.reset()
    assert first.matrices["time"][0, 3] == 2.5
