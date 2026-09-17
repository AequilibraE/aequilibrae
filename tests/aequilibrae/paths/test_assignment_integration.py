"""Assignment translation, output ownership and optimizer units."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aequilibrae import Graph, TrafficAssignment, TrafficClass
from aequilibrae.paths.assignment_context import AssignmentInputs, assignment_demand
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.outputs import LoadingOutputs, SkimmingOutputs
from aequilibrae.paths.vdf import bpr


def diamond(turn=False):
    graph = Graph()
    graph.network = pd.DataFrame(
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
    graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_skimming(["time", "distance"])
    if turn:
        graph.set_turn_restrictions(
            pd.DataFrame(
                {
                    "from_node": [10],
                    "via_node": [20],
                    "to_node": [40],
                    "penalty": [0.5],
                }
            )
        )
    return graph


def matrix_for(graph, demand=None):
    if demand is None:
        demand = np.zeros((graph.num_zones, graph.num_zones, 2))
        demand[0, -1] = [8.0, 12.0]
    return SimpleNamespace(
        matrix_view=demand,
        view_names=["work", "other"][: demand.shape[2] if demand.ndim == 3 else 1],
        index=graph.centroids.copy(),
        file_path=None,
        zones=graph.num_zones,
    )


def assignment_for(graph, matrix=None, algorithm="all-or-nothing", pce=2.5, cores=1, iterations=8):
    traffic = TrafficClass("cars", graph, matrix or matrix_for(graph))
    traffic.set_pce(pce)
    traffic.set_select_links({"upper": [(71, 1)], "either": [(71, 1), (55, 1)]})
    assignment = TrafficAssignment()
    assignment.set_classes([traffic])
    assignment.set_cores(cores, elementwise_cores=1)
    assignment.set_time_field("time")
    assignment.set_capacity_field("capacity")
    assignment.set_vdf(bpr, {"alpha": 0.15, "beta": 1.0})
    assignment.max_iter = iterations
    assignment.rgap_target = 0.0
    assignment.set_algorithm(algorithm)
    return assignment, traffic


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cores", [1, 2])
def test_aon_uses_compact_owners_and_keeps_demand_units(turn, cores):
    graph = diamond(turn)
    assignment, traffic = assignment_for(graph, cores=cores)
    original = traffic.matrix.matrix_view.copy()
    graph_costs = graph.compact_cost.copy()
    graph_skims = graph.compact_skims.copy()
    assignment.execute()

    results = traffic.results
    assert isinstance(results.skims, SkimmingOutputs)
    assert isinstance(results.output.loading, LoadingOutputs)
    inputs = assignment.assignment.inputs[traffic._id]
    assert isinstance(inputs.routing, TurnBasedContext if turn else NodeBasedContext)
    assert results.output.loading.link_count == graph.compact_num_links
    loads = results.get_load_results()
    for link in (71, 12):
        assert loads.loc[link, "work_ab"] == 8.0
        assert loads.loc[link, "other_ab"] == 12.0
    for link in (55, 24):
        assert loads.loc[link, "work_ab"] == 0.0
    np.testing.assert_allclose(results.link_loads, traffic._aon_results.link_loads)
    np.testing.assert_allclose(assignment.assignment.total_flow, 2.5 * results.total_link_loads)
    np.testing.assert_array_equal(results.select_link_od.matrices["upper"][0, 3], [8.0, 12.0])
    np.testing.assert_array_equal(results.select_link_loading["either"], results.compact_link_loads)
    assert results.total_turn_penalty == (10.0 if turn else 0.0)
    assert assignment.assignment.fw_total_turn_cost == 2.5 * results.total_turn_penalty
    assert results.skims.matrices["time"][0, 3] == (2.5 if turn else 2.0)
    assert results.skims.matrices["distance"][0, 3] == 7.0
    np.testing.assert_array_equal(traffic.matrix.matrix_view, original)
    np.testing.assert_array_equal(graph.compact_cost, graph_costs)
    np.testing.assert_array_equal(graph.compact_skims, graph_skims)


@pytest.mark.parametrize("algorithm", ["msa", "frank-wolfe", "cfw", "bfw"])
@pytest.mark.parametrize("line_search", ["exact", "trapezoidal"])
def test_turn_cost_and_selected_outputs_follow_accepted_solution(algorithm, line_search):
    assignment, traffic = assignment_for(diamond(True), algorithm=algorithm, iterations=12)
    assignment.set_line_search(line_search)
    assignment.set_algorithm(algorithm)
    assignment.execute()
    results = traffic.results
    upper = results.get_load_results().loc[71, "work_tot"] + results.get_load_results().loc[71, "other_tot"]
    assert results.total_turn_penalty == pytest.approx(0.5 * upper)
    np.testing.assert_allclose(results.select_link_loading["either"], results.compact_link_loads)
    np.testing.assert_allclose(results.select_link_od.matrices["either"][0, 3], [8.0, 12.0])
    assert results.select_link_od.matrices["upper"].sum() == pytest.approx(upper)

    optimizer = assignment.assignment
    current = traffic.pce * (
        np.dot(optimizer.congested_time + traffic.fixed_cost, results.total_link_loads) + results.total_turn_penalty
    )
    aon = traffic.pce * (
        np.dot(optimizer.congested_time + traffic.fixed_cost, traffic._aon_results.total_link_loads)
        + traffic._aon_results.total_turn_penalty
    )
    assert current >= aon - 1e-10
    assert optimizer.rgap == pytest.approx(abs(current - aon) / current)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_demand_rejected_at_assignment_entry(bad):
    graph = diamond()
    matrix = matrix_for(graph)
    matrix.matrix_view[0, 1, 0] = bad
    assignment, _ = assignment_for(graph, matrix)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        assignment.execute()


def test_strided_demand_is_copied_once_without_reshaping_caller():
    graph = diamond()
    source = np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4)
    view = source[:, :, 1:3]
    matrix = matrix_for(graph, view)
    packed = assignment_demand(matrix, graph.centroids)
    assert packed.flags.c_contiguous
    assert not packed.flags.writeable
    assert matrix.matrix_view is view
    assert not np.shares_memory(packed, source)
    np.testing.assert_array_equal(packed, view)

    matrix = matrix_for(graph, source[:, :, 0])
    packed = assignment_demand(matrix, graph.centroids)
    assert packed.shape == (4, 4, 1)
    assert matrix.matrix_view.ndim == 2


@pytest.mark.parametrize("cores", [1, 2])
def test_unreachable_demand_is_reported_without_loading_or_counting_intrazonal(caplog, cores):
    graph = diamond()
    matrix = matrix_for(graph)
    matrix.matrix_view[3, 0] = [3.0, 7.0]
    matrix.matrix_view[3, 3] = [100.0, 200.0]
    assignment, traffic = assignment_for(graph, matrix, cores=cores)
    assignment.execute()
    assert traffic.results.unassigned_demand == 10.0
    assert traffic._aon_results.unassigned_demand == 10.0
    assert "10 demand could not be assigned" in caplog.text
    assert np.isinf(traffic.results.skims.matrices["time"][3, 0])
    assert traffic.results.skims.matrices["time"][3, 3] == 0.0
    assert traffic.results.select_link_od.demand[3].sum() == 0.0
    assert traffic.results.total_link_loads.sum() == 40.0

    traffic.results.reset()
    assert traffic.results.unassigned_demand == 0.0
    assert traffic.results.total_turn_penalty == 0.0


def test_repeated_execution_clears_history_and_preserves_old_output_views():
    assignment, traffic = assignment_for(diamond(True), algorithm="bfw")
    assignment.execute()
    first = traffic.results.output
    retained = first.loading.link_loads
    expected = retained.copy()
    first_report_length = len(assignment.assignment.convergence_report["iteration"])
    assignment.execute()
    np.testing.assert_allclose(traffic.results.compact_link_loads, expected)
    np.testing.assert_array_equal(retained, expected)
    assert traffic.results.output is not first
    assert len(assignment.assignment.convergence_report["iteration"]) == first_report_length


def test_shared_graph_classes_have_independent_cost_buffers():
    graph = diamond()
    first = AssignmentInputs(graph, matrix_for(graph), "time", {}, 1)
    second = AssignmentInputs(graph, matrix_for(graph), "time", {}, 1)
    first.update_costs(np.ones(graph.num_links), np.zeros(graph.num_links))
    second.update_costs(np.full(graph.num_links, 3.0), np.zeros(graph.num_links))
    assert not np.shares_memory(first.costs, second.costs)
    assert not np.shares_memory(first.time_skim, second.time_skim)
    np.testing.assert_array_equal(first.routing.costs, np.ones(graph.compact_num_links))
    np.testing.assert_array_equal(second.routing.costs, np.full(graph.compact_num_links, 3.0))


def test_prepared_driver_reuses_output_storage_and_replaces_unassigned_total():
    graph = diamond(True)
    matrix = matrix_for(graph)
    inputs = AssignmentInputs(graph, matrix, "time", {"upper": [0]}, 2)
    state = inputs.make_state(1, 10000)
    output = state.output
    buffers = [output.loading.link_loads, output.skimming.skims, output.select_link.od.demand]
    addresses = [buffer.ctypes.data for buffer in buffers]
    for cost in (1.0, 2.0, np.inf, 1.0):
        inputs.update_costs(np.full(graph.num_links, cost), np.zeros(graph.num_links))
        inputs.driver.run(output)
        state.update_totals()
        assert [
            output.loading.link_loads.ctypes.data,
            output.skimming.skims.ctypes.data,
            output.select_link.od.demand.ctypes.data,
        ] == addresses
        assert output.unassigned_demand == (20.0 if np.isinf(cost) else 0.0)
    output.reset()
    assert not np.any(buffers[0])
    assert np.isinf(buffers[1]).all()
    assert not np.any(buffers[2])


def test_compressed_chain_and_removed_link_project_without_dummy_storage():
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [9, 2, 7],
            "a_node": [10, 20, 20],
            "b_node": [20, 30, 40],
            "direction": [0, 0, 0],
            "time": [1.0, 2.0, 3.0],
        }
    )
    graph.prepare_graph(np.array([10, 30]))
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    inputs = AssignmentInputs(graph, matrix_for(graph), "time", {}, 1)
    full = np.empty(graph.num_links)
    full[inputs.mapping.graph_ids] = graph.graph.time.to_numpy()
    inputs.update_costs(full, np.zeros_like(full))
    output = inputs.driver.run(inputs.driver.make_outputs())
    projected = inputs.mapping.full_loads(output.loading.link_loads)[inputs.mapping.graph_ids]
    for row, link in enumerate(graph.graph.itertuples()):
        expected = [8.0, 12.0] if link.link_id in (9, 2) and link.direction == 1 else [0.0, 0.0]
        np.testing.assert_array_equal(projected[row], expected)
    assert output.loading.link_count == graph.compact_num_links
    assert graph.compact_num_links < graph.num_links


@pytest.mark.parametrize("turn", [False, True])
def test_zero_demand_without_skims_leaves_no_optional_outputs(turn):
    graph = diamond(turn)
    graph.set_skimming([])
    matrix = matrix_for(graph, np.zeros((4, 4, 1)))
    inputs = AssignmentInputs(graph, matrix, "time", {}, 2)
    output = inputs.driver.run(inputs.driver.make_outputs())
    assert inputs.driver.origins.size == 0
    assert output.skimming is None
    assert output.select_link is None
    assert output.turn_cost_total == output.unassigned_demand == 0.0
    assert not np.any(output.loading.link_loads)


def test_unassigned_total_is_copied_blended_and_reset_with_output_group():
    graph = diamond()
    inputs = AssignmentInputs(graph, matrix_for(graph), "time", {}, 1)
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


def test_classes_sharing_a_graph_route_with_their_own_fixed_costs_and_pce():
    graph = diamond()
    graph.graph.loc[graph.graph.link_id == 71, "toll"] = 10.0
    cars = TrafficClass("cars", graph, matrix_for(graph))
    trucks = TrafficClass("trucks", graph, matrix_for(graph))
    cars.set_pce(0.5)
    trucks.set_pce(3.0)
    trucks.set_fixed_cost("toll")
    trucks.set_vot(2.0)
    assignment = TrafficAssignment()
    assignment.set_classes([cars, trucks])
    assignment.set_cores(2, elementwise_cores=1)
    assignment.set_time_field("time")
    assignment.set_capacity_field("capacity")
    assignment.set_vdf(bpr, {"alpha": 0.15, "beta": 1.0})
    assignment.set_algorithm("all-or-nothing")
    assignment.execute()
    assert cars.results.get_load_results().loc[71, "work_ab"] == 8.0
    assert trucks.results.get_load_results().loc[71, "work_ab"] == 0.0
    assert trucks.results.get_load_results().loc[55, "work_ab"] == 8.0
    expected = 0.5 * cars.results.total_link_loads + 3.0 * trucks.results.total_link_loads
    np.testing.assert_allclose(assignment.assignment.total_flow, expected)
    assert trucks.fixed_cost.max() == 5.0


@pytest.mark.parametrize("turn", [False, True])
def test_translation_preserves_existing_centroid_blocking_branch(turn):
    graph = diamond(turn)
    graph.set_blocked_centroid_flows(True)
    inputs = AssignmentInputs(graph, matrix_for(graph), "time", {}, 1)
    # The existing turn branch relies on Graph's generated connector bans,
    # rather than adding the routing context's blocked centroid prefix.
    expected = 0 if graph.has_turn_restrictions else graph.num_zones
    assert inputs.routing.blocked_centroid_count == expected
    if graph.has_turn_restrictions:
        np.testing.assert_array_equal(inputs.routing.turn_fs, graph.compact_turn_fs)
        np.testing.assert_array_equal(inputs.routing.turn_to_links, graph.compact_turn_to_arcs)
        np.testing.assert_array_equal(inputs.routing.turn_penalties, graph.compact_turn_penalties)


def test_core_settings_can_change_after_algorithm_selection():
    assignment, traffic = assignment_for(diamond(), cores=1)
    assignment.set_cores(2, elementwise_cores=2, threading_threshold=-1)
    assignment.execute()
    optimizer = assignment.assignment
    assert optimizer.cores == optimizer.elementwise_cores == 2
    assert optimizer.blend_options == {"cores": 2, "threading_threshold": -1}
    assert traffic.results.state.cores == 2


def test_reporting_snapshots_and_thread_changes_do_not_modify_output():
    assignment, traffic = assignment_for(diamond(True))
    assignment.execute()
    output = traffic.results.output
    snapshot = traffic.results.link_loads
    compact = traffic.results.compact_link_loads
    skim = traffic.results.skims.matrices["time"]
    expected = snapshot.copy()
    traffic.results.set_cores(2)
    assert traffic.results.output is output
    np.testing.assert_array_equal(traffic.results.link_loads, expected)
    assert not snapshot.flags.writeable
    traffic.results.reset()
    assert not np.any(compact)
    assert np.isinf(skim).all()
    np.testing.assert_array_equal(snapshot, expected)


def test_exports_use_named_skims_and_every_selected_demand_column(sioux_falls_single_class):
    project = sioux_falls_single_class
    assignment, traffic = assignment_for(diamond(True))
    assignment.execute()
    assignment.save_skims("new_skims", which_ones="all", project=project)
    assignment.save_select_link_matrices("new_selected", project=project)
    project.matrices.update_database()

    skims = project.matrices.get_matrix("new_skims_cars")
    selected = project.matrices.get_matrix("new_selected_omx")
    try:
        for name, values in traffic.results.skims.matrices.items():
            np.testing.assert_array_equal(skims.get_matrix(f"{name}_blended"), values)
            np.testing.assert_array_equal(skims.get_matrix(f"{name}_final"), values)
        expected_names = {f"{name}_cars_{column}" for name in ("upper", "either") for column in ("work", "other")}
        assert set(selected.names) == expected_names
        for name, values in traffic.results.select_link_od.matrices.items():
            for index, column in enumerate(("work", "other")):
                np.testing.assert_array_equal(selected.get_matrix(f"{name}_cars_{column}"), values[:, :, index])
    finally:
        skims.close()
        selected.close()


def test_congested_skims_use_final_costs_without_replacing_aon_or_graph_fields():
    graph = diamond(True)
    assignment, traffic = assignment_for(graph, algorithm="msa", iterations=4)
    assignment.execute()
    original_fields = graph.skim_fields.copy()
    original_skims = traffic._aon_results.skims
    original_values = original_skims.skims.copy()
    output = traffic.skim_congested("distance")
    assert isinstance(output, SkimmingOutputs)
    assert traffic.congested_skims is output
    assert traffic._aon_results.skims is original_skims
    np.testing.assert_array_equal(original_skims.skims, original_values)
    assert graph.skim_fields == original_fields
    full_times = traffic.congested_time[traffic.results.state.mapping.graph_ids]
    by_link = dict(zip(graph.graph.link_id, full_times, strict=True))
    shortest = min(by_link[71] + by_link[12] + 0.5, by_link[55] + by_link[24])
    assert output.matrices["__assignment_cost__"][0, 3] == pytest.approx(shortest)
    assert output.matrices["__congested_time__"][0, 3] == pytest.approx(shortest)
