import json
import pathlib
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths import TransitAssignment, TransitClass
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.create_example import create_example
from aequilibrae.transit import Transit
from aequilibrae.matrix import AequilibraeMatrix


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_traffic_assignment_scenarios(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    # Build graphs for the scenario
    scenario_example.network.build_graphs(fields=["distance", "capacity_ab", "capacity_ba"], modes=["c"])
    graph = scenario_example.network.graphs["c"]
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    try:
        mat = scenario_example.matrices.get_matrix("demand_omx")
    except Exception:
        # Expected to fail for non-sioux_falls scenarios as theres no demand_omx
        assert scenario != "root"
        return

    mat.computational_view()

    assigclass = TrafficClass("car", graph, mat)
    assignment = TrafficAssignment(scenario_example)
    assignment.add_class(assigclass)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("distance")
    assignment.max_iter = 5
    assignment.set_algorithm("msa")

    assignment.execute()

    assert assigclass.results.total_link_loads is not None
    assert len(assigclass.results.total_link_loads) > 0
    mat.close()


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_transit_assignment_scenarios(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    data = Transit(scenario_example)
    try:
        graph = data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method="overlapping_regions",
        )
    except ValueError:
        assert scenario != "coquimbo"
        return

    scenario_example.network.build_graphs(modes=["c"])
    graph.create_line_geometry(method="connector project match", graph="c")
    transit_graph = graph.to_transit_graph()
    zones_in_the_model = len(transit_graph.centroids)
    names_list = ["pt"]

    mat = AequilibraeMatrix()
    mat.create_empty(zones=zones_in_the_model, matrix_names=names_list, memory_only=True)
    mat.index = transit_graph.centroids[:]
    mat.matrices[:, :, 0] = np.full((zones_in_the_model, zones_in_the_model), 1.0)
    mat.computational_view()

    assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)
    assig = TransitAssignment()
    assig.add_class(assigclass)
    assig.set_time_field("trav_time")
    assig.set_frequency_field("freq")
    assig.set_algorithm("os")
    assigclass.set_demand_matrix_core("pt")

    assig.execute()

    results = assig.results()
    assert results is not None

    assig.save_results(table_name=f"transit_test_{scenario}")

    # Verify the result was saved
    saved_results = scenario_example.results.list()
    table_names = saved_results["table_name"].tolist() if len(saved_results) > 0 else []
    assert f"transit_test_{scenario}" in table_names


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_matrices_scenarios(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    matrices = scenario_example.matrices
    df = matrices.list()

    if len(df) > 0:
        first_matrix = df.iloc[0]["name"]
        rec = matrices.get_record(first_matrix)
        assert rec.name is not None
        assert rec.name == first_matrix
    else:
        assert scenario != "root"


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_route_choice_scenarios(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    scenario_example.network.build_graphs(fields=["distance"], modes=["c"])
    graph = scenario_example.network.graphs["c"]
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    if len(graph.centroids) >= 2:
        rc = RouteChoiceSet(graph)
        a, b = graph.centroids[0], graph.centroids[-1]
        shape = (graph.num_zones, graph.num_zones)

        results = rc.run(int(a), int(b), shape, max_routes=3, max_depth=2)

        assert isinstance(results, list)
        assert len(results) <= 3

        for route in results:
            assert isinstance(route, tuple)
            assert len(route) > 0
    else:
        assert scenario == "nauru"  # Only Nauru doesn't have centroids


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_results_scenarios(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    results = scenario_example.results
    table_name = f"test_result_{scenario}"

    # Create a new result record
    record = results.new_record(
        table_name,
        procedure="test_procedure",
        procedure_id=f"test_id_{scenario}",
        procedure_report=json.dumps({"status": "success", "scenario": scenario}),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description=f"Test result for {scenario} scenario",
    )

    # Verify the record was created
    assert record.table_name == table_name
    assert record.procedure == "test_procedure"
    assert results.check_exists(table_name)

    # Test saving data to the result
    test_data = pd.DataFrame({"id": [1, 2, 3], "scenario": [scenario] * 3, "value": [10, 20, 30]})

    record.set_data(test_data, index=False)

    # Verify data retrieval
    retrieved_data = record.get_data()
    assert len(retrieved_data) == 3
    assert list(retrieved_data.columns) == ["id", "scenario", "value"]

    # Clean up
    results.delete_record(table_name)
    assert not results.check_exists(table_name)


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_network_scenarios(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    network = scenario_example.network
    links1 = network.links.data
    nodes1 = network.nodes.data

    network.build_graphs(fields=["distance"], modes=["c"])
    graph1 = network.graphs["c"]

    other_scenarios = [s for s in ["root", "nauru", "coquimbo"] if s != scenario]
    if other_scenarios:
        other_scenario = other_scenarios[0]
        scenario_example.use_scenario(other_scenario)

        links2 = network.links.data
        nodes2 = network.nodes.data

        network.build_graphs(fields=["distance"], modes=["c"])
        graph2 = network.graphs["c"]

        # Pandas doesn't have a good way to assert frames not equal
        try:
            pd.testing.assert_frame_equal(links1, links2)
        except AssertionError:
            pass
        else:
            raise AssertionError

        try:
            pd.testing.assert_frame_equal(nodes1, nodes2)
        except AssertionError:
            pass
        else:
            raise AssertionError

        try:
            pd.testing.assert_frame_equal(graph1, graph2)
        except AssertionError:
            pass
        else:
            raise AssertionError


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_scenario_result_isolation(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    results = scenario_example.results
    table_name = f"isolation_test_{scenario}"

    # Create scenario-specific result
    _ = results.new_record(
        table_name,
        procedure="isolation_test",
        procedure_id=f"isolation_{scenario}",
        description=f"Testing isolation for {scenario}",
    )
    assert results.check_exists(table_name)

    # Switch to different scenario and verify isolation
    other_scenarios = [s for s in ["root", "nauru", "coquimbo"] if s != scenario]
    if other_scenarios:
        other_scenario = other_scenarios[0]
        scenario_example.use_scenario(other_scenario)
        results.reload()

        assert not results.check_exists(table_name)

        scenario_example.use_scenario(scenario)
        results.reload()

        results.delete_record(table_name)
        assert not results.check_exists(table_name)
