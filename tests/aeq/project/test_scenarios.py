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


@pytest.mark.parametrize("from_scenario", ["root", "nauru", "coquimbo"])
def test_create_empty_scenario_from_different_scenarios(scenario_example, from_scenario):
    """Test creating empty scenarios from different starting scenarios"""
    scenario_example.use_scenario(from_scenario)

    new_scenario_name = f"empty_from_{from_scenario}"
    description = f"Empty scenario created from {from_scenario}"

    scenario_example.create_empty_scenario(new_scenario_name, description)

    # Verify scenario was registered in root
    scenario_example.use_scenario("root")
    scenarios = scenario_example.list_scenarios()
    assert new_scenario_name in scenarios["scenario_name"].values

    # Check description
    scenario_row = scenarios[scenarios["scenario_name"] == new_scenario_name]
    assert scenario_row.iloc[0]["description"] == description

    # Switch to new scenario and verify it's empty
    scenario_example.use_scenario(new_scenario_name)
    assert len(scenario_example.network.links.data) == 0
    assert len(scenario_example.network.nodes.data) == 0
    assert len(scenario_example.matrices.list()) == 0
    assert len(scenario_example.results.list()) == 0

    # Verify scenario files exist
    scenario_path = scenario_example.root_scenario.base_path / "scenarios" / new_scenario_name
    assert scenario_path.exists()
    assert (scenario_path / "project_database.sqlite").exists()
    assert (scenario_path / "matrices").exists()


@pytest.mark.parametrize("from_scenario", ["root", "nauru", "coquimbo"])
def test_clone_scenario_from_different_scenarios(scenario_example, from_scenario):
    """Test cloning scenarios from different starting scenarios"""
    scenario_example.use_scenario(from_scenario)

    # Get source scenario data for comparison
    source_links = scenario_example.network.links.data.copy()
    source_nodes = scenario_example.network.nodes.data.copy()
    source_matrices = scenario_example.matrices.list().copy()
    source_results = scenario_example.results.list().copy()

    new_scenario_name = f"clone_from_{from_scenario}"
    description = f"Cloned scenario from {from_scenario}"

    scenario_example.clone_scenario(new_scenario_name, description)

    # Verify scenario was registered in root
    scenario_example.use_scenario("root")
    scenarios = scenario_example.list_scenarios()
    assert new_scenario_name in scenarios["scenario_name"].values

    # Check description
    scenario_row = scenarios[scenarios["scenario_name"] == new_scenario_name]
    assert scenario_row.iloc[0]["description"] == description

    # Switch to cloned scenario and verify data matches source
    scenario_example.use_scenario(new_scenario_name)
    cloned_links = scenario_example.network.links.data
    cloned_nodes = scenario_example.network.nodes.data
    cloned_matrices = scenario_example.matrices.list()
    cloned_results = scenario_example.results.list()

    # Compare data (should be identical)
    pd.testing.assert_frame_equal(source_links, cloned_links)
    pd.testing.assert_frame_equal(source_nodes, cloned_nodes)
    pd.testing.assert_frame_equal(source_matrices, cloned_matrices)
    pd.testing.assert_frame_equal(source_results, cloned_results)

    # Verify scenario files exist
    scenario_path = scenario_example.root_scenario.base_path / "scenarios" / new_scenario_name
    assert scenario_path.exists()
    assert (scenario_path / "project_database.sqlite").exists()
    assert (scenario_path / "matrices").exists()


def test_create_empty_scenario_duplicate_name(scenario_example):
    """Test that creating scenario with duplicate name fails"""
    scenario_example.use_scenario("root")

    scenario_example.create_empty_scenario("test_duplicate", "First scenario")

    # Attempt to create another with same name should fail
    with pytest.raises(ValueError, match="a scenario of that name already exists"):
        scenario_example.create_empty_scenario("test_duplicate", "Second scenario")


def test_clone_scenario_duplicate_name(scenario_example):
    """Test that cloning scenario with duplicate name fails"""
    scenario_example.use_scenario("root")

    scenario_example.clone_scenario("test_clone_duplicate", "First clone")

    # Attempt to clone another with same name should fail
    with pytest.raises(ValueError, match="a scenario of that name already exists"):
        scenario_example.clone_scenario("test_clone_duplicate", "Second clone")


def test_scenario_operations_return_to_original(scenario_example):
    """Test that scenario operations return to the original scenario"""
    original_scenario = "nauru"
    scenario_example.use_scenario(original_scenario)

    # Create empty scenario - should return to original
    scenario_example.create_empty_scenario("test_return_empty", "Test return")
    assert scenario_example.scenario.name == original_scenario

    # Clone scenario - should return to original
    scenario_example.clone_scenario("test_return_clone", "Test return")
    assert scenario_example.scenario.name == original_scenario


def test_scenario_isolation_after_creation(scenario_example):
    """Test that new scenarios are properly isolated"""
    scenario_example.use_scenario("root")

    # Add some data to root
    results = scenario_example.results
    results.new_record("root_specific", "test", "test_id", description="Root only data")

    # Create empty scenario
    scenario_example.create_empty_scenario("isolated_empty", "Isolated test")
    scenario_example.use_scenario("isolated_empty")

    # Verify isolation
    empty_results = scenario_example.results.list()
    if len(empty_results) > 0:
        assert "root_specific" not in empty_results["table_name"].values

    # Clone scenario
    scenario_example.use_scenario("root")
    scenario_example.clone_scenario("isolated_clone", "Isolated clone")
    scenario_example.use_scenario("isolated_clone")

    # Verify cloned data exists
    clone_results = scenario_example.results.list()
    assert len(clone_results) > 0, "No results found in cloned scenario"
    assert "root_specific" in clone_results["table_name"].values, "'root_specific' not found in cloned scenario results"

    # Modify clone data shouldn't affect root
    results = scenario_example.results
    results.new_record("clone_specific", "test", "test_id", description="Clone only data")

    scenario_example.use_scenario("root")
    root_results = scenario_example.results.list()
    assert "clone_specific" not in root_results["table_name"].values


@pytest.mark.parametrize("scenario", ["root", "nauru", "coquimbo"])
def test_scenario_run_module_persistence(scenario_example, scenario):
    scenario_example.use_scenario(scenario)

    # For the root module we should have one matrix in the summary
    if scenario == "root":
        assert "demand_omx" in scenario_example.run.matrix_summary()
    else:
        # For the others we shouldn't have any matrices, and the "run" dir shouldn't exist
        assert len(scenario_example.run.matrix_summary()) == 0
        assert not (scenario_example.project_base_path / "run").exists()


def test_scenario_use_scenario_must_exists(scenario_example):
    with pytest.raises(ValueError, match="scenario 'a scenario that doesn't exist' does not exist"):
        scenario_example.use_scenario("a scenario that doesn't exist")
