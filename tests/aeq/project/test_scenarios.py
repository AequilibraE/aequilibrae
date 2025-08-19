import json
import pathlib
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths import TransitAssignment, TransitClass
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.create_example import create_example
from aequilibrae.transit import Transit
from aequilibrae.matrix import AequilibraeMatrix


class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.project = create_example(self.root / "sioux_falls", "sioux_falls")
        self.nauru = create_example(self.root / "sioux_falls" / "scenarios" / "nauru", "nauru")
        self.coquimbo = create_example(self.root / "sioux_falls" / "scenarios" / "coquimbo", "coquimbo")

        with self.project.db_connection as conn:
            conn.executemany("INSERT INTO scenarios (scenario_name) VALUES (?)", [("nauru",), ("coquimbo",)])

        with self.nauru.db_connection as conn:
            conn.execute("DROP TABLE scenarios")

        with self.coquimbo.db_connection as conn:
            conn.execute("DROP TABLE scenarios")

        self.scenarios = ["root", "nauru", "coquimbo"]

    def tearDown(self):
        self.project.close()
        # self.tmp.cleanup()

    def test_traffic_assignment_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                # Build graphs for the scenario
                self.project.network.build_graphs(fields=["distance", "capacity_ab", "capacity_ba"], modes=["c"])
                graph = self.project.network.graphs["c"]
                graph.set_graph("distance")
                graph.set_blocked_centroid_flows(False)

                try:
                    mat = self.project.matrices.get_matrix("demand_omx")
                except Exception:
                    # Expected to fail for non-sioux_falls scenarios as theres no demand_omx
                    self.assertNotEqual(scenario, "root")
                    continue

                mat.computational_view()

                assigclass = TrafficClass("car", graph, mat)
                assignment = TrafficAssignment(self.project)
                assignment.add_class(assigclass)
                assignment.set_vdf("BPR")
                assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
                assignment.set_capacity_field("capacity")
                assignment.set_time_field("distance")
                assignment.max_iter = 5
                assignment.set_algorithm("msa")

                assignment.execute()

                self.assertIsNotNone(assigclass.results.total_link_loads)
                self.assertGreater(len(assigclass.results.total_link_loads), 0)
                mat.close()

    def test_transit_assignment_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                data = Transit(self.project)
                try:
                    graph = data.create_graph(
                        with_outer_stop_transfers=False,
                        with_walking_edges=False,
                        blocking_centroid_flows=False,
                        connector_method="overlapping_regions",
                    )
                except ValueError:
                    self.assertNotEqual(scenario, "coquimbo")
                    continue

                self.project.network.build_graphs(modes=["c"])
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
                self.assertIsNotNone(results)

                assig.save_results(table_name=f"transit_test_{scenario}")

                # Verify the result was saved
                saved_results = self.project.results.list()
                table_names = saved_results["table_name"].tolist() if len(saved_results) > 0 else []
                self.assertIn(f"transit_test_{scenario}", table_names)

    def test_matrices_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                matrices = self.project.matrices
                df = matrices.list()

                if len(df) > 0:
                    first_matrix = df.iloc[0]["name"]
                    rec = matrices.get_record(first_matrix)
                    self.assertIsNotNone(rec.name)
                    self.assertEqual(rec.name, first_matrix)
                else:
                    self.assertNotEqual(scenario, "root")

    def test_route_choice_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                self.project.network.build_graphs(fields=["distance"], modes=["c"])
                graph = self.project.network.graphs["c"]
                graph.set_graph("distance")
                graph.set_blocked_centroid_flows(False)

                if len(graph.centroids) >= 2:
                    rc = RouteChoiceSet(graph)
                    a, b = graph.centroids[0], graph.centroids[-1]
                    shape = (graph.num_zones, graph.num_zones)

                    results = rc.run(int(a), int(b), shape, max_routes=3, max_depth=2)

                    self.assertIsInstance(results, list)
                    self.assertLessEqual(len(results), 3)

                    for route in results:
                        self.assertIsInstance(route, tuple)
                        self.assertGreater(len(route), 0)
                else:
                    self.assertEqual(scenario, "nauru")  # Only Nauru doesn't have centroids

    def test_results_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                results = self.project.results
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
                self.assertEqual(record.table_name, table_name)
                self.assertEqual(record.procedure, "test_procedure")
                self.assertTrue(results.check_exists(table_name))

                # Test saving data to the result
                test_data = pd.DataFrame({"id": [1, 2, 3], "scenario": [scenario] * 3, "value": [10, 20, 30]})

                record.set_data(test_data, index=False)

                # Verify data retrieval
                retrieved_data = record.get_data()
                self.assertEqual(len(retrieved_data), 3)
                self.assertListEqual(list(retrieved_data.columns), ["id", "scenario", "value"])

                # Clean up
                results.delete_record(table_name)
                self.assertFalse(results.check_exists(table_name))

    def test_network_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                network = self.project.network
                links1 = network.links.data
                nodes1 = network.nodes.data

                network.build_graphs(fields=["distance"], modes=["c"])
                graph1 = network.graphs["c"]

                other_scenarios = [s for s in self.scenarios if s != scenario]
                if other_scenarios:
                    other_scenario = other_scenarios[0]
                    self.project.switch_scenario(other_scenario)

                    links2 = network.links.data
                    nodes2 = network.nodes.data

                    network.build_graphs(fields=["distance"], modes=["c"])
                    graph2 = network.graphs["c"]

                    # Pandas doesn't have a good way to assert frames not equal
                    with self.assertRaises(AssertionError):
                        pd.testing.assert_frame_equal(links1, links2)

                    with self.assertRaises(AssertionError):
                        pd.testing.assert_frame_equal(nodes1, nodes2)

                    with self.assertRaises(AssertionError):
                        pd.testing.assert_frame_equal(graph1, graph2)

    def test_scenario_result_isloation(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario):
                self.project.switch_scenario(scenario)

                results = self.project.results
                table_name = f"isolation_test_{scenario}"

                # Create scenario-specific result
                _ = results.new_record(
                    table_name,
                    procedure="isolation_test",
                    procedure_id=f"isolation_{scenario}",
                    description=f"Testing isolation for {scenario}",
                )
                self.assertTrue(results.check_exists(table_name))

                # Switch to different scenario and verify isolation
                other_scenarios = [s for s in self.scenarios if s != scenario]
                if other_scenarios:
                    other_scenario = other_scenarios[0]
                    self.project.switch_scenario(other_scenario)
                    results.reload()

                    self.assertFalse(results.check_exists(table_name))

                    self.project.switch_scenario(scenario)
                    results.reload()

                    results.delete_record(table_name)
                    self.assertFalse(results.check_exists(table_name))
