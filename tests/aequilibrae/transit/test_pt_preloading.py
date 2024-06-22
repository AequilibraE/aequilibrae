import numpy as np
import pandas as pd
import pytest
from typing import List

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, AequilibraeMatrix
from aequilibrae.utils.create_example import create_example

# Overall TODO:
# 1) Create test structure for the API for PT Preloading
# 2) Modify database structure to store PCE value for a service and have default info
# 3) Consider simple example scenarios to test correctness of algorithms
# 4) Decide whether more complex tests are necessary
# 5) Performance testing and speed ups in cython

# Build TODO:
# 1. Write assertions for build test.
# 2. Implement inclusion condition into build for start, end, middle
# 3. Implement pce & default pce
# 4. Optimisation & Additional Testing
# 5. Input timings surrounding midnight (ie going past 24hrs)

# Assignment TODO:
# 1. Write test(s)
# 2. Figure out where directions are accounted for (ie in the capacity vector 
#    and graph building are the directions split to not include an option with
#    both directions?) - this is important to understand just to make sure 
#    everything is working correctly!
# Test:
# Do aon, and check speed is reduced after preload on preloaded links.

# Extra TODO:
# 1. Remove unecessary inputs to test functions
# 2. Change fixtures to setUp and tearDown so project and matrices can be closed!


@pytest.fixture
def project(tmp_path):
    proj = create_example(str(tmp_path / "test_traffic_assignment"), from_model="coquimbo")
    proj.network.build_graphs()
    proj.activate()
    return proj

@pytest.fixture
def graphs(project: Project):
    return [project.network.graphs[c] for c in "ctwb"]

@pytest.fixture
def assignment(project: Project, graphs: List[Graph]):
    # NOT YET COMPLETED - COPIED FROM test_mc_traffic_assignment.py
    n = graphs[0].centroids.shape[0]
    for graph in graphs:
            graph.set_skimming(["travel_time"])
            graph.set_graph("travel_time")
            graph.set_blocked_centroid_flows(False)
            graph.graph['capacity'] = graph.graph['capacity'].fillna(1.0)
            graph.graph['travel_time'] = graph.graph['travel_time'].fillna(1.0)

    # THERE IS NO DEMAND MATRIX, WHAT SHOULD I DO?

    # Create a random matrix as a temporary fix so testing can begin!
    matrices = AequilibraeMatrix()
    matrices.create_empty(zones=n, matrix_names=['car', 'transit', 'walk', 'bike'])
    matrices.index = graphs[0].centroids

    np.random.seed(7)
    matrices.matrices[:, :, 0] = np.random.uniform(0, 50, size=(n, n))
    matrices.matrices[:, :, 1] = np.random.uniform(0, 50, size=(n, n))
    matrices.matrices[:, :, 2] = np.random.uniform(0, 50, size=(n, n))
    matrices.matrices[:, :, 3] = np.random.uniform(0, 50, size=(n, n))

    matrices.computational_view(['car', 'transit', 'walk', 'bike'])

    # car_matrix = project.matrices.get_matrix("demand_mc")
    # car_matrix.computational_view(["car"])
    # transit_matrix = project.matrices.get_matrix("demand_mc")
    # transit_matrix.computational_view(["transit"])
    # walk_matrix = project.matrices.get_matrix("demand_mc")
    # walk_matrix.computational_view(["walk"])
    # bike_matrix = project.matrices.get_matrix("demand_mc")
    # bike_matrix.computational_view(["bike"])

    assignment = TrafficAssignment()
    carclass = TrafficClass("car", graphs[0], matrices)
    carclass.set_pce(1.0)
    # transitclass = TrafficClass("transit", graphs[1], transit_matrix)
    # transitclass.set_pce(1.5)
    # walkclass = TrafficClass("walk", graphs[2], walk_matrix)
    # walkclass.set_pce(0.2)
    # bikeclass = TrafficClass("bike", graphs[3], walk_matrix)
    # bikeclass.set_pce(0.5)

    # assignment.set_classes([carclass, transitclass, walkclass, bikeclass])
    assignment.set_classes([carclass])

    for cls in assignment.classes:
            cls.graph.set_skimming(["travel_time", "distance"])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})

    assignment.set_capacity_field("capacity")
    assignment.set_time_field("travel_time")

    assignment.max_iter = 20
    assignment.set_algorithm("bfw")

    return assignment

class TestPTPreloaing:

    def test_run(self, project: Project, graphs: List[Graph], assignment: TrafficAssignment):
        """Tests a full run through of pt preloading."""
        # NOT YET COMPLETED!

        # Get preload
        preload = self.test_built_pt_preload(project, graphs)

        # Run non-preloaded assignment and get results
        assignment.execute()
        # without_preload = ... # dataframe containing all relevant fields for each link/dir
        
        # Run preloaded assignment
        assignment.set_pt_preload(preload)
        assignment.execute()

        # Get assignment results
        # with_preload = ... # dataframe containing all relevant fields for each link/dir

        # Check assignment run with preload gives lower speeds on preloaded links than without preload
        # MAKE SURE THAT THE ASSIGNMENT ACTUALLY RESETS BETWEEN EXECUTIONS!
        loaded = (preload != 0)

        assert False  # PLACEHOLDER

    def test_built_pt_preload(self, project: Project, graphs: List[Graph]):
        """
        Check that building pt preload works correctly for a basic example from
        the coquimbo network.
        """
        # Preload parameters
        to_24hrs = lambda hrs: int(hrs * 60 * 60) # hrs from midnight in seconds
        periods = [(to_24hrs(start), to_24hrs(end)) for start, end in [(7, 8), (6.5, 8.5), (5, 10)]]

        # Generate preloads
        preload_with_period = lambda period: project.network.build_pt_preload(graphs[0], *period)
        preloads = [preload_with_period(period) for period in periods]

        # Assertions about the preload and coquimbo network:
        # Check correct size
        for p in preloads:
            assert len(p) == len(graphs[0].graph)
        
        # Check preloads increase in size as time period increases
        for p1, p2 in zip(preloads, preloads[1:]):
            assert np.all(p1 <= p2)
            assert p1.sum() < p2.sum()

        # Return preload for further testing
        return preloads[1]

    def test_preloaded_assignment(self, project: Project, graphs: List[Graph], assignment: TrafficAssignment):
        """
        Check that the setting a preload and running an assignment works as intended.

        Preload several links with very high preload values and check nothing is assigned to those links.
        """
        # NOT YET COMPLETED!
        
        # Create custom preload data
        preload = np.zeros(len(graphs[0].graph)) # type: np.ndarray

        # Links chosen: (2274, 1), (43, 1) - both have modes ct
        l1, l2, d1, d2 = 2274, 10036, 1, 1

        # Convert to supernet_id indexing
        g = graphs[0].graph
        get_index = lambda link, dir: g[(g['link_id'] == link) & (g['direction'] == dir)]['__supernet_id__'].values[0]

        # Set values to infinity so no traffic can go onto them.
        preload[get_index(l1, d1)], preload[get_index(l2, d2)] = np.inf, np.inf

        # Run preloaded assignment
        assignment.set_pt_preload(preload) # NOT YET IMPLEMENTED!
        assignment.execute()

        # Check results
        # ASSERT NO TRAFFIC ON THESE LINKS!
        assert False
