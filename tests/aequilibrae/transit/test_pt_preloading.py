import numpy as np
import pandas as pd
import pytest
from typing import List

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project
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
# Test:
# Do aon, and check speed is reduced after preload on preloaded links.

# Assignment TODO:
# 1. Write test(s)
# 2. Figure out where directions are accounted for (ie in the capacity vector 
#    and graph building are the directions split to not include an option with
#    both directions?) - this is important to understand just to make sure 
#    everything is working correctly!

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
    car_graph = project.network.graphs["c"]
    transit_graph = project.network.graphs["t"]
    walk_graph = project.network.graphs["w"]
    bike_graph = project.network.graphs["b"]
    return [car_graph, transit_graph, walk_graph, bike_graph]

@pytest.fixture
def assignment(project: Project, graphs: List[Graph]):
    # NOT YET COMPLETED - COPIED FROM test_mc_traffic_assignment.py - INTENDED TO REPLACE __run_preloaded_assig
    for graph in graphs: # Do we ignore transit graph in this whole function?
            graph.set_skimming(["free_flow_time"])
            graph.set_graph("free_flow_time")
            graph.set_blocked_centroid_flows(False)

    car_matrix = project.matrices.get_matrix("demand_mc")
    car_matrix.computational_view(["car"])

    transit_matrix = project.matrices.get_matrix("demand_mc")
    transit_matrix.computational_view(["transit"])

    walk_matrix = project.matrices.get_matrix("demand_mc")
    walk_matrix.computational_view(["walk"])

    bike_matrix = project.matrices.get_matrix("demand_mc")
    bike_matrix.computational_view(["bike"])

    assignment = TrafficAssignment()
    carclass = TrafficClass("car", graphs[0], car_matrix)
    carclass.set_pce(1.0)
    transitclass = TrafficClass("transit", graphs[1], transit_matrix)
    transitclass.set_pce(0.2)
    walkclass = TrafficClass("walk", graphs[2], walk_matrix)
    walkclass.set_pce(2.5)
    bikeclass = TrafficClass("bike", graphs[2], walk_matrix)
    bikeclass.set_pce(2.5)

    assignment.set_classes([carclass, transitclass, walkclass, bikeclass])

    for cls in assignment.classes:
            cls.graph.set_skimming(["free_flow_time", "distance"])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")

    assignment.max_iter = 20
    assignment.set_algorithm("bfw")

    return assignment

class TestPTPreloaing:

    def test_run(self, project: Project, graphs: List[Graph], assignment: TrafficAssignment):
        """Tests a full run through of pt preloading."""
        # NOT YET COMPLETED!

        # Get preload
        preload = self.test_built_pt_preload(project, graphs)
        
        # Run preloaded assignment
        assignment.set_pt_preload(preload)
        assignment.execute()

        # Check results (NOT AS RIGOROUS AS test_preloaded_assignment)
        assert False  # PLACEHOLDER

    def test_built_pt_preload(self, project: Project, graphs: List[Graph]):
        """
        Check that building pt preload works correctly for a basic example from
        the coquimbo network.
        """
        # NOT YET COMPLETED!
        # TODO: Figure out a test that makes sense and is relatively simple to check build phase (maybe in a narrow time period?)

        # Preload parameters
        period_start = int(6.5 * 60 * 60) # 6:30am in seconds from midnight
        period_end = int(8.5 * 60 * 60)   # 8:30am in seconds from midnight
        # What if someone wants timings between 11pm and 1am (ie around midnight), 
        # how do I detemine these instead.

        # Get preload info from network
        preload = project.network.build_pt_preload(graphs[0], period_start, period_end)
        preload_old = project.network.old_build_pt_preload(graphs[0], period_start, period_end)

        from time import time
        
        t_old, t_new, n = 0, 0, 10
        for _ in range(n):
            t0 = time()
            project.network.build_pt_preload(graphs[0], period_start, period_end)
            t1 = time()
            t_new += t1 - t0
            project.network.old_build_pt_preload(graphs[0], period_start, period_end)
            t_old += time() - t1
        t_old /= n
        t_new /= n

        speed_inc = t_old/t_new

        # Assertions about the preload and coquimbo network:
        assert len(preload) == len(graphs[0].graph)
        np.testing.assert_array_equal(preload, preload_old)
        assert False # NOT YET COMPLETED - DECIDE WHAT ELSE TO TEST FOR

        # Return preloads for further testing
        return preload

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
