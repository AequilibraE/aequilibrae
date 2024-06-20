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

# Assignment TODO:
# 1. Write test(s)
# 2. Figure out where directions are accounted for (ie in the capacity vector 
#    and graph building are the directions split to not include an option with
#    both directions?) - this is important to understand just to make sure 
#    everything is working correctly!


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

class TestPTPreloaing:

    def test_run(self, project: Project, graphs: List[Graph]):
        """Tests a full run through of pt preloading."""
        # NOT YET COMPLETED!

        # Get preload (test will not run if prior test fails)
        preload = self.test_built_pt_preload(project, graphs)
        
        # Run preloaded assignment (test will fail if this cannot be run)
        self.__run_preloaded_assig(project, graphs, preload)

        # Check results (NOT AS RIGOROUS AS test_preloaded_assignment)
        assert False  # PLACEHOLDER

    def test_built_pt_preload(self, project: Project, graphs: List[Graph]):
        """
        Check that building pt preload works correctly for a basic example from
        the coquimbo network.
        """
        # NOT YET COMPLETED!

        # Preload parameters
        period_start = int(6.5 * 60 * 60) # 6:30am in seconds from midnight
        period_end = int(8.5 * 60 * 60)   # 8:30am in seconds from midnight
        # What if someone wants timings between 11pm and 1am (ie around midnight), 
        # how do I detemine these instead.

        # Get preload info from network
        preload = project.network.build_pt_preload(graphs[0], period_start, period_end)

        # Assertions about the preload and coquimbo network:
        assert len(preload) == len(graphs[0].graph)
        assert False # PLACEHOLDER

        # Return preloads for further testing
        return preload

    def test_preloaded_assignment(self, project: Project, graphs: List[Graph]):
        """
        Check that the setting a preload and running an assignment works as intended.

        Maybe put infinity on 1 or 2 links and check nothing is assigned to those links?
        """
        # NOT YET COMPLETED!
        
        # Create custom preload data
        preloads = None # type: np.ndarray

        # Run preloaded assignment
        assignment = self.__run_preloaded_assig(project, graphs, preloads)

        # Check results
        assert False

    def __run_preloaded_assig(
        self, proj: Project, all_graphs: List[Graph], preload: np.ndarray
        ) -> TrafficAssignment:
        """Runs an assignment with a pt preload"""
        # NEED TO CHECK WHICH INPUT PARAMETERS ARE ACTUALLY NEEDED!
        # MAY NEED TO ADD MORE TO ALLOW FOR MORE TESTING!
        # NOT YET COMPLETED!

        # Create Assignment object
        assignment = TrafficAssignment()

        # Set Traffic Classes

        # Set preload info into assig
        assignment.set_pt_preload(preload) # NOT YET IMPLEMENTED!
        
        # Set other assignment parameters

        # Run and return assignment object
        assignment.execute()
        return assignment
