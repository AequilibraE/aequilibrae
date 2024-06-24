import numpy as np
import pytest
from typing import List

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, AequilibraeMatrix
from aequilibrae.utils.create_example import create_example

# TODO:
# 1) Change fixtures to setUp and tearDown so project and matrices can be closed.
# 2) Add PCE to transit database schema for each trip
# 3) Input timings surrounding midnight (ie going past 24hrs)??

@pytest.fixture
def project(tmp_path):
    proj = create_example(str(tmp_path / "test_traffic_assignment"), from_model="coquimbo")
    proj.network.build_graphs()
    proj.activate()
    yield proj
    proj.close()

@pytest.fixture
def graphs(project: Project):
    return [project.network.graphs[c] for c in "ctwb"]

@pytest.fixture
def assignment(graphs: List[Graph]):
    g = graphs[0]
    n_zones = g.centroids.shape[0]
    g.set_skimming(["travel_time"])
    g.set_graph("travel_time") # FIGURE OUT WHY THESE ARE ALL NAN AND DEAL WITH THEM!!!
    g.set_blocked_centroid_flows(False)
    g.graph['capacity'] = 500
    g.graph['travel_time'] = g.graph['distance'] / 50

    # Create a random matrix for testing
    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=n_zones, matrix_names=['car'])
    matrix.index = graphs[0].centroids

    np.random.seed(7)
    matrix.matrices[:, :, 0] = np.random.uniform(0, 50, size=(n_zones, n_zones))
    matrix.computational_view('car')

    # Create assignment and set parameters
    assignment = TrafficAssignment()

    carclass = TrafficClass("car", graphs[0], matrix)
    assignment.set_classes([carclass])
    for cls in assignment.classes:
            cls.graph.set_skimming(["travel_time", "distance"])

    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("travel_time")
    assignment.max_iter = 1 # AON assignment
    assignment.set_algorithm("bfw")

    yield assignment
    matrix.close()

class TestPTPreloaing:

    def test_run(self, project: Project, graphs: List[Graph], assignment: TrafficAssignment):
        """Tests a full run through of pt preloading."""
        # Get preload
        preload = self.test_built_pt_preload(project, graphs)

        # Run non-preloaded assignment and get results
        assignment.execute()
        without_res = assignment.results() # dataframe containing all relevant fields for each link/dir
        
        # Run preloaded assignment and get results
        assignment.set_pt_preload(preload)
        assignment.execute()
        with_res = assignment.results() # dataframe containing all relevant fields for each link/dir

        # Check that average delay increases (ie the preload has reduced speeds)
        assert with_res["Delay_factor_AB"].mean() > without_res["Delay_factor_AB"].mean()

    def test_built_pt_preload(self, project: Project, graphs: List[Graph]):
        """
        Check that building pt preload works correctly for a basic example from
        the coquimbo network.
        """
        # Preload parameters
        to_24hrs = lambda hrs: int(hrs * 60 * 60) # hrs from midnight in seconds
        periods = [(to_24hrs(start), to_24hrs(end)) for start, end in [(7, 8), (6.5, 8.5), (5, 10)]]

        # Generate preloads
        preload_with_period = lambda period: project.network.build_pt_preload(graphs[0], *period, inclusion_cond="start")
        preloads = list(map(preload_with_period, periods))

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
