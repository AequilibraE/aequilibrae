import numpy as np
import pytest
from typing import List

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, AequilibraeMatrix
from aequilibrae.utils.create_example import create_example

# Overall TODO:
# 1) Add in inclusion conditions
# 2) Fix assignment object construction, and complete test (delete 3rd test)
# 3) Add PCE to transit database schema for each trip

# Build TODO:
# 1. PCE in database schema
# 2. Input timings surrounding midnight (ie going past 24hrs)

# SQL Code for including all trips which cover greater than some threshold proportion of the period
#
# WITH Intervals AS (
#     SELECT 
#         trip_id,
#         MIN(departure) AS min_departure,
#         MAX(arrival) AS max_arrival
#     FROM trips_schedule
#     GROUP BY trip_id
# ),
# Proportions AS (
#     SELECT
#         trip_id,
#         LEAST(max_arrival, x2) - GREATEST(min_departure, x1) AS overlap
#     FROM TripIntervals
#     WHERE max_arrival > x1 AND min_departure < x2
# )
# SELECT
#     trip_id
# FROM Proportions
# WHERE overlap / (x2 - x1) > threshold;

# Assignment TODO:
# Test: do aon, and check average delay (basically speed) is increased after preload.

# Extra TODO:
# 1. Remove unecessary inputs to test functions
# 2. Change fixtures to setUp and tearDown so project and matrices can be closed.


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
def assignment(graphs: List[Graph]):
    graph = graphs[0]
    n_zones = graph.centroids.shape[0]
    graph.set_skimming(["travel_time"])
    graph.set_graph("travel_time") # FIGURE OUT WHY THESE ARE ALL NAN AND DEAL WITH THEM!!!
    graph.set_blocked_centroid_flows(False)
    graph.graph['capacity'] = graph.graph['capacity'].fillna(1.0)
    graph.graph['travel_time'] = graph.graph['travel_time'].fillna(1.0)

    # Create a random matrix for testing
    matrices = AequilibraeMatrix()
    matrices.create_empty(zones=n_zones, matrix_names=['car'])
    matrices.index = graphs[0].centroids

    np.random.seed(7)
    matrices.matrices[:, :, 0] = np.random.uniform(0, 1, size=(n_zones, n_zones))
    matrices.computational_view('car')

    assignment = TrafficAssignment()
    carclass = TrafficClass("car", graphs[0], matrices)
    carclass.set_pce(1.0)

    assignment.set_classes([carclass])

    for cls in assignment.classes:
            cls.graph.set_skimming(["travel_time", "distance"])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})

    assignment.set_capacity_field("capacity")
    assignment.set_time_field("travel_time")

    assignment.max_iter = 1 # AON assignment
    assignment.set_algorithm("bfw")

    return assignment

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
        preload_with_period = lambda period: project.network.build_pt_preload(
            graphs[0], *period, inclusion_cond="start")
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
