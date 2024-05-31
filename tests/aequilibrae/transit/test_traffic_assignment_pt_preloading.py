import random
import sqlite3
import string
from os.path import join, isfile
from pathlib import Path
from random import choice

import numpy as np
import pandas as pd
import pytest
from typing import List

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project

# Overall TODO:
# 1) Create test structure for the API for PT Preloading
# 2) Figure out how to extract PT network data from public_transport.sqlite (NOT GTFS!!!)
# 3) Modify database structure to store PCE value for a service and have default info
# 4) Create algorithm to turn PT network data into a pre-load vector (of size no. edges in graph) with no. vehicles * PCE
# 5) Create modification to current traffic assignment algorithm to reduce capacity at end of each assignment iteration
# 6) Consider simple example scenarios to test correctness of algorithms
# 7) Decide whether more complex tests are necessary
# 8) Performance testing and speed ups in cython


# Today TODO:
# 3. Write docstring for built_pt_preload
# 4. Pull in latest addition to develop (fixed GTFS importer) and change relevant lines in function
# 4. Write simple test(s) based off coquimbo network for built_pt_preload
# 6. Start adding this into the TrafficAssignment...



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
        # Preload parameters
        period_start = int(6.5 * 60 * 60) # 6:30am in seconds from midnight
        period_end = int(8.5 * 60 * 60)   # 8:30am in seconds from midnight
        # What if someone wants timings between 11pm and 1am (ie around midnight), 
        # how do I detemine these instead.

        # Get preload info from network
        to_build = [True, False, False, False]
        preloads = project.network.build_pt_preload(graphs, to_build, period_start, period_end)

        # Create Assignment object
        assignment = TrafficAssignment()

        # Set Traffic Classes

        # Set preload info into assig
        assignment.set_pt_preload(preloads) # NOT YET IMPLEMENTED!
        
        # Set other assignment parameters

        # Run assignment

        # Check results
        assert False  # Ensure test fails for now
    
    # Figure out more specific tests for creation of pre-load vector!

    def test_built_pt_preload(self, project: Project, graphs: List[Graph]):
        """
        Check that building pt preload works correctly for a basic example from
        the coquimbo network (FIGURE OUT SPECIFIC TEST FOR THIS!)
        """
        # Preload parameters
        period_start = int(6.5 * 60 * 60) # 6:30am in seconds from midnight
        period_end = int(8.5 * 60 * 60)   # 8:30am in seconds from midnight
        # What if someone wants timings between 11pm and 1am (ie around midnight), 
        # how do I detemine these instead.

        # Get preload info from network
        to_build = [True, False, False, False]
        preloads = project.network.build_pt_preload(graphs, to_build, period_start, period_end)

        # Assertions about the preload and coquimbo network:
        assert False
