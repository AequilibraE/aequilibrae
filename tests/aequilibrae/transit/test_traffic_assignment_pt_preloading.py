import random
import sqlite3
import string
from os.path import join, isfile
from pathlib import Path
from random import choice

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project

# TODO:
# 1) Create test structure for the API for PT Preloading
# 2) Figure out how to extract PT network data from public_transport.sqlite (NOT GTFS!!!)
# 3) Modify database structure to store PCE value for a service and have default info
# 4) Create algorithm to turn PT network data into a pre-load vector (of size no. edges in graph) with no. vehicles * PCE
# 5) Create modification to current traffic assignment algorithm to reduce capacity at end of each assignment iteration
# 6) Consider simple example scenarios to test correctness of algorithms
# 7) Decide whether more complex tests are necessary
# 8) Performance testing and speed ups in cython


@pytest.fixture
def project(tmp_path):
    proj = create_example(str(tmp_path / "test_traffic_assignment"), from_model="coquimbo")
    proj.network.build_graphs()
    proj.activate()
    return proj


class TestPTPreloaing:

    def test_run(self, project: Project):
        project.network.build_graphs()
        graph = project.network.graphs["c"]

        # Get preload info from somewhere
        pre_load = project.network.build_pt_preload(graph, start_time=7*60*60, end_time=9*60*60, default_pce=3.0, inclusion_cond="start")

        assignment = TrafficAssignment()

        # Set preload info into assig
        assignment.set_pt_preload(pre_load)
        
        # Set other assignment parameters

        # Run assignment

        # Check results
        assert False  # Ensure test fails for now
