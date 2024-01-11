import os
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase
import pandas as pd
import numpy as np

from aequilibrae import Graph, Project
from aequilibrae.paths.bfs_le import RouteChoice
from ...data import siouxfalls_project


class TestRouteChoice(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_route_choice" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)
        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs()
        self.graph = self.project.network.graphs["c"]  # type: Graph
        self.graph.set_graph("free_flow_time")
        self.graph.set_blocked_centroid_flows(False)

    def tearDown(self) -> None:
        self.project.close()

    def test_route_choice(self):
        rc = RouteChoice(self.graph)

        # breakpoint()
        results = rc.run(1, 20, max_routes=5000, max_depth=6)
        # print(*results.items(), sep="\n")
        print(*(results), sep="\n")
        print(len(results))
        assert False
