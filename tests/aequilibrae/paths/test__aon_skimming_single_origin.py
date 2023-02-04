import unittest
from tempfile import gettempdir
from uuid import uuid4
from os.path import join
from aequilibrae.paths.results import SkimResults
from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
import numpy as np
from aequilibrae.utils.create_example import create_example

# Adds the folder with the data to the path and collects the paths to the files
from ...data import path_test, test_graph


class TestSkimming_single_origin(unittest.TestCase):
    def setUp(self) -> None:
        path = join(gettempdir(), "skim_test_" + uuid4().hex)
        self.project = create_example(path)

    def tearDown(self) -> None:
        self.project.close()

    def test_skimming_single_origin(self):
        self.project.network.build_graphs()
        g = self.project.network.graphs["c"]
        g.set_graph(cost_field="distance")
        g.set_skimming("distance")

        origin = np.random.choice(g.centroids[:-1], 1)[0]

        # skimming results
        res = SkimResults()
        res.prepare(g)
        aux_result = MultiThreadedNetworkSkimming()
        aux_result.prepare(g, res)

        a = skimming_single_origin(origin, g, res, aux_result, 0)
        tot = np.sum(res.skims.distance[origin, :])
        if tot > 10e10:
            self.fail("Skimming was not successful. At least one np.inf returned for origin {}.".format(origin))

        if a != origin:
            self.fail("Skimming returned an error: {} for origin {}".format(a, origin))
