import os
import sys
import unittest
from aequilibrae.paths import Graph
from aequilibrae.paths.results import SkimResults
from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
import numpy as np
from .parameters_test import centroids

# Adds the folder with the data to the path and collects the paths to the files
# lib_path = os.path.abspath(os.path.join('..', '../tests'))
# sys.path.append(lib_path)
from ...data import path_test, test_graph


class TestSkimming_single_origin(unittest.TestCase):
    def test_skimming_single_origin(self):

        origin = 1

        # graph
        g = Graph()
        g.load_from_disk(test_graph)
        g.set_graph(cost_field="distance", skim_fields=None)
        # g.block_centroid_flows = False
        # None implies that only the cost field will be skimmed

        # skimming results
        res = SkimResults()
        res.prepare(g)
        aux_result = MultiThreadedNetworkSkimming()
        aux_result.prepare(g, res)

        a = skimming_single_origin(origin, g, res, aux_result, 0)
        tot = np.sum(res.skims.distance[origin, :])
        if tot > 10e10:
            self.fail("Skimming was not successful. At least one np.inf returned.")

        if a != origin:
            self.fail("Skimming returned an error: " + a)
