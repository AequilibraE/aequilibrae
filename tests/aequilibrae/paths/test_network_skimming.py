from unittest import TestCase
import numpy as np
from aequilibrae.paths import Graph
from aequilibrae.paths import NetworkSkimming
from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.results import SkimResults
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming

# Adds the folder with the data to the path and collects the paths to the files
from ...data import test_graph


class TestNetwork_skimming(TestCase):
    def test_network_skimming(self):
        # graph
        g = Graph()
        g.load_from_disk(test_graph)
        g.set_graph(cost_field="distance")
        g.set_skimming("distance")

        # skimming results
        res = SkimResults()
        res.prepare(g)

        aux_res = MultiThreadedNetworkSkimming()
        aux_res.prepare(g, res)
        _ = skimming_single_origin(26, g, res, aux_res, 0)

        skm = NetworkSkimming(g, res)
        skm.execute()

        tot = np.nanmax(res.skims.distance[:, :])

        if tot > 10e10:
            self.fail("Skimming was not successful. At least one np.inf returned.")

        if skm.report:
            self.fail("Skimming returned an error:" + str(skm.report))
