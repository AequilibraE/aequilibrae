from unittest import TestCase

import numpy as np

from aequilibrae.paths import Graph
from aequilibrae.paths import NetworkSkimming
from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.results import SkimResults

# Adds the folder with the data to the path and collects the paths to the files
from ...data import test_graph


class MultiThreadedNetworkSkimming:
    def __init__(self):
        self.predecessors = None  # The predecessors for each node in the graph
        # holds the skims for all nodes in the network (during path finding)
        self.temporary_skims = None
        # Keeps the order in which the nodes were reached for the cascading network loading
        self.reached_first = None
        self.connectors = None  # The previous link for each node in the tree
        # holds the b_nodes in case of flows through centroid connectors are blocked
        self.temp_b_nodes = None

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph, results):
        itype = graph.default_types("int")
        ftype = graph.default_types("float")
        self.predecessors = np.zeros((results.nodes, results.cores), dtype=itype)
        self.temporary_skims = np.zeros((results.nodes, results.num_skims, results.cores), dtype=ftype)
        self.reached_first = np.zeros((results.nodes, results.cores), dtype=itype)
        self.connectors = np.zeros((results.nodes, results.cores), dtype=itype)
        self.temp_b_nodes = np.zeros((graph.b_node.shape[0], results.cores), dtype=itype)
        for i in range(results.cores):
            self.temp_b_nodes[:, i] = graph.b_node[:]


class TestNetwork_skimming(TestCase):
    def test_network_skimming(self):
        # graph
        g = Graph()
        g.load_from_disk(test_graph)
        g.set_graph(cost_field="distance", skim_fields=None)
        # None implies that only the cost field will be skimmed

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
