import os, sys
from unittest import TestCase
from aequilibrae.paths import Graph
from aequilibrae.paths.results import SkimResults
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import NetworkSkimming, skimming_single_origin
import numpy as np

# Adds the folder with the data to the path and collects the paths to the files
lib_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(lib_path)
from data import path_test, test_graph

from parameters_test import centroids

class MultiThreadedNetworkSkimming:
    def __init__(self):
        self.predecessors = None  # The predecessors for each node in the graph
        self.temporary_skims = None  # holds the skims for all nodes in the network (during path finding)
        self.reached_first = None    # Keeps the order in which the nodes were reached for the cascading network loading
        self.connectors = None  # The previous link for each node in the tree
        self.temp_b_nodes = None  #  holds the b_nodes in case of flows through centroid connectors are blocked

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph, results):
        itype = graph.default_types('int')
        ftype = graph.default_types('float')
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
        g.set_graph(cost_field='distance', skim_fields=None)
        # None implies that only the cost field will be skimmed

        # skimming results
        res = SkimResults()
        res.prepare(g)

        aux_res = MultiThreadedNetworkSkimming()
        aux_res.prepare(g, res)
        a = skimming_single_origin(26, g, res, aux_res, 0)


        skm = NetworkSkimming(g, res)
        skm.execute()

        tot = np.nanmax(res.skims.distance[:, :])

        if tot > 10e10:
            self.fail('Skimming was not successful. At least one np.inf returned.')

        if skm.report:
            self.fail('Skimming returned an error:' + str(skm.report))
