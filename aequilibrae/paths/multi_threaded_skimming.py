import numpy as np

from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths


class MultiThreadedNetworkSkimming(MultiThreadedPaths):
    def __init__(self):
        MultiThreadedPaths.__init__(self)

        # holds the skims for all nodes in the network (during path finding)
        self.temporary_skims = np.array([], np.int64)

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph, results):
        self.prepare_(graph, results)

        ftype = graph.default_types("float")
        self.temporary_skims = np.zeros((results.cores, results.nodes, results.num_skims), dtype=ftype)
