import numpy as np

from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths


class MultiThreadedNetworkSkimming(MultiThreadedPaths):
    def __init__(self):
        MultiThreadedPaths.__init__(self)

        # holds the skims for all nodes in the network (during path finding)
        self.temporary_skims = np.array([], np.int64)

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph, cores, nodes, num_skims):
        self.prepare_(graph, cores, nodes)

        ftype = graph.default_types("float")
        self.temporary_skims = np.zeros((cores, nodes, num_skims), dtype=ftype)
