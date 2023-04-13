from collections import OrderedDict


class PathStorage:
    """
    This class is designed to store path-computation objects to take advantage of the fact that AequilibraE
    preserves the entire shortest path tree when computing a path between two nodes and can re-trace the
    same tree for a path from the same origin to a different destination.

    Since this caching in memory can take too much memory, the *threshold* parameter exists to limit the number
    of path objects kept in memory.

    If you have a large amount of memory in your system, you can set the threshold class variable accordingly.
    """

    def __init__(self):
        self.graphs = {}
        self.storage = OrderedDict()
        self.uses = 0
        self.total_paths = 0
        self.threshold = 50

    def add_graph(self, graph, mode_id):
        if mode_id in self.graphs:
            return

        self.graphs[mode_id] = graph
        self.storage[mode_id] = OrderedDict()

    def get_path_results(self, origin, mode_id):
        from aequilibrae.paths import PathResults

        self.uses += 1
        if origin in self.storage[mode_id]:
            # We move the last used element from the list to the most recent position
            self.storage[mode_id].move_to_end(origin, last=True)
            return self.storage[mode_id][origin]

        # We check if our cache is getting too large
        stored_values = [len(data) for data in self.storage.values()]
        if sum(stored_values) > self.threshold:
            # If we reached the maximum number of paths to store, we remove the element that has not been used
            # in the longest time from the mode ID with the biggest cache
            mode_to_clean = list(self.storage.keys())[stored_values.index(max(stored_values))]
            self.storage[mode_to_clean].clear()

        graph = self.graphs[mode_id]
        res = PathResults()
        res.prepare(graph)
        d = graph.centroids[0] if origin != graph.centroids[0] else graph.centroids[1]
        res.compute_path(origin, d)
        self.storage[mode_id][origin] = res
        self.total_paths += 1

        return res

    def clear(self):
        self.storage.clear()
        self.graphs.clear()
        self.uses = 0
        self.total_paths = 0
