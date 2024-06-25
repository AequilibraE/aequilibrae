import threading
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from aequilibrae import global_logger
from aequilibrae.matrix import AequilibraeMatrix
from .multi_threaded_aon import MultiThreadedAoN

try:
    from aequilibrae.paths.AoN import one_to_all, assign_link_loads
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")

from aequilibrae.utils.signal import SIGNAL

if False:
    from .results import AssignmentResults
    from .graph import Graph


class allOrNothing:
    def __init__(self, class_name, matrix, graph, results):
        # type: (AequilibraeMatrix, Graph, AssignmentResults)->None
        self.assignment: SIGNAL = None

        self.class_name = class_name
        self.matrix = matrix
        self.graph = graph
        self.results = results
        self.aux_res = MultiThreadedAoN()

        if results._graph_id != graph._id:
            raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

        elif matrix.matrix_view is None:
            raise ValueError(
                "Matrix was not prepared for assignment. "
                "Please create a matrix_procedures view with all classes you want to assign"
            )

        elif not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")

    def _build_signal(self):
        if self.assignment is None:
            self.assignment = SIGNAL(object)
            self.assignment.emit(["start", self.matrix.zones, self.class_name])

    def doWork(self):
        self.execute()

    def execute(self):
        self._build_signal()
        self.report = []
        self.cumulative = 0
        self.aux_res.prepare(self.graph, self.results)
        self.matrix.matrix_view = self.matrix.matrix_view.reshape(
            (self.graph.num_zones, self.graph.num_zones, self.results.classes["number"])
        )
        mat = self.matrix.matrix_view
        pool = ThreadPool(self.results.cores)
        all_threads = {"count": 0}
        for orig in self.matrix.index:
            i = int(self.graph.nodes_to_indices[orig])
            if np.nansum(mat[i, :, :]) > 0 or self.results.num_skims > 0:
                if self.graph.fs[i] == self.graph.fs[i + 1]:
                    self.report.append("Centroid " + str(orig) + " is not connected")
                else:
                    pool.apply_async(self.func_assig_thread, args=(orig, all_threads))
        pool.close()
        pool.join()
        self.assignment.emit(["update", self.matrix.index.shape[0], self.class_name])
        # TODO: Multi-thread this sum
        self.results.compact_link_loads = np.sum(self.aux_res.temp_link_loads, axis=0)
        assign_link_loads(
            self.results.link_loads, self.results.compact_link_loads, self.results.crosswalk, self.results.cores
        )

    def func_assig_thread(self, origin, all_threads):
        thread_id = threading.get_ident()
        th = all_threads.get(thread_id, all_threads["count"])
        if th == all_threads["count"]:
            all_threads[thread_id] = all_threads["count"]
            all_threads["count"] += 1

        x = one_to_all(origin, self.matrix, self.graph, self.results, self.aux_res, th)
        self.cumulative += 1
        if x != origin:
            self.report.append(x)
        if self.cumulative % 10 == 0:
            self.assignment.emit(["update", self.cumulative, self.class_name])
