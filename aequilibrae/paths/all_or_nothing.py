import importlib.util as iutil
import threading
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from .AoN import one_to_all
from .multi_threaded_aon import MultiThreadedAoN
from ..utils import WorkerThread
from .results import AssignmentResults
from .graph import Graph
from aequilibrae.matrix import AequilibraeMatrix

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal as SIGNAL


class allOrNothing(WorkerThread):
    if pyqt:
        assignment = SIGNAL(object)

    def __init__(self, matrix: AequilibraeMatrix, graph: Graph, results: AssignmentResults):
        WorkerThread.__init__(self, None)

        self.matrix = matrix
        self.graph = graph
        self.results = results
        self.aux_res = MultiThreadedAoN()
        self.report = []
        self.cumulative = 0

        if results.__graph_id__ != graph.__id__:
            raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

        elif matrix.matrix_view is None:
            raise ValueError(
                "Matrix was not prepared for assignment. "
                "Please create a matrix_procedures view with all classes you want to assign"
            )

        elif not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")

    def doWork(self):
        self.execute()

    def execute(self):
        if pyqt:
            self.assignment.emit(["zones finalized", 0])

        self.aux_res.prepare(self.graph, self.results)
        self.matrix.matrix_view = self.matrix.matrix_view.reshape(
            (self.graph.num_zones, self.graph.num_zones, self.results.classes["number"])
        )
        mat = self.matrix.matrix_view
        pool = ThreadPool(self.results.cores)
        all_threads = {"count": 0}
        for orig in self.matrix.index:
            i = int(self.graph.nodes_to_indices[orig])
            if np.nansum(mat[i, :, :]) > 0:
                if self.graph.fs[i] == self.graph.fs[i + 1]:
                    self.report.append("Centroid " + str(orig) + " is not connected")
                else:
                    pool.apply_async(self.func_assig_thread, args=(orig, all_threads))
                    # self.func_assig_thread(orig, all_threads)
        pool.close()
        pool.join()
        self.results.link_loads = np.sum(self.aux_res.temp_link_loads, axis=2)

        if pyqt:
            self.assignment.emit(["text AoN", "Saving Outputs"])
            self.assignment.emit(["finished_threaded_procedure", None])

    def func_assig_thread(self, O, all_threads):
        thread_id = threading.get_ident()
        th = all_threads.get(thread_id, all_threads["count"])
        if th == all_threads["count"]:
            all_threads[thread_id] = all_threads["count"]
            all_threads["count"] += 1

        x = one_to_all(O, self.matrix, self.graph, self.results, self.aux_res, th)
        self.cumulative += 1
        if x != O:
            self.report.append(x)
        if pyqt:
            self.assignment.emit(["zones finalized", self.cumulative])
            txt = str(self.cumulative) + " / " + str(self.matrix.zones)
            self.assignment.emit(["text AoN", txt])
