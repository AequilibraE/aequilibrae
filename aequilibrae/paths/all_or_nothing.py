import importlib.util as iutil
import threading
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

from .multi_threaded_aon import MultiThreadedAoN
from ..utils import WorkerThread
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae import global_logger

try:
    from aequilibrae.paths.AoN import one_to_all, assign_link_loads
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal as SIGNAL

if False:
    from .results import AssignmentResults
    from .graph import Graph


class allOrNothing(WorkerThread):
    if pyqt:
        assignment = SIGNAL(object)

    def __init__(self, matrix, graph, results):
        # type: (AequilibraeMatrix, Graph, AssignmentResults)->None

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
            if np.nansum(mat[i, :, :]) > 0 or self.results.num_skims > 0:
                if self.graph.fs[i] == self.graph.fs[i + 1]:
                    self.report.append("Centroid " + str(orig) + " is not connected")
                else:
                    pool.apply_async(self.func_assig_thread, args=(orig, all_threads))
        pool.close()
        pool.join()
        # TODO: Multi-thread this sum
        self.results.compact_link_loads = np.sum(self.aux_res.temp_link_loads, axis=0)
        assign_link_loads(
            self.results.link_loads, self.results.compact_link_loads, self.results.crosswalk, self.results.cores
        )
        if pyqt:
            self.assignment.emit(["finished_threaded_procedure", None])

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
        if pyqt:
            self.assignment.emit(["zones finalized", self.cumulative])
            self.assignment.emit(["text AoN", f"{self.cumulative:,}/{self.matrix.zones:,}"])
