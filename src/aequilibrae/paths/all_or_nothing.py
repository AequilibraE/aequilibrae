import logging
import threading
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix
from aequilibrae.paths.cython.AoN import aon_parallel
from aequilibrae.paths.cython.parallel_numpy import assign_link_loads
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread
from aequilibrae.utils.logging_utils import debug_bridge

from .multi_threaded_aon import MultiThreadedAoN

logger = logging.getLogger(__name__)


class allOrNothing(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, class_name: str, matrix: AequilibraeMatrix, graph: Graph, results: AssignmentResults):
        WorkerThread.__init__(self, None)

        self.class_name = class_name
        self.matrix = matrix
        self.graph = graph
        self.results = results
        self.aux_res = MultiThreadedAoN()
        self.signal.emit(["start", self.matrix.zones, self.class_name])

        if results._graph_id != graph._id:
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
        """Runs the all-or-nothing assignment.

        Dispatches all origins to a single OpenMP-parallel Cython kernel
        (``aon_parallel``). This avoids the per-origin Python pool dispatch
        overhead the previous ThreadPool-based path paid. Path file saving
        requires the GIL, so that case keeps the per-origin thread pool.
        """
        msg = f"All-or-Nothing - Traffic Class: {self.class_name} - Zones: 0/{self.matrix.zones}"
        self.signal.emit(["set_text", msg])
        self.report = []
        self.cumulative = 0
        self.aux_res.prepare(self.graph, self.results)
        self.matrix.matrix_view = self.matrix.matrix_view.reshape(
            (self.graph.num_zones, self.graph.num_zones, self.results.classes["number"])
        )
        with debug_bridge(logger) as bridge:
            skipped = aon_parallel(
                self.matrix, self.graph, self.results, self.aux_res, self.results.cores, bridge=bridge
            )
            self.report.extend(skipped)
        val = self.matrix.index.shape[0]
        msg = f"All-or-Nothing - Traffic Class: {self.class_name} - Zones: {val}/{self.matrix.zones}"
        self.signal.emit(["set_text", msg])

        # TODO: Multi-thread this sum
        self.results.compact_link_loads = np.sum(self.aux_res.temp_link_loads, axis=0)
        assign_link_loads(
            self.results.link_loads,
            self.results.compact_link_loads,
            self.results.crosswalk,
            self.results.elementwise_cores,
            self.results.threading_threshold,
        )
