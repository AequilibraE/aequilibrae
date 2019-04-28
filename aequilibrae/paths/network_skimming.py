"""
-----------------------------------------------------------------------------------------------------------
 Package:    AequilibraE

 Name:       Network skimming
 Purpose:    Implement skimming algorithms based on Cython's path finding and skimming

 Original Author:  Pedro Camargo (c@margo.co)
 Contributors:
 Last edited by: Pedro Camrgo

 Website:    www.AequilibraE.com
 Repository:  https://github.com/AequilibraE/AequilibraE

 Created:    2017-07-03
 Updated:    2017-05-07
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
-----------------------------------------------------------------------------------------------------------
 """

import importlib
import sys
import threading
from multiprocessing.dummy import Pool as ThreadPool

from .AoN import skimming_single_origin
from .multi_threaded_skimming import MultiThreadedNetworkSkimming
from ..utils import WorkerThread

have_pyqt5 = importlib.util.find_spec("PyQt5")
if have_pyqt5 is None:
    pyqt = False
else:
    from PyQt5.QtCore import pyqtSignal as SIGNAL

    pyqt = True

sys.dont_write_bytecode = True


class NetworkSkimming(WorkerThread):

    skimming = SIGNAL(object)

    def __init__(self, graph, results, origins=None):
        WorkerThread.__init__(self, None)

        self.origins = origins
        self.graph = graph
        self.results = results
        self.aux_res = MultiThreadedNetworkSkimming()
        self.report = []
        self.cumulative = 0

        if results.__graph_id__ != graph.__id__:
            raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

        if results.__graph_id__ is None:
            raise ValueError("The results object was not prepared. Use results.prepare(graph)")

        elif results.__graph_id__ != graph.__id__:
            raise ValueError("The results object was prepared for a different graph")

    def doWork(self):
        self.execute()

    def execute(self):
        if pyqt:
            self.skimming.emit(["zones finalized", 0])

        self.aux_res.prepare(self.graph, self.results)

        pool = ThreadPool(self.results.cores)
        all_threads = {"count": 0}
        for orig in list(self.graph.centroids):
            i = int(self.graph.nodes_to_indices[orig])
            if i >= self.graph.nodes_to_indices.shape[0]:
                self.report.append("Centroid " + str(orig) + " is beyond the domain of the graph")
            elif self.graph.fs[int(i)] == self.graph.fs[int(i) + 1]:
                self.report.append("Centroid " + str(orig) + " does not exist in the graph")
            else:
                pool.apply_async(self.func_assig_thread, args=(orig, all_threads))
        pool.close()
        pool.join()

        if pyqt:
            self.skimming.emit(["text skimming", "Saving Outputs"])
            self.skimming.emit(["finished_threaded_procedure", None])

    def func_assig_thread(self, O, all_threads):
        if threading.get_ident() in all_threads:
            th = all_threads[threading.get_ident()]
        else:
            all_threads[threading.get_ident()] = all_threads["count"]
            th = all_threads["count"]
            all_threads["count"] += 1
        x = skimming_single_origin(O, self.graph, self.results, self.aux_res, th)
        self.cumulative += 1
        if x != O:
            self.report.append(x)
        if pyqt:
            self.skimming.emit(["zones finalized", self.cumulative])
            txt = str(self.cumulative) + " / " + str(self.matrix.zones)
            self.skimming.emit(["text skimming", txt])
