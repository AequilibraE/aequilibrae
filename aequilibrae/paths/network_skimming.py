import sys
import threading
import importlib.util as iutil
from uuid import uuid4
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime
from aequilibrae.context import get_active_project
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
from aequilibrae.paths.results.skim_results import SkimResults
from aequilibrae.utils import WorkerThread
from aequilibrae import global_logger

try:
    from aequilibrae.paths.AoN import skimming_single_origin
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

spec = iutil.find_spec("openmatrix")
has_omx = spec is not None

sys.dont_write_bytecode = True


class NetworkSkimming(WorkerThread):
    """

    ::

        from aequilibrae.paths.network_skimming import NetworkSkimming
        from aequilibrae.project import Project

        project = Project()
        project.open(self.proj_dir)
        network = self.project.network

        network.build_graphs()
        graph = network.graphs['c']
        graph.set_graph(cost_field="distance")
        graph.set_skimming("distance")

        skm = NetworkSkimming(graph)
        skm.execute()

        # The skim report (if any error generated) is available here
        skm.report

        # To access the skim matrix directly from its temporary file
        matrix = skm.results.skims

        # Or you can save the results to disk
        skm.save_to_project('skimming result')

        # Or specify the AequilibraE's matrix file format
        skm.save_to_project('skimming result', 'aem')

        project.close()
    """

    if pyqt:
        skimming = pyqtSignal(object)

    def __init__(self, graph, origins=None, project=None):
        WorkerThread.__init__(self, None)
        self.project = project or get_active_project()
        self.origins = origins
        self.graph = graph
        self.results = SkimResults()
        self.aux_res = MultiThreadedNetworkSkimming()
        self.report = []
        self.procedure_id = ""
        self.procedure_date = ""
        self.cumulative = 0

    def doWork(self):
        self.execute()

    def execute(self):
        """Runs the skimming process as specified in the graph"""
        if pyqt:
            self.skimming.emit(["zones finalized", 0])

        self.results.prepare(self.graph)
        self.aux_res = MultiThreadedNetworkSkimming()
        self.aux_res.prepare(self.graph, self.results)

        pool = ThreadPool(self.results.cores)
        all_threads = {"count": 0}
        for orig in list(self.graph.centroids):
            i = int(self.graph.nodes_to_indices[orig])
            if i >= self.graph.nodes_to_indices.shape[0]:
                self.report.append(f"Centroid {orig} is beyond the domain of the graph")
            elif self.graph.fs[int(i)] == self.graph.fs[int(i) + 1]:
                self.report.append(f"Centroid {orig} does not exist in the graph")
            else:
                pool.apply_async(self.__func_skim_thread, args=(orig, all_threads))
        pool.close()
        pool.join()
        self.aux_res = None
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        if pyqt:
            self.skimming.emit(["text skimming", "Saving Outputs"])
            self.skimming.emit(["finished_threaded_procedure", None])

    def save_to_project(self, name: str, format="omx") -> None:
        """Saves skim results to the project folder and creates record in the database

        Args:
            *name* (:obj:`str`): Name of the matrix. Same value for matrix record name and file (plus extension)
            *format* (:obj:`str`, `Optional`): File format ('aem' or 'omx'). Default is 'omx'
        """

        file_name = f"{name}.{format.lower()}"
        mats = self.project.matrices
        record = mats.new_record(name, file_name, self.results.skims)
        record.procedure_id = self.procedure_id
        record.timestamp = self.procedure_date
        record.procedure = "Network skimming"
        record.save()

    def __func_skim_thread(self, origin, all_threads):
        if threading.get_ident() in all_threads:
            th = all_threads[threading.get_ident()]
        else:
            all_threads[threading.get_ident()] = all_threads["count"]
            th = all_threads["count"]
            all_threads["count"] += 1
        x = skimming_single_origin(origin, self.graph, self.results, self.aux_res, th)
        self.cumulative += 1
        if x != origin:
            self.report.append(x)
        if pyqt:
            self.skimming.emit(["zones finalized", self.cumulative])
            txt = str(self.cumulative) + " / " + str(self.matrix.zones)
            self.skimming.emit(["text skimming", txt])
