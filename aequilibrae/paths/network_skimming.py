import multiprocessing as mp
import sys
import threading
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from uuid import uuid4

from aequilibrae.paths.AoN import skimming_single_origin

from aequilibrae.context import get_active_project
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
from aequilibrae.paths.results.skim_results import SkimResults
from aequilibrae.utils.core_setter import set_cores
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread

sys.dont_write_bytecode = True


class NetworkSkimming(WorkerThread):
    signal = SIGNAL(object)

    """

    .. code-block:: python

        >>> from aequilibrae.paths.network_skimming import NetworkSkimming

        >>> project = create_example(project_path)

        >>> network = project.network
        >>> network.build_graphs()

        >>> graph = network.graphs['c']
        >>> graph.set_graph(cost_field="distance")
        >>> graph.set_skimming("distance")

        >>> skm = NetworkSkimming(graph)
        >>> skm.execute()

        # The skim report (if any error generated) is available here
        >>> skm.report
        []

        # To access the skim matrix directly from its temporary file
        >>> matrix = skm.results.skims

        # Or you can save the results to disk
        >>> skm.save_to_project(os.path.join(project_path, 'matrices/skimming_result.omx'))

        # Or specify the AequilibraE's matrix file format
        >>> skm.save_to_project(os.path.join(project_path, 'matrices/skimming_result.aem'), 'aem')

        >>> project.close()
    """

    def __init__(self, graph, origins=None, project=None):
        WorkerThread.__init__(self, None)
        self.project = project
        self.origins = origins
        self.graph = graph
        self.cores = mp.cpu_count()
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
        self.signal.emit(["start", self.graph.num_zones, ""])
        self.results.cores = self.cores
        self.results.prepare(self.graph)
        self.aux_res = MultiThreadedNetworkSkimming()
        self.aux_res.prepare(self.graph, self.results.cores, self.results.nodes, self.results.num_skims)
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

        self.signal.emit(["set_text", "Saving Outputs"])
        self.signal.emit(["finished"])

    def set_cores(self, cores: int) -> None:
        """
        Sets number of cores (threads) to be used in computation

        Value of zero sets number of threads to all available in the system, while negative values indicate the number
        of threads to be left out of the computational effort.

        Resulting number of cores will be adjusted to a minimum of zero or the maximum available in the system if the
        inputs result in values outside those limits

        :Arguments:
            **cores** (:obj:`int`): Number of cores to be used in computation
        """
        self.cores = set_cores(cores)

    def save_to_project(self, name: str, format="omx", project=None) -> None:
        """Saves skim results to the project folder and creates record in the database

        :Arguments:
            **name** (:obj:`str`): Name of the matrix. Same value for matrix record name and file (plus extension)

            **format** (:obj:`str`, *Optional*): File format ('aem' or 'omx'). Default is 'omx'

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """

        file_name = f"{name}.{format.lower()}"
        if not project:
            project = self.project or get_active_project()
        mats = project.matrices
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

        self.signal.emit(["update", self.cumulative, f"{self.cumulative}/{self.graph.num_zones}"])
