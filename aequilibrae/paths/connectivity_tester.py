import multiprocessing as mp
import sys
import threading
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from uuid import uuid4

from aequilibrae import global_logger
from aequilibrae.context import get_active_project
from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
from aequilibrae.paths.results.skim_results import SkimResults

try:
    from aequilibrae.paths.AoN import skimming_single_origin
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")

from aequilibrae.utils.signal import SIGNAL

sys.dont_write_bytecode = True


class ConnectivityTester:
    """

    .. code-block:: python

        >>> from aequilibrae.paths.connectivity_tester import ConnectivityTester

        >>> project = create_example(project_path)

        >>> network = project.network
        >>> network.build_graphs()

        >>> graph = network.graphs['c']
        >>> graph.set_graph(cost_field="distance")

        >>> conn_test = ConectivityTester(graph)
        >>> conn_test.execute()

        # The connectivity tester report as a Pandas DataFrame
        >>> conn_test.report

        >>> project.close()
    """

    connectivity = SIGNAL(object)

    def __init__(self, graph, origins=None, project=None):
        self.project = project
        self.origins = origins
        self.graph = graph
        self.cores = mp.cpu_count()
        self.aux_res = MultiThreadedPaths()
        self.report = []
        self.procedure_id = ""
        self.procedure_date = ""
        self.cumulative = 0

    def doWork(self):
        self.execute()

    def execute(self):
        """Runs the skimming process as specified in the graph"""
        self.connectivity.emit(["zones finalized", 0])
        self.aux_res.prepare_(self.graph, self.cores, self.graph.compact_num_nodes + 1)

        pool = ThreadPool(self.cores)
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

        self.connectivity.emit(["text connectivity", "Saving Outputs"])
        self.connectivity.emit(["finished_threaded_procedure", None])

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

        if isinstance(cores, int):
            if cores < 0:
                self.cores = max(1, mp.cpu_count() + cores)
            if cores == 0:
                self.cores = mp.cpu_count()
            elif cores > 0:
                cores = min(mp.cpu_count(), cores)
                if self.cores != cores:
                    self.cores = cores
        else:
            raise ValueError("Number of cores needs to be an integer")

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

        self.connectivity.emit(["zones finalized", self.cumulative])
        txt = str(self.cumulative) + " / " + str(self.matrix.zones)
        self.connectivity.emit(["text connectivity", txt])
