import multiprocessing as mp
import sys
from datetime import datetime
from uuid import uuid4


from aequilibrae.context import get_active_project
from aequilibrae.paths.cython.skimming_core import skimming_parallel
from aequilibrae.paths.results.skim_results import SkimResults
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.core_setter import set_cores
from aequilibrae.utils.interface.worker_thread import WorkerThread

sys.dont_write_bytecode = True


class NetworkSkimming(WorkerThread):
    """

    .. code-block:: python

        >>> from aequilibrae.paths.network_skimming import NetworkSkimming

        >>> project = create_example(project_path)
        >>> project.network.build_graphs(modes=["c"])

        >>> graph = project.network.graphs['c']
        >>> graph.set_graph("distance")
        >>> graph.set_skimming("distance")

        >>> skm = NetworkSkimming(graph)
        >>> skm.execute()

        # The skim report (if any error generated) is available here
        >>> skm.report
        []

        # To access the skim matrix directly from its temporary file
        >>> matrix = skm.results.skims

        # Or you can save the results to disk
        >>> skm.save_to_project('skimming_result_omx', 'omx')

        >>> project.close()
    """

    signal = SIGNAL(object)

    def __init__(self, graph, origins=None, project=None):
        WorkerThread.__init__(self, None)
        self.project = project
        self.origins = origins
        self.graph = graph
        self.cores = mp.cpu_count()
        self.results = SkimResults()
        self.report = []
        self.procedure_id = ""
        self.procedure_date = ""
        self.cumulative = 0

    def doWork(self):
        self.execute()

    def execute(self):
        """Runs the skimming process as specified in the graph.

        Dispatches all origins to a single OpenMP-parallel Cython kernel
        (``skimming_parallel``). This avoids the per-origin Python pool
        dispatch overhead the previous ThreadPool-based path paid.
        """
        self.signal.emit(["start", self.graph.num_zones, ""])

        self.results.cores = self.cores
        self.results.prepare(self.graph)

        skipped = skimming_parallel(self.graph, self.results, self.results.cores)
        for _orig, msg in skipped:
            self.report.append(msg)

        self.signal.emit(["update", self.graph.num_zones, f"{self.graph.num_zones}/{self.graph.num_zones}"])

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

