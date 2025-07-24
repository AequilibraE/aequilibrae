import multiprocessing as mp
import sys

from aequilibrae.paths.AoN import connectivity_multi_threaded

from aequilibrae.utils.core_setter import set_cores
from aequilibrae.utils.aeq_signal import SIGNAL

sys.dont_write_bytecode = True


class ConnectivityAnalysis:
    """

    .. code-block:: python

        >>> from aequilibrae.paths.connectivity_analysis import ConnectivityAnalysis

        >>> project = create_example(project_path)

        >>> network = project.network
        >>> network.build_graphs()

        >>> graph = network.graphs['c']
        >>> graph.set_graph(cost_field="distance")
        >>> graph.set_blocked_centroid_flows(False)

        >>> conn_test = ConnectivityAnalysis(graph)
        >>> conn_test.execute()

        # The connectivity tester report as a Pandas DataFrame
        >>> disconnected = conn_test.disconnected_pairs

        >>> project.close()
    """

    connectivity = SIGNAL(object)

    def __init__(self, graph, origins=None, project=None):
        self.project = project
        self.origins = origins
        self.graph = graph
        self.cores = mp.cpu_count()
        self.report = []
        self.procedure_id = ""
        self.procedure_date = ""
        self.cumulative = 0

    def doWork(self):
        self.execute()

    def execute(self):
        """Runs the skimming process as specified in the graph"""

        self.disconnected_pairs = connectivity_multi_threaded(self)
        self.disconnected_pairs = self.disconnected_pairs.sort_values(["origin", "destination"])

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
