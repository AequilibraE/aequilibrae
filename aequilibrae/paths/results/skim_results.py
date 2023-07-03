import multiprocessing as mp

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix
from aequilibrae.paths.graph import Graph


class SkimResults:
    """
    Network skimming result holder.

    .. code-block:: python

          >>> from aequilibrae import Project
          >>> from aequilibrae.paths.results import SkimResults

          >>> proj = Project.from_path("/tmp/test_project")
          >>> proj.network.build_graphs()

          # Mode c is car in this project
          >>> car_graph = proj.network.graphs['c']

          # minimize travel time
          >>> car_graph.set_graph('free_flow_time')

          # Skims travel time and distance
          >>> car_graph.set_skimming(['free_flow_time', 'distance'])

          >>> res = SkimResults()
          >>> res.prepare(car_graph)

          >>> res.skims.export('/tmp/test_project/matrix.aem')
    """

    def __init__(self):
        self.skims = AequilibraeMatrix()
        self.cores = mp.cpu_count()

        self.links = -1
        self.nodes = -1
        self.zones = -1
        self.num_skims = -1
        self.__graph_id__ = None
        self.graph = Graph()

    def prepare(self, graph: Graph):
        """
        Prepares the object with dimensions corresponding to the graph objects

        :Arguments:
            **graph** (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)
        """

        if not graph.cost_field:
            raise Exception('Cost field needs to be set for computation. use graph.set_graph("your_cost_field")')

        self.nodes = graph.compact_num_nodes + 1
        self.zones = graph.num_zones
        self.links = graph.compact_num_links + 1
        self.num_skims = len(graph.skim_fields)

        self.skims = AequilibraeMatrix()
        self.skims.create_empty(
            file_name=AequilibraeMatrix().random_name(), zones=self.zones, matrix_names=graph.skim_fields
        )
        self.skims.index[:] = graph.centroids[:]
        self.skims.computational_view(core_list=self.skims.names)
        self.skims.matrix_view = self.skims.matrix_view.reshape(self.zones, self.zones, self.num_skims)
        self.__graph_id__ = graph.__id__
        self.graph = graph
