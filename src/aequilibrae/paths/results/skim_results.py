from typing import List

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix
from aequilibrae.paths.cython.dijkstra import HEAP_MAP
from aequilibrae.paths.graph import Graph
from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads


class SkimResults:
    """
    Network skimming result holder.

    .. code-block:: python

          >>> from aequilibrae.paths.results import SkimResults

          >>> project = create_example(project_path)
          >>> project.network.build_graphs()

          # Mode c is car in this project
          >>> car_graph = project.network.graphs['c']

          # minimize travel time
          >>> car_graph.set_graph('free_flow_time')

          # Skims travel time and distance
          >>> car_graph.set_skimming(['free_flow_time', 'distance'])

          >>> res = SkimResults()
          >>> res.prepare(car_graph)

          >>> res.skims.export(project_path / "skim_matrices.omx")

          >>> project.close()
    """

    def __init__(self):
        self.skims = AequilibraeMatrix()
        self.cores = omp_get_max_threads()

        self._heap = "4ary"

    def prepare(self, graph: Graph):
        """
        Prepares the object with dimensions corresponding to the graph objects

        :Arguments:
            **graph** (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)
        """

        if not graph.cost_field:
            raise Exception('Cost field needs to be set for computation. use graph.set_graph("your_cost_field")')

        zones = graph.num_zones
        num_skims = len(graph.skim_fields)

        self.skims = AequilibraeMatrix()
        self.skims.create_empty(
            file_name=AequilibraeMatrix().random_name(), zones=zones, matrix_names=graph.skim_fields
        )
        self.skims.index[:] = graph.centroids[:]
        self.skims.computational_view(core_list=self.skims.names)
        self.skims.matrix_view = self.skims.matrix_view.reshape(zones, zones, num_skims)

    def set_heap(self, heap: str) -> None:
        """
        Set the priority queue implementation used for path computation. Must be one of ``get_heaps()``.

        :Arguments:
            **heap** (:obj:`str`): Heap to use.
        """
        if heap not in HEAP_MAP:
            raise ValueError(f"heap must be one of {self.get_heaps()}")

        self._heap = heap

    def get_heaps(self) -> List[str]:
        """Return the available priority queue implementations."""
        return list(HEAP_MAP.keys())
