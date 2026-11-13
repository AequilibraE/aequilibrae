from typing import List

from aequilibrae.paths.cython.outputs import SkimmingOutputs
from aequilibrae.paths.cython.dijkstra import HEAP_MAP
from aequilibrae.paths.graph import Graph
from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads


class SkimResults:
    """
    Network skimming result holder.

    .. code-block:: python

          >>> from aequilibrae.paths.results import SkimResults

          >>> project = create_example(project_path)

          # Mode c is car in this project
          # minimize travel time
          # Skims travel time and distance
          >>> car_graph = project.network.build_graph(
          ...     'c', 'free_flow_time', skimming=['free_flow_time', 'distance'], compact=True
          ... )

          >>> res = SkimResults()
          >>> res.prepare(car_graph)

          >>> res.skims.save_as_omx(project_path / "skim_matrices.omx")

          >>> project.close()
    """

    def __init__(self):
        self.skims = None
        self.cores = omp_get_max_threads()

        self._heap = "4ary"

    def prepare(self, graph: Graph, output: SkimmingOutputs | None = None):
        """
        Prepares the object with dimensions corresponding to the graph objects

        :Arguments:
            **graph** (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)

            **output** (:obj:`SkimmingOutputs`, *Optional*): Existing output buffer to share.
        """

        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph built by Network.build_graph()")
        if not graph.cost_field:
            raise ValueError("Build the graph with a cost field before skimming")

        if output is None:
            output = SkimmingOutputs(graph.num_zones, graph.num_zones, graph.skim_fields)
        self.skims = output.to_matrix(graph.centroids)

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
