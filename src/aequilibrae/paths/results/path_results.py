import numpy as np

from aequilibrae.paths.cython.basic_path_finding import available_heaps
from aequilibrae.paths.cython.context import SkimmingContext
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults
from aequilibrae.paths.cython.skimming import skimming
from aequilibrae.paths.cython.workspaces import SkimmingWorkspace
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.routing_context import GraphMapping, make_routing_context


class PathResults:
    """Path computation result holder

    .. code-block:: python

        >>> from aequilibrae.paths.results import PathResults
        >>> project = create_example(project_path)
        >>> project.network.build_graphs()
        >>> graph = project.network.graphs['c']
        >>> graph.set_graph('distance')
        >>> graph.set_skimming(['distance', 'free_flow_time'])
        >>> res = PathResults(graph, 1, 17)
        >>> res.update_trace(9)
        >>> project.close()
    """

    def __init__(
        self,
        graph: Graph,
        origin: int,
        destination: int,
        early_exit: bool = False,
        a_star: bool = False,
        heuristic: str | None = None,
        heap: str | None = None,
    ) -> None:
        """Prepare graph data and compute the initial path.

        :Arguments:
            **graph** (:obj:`Graph`): Prepared graph with a cost field.
            **origin** (:obj:`int`): External ID of the path origin.
            **destination** (:obj:`int`): External ID of the initial destination.
            **early_exit** (:obj:`bool`): Stop after finalising the destination.
            **a_star** (:obj:`bool`): Whether to use A* for this search.
            **heuristic** (:obj:`str` or None): Heuristic used when A* is enabled.
            **heap** (:obj:`str` or None): Priority queue implementation to use.

        :Raises:
            **ValueError**: If an external node ID is not in the graph snapshot.
        """
        self._check_search_options(a_star, heuristic, heap)
        self._heap = "4ary"
        self.set_graph_data(graph)
        if heap is not None:
            self.set_heap(heap)
        self.compute_path(origin, destination, early_exit=early_exit)

    @staticmethod
    def _check_search_options(a_star, heuristic, heap):
        if a_star or heuristic is not None:
            raise NotImplementedError("PathResults does not support A* or its heuristics")
        if heap is not None and heap not in available_heaps():
            raise ValueError(f"heap must be one of {available_heaps()}")

    def set_graph_data(self, graph: Graph) -> None:
        """Prepare graph data for path and skim computation.

        :Arguments:
            **graph** (:obj:`Graph`): Prepared graph with a cost field. Its
                topology, costs, skim fields and external IDs are copied.

        :Returns:
            ``None``. The existing result buffers are replaced for the new
            snapshot and cleared before the next search.

        :Raises:
            **ValueError**: If the graph does not have context-compatible IDs
                or a required node/link mapping is invalid.
        """
        context = make_routing_context(graph)
        mapping = GraphMapping(context, graph.all_nodes, graph.graph.link_id, graph.graph.direction)

        fields = {}
        for name in graph.skim_fields:
            if name != graph.cost_field:
                values = np.array(graph.graph[name], dtype=np.float64, order="C", copy=True)
                values.flags.writeable = False
                fields[name] = values

        skim_context = SkimmingContext(
            context.link_count,
            link_fields=fields,
            cost_name=graph.cost_field if graph.cost_field in graph.skim_fields else None,
        )

        self.context = context
        self._mapping = mapping
        self.nodes = context.node_count
        self.links = context.link_count
        self.num_skims = len(skim_context.field_names)
        self.search_results = SearchResults(self.nodes, context.state_count, self.links)
        self._skimming = skim_context
        self._skim_workspace = (
            SkimmingWorkspace(context.state_count, skim_context.additive_field_count)
            if skim_context.additive_field_count
            else None
        )
        self.skims = skim_context.make_outputs(self.nodes) if self.num_skims else None
        self.reset()

    def compute_path(
        self,
        origin: int,
        destination: int,
        early_exit: bool = False,
        a_star: bool = False,
        heuristic: str | None = None,
        heap: str | None = None,
    ) -> None:
        """Search the prepared graph and trace a path between external IDs.

        With ``early_exit``, stop when the destination is finalised. Other
        missing terminals may still be reachable; ``update_trace`` searches
        again if needed.

        :Arguments:
            **origin** (:obj:`int`): External ID of the search origin.
            **destination** (:obj:`int`): External ID of the traced destination.
            **early_exit** (:obj:`bool`): Stop after finalising the destination.
            **a_star** (:obj:`bool`): Whether to use A* for this search.
            **heuristic** (:obj:`str` or None): Heuristic used when A* is enabled.
            **heap** (:obj:`str` or None): Priority queue implementation to use.

        :Returns:
            ``None``. The search arrays, skims and traced path are updated in place.

        :Raises:
            **ValueError**: If an external node ID is not in the prepared graph.
        """
        self._check_search_options(a_star, heuristic, heap)
        origin_index = self._mapping.node_index(origin)
        destination_index = self._mapping.node_index(destination)
        targets = None

        if early_exit:
            targets = np.zeros(self.nodes, dtype=bool)
            targets[destination_index] = True

        query = SearchQuery(self.nodes, origin_index, targets)
        dijkstra(self.context, query, self.search_results, heap=self._heap if heap is None else heap)

        self.origin = origin
        self.destination = destination
        self.early_exit = early_exit
        if self.skims is not None:
            skimming(self.search_results, self._skimming, self._skim_workspace, self.skims)

        self._trace(destination_index)

    def update_trace(self, destination: int) -> None:
        """Update the traced destination using the current search tree.

        A partial early-exit search is rerun when it has not finalised the
        requested destination. A full search reuses its finalised state tree.

        :Arguments:
            **destination** (:obj:`int`): External ID of the new destination.

        :Returns:
            ``None``. ``path``, ``path_nodes``, ``path_link_directions`` and
            ``milepost`` are replaced for the requested destination.

        :Raises:
            **RuntimeError**: If no search has been performed.
            **ValueError**: If the destination is not in the prepared graph.
        """
        index = self._mapping.node_index(destination)
        results = self.search_results
        if results.origin is None:
            raise RuntimeError("Compute a path before updating its trace")

        if not results.reachable_to(index) and not results.exhausted:
            self.compute_path(self.origin, destination, early_exit=self.early_exit)
            return

        self.destination = destination
        self._trace(index)

    def _trace(self, destination: int) -> None:
        results = self.search_results
        states = results.path_states_to(destination)
        if states.size == 0:
            self._clear_path()
            return

        links = results.connectors[states[1:]]

        self.path = self._mapping.link_ids[links]
        self.path_link_directions = self._mapping.directions[links]
        self.path_nodes = self._mapping.path_nodes(results.origin, links)
        # Stored labels preserve the cost of each actual arrival, including turns.
        self.milepost = results.distances[states]

    def _clear_path(self) -> None:
        self.path = None
        self.path_nodes = None
        self.path_link_directions = None
        self.milepost = None

    def reset(self) -> None:
        """Clear search results, paths, labels and skims in place.

        :Returns:
            ``None``. Allocated result buffers and their read-only views remain
            valid, but contain their reset sentinel or infinity values.
        """
        self.search_results.reset()
        if self.skims is not None:
            self.skims.reset()
        self._clear_path()
        self.origin = None
        self.destination = None
        self.early_exit = False

    def set_heap(self, heap: str) -> None:
        """Select the priority queue implementation used for path computation.

        :Arguments:
            **heap** (:obj:`str`): Name returned by :meth:`get_heaps`.

        :Returns:
            ``None``. The selected heap is used by subsequent searches.
        """
        if heap not in available_heaps():
            raise ValueError(f"heap must be one of {available_heaps()}")
        self._heap = heap

    def get_heaps(self) -> list[str]:
        """Return the available priority queue implementation names.

        :Returns:
            :obj:`list[str]`: Names accepted by :meth:`set_heap`.
        """
        return available_heaps()

    def set_heuristic(self, heuristic: str) -> None:
        """Select the heuristic used by A* path computation.

        :Arguments:
            **heuristic** (:obj:`str`): Name returned by :meth:`get_heuristics`.

        :Returns:
            ``None``. The selected heuristic is used by subsequent A* searches.
        """
        raise NotImplementedError("PathResults does not support A* or its heuristics")

    def get_heuristics(self) -> list[str]:
        """Return the available A* heuristic names.

        :Returns:
            :obj:`list[str]`: Names accepted by :meth:`set_heuristic`.
        """
        return []

    @property
    def node_ids(self) -> np.ndarray:
        """External node IDs in context and skim destination order."""
        return self._mapping.node_ids

    @property
    def predecessors(self) -> np.ndarray:
        return self.search_results.predecessors

    @property
    def connectors(self) -> np.ndarray:
        return self.search_results.connectors

    @property
    def settlement_order(self) -> np.ndarray:
        return self.search_results.settlement_order

    @property
    def terminal_states(self) -> np.ndarray:
        return self.search_results.terminal_states

    @property
    def distances(self) -> np.ndarray:
        return self.search_results.distances

    @property
    def turn_costs(self) -> np.ndarray:
        return self.search_results.turn_costs
