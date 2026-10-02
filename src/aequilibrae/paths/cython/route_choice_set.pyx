# cython: language_level=3str
"""
Generate route sets with BFSLE or link penalisation.

BFSLE follows Rieser-Schüssler, Balmer and Axhausen, "Route Choice Sets for
Very High-Resolution Data": https://doi.org/10.1080/18128602.2012.671383

BFSLE starts with the unmodified graph, finds a route, and adds a new subgraph
for each route link by banning that link together with previously banned links.
Duplicate banned-link sets are skipped. Link penalisation instead the costs of
links in each found route before searching again.

The route set stores pointers to RouteCandidate objects. Ordered links identify
unique routes with optional turn-cost increments so PSL can use the costs
later. The removed-link sets and current/next queues are also pointer-based:
queues are swapped at each depth, and duplicate sets are discarded. The custom
hash and equality operations that compare pointed-to values are declared in
route_choice_types.pxd.

Search results are overwritten by the next search, so turn-cost increments must
be copied and routes must be constructed from the predecessors. The queue is
shuffled when it may fill the route set in that depth.
"""

from cython.operator cimport dereference as d
from cython.parallel cimport parallel, prange, threadid
from libc.limits cimport UINT_MAX
from libc.math cimport INFINITY, isfinite
from libc.string cimport memcpy
from libcpp cimport bool
from libcpp cimport nullptr
from libcpp.algorithm cimport reverse, copy
from libcpp.memory cimport shared_ptr
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from openmp cimport omp_get_max_threads

from aequilibrae.matrix.coo_demand cimport GeneralisedCOODemand
from aequilibrae.paths.cython.a_star cimport EuclideanContext, HaversineContext, cpp_a_star
from aequilibrae.paths.cython.a_star import validate_scale
from aequilibrae.paths.cython.context cimport NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.dijkstra cimport cpp_dijkstra
from aequilibrae.paths.cython.route_choice_types cimport (
    LinkSet_t, RouteVec_t,
    RouteCandidate,
    RouteCandidateSet_t,
    minstd_rand,
    shuffle,
)
from aequilibrae.paths.cython.search_results cimport SearchResults
from aequilibrae.paths.cython.workspaces cimport SearchWorkspace, AStarWorkspace, CppAStarWorkspace
from aequilibrae.paths.graph import Graph, _get_graph_to_network_mapping
from aequilibrae.paths.routing_context import make_routing_context
from aequilibrae.paths.path_heuristics import HEURISTICS, make_heuristic_context
from aequilibrae.paths.cython.route_choice_set_results cimport (
    recompute_route_probabilities,
    imported_route_dataframe,
)
from aequilibrae.paths.cython.route_choice_set_results import check_disutility_cutoff_values
from aequilibrae.utils.cython.bar cimport Bar
from aequilibrae.utils.cython.bridge cimport Bridge, log, aeq_format_string as f, DEBUG, WARNING

from typing import Tuple
import builtins
import logging
import operator

import cython
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@cython.embedsignature(True)
cdef class RouteChoiceSet:
    """
    Route choice via BFSLE or link penalisation, using the compact routing graph.
    See the module documentation for the algorithms and reference.
    """

    def __init__(self, graph: Graph, *, coordinates: pd.DataFrame | None = None):
        self.graph = graph
        self.routing = make_routing_context(graph, compact=True)
        self.coordinates = coordinates.copy() if coordinates is not None else None
        self.lonlat = graph.lonlat_index.copy()
        self.node_ids = graph.compact_all_nodes.copy()
        self.turn_based = isinstance(self.routing, TurnBasedContext)
        # Directed link indices use full-graph row order, before compression can hide invalid steps.
        self.full_link_ids = graph.graph.link_id.to_numpy(copy=False)
        self.full_directions = graph.graph.direction.to_numpy(copy=False)
        self.full_link_indices = {
            int(link) * int(direction): i
            for i, (link, direction) in enumerate(zip(self.full_link_ids, self.full_directions))
        }
        self.centroids = graph.centroids
        self.network_link_ids = np.unique(self.full_link_ids)
        self.network_mapping = _get_graph_to_network_mapping(
            np.asarray(self.full_link_ids), np.asarray(self.full_directions)
        )
        for values in (self.network_link_ids, *vars(self.network_mapping).values()):
            values.flags.writeable = False

        # We only need to do extra work when there's penalties, not just bans
        self.has_turn_costs = False
        if self.turn_based:
            penalties = self.routing.turn_penalties
            self.has_turn_costs = np.any((penalties > 0) & np.isfinite(penalties))

        self.cost_view = self.routing.costs

        # The search loop needs a dense lookup without the GIL.
        self.nodes_to_indices_view = graph.compact_nodes_to_indices
        self.num_nodes = self.routing.node_count
        self.num_links = self.routing.link_count

        # Keep the full-to-compact link crosswalk in graph-row order for loading.
        self.graph_compressed_id_view = graph.graph.__compressed_id__.to_numpy(copy=False)
        idx, data, _ = graph.create_compressed_link_network_mapping()
        self.mapping_idx = idx
        self.mapping_data = data

        # Compressed route expansion uses supernetwork order, unlike imported links.
        signed_ids = np.empty(graph.num_links, dtype=np.int64)
        signed_ids[graph.graph.__supernet_id__.to_numpy(copy=False)] = (
            np.asarray(self.full_link_ids) * np.asarray(self.full_directions)
        )
        self.link_id_direction = signed_ids

        self.results = None
        self.ll_results = None

    @cython.embedsignature(True)
    def run(self, origin: int, destination: int, shape: Tuple[int, int], demand: float = 0.0, *args, **kwargs):
        """Compute the a route set for a single OD pair.

        Often the returned list's length is ``max_routes``, however, it may be limited by ``max_depth`` or if all
        unique possible paths have been found then a smaller set will be returned.

        Additional arguments are forwarded to ``RouteChoiceSet.batched``.

        :Arguments:
            **origin** (:obj:`int`): Origin node ID. Must be present within compact graph. Recommended to choose a
                centroid.
            **destination** (:obj:`int`): Destination node ID. Must be present within compact graph. Recommended to
                choose a centroid.
            **demand** (:obj:`double`): Demand for this single OD pair.

        :Returns: **route set** (:obj:`list[tuple[int, ...]]): Returns a list of unique variable length tuples of
            link IDs. Represents paths from ``origin`` to ``destination``.
        """
        df = pd.DataFrame({
            "origin id": [origin],
            "destination id": [destination],
            "demand": [demand]
        }).set_index(["origin id", "destination id"])
        demand_coo = GeneralisedCOODemand("origin id", "destination id", np.asarray(self.nodes_to_indices_view), shape)
        demand_coo.add_df(df)

        self.batched(demand_coo, {}, *args, **kwargs)
        where = kwargs.get("where", None)
        if where is not None:
            results = RouteChoiceSetResults.read_dataset(where)
        else:
            results = self.get_results()
        return [tuple(x) for x in results["route set"]]

    @staticmethod
    def _validate_search_options(a_star, heuristic, heuristic_scale, penalty):
        if not np.isfinite(penalty) or penalty < 1.0:
            raise ValueError("`penalty` must be finite and >= 1")

        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic must be one of {list(HEURISTICS)}")

        if a_star or heuristic_scale is not None:
            validate_scale(heuristic_scale)

    # Bounds checking doesn't really need to be disabled here but the warning is annoying
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    def batched(
        self,
        demand: GeneralisedCOODemand,
        select_links: Dict[str, FrozenSet[FrozenSet[int]]] = None,
        sl_link_loading: bool = True,
        max_routes: int = 0,
        max_depth: int = 0,
        max_misses: int = 100,
        seed: int = 0,
        cores: int = 0,
        a_star: bool = False,
        bfsle: bool = True,
        penalty: float = 1.0,
        where: Optional[str] = None,
        to_parquet_kwargs: dict | None = None,
        store_results: bool = True,
        path_size_logit: bool = False,
        beta: float = 1.0,
        disutility_cutoff_constant: float = float('inf'),
        disutility_cutoff_coefficient: float = float('inf'),
        *,
        heuristic: str = "euclidean",
        heuristic_scale: float | None = None,
        bridge: Bridge
    ):
        """Compute the a route set for a list of OD pairs.

        Often the returned list for each OD pair's length is ``max_routes``, however, it may be limited by ``max_depth``
        or if all unique possible paths have been found then a smaller set will be returned.

        :Arguments:
            **ods** (:obj:`list[tuple[int, int]]`): List of OD pairs ``(origin, destination)``. Origin and destination
                node ID must be present within compact graph. Recommended to choose a centroids.
            **max_routes** (:obj:`int`): Maximum size of the generated route set. Must be non-negative. Default of
                ``0`` for unlimited.
            **max_depth** (:obj:`int`): Maximum depth BFSLE can explore, or maximum number of iterations for link
                penalisation. Must be non-negative. Default of ``0`` for unlimited.
            **max_misses** (:obj:`int`): Maximum number of collective duplicate routes found for a single OD pair.
                Terminates if exceeded.
            **seed** (:obj:`int`): Seed used for rng. Must be non-negative. Default of ``0``.
            **cores** (:obj:`int`): Number of cores to use when parallelising over OD pairs. Must be non-negative.
                Default of ``0`` for all available.
            **bfsle** (:obj:`bool`): Whether to use Breadth First Search with Link Removal (BFSLE) over link
                penalisation. Default ``True``.
            **penalty** (:obj:`float`): Penalty to use for Link Penalisation and BFSLE with LP.
            **a_star** (:obj:`bool`): Use A* instead of Dijkstra. Default ``False``.
            **heuristic** (:obj:`str`): ``euclidean`` (default) uses the constructor's planar coordinates;
                ``haversine`` uses the graph's longitude/latitude coordinates in degrees.
            **heuristic_scale** (:obj:`float` or None): Explicit finite, nonnegative coefficient required for A*.
                A scale above a consistent bound can give non-shortest paths. Use ``estimate_heuristic_scale``
                to calculate a conservative value before searching.
            **where** (:obj:`str`): Optional file path to save results to immediately. Will return None.
            **to_parquet_kwargs** (:obj:`dict`): Keyword arguments to supply to the underlying ``to_parquet`` call.
        """
        cdef:
            long long origin, dest
            long int i

        self._validate_search_options(a_star, heuristic, heuristic_scale, penalty)
        cdef RouteChoiceHeuristic search_heuristic

        search_heuristic.a_star = a_star
        search_heuristic.haversine = heuristic == "haversine"
        # The owner keeps coordinate buffers alive while workers share read-only views.
        heuristic_context = None
        if a_star:
            heuristic_context = make_heuristic_context(
                self.node_ids, self.coordinates, self.lonlat, heuristic, heuristic_scale
            )

            if search_heuristic.haversine:
                search_heuristic.haversine_context = (<HaversineContext>heuristic_context).view()
            else:
                search_heuristic.euclidean_context = (<EuclideanContext>heuristic_context).view()

        if select_links is None:
            select_links = {}

        if max_routes == 0 and max_depth == 0:
            raise ValueError("Either `max_routes` or `max_depth` must be > 0")

        if max_routes < 0 or max_depth < 0:
            raise ValueError("`max_routes`, `max_depth`, and `cores` must be non-negative")

        if path_size_logit and beta < 0:
            raise ValueError("`beta` must be >= 0 for path sized logit model")

        if path_size_logit:
            check_disutility_cutoff_values(disutility_cutoff_constant, disutility_cutoff_coefficient)

        for origin, dest in demand.df.index:
            if self.nodes_to_indices_view[origin] == -1:
                raise ValueError(f"Origin {origin} is not present within the compact graph")
            if self.nodes_to_indices_view[dest] == -1:
                raise ValueError(f"Destination {dest} is not present within the compact graph")

        cdef Bar bar = bridge.new_bar("{}/{} ODs processed", len(demand.df))

        cdef:
            long long origin_index, dest_index
            unsigned int c_max_routes = max_routes
            unsigned int c_max_depth = max_depth
            unsigned int c_max_misses = max_misses
            unsigned int c_seed = seed
            long int c_cores = cores if cores > 0 else omp_get_max_threads()

            double [:, ::1] cost_matrix = np.zeros((c_cores, self.num_links), dtype=np.float64)
            bool [:, ::1] targets = np.zeros((c_cores, self.num_nodes), dtype=np.bool_)
            vector[CppSearchQuery] query_views
            vector[CppMutableSearchResults] search_views
            vector[CppAStarWorkspace] workspace_views
            SearchWorkspace workspace
            vector[CppNodeBasedContext] node_views
            vector[CppTurnBasedContext] turn_views
            SearchResults search

        # Each worker keeps its own cost binding, target and result buffers.
        workers = []
        worker_contexts = []
        query_views.resize(c_cores)
        search_views.resize(c_cores)
        workspace_views.resize(c_cores)
        node_views.resize(c_cores)
        turn_views.resize(c_cores)
        for j in range(c_cores):
            worker_context = self.routing.with_costs(cost_matrix[j])
            worker_contexts.append(worker_context)

            search = SearchResults(self.num_nodes, self.routing.state_count, self.num_links)
            if search_heuristic.a_star:
                workspace = AStarWorkspace(self.num_nodes, self.routing.state_count)
                workspace_views[j] = (<AStarWorkspace>workspace).a_star_view()
            else:
                workspace = SearchWorkspace(self.num_nodes, self.routing.state_count)
                workspace_views[j].search = workspace.view()
            workers.append((search, workspace))

            query_views[j].node_count = self.num_nodes
            query_views[j].target_mask = &targets[j, 0]
            query_views[j].target_count = 1
            search_views[j] = search.view()

            # Fused types are not nice as class attributes so we need to have
            # both, only one will ever be used at once
            if self.turn_based:
                turn_views[j] = (<TurnBasedContext>worker_context).view()
            else:
                node_views[j] = (<NodeBasedContext>worker_context).view()

        cdef:
            RouteCandidateSet_t *route_set
            vector[vector[double]] *turn_vecs
            shared_ptr[vector[double]] prob_vec
            int thread_id
            bint found_zero_cost

        demand._initalise_col_names()
        self.ll_results = LinkLoadingResults(demand, select_links, self.num_links, sl_link_loading, c_cores)

        for _, grouped_demand_df in (demand.batches() if where is not None else ((None, None),)):
            if bridge.should_stop():
                break

            demand._initalise_c_data(grouped_demand_df)

            self.results = RouteChoiceSetResults(
                demand,
                disutility_cutoff_constant,
                disutility_cutoff_coefficient,
                beta,
                self.num_links,
                self.cost_view,
                self.mapping_idx,
                self.mapping_data,
                self.link_id_direction,
                store_results=store_results,
                perform_assignment=path_size_logit,
            )

            with nogil, parallel(num_threads=c_cores):
                # Make the variables thread local
                route_set = new RouteCandidateSet_t()
                turn_vecs = new vector[vector[double]]()
                thread_id = threadid()
                found_zero_cost = False

                for i in prange(<long int>demand.ods.size(), schedule="guided"):
                    if bridge.should_stop():
                        break

                    origin_index = self.nodes_to_indices_view[demand.ods[i].first]
                    dest_index = self.nodes_to_indices_view[demand.ods[i].second]
                    log(bridge.c, DEBUG, f("Route choice: ", origin_index, ", ", dest_index))

                    if origin_index == dest_index:
                        bar.inc()
                        continue

                    query_views[thread_id].origin = origin_index
                    targets[thread_id, dest_index] = True

                    if bfsle:
                        RouteChoiceSet.bfsle(
                            self, d(route_set), origin_index, dest_index,
                            c_max_routes, c_max_depth, c_max_misses, cost_matrix[thread_id],
                            query_views[thread_id], search_views[thread_id], workspace_views[thread_id],
                            node_views[thread_id], turn_views[thread_id], search_heuristic, penalty, c_seed,
                            path_size_logit and self.has_turn_costs,
                        )
                    else:
                        RouteChoiceSet.link_penalisation(
                            self, d(route_set), origin_index, dest_index,
                            c_max_routes, c_max_depth, c_max_misses, cost_matrix[thread_id],
                            query_views[thread_id], search_views[thread_id], workspace_views[thread_id],
                            node_views[thread_id], turn_views[thread_id], search_heuristic, penalty, c_seed,
                            path_size_logit and self.has_turn_costs,
                        )

                    # Move links and optional turn steps into the same route order.
                    route_vec = self.results.get_route_vec(i)
                    RouteChoiceSetResults.route_set_to_route_vec(
                        d(route_vec),
                        d(turn_vecs),
                        d(route_set),
                        path_size_logit and self.has_turn_costs,
                    )

                    if path_size_logit:
                        prob_vec = self.results.compute_result(
                            i, d(route_vec), d(turn_vecs), &found_zero_cost, thread_id
                        )
                        self.ll_results.link_load_single_route_set(i, d(route_vec), d(prob_vec), thread_id)
                        self.ll_results.sl_link_load_single_route_set(
                            i, d(route_vec),
                            d(prob_vec),
                            origin_index,
                            dest_index,
                            thread_id
                        )

                    if found_zero_cost:
                        log(
                            bridge.c,
                            WARNING,
                            f(
                                "Found zero cost route for: ",
                                demand.ods[i].first,
                                ", ",
                                demand.ods[i].second,
                                ". The entire route set has been masked.",
                            ),
                        )

                    if d(route_vec).size() == 0:
                        log(
                            bridge.c,
                            WARNING,
                            f(
                                "Found unreachable: ",
                                demand.ods[i].first,
                                ", ",
                                demand.ods[i].second,
                                ". No choice sets were generated.",
                            ),
                        )

                    d(turn_vecs).clear()
                    targets[thread_id, dest_index] = False
                    bar.inc()

                del route_set
                del turn_vecs

            if store_results:
                self.get_results()
                if where is not None:
                    self.results.write(where, to_parquet_kwargs if to_parquet_kwargs is not None else {})

        if path_size_logit:
            self.ll_results.reduce_link_loading()
            self.ll_results.reduce_sl_link_loading()
            self.ll_results.reduce_sl_od_matrix()

            self.get_link_loading(cores=c_cores)
            self.get_sl_link_loading(cores=c_cores)
            self.get_sl_od_matrices()

    def make_demand(self, origin_column, destination_column):
        """Create demand using the borrowed node mapping and centroid count."""
        return GeneralisedCOODemand(
            origin_column, destination_column, self.graph.nodes_to_indices,
            shape=(len(self.centroids), len(self.centroids))
        )

    def compact_links(self, link_id, direction):
        """Map a selected link and direction using the graph's link mapping."""
        directions = (-1, 1) if direction == 0 else (direction,)
        if direction not in (-1, 0, 1):
            raise ValueError(f"link_id or direction {(link_id, direction)} is not present within graph.")

        matches = [
            self.full_link_indices[link_id * sign]
            for sign in directions
            if link_id * sign in self.full_link_indices
        ]

        if not matches:
            raise ValueError(f"link_id or direction {(link_id, direction)} is not present within graph.")

        cdef size_t local
        result = []
        for local in matches:
            compact = self.graph_compressed_id_view[local]
            if not 0 <= compact < self.num_links:
                raise ValueError(
                    f"link ID {link_id} and direction {direction} is not present in compressed graph. "
                    "It may have been removed during dead-end removal."
                )

            result.append(compact)
        return result

    def recompute_psl(
        self, df, *, beta=1.0, disutility_cutoff_constant,
        disutility_cutoff_coefficient, log_warnings=True
    ):
        """Validate supplied routes and recompute the PSL results."""
        cdef RouteVec_t routes
        cdef vector[vector[double]] turns
        cdef vector[double] costs, overlap, probabilities
        cdef vector[bint] mask

        table = self.import_dataframe(df, log_warnings, routes, mask)
        self.recost_routes(table, routes, turns, costs, mask, log_warnings)

        recompute_route_probabilities(
            table,
            routes,
            turns,
            costs,
            mask,
            overlap,
            probabilities,
            self.graph.cost,
            beta,
            disutility_cutoff_constant,
            disutility_cutoff_coefficient
        )
        return imported_route_dataframe(table, costs, mask, overlap, probabilities)

    cdef object import_dataframe(self, object df, bint log_warnings, RouteVec_t &routes, vector[bint] &mask):
        """Convert signed IDs to full-link vectors."""
        cdef vector[long long] route
        cdef size_t local, i

        for column in ("origin id", "destination id", "route set"):
            if column not in df.columns:
                raise ValueError(f"provided DataFrame is missing required column '{column}'")

        keep = []
        for position, (index, row) in enumerate(df.iterrows()):
            route_ids = row["route set"]
            if not isinstance(route_ids, (list, np.ndarray)):
                raise TypeError(f"route sets must be a list or Numpy array, found {type(route_ids)}")

            if len(route_ids) == 0:
                if log_warnings:
                    logger.warning("Ignoring empty route at row %s for OD (%s, %s)",
                                   index, row["origin id"], row["destination id"])
                continue

            route.clear()
            for link in route_ids:
                if isinstance(link, (builtins.bool, np.bool_)):
                    raise TypeError("Route link IDs must be integers, not booleans")

                link = operator.index(link)
                if link not in self.full_link_indices:
                    raise ValueError(f"Imported route contains a link or direction absent from the graph: {link}")

                local = self.full_link_indices[link]
                if not 0 <= self.graph_compressed_id_view[local] < self.num_links:
                    raise ValueError(f"Imported route contains a link absent from the compact graph: {link}")

                route.push_back(local)
            keep.append(position)
            routes.emplace_back(new vector[long long]())
            d(routes.back()).swap(route)

        table = df.iloc[keep].copy()
        if "mask" in table:
            # Older path files stored the Cython boolean buffer as integer zeroes and ones.
            if table["mask"].isna().any() or not table["mask"].isin([False, True]).all():
                raise TypeError("The route mask must contain booleans without missing values")

            supplied_mask = table["mask"].to_numpy(dtype=np.bool_)
            for i in range(routes.size()):
                mask.push_back(supplied_mask[i])
        else:
            mask.resize(routes.size(), True)

        return table

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void recost_routes(
        self,
        object table,
        const RouteVec_t &routes,
        vector[vector[double]] &turns,
        vector[double] &costs,
        vector[bint] &mask,
        bint log_warnings
    ):
        """Walk full-link paths once, costing and checking each step before PSL."""
        cdef const int64_t[::1] node_ids = self.graph.all_nodes
        cdef const int64_t[::1] tails = self.graph.graph.a_node.to_numpy(copy=False)
        cdef const int64_t[::1] heads = self.graph.graph.b_node.to_numpy(copy=False)
        cdef const double[::1] link_costs = self.graph.cost[:tails.shape[0]]
        cdef const int64_t[::1] turn_fs, turn_to_arcs
        cdef const double[::1] turn_penalties
        cdef const vector[long long] *route
        cdef size_t i, j, previous, current
        cdef int64_t origin, destination, lo, hi, mid
        cdef int64_t blocked_centroids = (
            self.graph.num_zones if self.graph.block_centroid_flows and not self.turn_based else 0
        )
        cdef bint allow_uturns = self.graph.allow_path_uturns
        cdef bint connected, explicit_turn, valid, connectivity_ok, turns_ok, uturns_ok, centroids_ok
        cdef double cost, penalty

        if self.turn_based:
            turn_fs = self.graph.turn_fs
            turn_to_arcs = self.graph.turn_to_arcs
            turn_penalties = self.graph.turn_penalties

        costs.resize(routes.size(), 0.0)
        turns.resize(routes.size())

        for i, (index, row) in enumerate(table.iterrows()):
            route = &d(routes[i])
            origin, destination = operator.index(row["origin id"]), operator.index(row["destination id"])
            valid = connectivity_ok = turns_ok = uturns_ok = centroids_ok = True
            cost = 0.0
            turns[i].assign(d(route).size(), 0.0)

            if node_ids[tails[d(route)[0]]] != origin:
                valid = False
                if log_warnings:
                    logger.warning(
                        "Invalid route %s at row %s for OD (%s, %s): route starts at the wrong node",
                        row["route set"],
                        index,
                        origin,
                        destination
                    )

            if node_ids[heads[d(route).back()]] != destination:
                valid = False
                if log_warnings:
                    logger.warning(
                        "Invalid route %s at row %s for OD (%s, %s): route ends at the wrong node",
                        row["route set"],
                        index,
                        origin,
                        destination
                    )

            for j in range(d(route).size()):
                current = d(route)[j]
                cost += link_costs[current]
                if j == 0:
                    continue

                previous = d(route)[j - 1]
                connected = heads[previous] == tails[current]
                if not connected and connectivity_ok:
                    connectivity_ok = False
                    if log_warnings:
                        logger.warning(
                            "Invalid route %s at row %s for OD (%s, %s): disconnected links",
                            row["route set"],
                            index,
                            origin,
                            destination
                        )

                penalty = 0.0
                explicit_turn = False

                if self.turn_based:
                    lo, hi = turn_fs[previous], turn_fs[previous + 1]
                    while lo < hi:
                        mid = lo + (hi - lo) // 2
                        if turn_to_arcs[mid] < <int64_t>current:
                            lo = mid + 1
                        else:
                            hi = mid

                    if lo < turn_fs[previous + 1] and turn_to_arcs[lo] == <int64_t>current:
                        explicit_turn = True
                        penalty = turn_penalties[lo]

                turns[i][j] = penalty
                cost += penalty

                if not isfinite(penalty) and turns_ok:
                    turns_ok = False
                    if log_warnings:
                        logger.warning(
                            "Invalid route %s at row %s for OD (%s, %s): prohibited turn",
                            row["route set"],
                            index,
                            origin,
                            destination
                        )

                if connected and not allow_uturns and heads[current] == tails[previous]:
                    # An explicit finite turn overrides the default U-turn ban.
                    if (not explicit_turn or not isfinite(penalty)) and uturns_ok:
                        uturns_ok = False
                        if log_warnings:
                            logger.warning(
                                "Invalid route %s at row %s for OD (%s, %s): disallowed U-turn",
                                row["route set"],
                                index,
                                origin,
                                destination
                            )

                if connected and tails[current] < blocked_centroids:
                    if node_ids[tails[current]] != origin and centroids_ok:
                        centroids_ok = False
                        if log_warnings:
                            logger.warning(
                                "Invalid route %s at row %s for OD (%s, %s): blocked centroid flow",
                                row["route set"], index,
                                origin,
                                destination
                            )

            valid = valid and connectivity_ok and turns_ok and uturns_ok and centroids_ok
            if not valid:
                cost = INFINITY
            elif not isfinite(cost) and log_warnings:
                logger.warning(
                    "Invalid route %s at row %s for OD (%s, %s): infinite route cost",
                    row["route set"],
                    index,
                    origin,
                    destination
                )

            costs[i] = cost
            mask[i] = mask[i] and isfinite(cost)

    def assign_from_df(
        self,
        df: pd.DataFrame,
        demand: GeneralisedCOODemand,
        select_links: Dict[str, FrozenSet[FrozenSet[int]]] = None,
        recompute_psl: bool = False,
        sl_link_loading: bool = True,
        store_results: bool = True,
        beta: float = 1.0,
        disutility_cutoff_constant: float = float('inf'),
        disutility_cutoff_coefficient: float = float('inf'),
        *,
        log_warnings: bool = True,
    ):
        """
        Load supplied routes and their unmasked probabilities.

        PSL recomputation validates paths and replaces costs and probabilities. Otherwise, supplied results are kept and
        masked probabilities are used.
        """
        cdef RouteVec_t routes
        cdef vector[vector[double]] turns
        cdef vector[double] costs, overlap, route_probabilities
        cdef vector[bint] mask
        cdef size_t position, link
        cdef long long previous, compact
        cdef RouteVec_t compact_routes
        cdef vector[double] probabilities
        cdef vector[size_t] positions
        cdef vector[long long] *route

        df = self.import_dataframe(df, log_warnings, routes, mask)

        if recompute_psl:
            self.recost_routes(df, routes, turns, costs, mask, log_warnings)
            recompute_route_probabilities(
                df, routes, turns, costs, mask, overlap, route_probabilities,
                self.graph.cost, beta, disutility_cutoff_constant, disutility_cutoff_coefficient
            )
        else:
            if "probability" not in df:
                raise ValueError("provided DataFrame is missing required column 'probability'")

            route_probabilities = df["probability"].to_numpy(dtype=np.float64)

            # Without PSL recomputation, keep any supplied results. No route validation
            costs = df["cost"].to_numpy(dtype=np.float64) if "cost" in df else np.zeros(len(df))
            overlap = df["path overlap"].to_numpy(dtype=np.float64) if "path overlap" in df else np.zeros(len(df))
            for position in range(mask.size()):
                if not mask[position]:
                    route_probabilities[position] = 0.0

        cdef:
            long int c_cores = 1  # Single threaded only due to high python interop, this should be fast anyway
            int thread_id = 0

        # An OD with no routes loads no demand. Import has removed empty route rows.
        demand_indices = {od: i for i, od in enumerate(demand.df.index)}
        groups = df.groupby(list(demand.df.index.names), sort=False).indices

        # Now we initialise the demand matrix and prepare to insert the route sets
        demand._initalise_col_names()
        demand._initalise_c_data(None)

        self.results = RouteChoiceSetResults(
            demand,
            disutility_cutoff_constant,
            disutility_cutoff_coefficient,
            beta,
            self.num_links,
            self.cost_view,
            self.mapping_idx,
            self.mapping_data,
            self.link_id_direction,
            store_results=store_results,
            perform_assignment=True,
            full_link_ids=self.full_link_ids,
            full_directions=self.full_directions,
        )

        self.ll_results = LinkLoadingResults(demand, select_links, self.num_links, sl_link_loading, c_cores)

        # We iterate over the OD pairs in the path files. ODs without demand are omitted.
        for od, group in groups.items():
            if od[0] == od[1] or od not in demand_indices:
                continue

            od_idx = demand_indices[od]
            positions = group
            origin_index = self.nodes_to_indices_view[demand.ods[od_idx].first]
            dest_index = self.nodes_to_indices_view[demand.ods[od_idx].second]

            compact_routes.clear()
            probabilities.clear()
            for position in positions:
                if not mask[position]:
                    continue

                route = new vector[long long]()
                compact_routes.emplace_back(route)

                previous = -1
                # De-duplicate adjacent compact IDs without changing their order.
                # PSL validation checks original links before compression when requested.
                for link in d(routes[position]):
                    compact = self.graph_compressed_id_view[link]
                    if compact != previous:
                        route.push_back(compact)
                        previous = compact

                probabilities.push_back(route_probabilities[position])

            # Route and probability vectors can now be used for LL and SLL.
            self.ll_results.link_load_single_route_set(od_idx, compact_routes, probabilities, thread_id)
            self.ll_results.sl_link_load_single_route_set(
                od_idx, compact_routes, probabilities, origin_index, dest_index, thread_id
            )
            self.results.store_imported_result(od_idx, routes, costs, mask, overlap, route_probabilities, positions)

        # Clean up and reduce any results from the threaded storage
        self.ll_results.reduce_link_loading()
        self.ll_results.reduce_sl_link_loading()
        self.ll_results.reduce_sl_od_matrix()

        self.get_link_loading(cores=c_cores)
        self.get_sl_link_loading(cores=c_cores)
        self.get_sl_od_matrices()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef RouteCandidate *trace_route(
        RouteChoiceSet self,
        const CppMutableSearchResults &result,
        size_t destination,
        bint save_turns
    ) noexcept nogil:
        """Copy one settled state path before the next search replaces its labels."""
        cdef RouteCandidate *vec = new RouteCandidate()
        cdef size_t p = result.terminal_states[destination]

        # Walk the search tree from the destination and build the route backwards.
        while p != result.metadata.root:
            if save_turns:
                # Save this step before the next search replaces these labels.
                vec.turn_steps.push_back(result.turn_costs[p] - result.turn_costs[result.predecessors[p]])
            vec.links.push_back(result.connectors[p])
            p = result.predecessors[p]

        reverse(vec.links.begin(), vec.links.end())
        reverse(vec.turn_steps.begin(), vec.turn_steps.end())
        return vec

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef void path_find(
        RouteChoiceSet self,
        CppSearchQuery &query,
        const CppMutableSearchResults &result,
        const CppAStarWorkspace &workspace,
        const CppNodeBasedContext &node_context,
        const CppTurnBasedContext &turn_context,
        size_t destination,
        const RouteChoiceHeuristic &heuristic
    ) noexcept nogil:
        """Search one destination with the worker's current link costs."""
        if heuristic.a_star:
            if self.turn_based:
                if heuristic.haversine:
                    cpp_a_star(
                        turn_context, query, destination, heuristic.haversine_context, result, workspace
                    )
                else:
                    cpp_a_star(
                        turn_context, query, destination, heuristic.euclidean_context, result, workspace
                    )
            elif heuristic.haversine:
                cpp_a_star(
                    node_context, query, destination, heuristic.haversine_context, result, workspace
                )
            else:
                cpp_a_star(
                    node_context, query, destination, heuristic.euclidean_context, result, workspace
                )
        elif self.turn_based:
            cpp_dijkstra(turn_context, query, result, workspace.search)
        else:
            cpp_dijkstra(node_context, query, result, workspace.search)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef void bfsle(
        RouteChoiceSet self,
        RouteCandidateSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [::1] thread_cost,
        CppSearchQuery &query,
        const CppMutableSearchResults &result,
        const CppAStarWorkspace &workspace,
        const CppNodeBasedContext &node_context,
        const CppTurnBasedContext &turn_context,
        const RouteChoiceHeuristic &heuristic,
        double penalty,
        unsigned int seed,
        bint save_turns
    ) noexcept nogil:
        """Main method for route set generation. See top of file for commentary."""
        cdef:
            # Scratch objects
            LinkSet_t removed_links
            minstd_rand rng

            # These objects are juggled to prevent more allocations than necessary
            vector[unordered_set[long long] *] queue
            vector[unordered_set[long long] *] next_queue
            unordered_set[long long] *banned
            unordered_set[long long] *new_banned

            # Local variables, Cython doesn't allow conditional declarations
            RouteCandidate *vec
            pair[RouteCandidateSet_t.iterator, bool] status
            pair[LinkSet_t.iterator, bool] banned_status
            unsigned int miss_count = 0
            long long connector

            # Link penalisation, only used when penalty != 1.0
            bint lp = penalty != 1.0
            vector[double] *penalised_cost = <vector[double] *>nullptr
            vector[double] *next_penalised_cost = <vector[double] *>nullptr

            # Because we can have duplicate banned link sets, the insertion may fail, in that case we free the set
            # immediately. However, by doing so we then can't tell (without a method to track it), which sets have
            # already been freed in the queue if we happened to early exit from it, so we use another variable to just
            # free the remaining items in the queue.
            bool free_remaining = False

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX

        queue.push_back(new unordered_set[long long]())  # Start with no edges banned
        rng.seed(seed)

        if lp:
            # Although we don't need the dynamic ability of vectors here, Cython doesn't have the std::array module.
            penalised_cost = new vector[double](self.cost_view.shape[0])
            next_penalised_cost = new vector[double](self.cost_view.shape[0])
            copy(&self.cost_view[0], &self.cost_view[0] + self.cost_view.shape[0], penalised_cost.begin())
            copy(&self.cost_view[0], &self.cost_view[0] + self.cost_view.shape[0], next_penalised_cost.begin())

        # We'll go at most `max_depth` iterations down, at each depth we maintain a queue of the next set of banned
        # edges to consider
        for _depth in range(max_depth):
            if miss_count > max_misses or route_set.size() >= max_routes or queue.size() == 0:
                break

            # If we could potentially fill the route_set after this depth, shuffle the queue
            if queue.size() + route_set.size() >= max_routes:
                shuffle(queue.begin(), queue.end(), rng)

            next_queue.clear()
            for banned in queue:
                if free_remaining:
                    del banned
                    continue

                if lp:
                    # We copy the penalised cost buffer into the thread cost buffer to allow us to apply link
                    # penalisation,
                    copy(penalised_cost.cbegin(), penalised_cost.cend(), &thread_cost[0])
                else:
                    # ...otherwise we just copy directly from the cost view.
                    memcpy(&thread_cost[0], &self.cost_view[0], self.cost_view.shape[0] * sizeof(double))

                for connector in d(banned):
                    thread_cost[connector] = INFINITY

                RouteChoiceSet.path_find(
                    self, query, result, workspace, node_context, turn_context, dest_index, heuristic
                )

                # Mark this set of banned links as seen
                banned_status = removed_links.insert(banned)
                if not banned_status.second:
                    # If we failed to insert this banned set then an equal set already exists within the removed links
                    del banned
                    banned = d(banned_status.first)

                # If the destination is reachable we must build the path and re-add
                if result.terminal_states[dest_index] != <size_t>-1:
                    # Trace the settled state path, including its turn steps.
                    vec = RouteChoiceSet.trace_route(self, result, dest_index, save_turns)

                    if lp:
                        # Here we penalise all seen links for the *next* depth. If we penalised on the current depth
                        # then we would introduce a bias for earlier seen paths
                        for connector in vec.links:
                            # *= does not work
                            d(next_penalised_cost)[connector] = penalty * d(next_penalised_cost)[connector]

                    for connector in vec.links:
                        # This is one area for potential improvement. Here we construct a new set from the old one,
                        # copying all the elements then add a single element. An incremental set hash function could be
                        # of use. However, the since of this set is directly dependent on the current depth and as the
                        # route set size grows so incredibly fast the depth will rarely get high enough for this to
                        # matter. Copy the previously banned links, then for each vector in the path we add one and
                        # push it onto our queue
                        new_banned = new unordered_set[long long](d(banned))
                        new_banned.insert(connector)
                        # If we've already seen this set of removed links before we already know what the path is and
                        # its in our route set.
                        if removed_links.find(new_banned) != removed_links.end():
                            del new_banned
                        else:
                            next_queue.push_back(new_banned)

                    # The de-duplication of routes occurs here
                    status = route_set.insert(vec)
                    if not status.second:
                        del vec  # If the insertion failed, free this vector, we already have one that is equal to it
                        miss_count = miss_count + 1

                    if miss_count > max_misses or route_set.size() >= max_routes:
                        free_remaining = True
                        # This condition will be hit again at the start of the loop, we just don't want to
                        # iterate over the rest of the things in queue when we know there is not more space.
                        continue
                else:
                    pass

            queue.swap(next_queue)

            if lp:
                # Update the penalised_cost vector, since next_penalised_cost is always the one updated we just need to
                # bring penalised_cost up to date.
                copy(next_penalised_cost.cbegin(), next_penalised_cost.cend(), penalised_cost.begin())

        # We may have added more banned link sets to the queue then found out we hit the max depth, we should free those
        for banned in queue:
            del banned

        # We should also free all the sets in next_queue, we don't be needing them.  We remove next_queue before
        # removed_links because we just swapped it with queue, and removed_links contains a subset of those that were
        # added to queue (pre-swap). It may share elements so we make sure to erase them from the set before freeing
        # them to avoid a use-after free.

        for banned in removed_links:
            del banned

        if lp:
            # If we had enabled link penalisation, we'll need to free those vectors as well
            del penalised_cost
            del next_penalised_cost

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void link_penalisation(
        RouteChoiceSet self,
        RouteCandidateSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [::1] thread_cost,
        CppSearchQuery &query,
        const CppMutableSearchResults &result,
        const CppAStarWorkspace &workspace,
        const CppNodeBasedContext &node_context,
        const CppTurnBasedContext &turn_context,
        const RouteChoiceHeuristic &heuristic,
        double penalty,
        unsigned int seed,
        bint save_turns
    ) noexcept nogil:
        """Link penalisation algorithm for choice set generation."""
        cdef:
            # Scratch objects
            RouteCandidate *vec
            long long connector
            pair[RouteCandidateSet_t.iterator, bool] status
            unsigned int miss_count = 0

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX
        memcpy(&thread_cost[0], &self.cost_view[0], self.cost_view.shape[0] * sizeof(double))

        for _depth in range(max_depth):
            if route_set.size() >= max_routes:
                break

            RouteChoiceSet.path_find(self, query, result, workspace, node_context, turn_context, dest_index, heuristic)

            if result.terminal_states[dest_index] != <size_t>-1:
                # Trace the settled state path, including its turn steps.
                vec = RouteChoiceSet.trace_route(self, result, dest_index, save_turns)

                for connector in vec.links:
                    thread_cost[connector] = penalty * thread_cost[connector]

                # To prevent runaway algorithms if we find N duplicate routes we should stop
                status = route_set.insert(vec)
                if not status.second:
                    del vec  # If the insertion failed, free this vector, we already have one that is equal to it
                    miss_count = miss_count + 1

                if miss_count > max_misses:
                    break
            else:
                break

    def get_results(self):
        """
        :Returns:
            **route sets** (:obj:`pa.DataFrame`): Returns a table of OD pairs to lists of link IDs for
                each OD pair provided (as columns). Represents paths from ``origin`` to ``destination``.
        """
        if self.results is None:
            raise RuntimeError("Route Choice results not computed yet")

        return self.results.make_df_from_results()

    def get_link_loading(RouteChoiceSet self, cores: int = 0):
        """
        :Returns:
            **link loading results** (:obj:`Dict[str, np.array]`): Returns a dict of demand column names to
                uncompressed link loads.
        """
        if self.ll_results is None:
            raise RuntimeError("Link loading results not computed yet")

        return self.ll_results.link_loading_to_objects(
            self.graph_compressed_id_view,
            cores if cores > 0 else omp_get_max_threads()
        )

    def get_sl_link_loading(RouteChoiceSet self, cores: int = 0):
        """
        :Returns:
            **select link loading results** (:obj:`Dict[str, Dict[str, np.array]]`): Returns a dict of select link set
                names to a dict of demand column names to uncompressed select link loads.
        """
        if self.ll_results is None:
            raise RuntimeError("Link loading results not computed yet")

        return self.ll_results.sl_link_loading_to_objects(
            self.graph_compressed_id_view,
            cores if cores > 0 else omp_get_max_threads()
        )

    def get_sl_od_matrices(RouteChoiceSet self):
        """
        :Returns:
            **select link OD matrix results** (:obj:`Dict[str, Dict[str, scipy.sparse.coo_matrix]]`): Returns a dict of
                select link set names to a dict of demand column names to a sparse OD matrix
        """
        if self.ll_results is None:
            raise RuntimeError("Link loading results not computed yet")

        return self.ll_results.sl_od_matrices_structs_to_objects()

    def write_path_files(RouteChoiceSet self, where, to_parquet_kwargs):
        """
        Write the path-files to the directory specified

        :Arguments:
            **where** (:obj:`pathlib.Path`): Directory to save the dataset to.
        """
        if self.results is None:
            raise RuntimeError("Route Choice results not computed yet")

        self.results.write(where, to_parquet_kwargs)
