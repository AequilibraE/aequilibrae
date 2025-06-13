# cython: language_level=3str
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.cython.route_choice_types cimport LinkSet_t, minstd_rand, shuffle
from aequilibrae.matrix.coo_demand cimport GeneralisedCOODemand

from cython.operator cimport dereference as d
from cython.parallel cimport parallel, prange, threadid
from libc.limits cimport UINT_MAX
from libc.string cimport memcpy
from libcpp cimport nullptr
from libcpp.algorithm cimport reverse, copy
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool
from openmp cimport omp_get_max_threads

from libcpp.memory cimport shared_ptr

from typing import Tuple
import itertools
import warnings

import numpy as np
import pandas as pd


"""This module aims to implemented the BFS-LE algorithm as described in Rieser-Schüssler, Balmer, and Axhausen, 'Route
Choice Sets for Very High-Resolution Data'.  https://doi.org/10.1080/18128602.2012.671383

A rough overview of the algorithm is as follows.
    1. Prepare the initial graph, this is depth 0 with no links removed.
    2. Find a short path, P. If P is not empty add P to the path set.
    3. For all links p in P, remove p from E, compounding with the previously removed links.
    4. De-duplicate the sub-graphs, we only care about unique sub-graphs.
    5. Go to 2.

Details: The general idea of the algorithm is pretty simple, as is the implementation. The caveats here is that there is
a lot of cpp interop and memory management. A description of the purpose of variables is in order:

route_set: See route_choice.pxd for full type signature. It's an unordered set (hash set) of pointers to vectors of link
IDs. It uses a custom hashing function and comparator. The hashing function is defined in a string that in inlined
directly into the output cpp. This is done allow declaring of the `()` operator, which is required and AFAIK not
possible in Cython. The hash is designed to dereference then hash order dependent vectors. One isn't provided by
stdlib. The comparator simply dereferences the pointer and uses the vector comparator. It's designed to store the
outputted paths. Heap allocated (needs to be returned).

removed_links: See route_choice.pxd for full type signature. It's an unordered set of pointers to unordered sets of link
IDs. Similarly to `route_set` is uses a custom hash function and comparator. This hash function is designed to be order
independent and should only use commutative operations. The comparator is the same. It's designed to store all of the
removed link sets we've seen before. This allows us to detected duplicated graphs.

rng: A custom imported version of std::linear_congruential_engine. libcpp doesn't provide one so we do. It should be
significantly faster than the std::mersenne_twister_engine without sacrificing much. We don't need amazing RNG, just ok
and fast. This is only used to shuffle the queue.

queue, next_queue: These are vectors of pointers to sets of removed links. We never need to push to the front of these
so a vector is best. We maintain two queues, one that we are currently iterating over, and one that we can add to,
building up with all the newly removed link sets. These two are swapped at the end of an iteration, next_queue is then
cleared. These store sets of removed links.

banned, next_banned: `banned` is the iterator variable for `queue`. `banned` is copied into `next_banned` where another
link can be added without mutating `banned`. If we've already seen this set of removed links `next_banned` is
immediately deallocated. Otherwise it's placed into `next_queue`.

vec: `vec` is a scratch variable to store pointers to new vectors, or rather, paths while we are building them. Each
time a path is found a new one is allocated, built, and stored in the route_set.

p, connector: Scratch variables for iteration.

Optimisations: As described in the paper, both optimisations have been implemented. The path finding operates on the
compressed graph and the queue is shuffled if its possible to fill the route set that iteration. The route set may not
be filled due to duplicate paths but we can't know that in advance so we shuffle anyway.

Any further optimisations should focus on the path finding, from benchmarks it dominates the run time (~98%). Since huge
routes aren't required small-ish things like the memcpy and banned link set copy aren't high priority.

"""

# It would really be nice if these were modules. The 'include' syntax is long deprecated and adds a lot to compilation
# times
include 'basic_path_finding.pyx'


@cython.embedsignature(True)
cdef class RouteChoiceSet:
    """
    Route choice implemented via breadth first search with link removal (BFS-LE) as described in Rieser-Schüssler,
    Balmer, and Axhausen, 'Route Choice Sets for Very High-Resolution Data'
    """

    def __init__(self, graph: Graph):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        # self.heuristic = HEURISTIC_MAP[self.res._heuristic]
        self.cost_view = graph.compact_cost
        self.graph_fs_view = graph.compact_fs
        self.b_nodes_view = graph.compact_graph.b_node.values
        self.nodes_to_indices_view = graph.compact_nodes_to_indices

        # tmp = graph.lonlat_index.loc[graph.compact_all_nodes]
        # self.lat_view = tmp.lat.values
        # self.lon_view = tmp.lon.values
        self.a_star = False

        self.ids_graph_view = graph.compact_graph.id.values

        # We explicitly don't want the links that have been removed from the graph
        self.graph_compressed_id_view = graph.graph.__compressed_id__.values
        self.num_nodes = graph.compact_num_nodes
        self.num_links = graph.compact_num_links
        self.zones = graph.num_zones
        self.block_flows_through_centroids = graph.block_centroid_flows

        self.mapping_idx, self.mapping_data, _ = graph.create_compressed_link_network_mapping()

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
        a_star: bool = True,
        bfsle: bool = True,
        penalty: float = 1.0,
        where: Optional[str] = None,
        store_results: bool = True,
        path_size_logit: bool = False,
        beta: float = 1.0,
        cutoff_prob: float = 0.0,
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
            **where** (:obj:`str`): Optional file path to save results to immediately. Will return None.
        """
        cdef:
            long long origin, dest
            long int i

        if select_links is None:
            select_links = {}

        if max_routes == 0 and max_depth == 0:
            raise ValueError("Either `max_routes` or `max_depth` must be > 0")

        if max_routes < 0 or max_depth < 0:
            raise ValueError("`max_routes`, `max_depth`, and `cores` must be non-negative")

        if path_size_logit and beta < 0:
            raise ValueError("`beta` must be >= 0 for path sized logit model")

        if path_size_logit and not 0.0 <= cutoff_prob <= 1.0:
            raise ValueError("`cutoff_prob` must be 0 <= `cutoff_prob` <= 1 for path sized logit model")

        for origin, dest in demand.df.index:
            if self.nodes_to_indices_view[origin] == -1:
                raise ValueError(f"Origin {origin} is not present within the compact graph")
            if self.nodes_to_indices_view[dest] == -1:
                raise ValueError(f"Destination {dest} is not present within the compact graph")

        cdef:
            long long origin_index, dest_index
            unsigned int c_max_routes = max_routes
            unsigned int c_max_depth = max_depth
            unsigned int c_max_misses = max_misses
            unsigned int c_seed = seed
            long int c_cores = cores if cores > 0 else omp_get_max_threads()

            # Scale cutoff prob from [0, 1] -> [0.5, 1]. Values below 0.5 produce negative inverse binary logit values.
            double scaled_cutoff_prob = (1.0 - cutoff_prob) * 0.5 + 0.5

            # A* (and Dijkstra's) require memory views, so we must allocate here and take slices. Python can handle this
            # memory
            double [:, :] cost_matrix = np.empty((c_cores, self.cost_view.shape[0]), dtype=float)
            long long [:, :] predecessors_matrix = np.empty((c_cores, self.num_nodes + 1), dtype=np.int64)
            long long [:, :] conn_matrix = np.empty((c_cores, self.num_nodes + 1), dtype=np.int64)
            long long [:, :] b_nodes_matrix = np.broadcast_to(
                self.b_nodes_view,
                (c_cores, self.b_nodes_view.shape[0])
            ).copy()

            # This matrix is never read from, it exists to allow using the Dijkstra's method without changing the
            # interface.
            long long [:, :] _reached_first_matrix

            unsigned char [:, :] destinations_matrix = np.zeros((c_cores, self.num_nodes), dtype="bool")

            # self.a_star = a_star

        if self.a_star:
            _reached_first_matrix = np.zeros((c_cores, 1), dtype=np.int64)  # Dummy array to allow slicing
        else:
            _reached_first_matrix = np.zeros((c_cores, self.num_nodes + 1), dtype=np.int64)

        cdef:
            RouteSet_t *route_set
            shared_ptr[vector[double]] prob_vec
            int thread_id
            bint found_zero_cost

        demand._initalise_col_names()
        self.ll_results = LinkLoadingResults(demand, select_links, self.num_links, sl_link_loading, c_cores)

        # These are accessed with the gil and used for error reporting
        zero_cost_ods: list[tuple[int]] = []
        unreachable_ods: list[tuple[int]] = []

        for _, grouped_demand_df in (demand.batches() if where is not None else ((None, None),)):
            demand._initalise_c_data(grouped_demand_df)

            self.results = RouteChoiceSetResults(
                demand,
                scaled_cutoff_prob,
                beta,
                self.num_links,
                self.cost_view,
                self.mapping_idx,
                self.mapping_data,
                store_results=store_results,
                perform_assignment=path_size_logit,
            )

            with nogil, parallel(num_threads=c_cores):
                route_set = new RouteSet_t()
                thread_id = threadid()
                found_zero_cost = False  # Make the variable thread local
                for i in prange(<long int>demand.ods.size(), schedule="guided"):
                    origin_index = self.nodes_to_indices_view[demand.ods[i].first]
                    dest_index = self.nodes_to_indices_view[demand.ods[i].second]

                    if origin_index == dest_index:
                        continue

                    if self.block_flows_through_centroids:
                        blocking_centroid_flows(
                            0,  # Always blocking
                            origin_index,
                            self.zones,
                            self.graph_fs_view,
                            b_nodes_matrix[thread_id],
                            self.b_nodes_view,
                        )

                    if bfsle:
                        RouteChoiceSet.bfsle(
                            self,
                            d(route_set),
                            origin_index,
                            dest_index,
                            c_max_routes,
                            c_max_depth,
                            c_max_misses,
                            cost_matrix[thread_id],
                            predecessors_matrix[thread_id],
                            conn_matrix[thread_id],
                            b_nodes_matrix[thread_id],
                            _reached_first_matrix[thread_id],
                            destinations_matrix[thread_id],
                            penalty,
                            c_seed,
                        )
                    else:
                        RouteChoiceSet.link_penalisation(
                            self,
                            d(route_set),
                            origin_index,
                            dest_index,
                            c_max_routes,
                            c_max_depth,
                            c_max_misses,
                            cost_matrix[thread_id],
                            predecessors_matrix[thread_id],
                            conn_matrix[thread_id],
                            b_nodes_matrix[thread_id],
                            _reached_first_matrix[thread_id],
                            destinations_matrix[thread_id],
                            penalty,
                            c_seed,
                        )

                    # Here we transform the set of raw pointers to routes (vectors) into a vector of unique points to
                    # routes. This is done to simplify memory management later on.
                    route_vec = self.results.get_route_vec(i)
                    RouteChoiceSetResults.route_set_to_route_vec(d(route_vec), d(route_set))

                    if path_size_logit:
                        prob_vec = self.results.compute_result(i, d(route_vec), &found_zero_cost, thread_id)
                        self.ll_results.link_load_single_route_set(i, d(route_vec), d(prob_vec), thread_id)
                        self.ll_results.sl_link_load_single_route_set(
                            i, d(route_vec),
                            d(prob_vec),
                            origin_index,
                            dest_index,
                            thread_id
                        )


                    if d(route_vec).size() == 0 or found_zero_cost:
                        with gil:
                            if found_zero_cost:
                                zero_cost_ods.append(tuple(demand.ods[i]))
                            if d(route_vec).size() == 0:
                                unreachable_ods.append(tuple(demand.ods[i]))

                    if self.block_flows_through_centroids:
                        blocking_centroid_flows(
                            1,  # Always unblocking
                            origin_index,
                            self.zones,
                            self.graph_fs_view,
                            b_nodes_matrix[thread_id],
                            self.b_nodes_view,
                        )

                del route_set

            if store_results:
                self.get_results()
                if where is not None:
                    self.results.write(where)

        if path_size_logit:
            self.ll_results.reduce_link_loading()
            self.ll_results.reduce_sl_link_loading()
            self.ll_results.reduce_sl_od_matrix()

            self.get_link_loading(cores=c_cores)
            self.get_sl_link_loading(cores=c_cores)
            self.get_sl_od_matrices()

        if zero_cost_ods:
            warnings.warn(
                f"found zero cost routes for OD pairs, the entire route set has been masked for: {zero_cost_ods}"
            )
        if unreachable_ods:
            warnings.warn(
                f"found unreachable OD pairs, no choice sets generated for: {unreachable_ods}"
            )

    def assign_from_df(
        self,
        graph: pd.DataFrame,
        df: pd.DataFrame,
        demand: GeneralisedCOODemand,
        select_links: Dict[str, FrozenSet[FrozenSet[int]]] = None,
        recompute_psl: bool = False,
        sl_link_loading: bool = True,
        store_results: bool = True,
        beta: float = 1.0,
        cutoff_prob: float = 0.0,
    ):
        cdef:
            long int c_cores = 1  # Single threaded only due to high python interop, this shoud be fast anyway
            int thread_id = 0

            # Scale cutoff prob from [0, 1] -> [0.5, 1]. Values below 0.5 produce negative inverse binary logit values.
            double scaled_cutoff_prob = (1.0 - cutoff_prob) * 0.5 + 0.5

        for _, route_list in df["route set"].items():
            if not isinstance(route_list, (list, np.ndarray)):
                raise TypeError(f"route sets must be a list or Numpy array, found {type(route_list)}")

        # We want to enforce that if the demand matrix cell is non-cell for an OD pair, then at least one route exists to assign to it
        demand_df = demand.df.assign(idx=np.arange(len(demand.df)))
        demand_df = demand_df[demand_df.index.get_level_values(0) != demand_df.index.get_level_values(1)]

        df = df.set_index(demand_df.index.names)
        if not demand_df.index.drop_duplicates().isin(df.index).all():
            raise KeyError("not all origin and destinations IDs from the demand matrix are present within the path files")

        # We also store those indices along side the route sets themselves so it's easier to keep track
        df = demand_df[["idx"]].merge(df, how="left", left_index=True, right_index=True).reset_index()
        gb = df.groupby(by="idx")

        # In order to map the network link IDs to compressed links we'll use the graph
        graph_to_compressed = graph[["link_id", "direction"]].prod(axis=1).reset_index().set_index(0)

        # Now we initialise the demand matrix and prepare to insert the route sets
        demand._initalise_col_names()
        demand._initalise_c_data(None)

        self.results = RouteChoiceSetResults(
            demand,
            scaled_cutoff_prob,
            beta,
            self.num_links,
            self.cost_view,
            self.mapping_idx,
            self.mapping_data,
            store_results=store_results,
            perform_assignment=True,
        )

        self.ll_results = LinkLoadingResults(demand, select_links, self.num_links, sl_link_loading, c_cores)

        cdef:
            vector[long long] *route
            bint found_zero_cost

        # We iterate over the OD pairs in the path files
        for od_idx, df in gb:
            # We obtain a reference to the route vector, we then need to insert the right *compressed* link IDs
            route_vec = self.results.get_route_vec(od_idx)

            # If we are reusing the probabilities then we need to a similar thing for this
            if not recompute_psl:
                prob_vec = self.results.get_prob_vec(od_idx)

            d(route_vec).reserve(len(df))
            for _, row in df.iterrows():
                # We find the indices for the compressed id that corresponds to the direction link id pair (as a
                # product)
                compressed_link_indices = graph_to_compressed.loc[row["route set"]]["index"].to_numpy()

                route = new vector[long long]()
                # Then use itertools.groupby to de-duplicate them without modifying the order. The order is not required
                # for assignment but it is if we wish to output this route set again.
                for compressed_link_id, _ in itertools.groupby(graph.__compressed_id__.iloc[compressed_link_indices]):
                    route.push_back(compressed_link_id)

                d(route_vec).emplace_back(route)
                if not recompute_psl:
                    d(prob_vec).push_back(row["probability"])

            # If we are recomputing the probabilities then we do so here. This also has the side effect of recompute the
            # cost, masking, and path overlap with new parameters
            if recompute_psl:
                prob_vec = self.results.compute_result(od_idx, d(route_vec), &found_zero_cost, thread_id)

            # We have now have both the route and probability vectors restored so we can do LL and SLL.
            self.ll_results.link_load_single_route_set(od_idx, d(route_vec), d(prob_vec), thread_id)

            origin_index = self.nodes_to_indices_view[demand.ods[od_idx].first]
            dest_index = self.nodes_to_indices_view[demand.ods[od_idx].second]
            self.ll_results.sl_link_load_single_route_set(
                od_idx,
                d(route_vec),
                d(prob_vec),
                origin_index,
                dest_index,
                thread_id
            )

        # Clean up and reduce any results from the threaded storage
        self.ll_results.reduce_link_loading()
        self.ll_results.reduce_sl_link_loading()
        self.ll_results.reduce_sl_od_matrix()

        self.get_link_loading(cores=c_cores)
        self.get_sl_link_loading(cores=c_cores)
        self.get_sl_od_matrices()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef void path_find(
        RouteChoiceSet self,
        long origin_index,
        long dest_index,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        unsigned char [:] thread_destinations
    ) noexcept nogil:
        """Small wrapper around path finding, thread locals should be passes as arguments."""
        if self.a_star:
            path_finding_a_star(
                origin_index,
                dest_index,
                thread_cost,
                thread_b_nodes,
                self.graph_fs_view,
                self.nodes_to_indices_view,
                self.lat_view,
                self.lon_view,
                thread_predecessors,
                self.ids_graph_view,
                thread_conn,
                EQUIRECTANGULAR  # FIXME: enum import failing due to redefinition
            )
        else:
            thread_destinations[dest_index] = True
            path_finding(
                origin_index,
                thread_destinations,
                1,  # Single destination
                thread_cost,
                thread_b_nodes,
                self.graph_fs_view,
                thread_predecessors,
                self.ids_graph_view,
                thread_conn,
                _thread_reached_first
            )
            thread_destinations[dest_index] = False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef void bfsle(
        RouteChoiceSet self,
        RouteSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        unsigned char [:] thread_destinations,
        double penalty,
        unsigned int seed
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
            vector[long long] *vec
            pair[RouteSet_t.iterator, bool] status
            pair[LinkSet_t.iterator, bool] banned_status
            unsigned int miss_count = 0
            long long p, connector

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
        for depth in range(max_depth):
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
                    self,
                    origin_index,
                    dest_index,
                    thread_cost,
                    thread_predecessors,
                    thread_conn,
                    thread_b_nodes,
                    _thread_reached_first,
                    thread_destinations
                )

                # Mark this set of banned links as seen
                banned_status = removed_links.insert(banned)
                if not banned_status.second:
                    # If we failed to insert this banned set then an equal set already exists within the removed links
                    del banned
                    banned = d(banned_status.first)

                # If the destination is reachable we must build the path and readd
                if thread_predecessors[dest_index] >= 0:
                    vec = new vector[long long]()
                    # Walk the predecessors tree to find our path, we build it up in a C++ vector because we can't know
                    # how long it'll be
                    p = dest_index
                    while p != origin_index:
                        connector = thread_conn[p]
                        p = thread_predecessors[p]
                        vec.push_back(connector)

                    if lp:
                        # Here we penalise all seen links for the *next* depth. If we penalised on the current depth
                        # then we would introduce a bias for earlier seen paths
                        for connector in d(vec):
                            # *= does not work
                            d(next_penalised_cost)[connector] = penalty * d(next_penalised_cost)[connector]

                    reverse(vec.begin(), vec.end())

                    for connector in d(vec):
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
                        continue  # This condition will be hit again at the start of the loop, we just don't want to
                                  # iterate over the rest of the things in queue when we know there is not more space.
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
        RouteSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        unsigned char [:] thread_destinations,
        double penalty,
        unsigned int seed
    ) noexcept nogil:
        """Link penalisation algorithm for choice set generation."""
        cdef:
            # Scratch objects
            vector[long long] *vec
            long long p, connector
            pair[RouteSet_t.iterator, bool] status
            unsigned int miss_count = 0

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX
        memcpy(&thread_cost[0], &self.cost_view[0], self.cost_view.shape[0] * sizeof(double))

        for depth in range(max_depth):
            if route_set.size() >= max_routes:
                break

            RouteChoiceSet.path_find(
                self,
                origin_index,
                dest_index,
                thread_cost,
                thread_predecessors,
                thread_conn,
                thread_b_nodes,
                _thread_reached_first,
                thread_destinations
            )

            if thread_predecessors[dest_index] >= 0:
                vec = new vector[long long]()
                # Walk the predecessors tree to find our path, we build it up in a C++ vector because we can't know how
                # long it'll be
                p = dest_index
                while p != origin_index:
                    connector = thread_conn[p]
                    p = thread_predecessors[p]
                    vec.push_back(connector)

                for connector in d(vec):
                    thread_cost[connector] = penalty * thread_cost[connector]

                reverse(vec.begin(), vec.end())

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

    def write_path_files(RouteChoiceSet self, where):
        """
        Write the path-files to the directory specified

        :Arguments:
            **where** (:obj:`pathlib.Path`): Directory to save the dataset to.
        """
        if self.results is None:
            raise RuntimeError("Route Choice results not computed yet")

        self.results.write(where)
