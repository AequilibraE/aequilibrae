# cython: language_level=3str
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.cython.route_choice_types cimport LinkSet_t, minstd_rand, vector_bool_ptr, shuffle
from aequilibrae.paths.cython.coo_demand cimport GeneralisedCOODemand

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

import itertools
import logging
import pathlib
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


"""This module aims to implemented the BFS-LE algorithm as described in Rieser-Schüssler, Balmer, and Axhausen, 'Route
Choice Sets for Very High-Resolution Data'.  https://doi.org/10.1080/18128602.2012.671383

A rough overview of the algorithm is as follows.  1. Prepare the initial graph, this is depth 0 with no links removed.
    2. Find a short path, P. If P is not empty add P to the path set.  3. For all links p in P, remove p from E,
    compounding with the previously removed links.  4. De-duplicate the sub-graphs, we only care about unique
    sub-graphs.  5. Go to 2.

Details: The general idea of the algorithm is pretty simple, as is the implementation. The caveats here is that there is
a lot of cpp interop and memory management. A description of the purpose of variables is in order:

route_set: See route_choice.pxd for full type signature. It's an unordered set (hash set) of pointers to vectors of link
IDs. It uses a custom hashing function and comparator. The hashing function is defined in a string that in inlined
directly into the output ccp. This is done allow declaring of the `()` operator, which is required and AFAIK not
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

    route_set_dtype = pa.list_(pa.uint32())

    schema = pa.schema([
        pa.field("origin id", pa.uint32(), nullable=False),
        pa.field("destination id", pa.uint32(), nullable=False),
        pa.field("route set", route_set_dtype, nullable=False),
    ])

    psl_schema = pa.schema([
        pa.field("origin id", pa.uint32(), nullable=False),
        pa.field("destination id", pa.uint32(), nullable=False),
        pa.field("route set", route_set_dtype, nullable=False),
        pa.field("cost", pa.float64(), nullable=False),
        pa.field("mask", pa.bool_(), nullable=False),
        pa.field("path overlap", pa.float64(), nullable=False),
        pa.field("probability", pa.float64(), nullable=False),
    ])

    def __cinit__(self):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""
        results = <vector[RouteSet_t *] *>nullptr
        link_union_set = <vector[vector[long long] *] *>nullptr
        cost_set = <vector[vector[double] *] *>nullptr
        mask_set = <vector[vector_bool_ptr] *>nullptr
        path_overlap_set = <vector[vector[double] *] *>nullptr
        prob_set = <vector[vector[double] *] *>nullptr
        ods = <vector[pair[long long, long long]] *>nullptr

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
            schema = self.psl_schema if kwargs.get("path_size_logit", False) else self.schema
            results = pa.dataset.dataset(
                where, format="parquet", partitioning=pa.dataset.HivePartitioning(schema)
            ).to_table()
        else:
            results = self.get_results()
        return [tuple(x) for x in results.column("route set").to_pylist()]

    # Bounds checking doesn't really need to be disabled here but the warning is annoying
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    def batched(
            self,
            demand: GeneralisedCOODemand,
            select_links: Dict[str, FrozenSet[FrozenSet[int]]] = None,
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
            size_t i

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
            unsigned int c_cores = cores if cores > 0 else omp_get_max_threads()

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

            size_t max_results_len, j

        # self.a_star = a_star

        pa.set_io_thread_count(c_cores)

        if self.a_star:
            _reached_first_matrix = np.zeros((c_cores, 1), dtype=np.int64)  # Dummy array to allow slicing
        else:
            _reached_first_matrix = np.zeros((c_cores, self.num_nodes + 1), dtype=np.int64)

        demand._initalise_c_data()

        cdef:
            RouteSet_t *route_set
            shared_ptr[vector[double]] prob_vec
            int thread_id

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
        self.ll_results = LinkLoadingResults(demand, select_links, self.num_links, c_cores)

        with nogil, parallel(num_threads=c_cores):
            route_set = new RouteSet_t()
            thread_id = threadid()
            for i in prange(demand.ods.size()):
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
                        penalty,
                        c_seed,
                    )

                # Here we transform the set of raw pointers to routes (vectors) into a vector of unique points to
                # routes. This is done to simplify memory management later on.
                route_vec = self.results.get_route_vec(i)
                RouteChoiceSetResults.route_set_to_route_vec(d(route_vec), d(route_set))
                # We now drop all references to those raw pointers. The unique pointers now own those vectors.
                route_set.clear()

                if path_size_logit:
                    prob_vec = self.results.compute_result(i, d(route_vec), thread_id)
                    self.ll_results.link_load_single_route_set(i, d(route_vec), d(prob_vec), thread_id)
                    self.ll_results.sl_link_load_single_route_set(i, d(route_vec), d(prob_vec), origin_index, dest_index, thread_id)

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

        self.get_results()
        if path_size_logit:
            self.ll_results.reduce_link_loading()
            self.ll_results.reduce_sl_link_loading()
            self.ll_results.reduce_sl_od_matrix()

            self.get_link_loading(cores=c_cores)
            self.get_sl_link_loading(cores=c_cores)
            self.get_sl_od_matrices()

    @cython.initializedcheck(False)
    cdef void path_find(
        RouteChoiceSet self,
        long origin_index,
        long dest_index,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first
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
            path_finding(
                origin_index,
                dest_index,
                thread_cost,
                thread_b_nodes,
                self.graph_fs_view,
                thread_predecessors,
                self.ids_graph_view,
                thread_conn,
                _thread_reached_first
            )

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
            unsigned int miss_count = 0
            long long p, connector

            # Link penalisation, only used when penalty != 1.0
            bint lp = penalty != 1.0
            vector[double] *penalised_cost = <vector[double] *>nullptr
            vector[double] *next_penalised_cost = <vector[double] *>nullptr

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

            for banned in queue:
                if lp:
                    # We copy the penalised cost buffer into the thread cost buffer to allow us to apply link penalisation,
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
                    _thread_reached_first
                )

                # Mark this set of banned links as seen
                removed_links.insert(banned)

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
                        # its in our route set
                        if removed_links.find(new_banned) != removed_links.end():
                            del new_banned
                        else:
                            next_queue.push_back(new_banned)

                    # The deduplication of routes occurs here
                    status = route_set.insert(vec)
                    miss_count = miss_count + (not status.second)
                    if miss_count > max_misses or route_set.size() >= max_routes:
                        break

            queue.swap(next_queue)
            next_queue.clear()

            if lp:
                # Update the penalised_cost vector, since next_penalised_cost is always the one updated we just need to
                # bring penalised_cost up to date.
                copy(next_penalised_cost.cbegin(), next_penalised_cost.cend(), penalised_cost.begin())

        # We may have added more banned link sets to the queue then found out we hit the max depth, we should free those
        for banned in queue:
            del banned

        # We should also free all the sets in removed_links, we don't be needing them
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
                _thread_reached_first
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
                miss_count = miss_count + (not status.second)
                if miss_count > max_misses:
                    break
            else:
                break

    def get_results(self):  # Cython doesn't like this type annotation... -> pa.Table:
        """
        :Returns:
            **route sets** (:obj:`pyarrow.Table`): Returns a table of OD pairs to lists of link IDs for
                each OD pair provided (as columns). Represents paths from ``origin`` to ``destination``.
        """
        if self.results is None:
            raise RuntimeError("Route Choice results not computed yet")

        return self.results.make_table_from_results()

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


@cython.embedsignature(True)
cdef class Checkpoint:
    """
    A small wrapper class to write a dataset partition by partition
    """

    def __init__(self, where, schema, partition_cols=None):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        self.where = pathlib.Path(where)
        self.schema = schema
        self.partition_cols = partition_cols

    def write(self, table):
        logger = logging.getLogger("aequilibrae")
        pq.write_to_dataset(
            table,
            self.where,
            partitioning=self.partition_cols,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=True,
            existing_data_behavior="overwrite_or_ignore",
            file_visitor=lambda written_file: logger.info(f"Wrote partition dataset at {written_file.path}")
        )

    def read_dataset(self):
        return pa.dataset.dataset(self.where, format="parquet", partitioning=pa.dataset.HivePartitioning(self.schema))

    @staticmethod
    def batches(ods: List[Tuple[int, int]]):
        return (list(g) for k, g in itertools.groupby(sorted(ods), key=lambda x: x[0]))
