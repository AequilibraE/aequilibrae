# cython: language_level=3str

from aequilibrae.paths.graph import Graph
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.matrix.sparse_matrix cimport COO
from scipy.sparse import coo_matrix as sp_coo_matrix

from cython.operator cimport dereference as d
from cython.operator cimport postincrement as inc
from cython.parallel cimport parallel, prange, threadid
from libc.limits cimport UINT_MAX
from libc.math cimport INFINITY, exp, pow, log
from libc.stdlib cimport abort
from libc.string cimport memcpy
from libcpp cimport nullptr
from libcpp.algorithm cimport lower_bound, reverse, sort, copy, min_element
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from openmp cimport omp_get_max_threads

from libc.stdio cimport fprintf, stderr

import random
import itertools
import logging
import pathlib
import warnings
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq
import pandas as pd

cimport numpy as np  # Numpy *must* be cimport'd BEFORE pyarrow.lib, there's nothing quite like Cython.
cimport pyarrow as pa
cimport pyarrow.lib as libpa

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
include 'parallel_numpy.pyx'


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
    def run(self, origin: int, destination: int, demand: float = 0.0, *args, **kwargs):
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
        demand_coo = GeneralisedCOODemand("origin id", "destination id")
        demand_coo.add_df(df)

        self.batched(demand_coo, *args, **kwargs)
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
            eager_link_loading: bool = False,
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
        self.ll_results = LinkLoadingResults(demand, self.num_links, c_cores)


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

                prob_vec = self.results.compute_result(i, d(route_vec), thread_id)

                if eager_link_loading:
                    self.ll_results.link_load_single_route_set(i, d(route_vec), d(prob_vec), thread_id)

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

        if eager_link_loading:
            self.ll_results.reduce_link_loading()

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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.embedsignature(True)
    # @cython.initializedcheck(False)
    # @staticmethod
    # cdef vector[vector[double] *] *compute_path_files(
    #     vector[pair[long long, long long]] &ods,
    #     vector[RouteSet_t *] &results,
    #     vector[vector[long long] *] &link_union_set,
    #     vector[vector[double] *] &prob_set,
    #     unsigned int cores
    # ) noexcept nogil:
    #     """
    #     Computes the path files for the provided vector of RouteSets.

    #     Returns vector of vectors of link loads corresponding to each link in it's link_union_set.
    #     """
    #     cdef:
    #         vector[vector[double] *] *link_loads = new vector[vector[double] *](ods.size())
    #         vector[long long] *link_union
    #         vector[double] *loads
    #         vector[long long] *links

    #         vector[long long].const_iterator link_union_iter
    #         vector[long long].const_iterator link_iter

    #         size_t link_loc
    #         double prob
    #         long long i

    #     with parallel(num_threads=cores):
    #         for i in prange(ods.size()):
    #             link_union = link_union_set[i]
    #             loads = new vector[double](link_union.size(), 0.0)

    #             # We now iterate over all routes in the route_set, each route has an associated probability
    #             route_prob_iter = prob_set[i].cbegin()
    #             for route in d(results[i]):
    #                 prob = d(route_prob_iter)
    #                 inc(route_prob_iter)

    #                 if prob == 0.0:
    #                     continue

    #                 # For each link in the route, we need to assign the appropriate demand * prob Because the link union
    #                 # is known to be sorted, if the links in the route are also sorted we can just step along both
    #                 # arrays simultaneously, skipping elements in the link_union when appropriate. This allows us to
    #                 # operate on the link loads as a sparse map and avoid blowing up memory usage when using a dense
    #                 # formulation. This is also more cache efficient, the only downsides are that the code is
    #                 # harder to read and it requires sorting the route.

    #                 # NOTE: the sorting of routes is technically something that is already computed, during the
    #                 # computation of the link frequency we merge and sort all links, if we instead sorted then used an
    #                 # N-way merge we could reuse the sorted routes and the sorted link union.

    #                 # We copy the links in case the routes haven't already been saved
    #                 links = new vector[long long](d(route))
    #                 sort(links.begin(), links.end())

    #                 # links and link_union are sorted, and links is a subset of link_union
    #                 link_union_iter = link_union.cbegin()
    #                 link_iter = links.cbegin()

    #                 while link_iter != links.cend():
    #                     # Find the next location for the current link in links
    #                     while d(link_iter) != d(link_union_iter) and link_iter != links.cend():
    #                         inc(link_union_iter)

    #                     link_loc = link_union_iter - link_union.cbegin()
    #                     d(loads)[link_loc] = d(loads)[link_loc] + prob  # += here results in all zeros? Odd

    #                     inc(link_iter)

    #                 del links

    #             d(link_loads)[i] = loads

    #     return link_loads

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.embedsignature(True)
    # @cython.initializedcheck(False)
    # cdef vector[double] *apply_link_loading_from_path_files(
    #     RouteChoiceSet self,
    #     double[:, :] matrix_view,
    #     vector[vector[double] *] &path_files
    # ) noexcept nogil:
    #     """
    #     Apply link loading from path files.

    #     Returns a vector of link loads indexed by compressed link ID.
    #     """
    #     cdef:
    #         vector[double] *loads
    #         vector[long long] *link_union
    #         long origin_index, dest_index
    #         double demand

    #         vector[double] *link_loads = new vector[double](self.num_links)

    #     for i in range(self.ods.size()):
    #         loads = path_files[i]
    #         link_union = d(self.link_union_set)[i]

    #         origin_index = self.nodes_to_indices_view[d(self.ods)[i].first]
    #         dest_index = self.nodes_to_indices_view[d(self.ods)[i].second]
    #         demand = matrix_view[origin_index, dest_index]

    #         for j in range(link_union.size()):
    #             link = d(link_union)[j]
    #             d(link_loads)[link] = d(link_loads)[link] + demand * d(loads)[j]

    #     return link_loads


    # @cython.embedsignature(True)
    # def select_link_loading(RouteChoiceSet self, matrix, select_links: Dict[str, List[long]], cores: int = 0):
    #     """
    #     Apply link loading to the network using the demand matrix and the previously computed route sets.
    #     """
    #     if self.ods == nullptr \
    #        or self.link_union_set == nullptr \
    #        or self.prob_set == nullptr:
    #         raise ValueError("select link loading requires Route Choice path_size_logit results")

    #     if not isinstance(matrix, AequilibraeMatrix):
    #         raise ValueError("`matrix` is not an AequilibraE matrix")

    #     cores = cores if cores > 0 else omp_get_max_threads()

    #     cdef:
    #         unordered_set[long] select_link_set
    #         vector[double] *ll

    #     link_loads = {}

    #     for i, name in enumerate(matrix.names):
    #         matrix_ll = {}
    #         m = matrix.matrix_view if len(matrix.view_names) == 1 else matrix.matrix_view[:, :, i]
    #         for (k, v) in select_links.items():
    #             select_link_set = <unordered_set[long]> v

    #             coo = COO((self.zones, self.zones))

    #             ll = self.apply_select_link_loading(coo, m, select_link_set)
    #             res = self.apply_link_loading_func(ll, cores)
    #             del ll

    #             matrix_ll[k] = (coo, res)
    #         link_loads[name] = matrix_ll

    #     return link_loads

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.embedsignature(True)
    # @cython.initializedcheck(False)
    # cdef vector[double] *apply_select_link_loading(
    #     RouteChoiceSet self,
    #     COO sparse_mat,
    #     double[:, :] matrix_view,
    #     unordered_set[long] &select_link_set
    # ) noexcept nogil:
    #     """
    #     Apply select link loading.

    #     Returns a vector of link loads indexed by compressed link ID.
    #     """
    #     cdef:
    #         RouteSet_t *route_set
    #         vector[double] *route_set_prob
    #         vector[double].const_iterator route_prob_iter
    #         long origin_index, dest_index, o, d
    #         double demand, prob, load

    #         vector[double] *link_loads = new vector[double](self.num_links)

    #         bool link_present

    #     # For each OD pair, if a route contains one or more links in a select link set, add that ODs demand to
    #     # a sparse matrix of Os to Ds

    #     # For each route, if it contains one or more links in a select link set, apply the link loading for
    #     # that route

    #     for i in range(self.ods.size()):
    #         route_set = d(self.results)[i]
    #         route_set_prob = d(self.prob_set)[i]

    #         origin_index = self.nodes_to_indices_view[d(self.ods)[i].first]
    #         dest_index = self.nodes_to_indices_view[d(self.ods)[i].second]
    #         demand = matrix_view[origin_index, dest_index]

    #         route_prob_iter = route_set_prob.cbegin()
    #         for route in d(route_set):
    #             prob = d(route_prob_iter)
    #             inc(route_prob_iter)
    #             load = prob * demand

    #             link_present = False
    #             for link in d(route):
    #                 if select_link_set.find(link) != select_link_set.end():
    #                     sparse_mat.append(origin_index, dest_index, load)
    #                     link_present = True
    #                     break

    #             if link_present:
    #                 for link in d(route):
    #                     d(link_loads)[link] = d(link_loads)[link] + load  # += here results in all zeros? Odd

    #     return link_loads

    def get_results(self):  # Cython doesn't like this type annotation... -> pa.Table:
        """
        :Returns:
            **route sets** (:obj:`pyarrow.Table`): Returns a table of OD pairs to lists of link IDs for
                each OD pair provided (as columns). Represents paths from ``origin`` to ``destination``.
        """
        if self.results is None:
            raise RuntimeError("Route Choice results not computed yet")

        return libpa.pyarrow_wrap_table(self.results.make_table_from_results())

    def get_link_loading(RouteChoiceSet self, cores: int = 0):
        """
        :Returns:
            **link loading results** (:obj:`Dict[str, np.array]`): Returns a dict of demand column names to
                uncompressed link loads.
        """
        if self.ll_results is None:
            raise RuntimeError("Link loading results not computed yet")

        return dict(
            zip(
                *self.ll_results.apply_link_loading(
                    self.graph_compressed_id_view,
                    cores if cores > 0 else omp_get_max_threads()
                )
            )
        )


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


@cython.embedsignature(True)
cdef class RouteChoiceSetResults:
    """
    This class is supposed to help manage and compute the results of the route choice set generation. It also
    provides method to perform an assignment and link loading.
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

    def __init__(
            self,
            demand: GeneralisedCOODemand,
            cutoff_prob: float,
            beta: float,
            num_links: int,
            double[:] cost_view,
            unsigned int [:] mapping_idx,
            unsigned int [:] mapping_data,
            store_results: bool = True,
            perform_assignment: bool = True,
    ):
        """

        :Arguments:
            **demand** (`obj`: GeneralisedCOODemand): A GeneralisedCOODemand object stores the ODs pairs and various
              demand values in a COO form. No verification of these is performed here.

            **cutoff_prob** (`obj`: float): The cut-off probability for the inverse binary logit filter.

            **beta** (`obj`: float): The beta parameter for the path-sized logit.

            **store_results** (`obj`: bool): Whether or not to store the route set computation results. At a minimum stores
              the route sets per OD. If `perform_assignment` is True then the assignment results are stored as well.

            **perform_assignment** (`obj`: bool): Whether or not to perform a path-sized logit assignment.

        NOTE: This class makes no attempt to be thread safe when improperly accessed. Multithreaded accesses should be
        coordinated to not collide. Each index of `ods` should only ever be accessed by a single thread.

        NOTE: Depending on `store_results` the behaviour of accessing a single `ods` index multiple times will differ. When
        True the previous internal buffers will be reused. This will highly likely result incorrect results. When False some
        new internal buffers will used, link loading results will still be incorrect. Thus A SINGLE `ods` INDEX SHOULD NOT
        BE ACCESSED MULTIPLE TIMES.
        """

        if not store_results and not perform_assignment:
            raise ValueError("either `store_results` or `perform_assignment` must be True")

        self.demand = demand
        self.cutoff_prob = cutoff_prob
        self.beta = beta
        self.store_results = store_results
        self.perform_assignment = perform_assignment
        self.cost_view = cost_view
        self.mapping_idx = mapping_idx
        self.mapping_data = mapping_data

        cdef size_t size = self.demand.ods.size()

        # As the objects are attribute of the extension class they will be allocated before the object is
        # initialised. This ensures that accessing them is always valid and that they are just empty. We resize the ones
        # we will be using here and allocate the objects they store for the same reasons.
        #
        # We can't know how big they will be so we'll need to resize later as well.
        if self.store_results:
            self.__route_vecs.resize(size)
            for i in range(size):
                self.__route_vecs[i] = make_shared[RouteVec_t]()

        if self.perform_assignment and self.store_results:
            self.__cost_set.resize(size)
            self.__mask_set.resize(size)
            self.__path_overlap_set.resize(size)
            self.__prob_set.resize(size)
            for i in range(size):
                self.__cost_set[i] = make_shared[vector[double]]()
                self.__mask_set[i] = make_shared[vector[bool]]()
                self.__path_overlap_set[i] = make_shared[vector[double]]()
                self.__prob_set[i] = make_shared[vector[double]]()

    @staticmethod
    cdef void route_set_to_route_vec(RouteVec_t &route_vec, RouteSet_t &route_set) noexcept nogil:
        """
        Transform a set of raw pointers to routes (vectors) into a vector of unique points to
        routes.
        """
        cdef vector[long long] *route

        route_vec.reserve(route_set.size())
        for route in route_set:
            route_vec.emplace_back(route)

    cdef shared_ptr[RouteVec_t] get_route_vec(RouteChoiceSetResults self, size_t i) noexcept nogil:
        """
        Return either a new empty RouteSet_t, or the RouteSet_t (initially empty) corresponding to a OD pair index.

        If `self.store_results` is False no attempt is made to store the route set. The caller is responsible for maintaining
        a reference to it.

        Requires that 0 <= i < self.ods.size().
        """
        if self.store_results:
            # All elements of self.__route_vecs have been initialised in self.__init__.
            return self.__route_vecs[i]
        else:
            # We make a new empty RouteSet_t here, we don't attempt to store it.
            return make_shared[RouteVec_t]()

    cdef shared_ptr[vector[double]] __get_cost_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__cost_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[bool]] __get_mask_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__mask_set[i] if self.store_results else make_shared[vector[bool]]()

    cdef shared_ptr[vector[double]] __get_path_overlap_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__path_overlap_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[double]] __get_prob_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__prob_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[double]] compute_result(RouteChoiceSetResults self, size_t i, RouteVec_t &route_set, size_t thread_id) noexcept nogil:
        """
        Compute the desired results for the OD pair index with the provided route set. The route set is required as
        an argument here to facilitate not storing them. The route set should correspond to the provided OD pair index,
        however that is not enforced.

        Requires that 0 <= i < self.ods.size().

        Returns a shared pointer to the probability vector.
        """
        cdef:
            shared_ptr[vector[double]] cost_vec
            shared_ptr[vector[bool]] route_mask
            vector[long long] keys, counts
            shared_ptr[vector[double]] path_overlap_vec
            shared_ptr[vector[double]] prob_vec

        if not self.perform_assignment:
            # If we're not performing an assignment then we must be storing the routes and the routes most already be
            # stored when they were acquired, thus we don't need to do anything here.
            return shared_ptr[vector[double]]()

        cost_vec = self.__get_cost_set(i)
        route_mask = self.__get_mask_set(i)
        path_overlap_vec = self.__get_path_overlap_set(i)
        prob_vec = self.__get_prob_set(i)

        self.compute_cost(d(cost_vec), route_set, self.cost_view)
        if self.compute_mask(d(route_mask), d(cost_vec)):
            with gil:
                warnings.warn(
                    f"Zero cost route found for ({self.demand.ods[i].first}, {self.demand.ods[i].second}). "
                    "Entire route set masked"
                )
        self.compute_frequency(keys, counts, route_set, d(route_mask))
        self.compute_path_overlap(d(path_overlap_vec), route_set, keys, counts, d(cost_vec), d(route_mask), self.cost_view)
        self.compute_prob(d(prob_vec), d(cost_vec), d(path_overlap_vec), d(route_mask))

        return prob_vec

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_cost(RouteChoiceSetResults self, vector[double] &cost_vec, const RouteVec_t &route_set, const double[:] cost_view) noexcept nogil:
        """Compute the cost each route."""
        cdef:
            # Scratch objects
            double cost
            long long link
            size_t i

        cost_vec.resize(route_set.size())

        for i in range(route_set.size()):
            cost = 0.0
            for link in d(route_set[i]):
                cost = cost + cost_view[link]

            cost_vec[i] = cost

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef bool compute_mask(RouteChoiceSetResults self, vector[bool] &route_mask, const vector[double] &total_cost) noexcept nogil:
        """
        Computes a binary logit between the minimum cost path and each path, if the total cost is greater than the
        minimum + the difference in utilities required to produce the cut-off probability then the route is excluded from
        the route set.
        """
        cdef:
            bool found_zero_cost = False
            size_t i

            vector[double].const_iterator min = min_element(total_cost.cbegin(), total_cost.cend())
            double cutoff_cost = d(min) \
                + inverse_binary_logit(self.cutoff_prob, 0.0, 1.0)

        route_mask.resize(total_cost.size())

        # The route mask should be True for the routes we wish to include.
        for i in range(total_cost.size()):
            if total_cost[i] == 0.0:
                found_zero_cost = True
                break
            elif total_cost[i] <= cutoff_cost:
               route_mask[i] = True

        if found_zero_cost:
            # If we've found a zero cost path we must abandon the whole route set.
            for i in range(total_cost.size()):
                route_mask[i] = False
        elif min != total_cost.cend():
            # Always include the min element. It should already be but I don't trust floating math to do this correctly.
            # But only if there actually was a min element (i.e. empty route set)
            route_mask[min - total_cost.cbegin()] = True

        return found_zero_cost

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_frequency(
        RouteChoiceSetResults self,
        vector[long long] &keys,
        vector[long long] &counts,
        const RouteVec_t &route_set,
        const vector[bool] &route_mask
    ) noexcept nogil:
        """
        Compute a frequency map for each route with the route_mask applied.

        Each node at index i in the first returned vector has frequency at index i in the second vector.
        """
        cdef:
            vector[long long] link_union
            vector[long long].const_iterator union_iter

            # Scratch objects
            size_t length, count, i
            long long link

        # When calculating the frequency of routes, we need to exclude those not in the mask.
        length = 0
        for i in range(route_set.size()):
            # We do so here ...
            if not route_mask[i]:
                continue

            length = length + d(route_set[i]).size()
        link_union.reserve(length)

        for i in range(route_set.size()):
            # ... and here.
            if not route_mask[i]:
                continue

            link_union.insert(link_union.end(), d(route_set[i]).begin(), d(route_set[i]).end())

        sort(link_union.begin(), link_union.end())

        union_iter = link_union.cbegin()
        while union_iter != link_union.cend():
            count = 0
            link = d(union_iter)
            while link == d(union_iter) and union_iter != link_union.cend():
                count = count + 1
                inc(union_iter)

            keys.push_back(link)
            counts.push_back(count)

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_path_overlap(
        RouteChoiceSetResults self,
        vector[double] &path_overlap_vec,
        const RouteVec_t &route_set,
        const vector[long long] &keys,
        const vector[long long] &counts,
        const vector[double] &total_cost,
        const vector[bool] &route_mask,
        const double[:] cost_view
    ) noexcept nogil:
        """
        Compute the path overlap figure based on the route cost and frequency.

        Notation changes:
            a: link
            t_a: cost_view
            c_i: total_costs
            A_i: route
            sum_{k in R}: delta_{a,k}: freq_set
        """
        cdef:
            # Scratch objects
            vector[long long].const_iterator link_iter
            double path_overlap
            long long link
            size_t i

        path_overlap_vec.resize(route_set.size())

        for i in range(route_set.size()):
            # Skip masked routes
            if not route_mask[i]:
                continue

            path_overlap = 0.0
            for link in d(route_set[i]):
                # We know the frequency table is ordered and contains every link in the union of the routes.
                # We want to find the index of the link, and use that to look up it's frequency
                link_iter = lower_bound(keys.cbegin(), keys.cend(), link)

                # lower_bound returns keys.end() when no link is found.
                # This /should/ never happen.
                if link_iter == keys.cend():
                    continue
                path_overlap = path_overlap + cost_view[link] / counts[link_iter - keys.cbegin()]

            path_overlap_vec[i] = path_overlap / total_cost[i]

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_prob(
        RouteChoiceSetResults self,
        vector[double] &prob_vec,
        const vector[double] &total_cost,
        const vector[double] &path_overlap_vec,
        const vector[bool] &route_mask
    ) noexcept nogil:
        """Compute a probability for each route in the route set based on the path overlap."""
        cdef:
            # Scratch objects
            double inv_prob
            size_t i, j

        prob_vec.resize(total_cost.size())

        # Beware when refactoring the below, the scale of the costs may cause floating point errors. Large costs will
        # lead to NaN results
        for i in range(total_cost.size()):
            # The probability of choosing a route that has been masked out is 0.
            if not route_mask[i]:
                continue

            inv_prob = 0.0
            for j in range(total_cost.size()):
                # We must skip any other routes that are not included in the mask otherwise our probabilities won't
                # add up.
                if not route_mask[j]:
                    continue

                inv_prob = inv_prob + pow(path_overlap_vec[j] / path_overlap_vec[i], self.beta) \
                    * exp((total_cost[i] - total_cost[j]))  # Assuming theta=1.0

            prob_vec[i] = 1.0 / inv_prob

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef shared_ptr[libpa.CTable] make_table_from_results(RouteChoiceSetResults self):
        """
        Construct an Arrow table from C++ stdlib structures.

        Note: this function directly utilises the Arrow C++ API, the Arrow Cython API is not sufficient.
        See `route_choice_set.pxd` for Cython declarations.

        Returns a shared pointer to a Arrow CTable. This should be wrapped in a Python table before use.
        Compressed link IDs are expanded to full network link IDs.
        """

        if not self.store_results:
            raise RuntimeError("route set table construction requires `store_results` is True")

        cdef:
            shared_ptr[libpa.CArray] paths
            shared_ptr[libpa.CArray] offsets

            libpa.CMemoryPool *pool = libpa.c_get_memory_pool()

            # Custom imports, these are declared in route_choice.pxd *not* libarrow.  We have to use new here because
            # Cython doesn't support passing arguments to the default constructor as it implicitly constructs them and
            # Pyarrow only exposes the single constructor in Cython.
            CUInt32Builder *path_builder = new CUInt32Builder(pool)
            CDoubleBuilder *cost_col = <CDoubleBuilder *>nullptr
            CBooleanBuilder *mask_col = <CBooleanBuilder *>nullptr
            CDoubleBuilder *path_overlap_col = <CDoubleBuilder *>nullptr
            CDoubleBuilder *prob_col = <CDoubleBuilder *>nullptr

            libpa.CInt32Builder *offset_builder = new libpa.CInt32Builder(pool)  # Must be Int32 *not* UInt32
            libpa.CUInt32Builder *o_col = new libpa.CUInt32Builder(pool)
            libpa.CUInt32Builder *d_col = new libpa.CUInt32Builder(pool)
            vector[shared_ptr[libpa.CArray]] columns
            shared_ptr[libpa.CDataType] route_set_dtype = libpa.pyarrow_unwrap_data_type(self.route_set_dtype)

            libpa.CResult[shared_ptr[libpa.CArray]] route_set_results

            int offset = 0
            size_t network_link_begin, network_link_end, link

        # Origins, Destination, Route set, [Cost for route, Mask, Path_Overlap for route, Probability for route]
        columns.resize(len(self.psl_schema) if self.perform_assignment else len(self.schema))

        if self.perform_assignment:
            cost_col = new CDoubleBuilder(pool)
            mask_col = new CBooleanBuilder(pool)
            path_overlap_col = new CDoubleBuilder(pool)
            prob_col = new CDoubleBuilder(pool)

            for i in range(self.demand.ods.size()):
                cost_col.AppendValues(d(self.__cost_set[i]))
                mask_col.AppendValues(d(self.__mask_set[i]))
                path_overlap_col.AppendValues(d(self.__path_overlap_set[i]))
                prob_col.AppendValues(d(self.__prob_set[i]))

        for i in range(self.demand.ods.size()):
            route_set = self.__route_vecs[i]

            # Instead of constructing a "list of lists" style object for storing the route sets we instead will
            # construct one big array of link IDs with a corresponding offsets array that indicates where each new row
            # (path) starts.
            for j in range(d(route_set).size()):
                o_col.Append(self.demand.ods[i].first)
                d_col.Append(self.demand.ods[i].second)

                offset_builder.Append(offset)

                for link in d(d(route_set)[j]):
                    # Translate the compressed link IDs in route to network link IDs, this is a 1:n mapping
                    network_link_begin = self.mapping_idx[link]
                    network_link_end = self.mapping_idx[link + 1]
                    path_builder.AppendValues(
                        &self.mapping_data[network_link_begin],
                        network_link_end - network_link_begin
                    )

                    offset += network_link_end - network_link_begin

        path_builder.Finish(&paths)

        offset_builder.Append(offset)  # Mark the end of the array in offsets
        offset_builder.Finish(&offsets)

        route_set_results = libpa.CListArray.FromArraysAndType(
            route_set_dtype,
            d(offsets.get()),
            d(paths.get()),
            pool,
            shared_ptr[libpa.CBuffer]()
        )

        o_col.Finish(&columns[0])
        d_col.Finish(&columns[1])
        columns[2] = d(route_set_results)

        if self.perform_assignment:
            cost_col.Finish(&columns[3])
            mask_col.Finish(&columns[4])
            path_overlap_col.Finish(&columns[5])
            prob_col.Finish(&columns[6])

        cdef shared_ptr[libpa.CSchema] schema = libpa.pyarrow_unwrap_schema(
            self.psl_schema if self.perform_assignment else self.schema
        )
        cdef shared_ptr[libpa.CTable] table = libpa.CTable.MakeFromArrays(schema, columns)

        del path_builder
        del offset_builder
        del o_col
        del d_col

        if self.perform_assignment:
            del cost_col
            del mask_col
            del path_overlap_col
            del prob_col

        return table


cdef class GeneralisedCOODemand:
    def __init__(self, origin_col: str, destination_col: str):
        """
        A container for access to the float64 and float32 fields of a data frame.
        """
        self.df = pd.DataFrame(columns=[origin_col, destination_col]).set_index([origin_col, destination_col])

    def add_df(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]], demand_cols=None, fill: float = 0.0):
        """
        Add a DataFrame to the existing ones.

        Expects a DataFrame with a multi-index of (o, d).

        If no demand cols are provided, all floating point columns are used.
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = (dfs,)

        new_dfs = [self.df]
        for df in dfs:
            if df.index.nlevels != 2:
                raise ValueError("provided pd.DataFrame doesn't have a 2-level multi-index")
            elif df.index.names != self.df.index.names:
                raise ValueError(f"mismatched index names. Expect {self.df.index.names}, provided {df.index.names}")

            if demand_cols is None:
                new_dfs.append(df.select_dtypes(["float64", "float32"]))
            else:
                for col in demand_cols:
                    if not (df.dtypes[col] == "float64" or df.dtypes[col] == "float32"):
                        raise TypeError(f"demand column ({col}) is not a float64 or float32")
                new_dfs.append(df[demand_cols])

        self.df = pd.concat(new_dfs, axis=1).fillna(fill)

    def add_matrix(self, matrix: AequilibraeMatrix, fill: float = 0.0):
        """
        Add an AequilibraE matrix to the existing demand in a sparse manner.
        """
        dfs = []
        for i, name in enumerate(matrix.view_names):
            m = matrix.matrix_view if len(matrix.view_names) == 1 else matrix.matrix_view[:, :, i]
            if np.nansum(m) == 0:
                continue

            coo_ = sp_coo_matrix(m)
            df = pd.DataFrame(
                data={
                    self.df.index.names[0]: matrix.index[coo_.row],
                    self.df.index.names[1]: matrix.index[coo_.col],
                    matrix.names[i]: coo_.data,
                },
            ).set_index([self.df.index.names[0], self.df.index.names[1]])
            dfs.append(df.dropna())

        self.add_df(dfs, fill=fill)
        logging.info(f"There {len(self.df):,} are OD pairs with non-zero flows")

    def _initalise_c_data(self):
        self.ods = self.df.index  # MultiIndex[int, int] -> vector[pair[long long, long long]]

        self.f64.clear()
        self.f32.clear()
        self.f64_names = []
        self.f32_names = []

        cdef:
            double[::1] f64_array
            float[::1] f32_array
            vector[double] *f64_vec
            vector[float] *f32_vec

        for col in self.df:
            if self.df.dtypes[col] == "float64":
                f64_array = self.df[col].to_numpy()
                self.f64_names.append(col)

                # The unique pointer will take ownership of this allocation
                f64_vec = new vector[double]()
                f64_vec.insert(f64_vec.begin(), &f64_array[0], &f64_array[0] + len(f64_array))
                self.f64.emplace_back(f64_vec)  # From here f63_vec should not be accessed. It is owned by the unique pointer

            elif self.df.dtypes[col] == "float32":
                f32_array = self.df[col].to_numpy()
                self.f32_names.append(col)

                # The unique pointer will take ownership of this allocation
                f32_vec = new vector[float]()
                f32_vec.insert(f32_vec.begin(), &f32_array[0], &f32_array[0] + len(f32_array))
                self.f32.emplace_back(f32_vec)  # From here f32_vec should not be accessed. It is owned by the unique pointer
            else:
                raise TypeError(f"non-floating point column ({col}) in self.df. Something has gone wrong")

    def no_demand(GeneralisedCOODemand self) -> bool:
        return len(self.df.columns) == 0

    def is_empty(self) -> bool:
        return self.df.index.empty

    cpdef _hello(GeneralisedCOODemand self):
        cdef size_t i

        print(self.df)

        print("ods:", self.ods.size(), self.ods)

        for i in range(self.f64.size()):
            array = np.asarray(d(self.f64[i]))
            print("f64:", d(self.f64[i]).size(), array.dtype, array)

        for i in range(self.f32.size()):
            array = np.asarray(d(self.f32[i]), dtype=np.float32)
            print("f32:", d(self.f32[i]).size(), array.dtype, array)

# See note in route_choice_set.pxd
cdef class LinkLoadingResults:
    def __cinit__(self, demand: GeneralisedCOODemand, num_links: int, threads: int):
        if threads <= 0:
            raise ValueError(f"threads must be positive ({threads})")
        elif num_links <= 0:
            raise ValueError(f"num_links must be positive ({num_links})")

        self.demand = demand
        self.num_links = num_links

        cdef:
            vector[unique_ptr[vector[double]]] *f64_demand_cols
            vector[unique_ptr[vector[float]]] *f32_demand_cols

        # Allocate the threaded f64 link loading.
        self.f64_link_loading_threaded.reserve(threads)
        for i in range(threads):
            f64_demand_cols = new vector[unique_ptr[vector[double]]]()
            f64_demand_cols.reserve(self.demand.f64.size())

            for j in range(self.demand.f64.size()):
                f64_demand_cols.emplace_back(new vector[double](self.num_links))

            self.f64_link_loading_threaded.emplace_back(f64_demand_cols)

        # Allocate the threaded f32 link loading.
        self.f32_link_loading_threaded.reserve(threads)
        for i in range(threads):
            f32_demand_cols = new vector[unique_ptr[vector[float]]]()
            f32_demand_cols.reserve(self.demand.f32.size())

            for j in range(self.demand.f32.size()):
                f32_demand_cols.emplace_back(new vector[float](self.num_links))

            self.f32_link_loading_threaded.emplace_back(f32_demand_cols)

        # self.f64_link_loading and self.f32_link_loading are not allocated here. The objects are initialised to empty
        # vectors but elements are created in self.reduce_link_loading

    def _sizes(self):
        print("-" * 10, "threaded", "-" * 10)
        print("f64 threads:", self.f64_link_loading_threaded.size())
        for i in range(self.f64_link_loading_threaded.size()):
            print("  f64 cols:", d(self.f64_link_loading_threaded[i]).size())
            for j in range(d(self.f64_link_loading_threaded[i]).size()):
                print("    f64 links:", d(d(self.f64_link_loading_threaded[i])[j]).size())

        print("f32 threads:", self.f32_link_loading_threaded.size())
        for i in range(self.f32_link_loading_threaded.size()):
            print("  f32 cols:", d(self.f32_link_loading_threaded[i]).size())
            for j in range(d(self.f32_link_loading_threaded[i]).size()):
                print("    f32 links:", d(d(self.f32_link_loading_threaded[i])[j]).size())

        print("-" * 10, "results", "-" * 10)
        print("f64 results:", self.f64_link_loading.size())
        for j in range(self.f64_link_loading.size()):
            print("  f64 cols:", d(self.f64_link_loading[j]).size())

        print("f32 results:", self.f32_link_loading.size())
        for j in range(self.f32_link_loading.size()):
            print("  f32 cols:", d(self.f32_link_loading[j]).size())

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void link_load_single_route_set(
        LinkLoadingResults self,
        const size_t od_idx,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        const size_t thread_id
    ) noexcept nogil:
        cdef:
            vector[double].const_iterator route_prob_iter

            # Cython doesn't allow declaring references to objects outside of function signatures. So we get the raw
            # pointer instead. It is still owned by the unique_ptr.
            vector[unique_ptr[vector[double]]] *f64_ll_cols = self.f64_link_loading_threaded[thread_id].get()
            vector[unique_ptr[vector[float]]] *f32_ll_cols = self.f32_link_loading_threaded[thread_id].get()
            vector[double] *f64_ll
            vector[float] *f32_ll

            double f64_demand, f64_load
            float f32_demand, f32_load
            size_t i, j, k

        # For each demand column
        for i in range(self.demand.f64.size()):
            # we grab a pointer to the relevant link loading vector and demand value.
            f64_demand = d(self.demand.f64[i])[od_idx]
            f64_ll = d(f64_ll_cols)[i].get()

            # For each route in the route set
            route_prob_iter = prob_vec.cbegin()
            for j in range(route_set.size()):
                # we compute out load,
                f64_load = d(route_prob_iter) * f64_demand
                inc(route_prob_iter)

                # then apply that to every link in the route
                for link in d(route_set[j]):
                    d(f64_ll)[link] = d(f64_ll)[link] + f64_load

        # Then we do it all over again for the floats.
        for i in range(self.demand.f32.size()):
            f32_demand = d(self.demand.f32[i])[od_idx]
            f32_ll = d(f32_ll_cols)[i].get()

            route_prob_iter = prob_vec.cbegin()
            for j in range(route_set.size()):
                f32_load = d(route_prob_iter) * f32_demand
                inc(route_prob_iter)

                for link in d(route_set[j]):
                    d(f32_ll)[link] = d(f32_ll)[link] + f32_load

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void reduce_link_loading(LinkLoadingResults self):
        """
        NOTE: doesn't require the GIL but should NOT be called in a multithreaded environment. Thus the function requires the GIL.
        """
        cdef:
            vector[unique_ptr[vector[double]]] *f64_ll_cols
            vector[unique_ptr[vector[float]]] *f32_ll_cols
            vector[double] *f64_ll_result
            vector[float] *f32_ll_result
            vector[double] *f64_ll
            vector[float] *f32_ll

            size_t thread_id, i, j

        # Allocate the result link loads
        self.f64_link_loading.reserve(self.demand.f64.size())
        for j in range(self.demand.f64.size()):
            self.f64_link_loading.emplace_back(new vector[double](self.num_links))

        self.f32_link_loading.reserve(self.demand.f32.size())
        for j in range(self.demand.f32.size()):
            self.f32_link_loading.emplace_back(new vector[float](self.num_links))

        # Here we sum all threads link loads into the results.
        # For each thread
        for thread_id in range(self.f64_link_loading_threaded.size()):
            # we grab the columns
            f64_ll_cols = self.f64_link_loading_threaded[thread_id].get()

            # for each column
            for i in range(d(f64_ll_cols).size()):
                # we get the link loading vector
                f64_ll_result = self.f64_link_loading[i].get()
                f64_ll = d(f64_ll_cols)[i].get()

                # for each link
                for j in range(d(f64_ll).size()):
                    d(f64_ll_result)[j] = d(f64_ll_result)[j] + d(f64_ll)[j]

        # Then we do it all over again for the floats.
        for thread_id in range(self.f32_link_loading_threaded.size()):
            f32_ll_cols = self.f32_link_loading_threaded[thread_id].get()

            for i in range(d(f32_ll_cols).size()):
                f32_ll_result = self.f32_link_loading[i].get()
                f32_ll = d(f32_ll_cols)[i].get()

                for j in range(d(f32_ll).size()):
                    d(f32_ll_result)[j] = d(f32_ll_result)[j] + d(f32_ll)[j]

        # Here we discard all the intermediate results
        self.f64_link_loading_threaded.clear()
        self.f32_link_loading_threaded.clear()
        self.f64_link_loading_threaded.shrink_to_fit()
        self.f32_link_loading_threaded.shrink_to_fit()

    cdef object apply_link_loading(LinkLoadingResults self, long long[:] compressed_id_view, int cores):
        cdef:
            vector[double] *f64_ll
            vector[float] *f32_ll
            double[:, :] f64_ll_view
            double[:, :] f64_actual
            float[:, :] f32_ll_view
            float[:, :] f32_actual
            size_t length

        f64_ll_vectors = []
        for i in range(self.f64_link_loading.size()):
            # Push a single element to the back to prevent an issue caused by links that are not present in the
            # compressed graph causing a OOB access
            f64_ll = self.f64_link_loading[i].get()
            f64_ll.push_back(0.0)

            # Cast the vector to a memory view
            f64_ll_view = <double [:f64_ll.size(), :1]> f64_ll.data()
            f64_actual = np.zeros((compressed_id_view.shape[0], 1), dtype=np.float64)

            # Assign the compressed link loads to the uncompressed graph
            assign_link_loads_cython(f64_actual, f64_ll_view, compressed_id_view, cores)

            # Delete the memory view object and pop that element of the end.
            del f64_ll_view
            f64_ll.pop_back()

            # Just return the actual link loads. If compressed are required then the memory view will have to be copied
            # before it is deleted. It does not own the vectors data.
            f64_ll_vectors.append(np.asarray(f64_actual).reshape(-1))

        # We do the same for the floats, again
        f32_ll_vectors = []
        for i in range(self.f32_link_loading.size()):
            f32_ll = self.f32_link_loading[i].get()
            f32_ll.push_back(0.0)

            f32_ll_view = <float [:f32_ll.size(), :1]>f32_ll.data()
            f32_actual = np.zeros((compressed_id_view.shape[0], 1), dtype=np.float32)

            assign_link_loads_cython(f32_actual, f32_ll_view, compressed_id_view, cores)

            del f32_ll_view
            f32_ll.pop_back()

            f32_ll_vectors.append(np.asarray(f32_actual).reshape(-1))

        return itertools.chain(self.demand.f64_names, self.demand.f32_names), itertools.chain(f64_ll_vectors, f32_ll_vectors)



cdef double inverse_binary_logit(double prob, double beta0, double beta1) noexcept nogil:
    if prob == 1.0:
        return INFINITY
    elif prob == 0.0:
        return -INFINITY
    else:
        return (log(prob / (1.0 - prob)) - beta0) / beta1
