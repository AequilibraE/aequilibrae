# cython: language_level=3str

from aequilibrae.paths.graph import Graph
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.matrix.sparse_matrix cimport COO

from cython.operator cimport dereference as d
from cython.operator cimport postincrement as inc
from cython.operator cimport predecrement as pre_dec
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

from libc.stdio cimport printf, fprintf, stderr

import random
import itertools
import logging
import pathlib
import warnings
from typing import List, Tuple, FrozenSet

import numpy as np
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq

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

    def __dealloc__(self):
        """
        C level deallocation. For freeing memory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already. Do not call any other Python method.
        """
        self.deallocate()

    cdef void deallocate(RouteChoiceSet self) nogil:
        """__dealloc__ cannot be called from normal code."""
        cdef:
            RouteSet_t *route_set
            vector[long long] *link_vec
            vector[double] *double_vec
            vector[bool] *bool_vec

        if self.results != nullptr:
            for route_set in d(self.results):
                for link_vec in d(route_set):
                    del link_vec
                del route_set
            del self.results
            self.results = <vector[RouteSet_t *] *>nullptr

        if self.link_union_set != nullptr:
            for link_vec in d(self.link_union_set):
                del link_vec
            del self.link_union_set
            self.link_union_set = <vector[vector[long long] *] *>nullptr

        if self.cost_set != nullptr:
            for double_vec in d(self.cost_set):
                del double_vec
            del self.cost_set
            self.cost_set = <vector[vector[double] *] *>nullptr

        if self.mask_set != nullptr:
            for bool_vec in d(self.mask_set):
                del bool_vec
            del self.mask_set
            self.mask_set = <vector[vector_bool_ptr] *>nullptr

        if self.path_overlap_set != nullptr:
            for double_vec in d(self.path_overlap_set):
                del double_vec
            del self.path_overlap_set
            self.path_overlap_set = <vector[vector[double] *] *>nullptr

        if self.prob_set != nullptr:
            for double_vec in d(self.prob_set):
                del double_vec
            del self.prob_set
            self.prob_set = <vector[vector[double] *] *>nullptr

        if self.ods != nullptr:
            del self.ods
            self.ods = prob_set = <vector[pair[long long, long long]] *>nullptr

    @cython.embedsignature(True)
    def run(self, origin: int, destination: int, *args, **kwargs):
        """Compute the a route set for a single OD pair.

        Often the returned list's length is ``max_routes``, however, it may be limited by ``max_depth`` or if all
        unique possible paths have been found then a smaller set will be returned.

        Additional arguments are forwarded to ``RouteChoiceSet.batched``.

        :Arguments:
            **origin** (:obj:`int`): Origin node ID. Must be present within compact graph. Recommended to choose a
                centroid.
            **destination** (:obj:`int`): Destination node ID. Must be present within compact graph. Recommended to
                choose a centroid.

        :Returns: **route set** (:obj:`list[tuple[int, ...]]): Returns a list of unique variable length tuples of
            link IDs. Represents paths from ``origin`` to ``destination``.
        """
        self.batched([(origin, destination)], *args, **kwargs)
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
            ods: List[Tuple[int, int]],
            max_routes: int = 0,
            max_depth: int = 0,
            max_misses: int = 100,
            seed: int = 0,
            cores: int = 0,
            a_star: bool = True,
            bfsle: bool = True,
            penalty: float = 1.0,
            where: Optional[str] = None,
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

        if max_routes == 0 and max_depth == 0:
            raise ValueError("Either `max_routes` or `max_depth` must be > 0")

        if max_routes < 0 or max_depth < 0:
            raise ValueError("`max_routes`, `max_depth`, and `cores` must be non-negative")

        if path_size_logit and beta < 0:
            raise ValueError("`beta` must be >= 0 for path sized logit model")

        if path_size_logit and not 0.0 <= cutoff_prob <= 1.0:
            raise ValueError("`cutoff_prob` must be 0 <= `cutoff_prob` <= 1 for path sized logit model")

        for origin, dest in ods:
            if self.nodes_to_indices_view[origin] == -1:
                raise ValueError(f"Origin {origin} is not present within the compact graph")
            if self.nodes_to_indices_view[dest] == -1:
                raise ValueError(f"Destination {dest} is not present within the compact graph")

        cdef:
            long long origin_index, dest_index, i
            unsigned int c_max_routes = max_routes
            unsigned int c_max_depth = max_depth
            unsigned int c_max_misses = max_misses
            unsigned int c_seed = seed
            unsigned int c_cores = cores if cores > 0 else omp_get_max_threads()

            # Scale cutoff prob from [0, 1] -> [0.5, 1]. Values below 0.5 produce negative inverse binary logit values.
            double scaled_cutoff_prob = (1.0 - cutoff_prob) * 0.5 + 0.5

            vector[pair[long long, long long]] c_ods

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

            vector[RouteSet_t *] *results
            size_t max_results_len, batch_len, j

        # self.a_star = a_star

        pa.set_io_thread_count(c_cores)

        if self.a_star:
            _reached_first_matrix = np.zeros((c_cores, 1), dtype=np.int64)  # Dummy array to allow slicing
        else:
            _reached_first_matrix = np.zeros((c_cores, self.num_nodes + 1), dtype=np.int64)

        # Shuffling the jobs improves load balancing where nodes pairs are geographically ordered
        set_ods = list(set(ods))
        if len(set_ods) != len(ods):
            warnings.warn(f"Duplicate OD pairs found, dropping {len(ods) - len(set_ods)} OD pairs")

        if where is not None:
            checkpoint = Checkpoint(
                where,
                self.psl_schema if path_size_logit else self.schema, partition_cols=["origin id"]
            )
            batches = list(Checkpoint.batches(set_ods))
            max_results_len = <size_t>max(len(batch) for batch in batches)
        else:
            random.shuffle(set_ods)
            batches = [set_ods]
            max_results_len = len(set_ods)

        results = new vector[RouteSet_t *](max_results_len)

        cdef:
            RouteSet_t *route_set
            pair[vector[long long] *, vector[long long] *] freq_pair
            vector[vector[long long] *] *link_union_set = <vector[vector[long long] *] *>nullptr
            vector[vector[double] *] *cost_set = <vector[vector[double] *] *>nullptr
            vector[vector_bool_ptr] *mask_set = <vector[vector_bool_ptr] *>nullptr
            vector[vector[double] *] *path_overlap_set = <vector[vector[double] *] *>nullptr
            vector[vector[double] *] *prob_set = <vector[vector[double] *] *>nullptr

        if path_size_logit:
            link_union_set = new vector[vector[long long] *](max_results_len)
            cost_set = new vector[vector[double] *](max_results_len)
            mask_set = new vector[vector_bool_ptr](max_results_len)
            path_overlap_set = new vector[vector[double] *](max_results_len)
            prob_set = new vector[vector[double] *](max_results_len)

        self.deallocate()  # We may be storing results from a previous run

        for batch in batches:
            c_ods = batch  # Convert the batch to a C++ vector, this isn't strictly efficient but is nicer
            batch_len = c_ods.size()
            # We know we've allocated enough size to store all max length batch but we resize to a smaller size when not
            # needed
            results.resize(batch_len)

            if path_size_logit:
                # We may clear these objects because it's either:
                # - the first iteration and they contain no elements, thus no memory to leak
                # - the internal objects were freed by the previous iteration
                link_union_set.clear()
                cost_set.clear()
                mask_set.clear()
                path_overlap_set.clear()
                prob_set.clear()

                link_union_set.resize(batch_len)
                cost_set.resize(batch_len)
                mask_set.resize(batch_len)
                path_overlap_set.resize(batch_len)
                prob_set.resize(batch_len)

            with nogil, parallel(num_threads=c_cores):
                for i in prange(batch_len, schedule= "dynamic", chunksize=1):
                    origin_index = self.nodes_to_indices_view[c_ods[i].first]
                    dest_index = self.nodes_to_indices_view[c_ods[i].second]

                    if self.block_flows_through_centroids:
                        blocking_centroid_flows(
                            0,  # Always blocking
                            origin_index,
                            self.zones,
                            self.graph_fs_view,
                            b_nodes_matrix[threadid()],
                            self.b_nodes_view,
                        )

                    if bfsle:
                        route_set = RouteChoiceSet.bfsle(
                            self,
                            origin_index,
                            dest_index,
                            c_max_routes,
                            c_max_depth,
                            c_max_misses,
                            cost_matrix[threadid()],
                            predecessors_matrix[threadid()],
                            conn_matrix[threadid()],
                            b_nodes_matrix[threadid()],
                            _reached_first_matrix[threadid()],
                            penalty,
                            c_seed,
                        )
                    else:
                        route_set = RouteChoiceSet.link_penalisation(
                            self,
                            origin_index,
                            dest_index,
                            c_max_routes,
                            c_max_depth,
                            c_max_misses,
                            cost_matrix[threadid()],
                            predecessors_matrix[threadid()],
                            conn_matrix[threadid()],
                            b_nodes_matrix[threadid()],
                            _reached_first_matrix[threadid()],
                            penalty,
                            c_seed,
                        )

                    if path_size_logit:
                        d(cost_set)[i] = RouteChoiceSet.compute_cost(route_set, self.cost_view)
                        d(mask_set)[i] = RouteChoiceSet.compute_mask(scaled_cutoff_prob, d(d(cost_set)[i]))

                        freq_pair = RouteChoiceSet.compute_frequency(route_set, d(d(mask_set)[i]))
                        d(link_union_set)[i] = freq_pair.first
                        d(path_overlap_set)[i] = RouteChoiceSet.compute_path_overlap(
                            route_set,
                            freq_pair,
                            d(d(cost_set)[i]),
                            d(d(mask_set)[i]),
                            self.cost_view
                        )
                        d(prob_set)[i] = RouteChoiceSet.compute_prob(
                            d(d(cost_set)[i]),
                            d(d(path_overlap_set)[i]),
                            d(d(mask_set)[i]),
                            beta
                        )
                        # While we need the unique sorted links (.first), we don't need the frequencies (.second)
                        del freq_pair.second

                    d(results)[i] = route_set

                    if self.block_flows_through_centroids:
                        blocking_centroid_flows(
                            1,  # Always unblocking
                            origin_index,
                            self.zones,
                            self.graph_fs_view,
                            b_nodes_matrix[threadid()],
                            self.b_nodes_view,
                        )

            if where is not None:
                table = libpa.pyarrow_wrap_table(
                    self.make_table_from_results(c_ods, d(results), cost_set, mask_set, path_overlap_set, prob_set)
                )

                # Once we've made the table all results have been copied into some pyarrow structure, we can free our
                # inner internal structures
                if path_size_logit:
                    for j in range(batch_len):
                        del d(link_union_set)[j]
                        del d(cost_set)[j]
                        del d(mask_set)[j]
                        del d(path_overlap_set)[j]
                        del d(prob_set)[j]

                for j in range(batch_len):
                    for route in d(d(results)[j]):
                        del route
                    del d(results)[j]

                checkpoint.write(table)
                del table
            else:
                # where is None implies len(batches) == 1, i.e. there was only one batch and we should keep everything
                # in memory
                pass

        # Here we decide if we wish to preserve our results for later saving/link loading
        if where is not None:
            # We're done with everything now, we can free the outer internal structures
            del results
            if path_size_logit:
                del link_union_set
                del cost_set
                del mask_set
                del path_overlap_set
                del prob_set
        else:
            self.results = results
            self.link_union_set = link_union_set
            self.cost_set = cost_set
            self.mask_set = mask_set
            self.path_overlap_set = path_overlap_set
            self.prob_set = prob_set

            # Copy the c_ods vector, it was provided by the auto Cython conversion and is allocated on the stack,
            # we should copy it to keep it around
            self.ods = new vector[pair[long long, long long]](c_ods)

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
    cdef RouteSet_t *bfsle(
        RouteChoiceSet self,
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
        double penatly,
        unsigned int seed
    ) noexcept nogil:
        """Main method for route set generation. See top of file for commentary."""
        cdef:
            # Output
            RouteSet_t *route_set

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
            bint lp = penatly != 1.0
            vector[double] *penalised_cost = <vector[double] *>nullptr
            vector[double] *next_penalised_cost = <vector[double] *>nullptr

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX

        queue.push_back(new unordered_set[long long]())  # Start with no edges banned
        route_set = new RouteSet_t()
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
                            d(next_penalised_cost)[connector] = penatly * d(next_penalised_cost)[connector]

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

        return route_set

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef RouteSet_t *link_penalisation(
        RouteChoiceSet self,
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
        double penatly,
        unsigned int seed
    ) noexcept nogil:
        """Link penalisation algorithm for choice set generation."""
        cdef:
            RouteSet_t *route_set

            # Scratch objects
            vector[long long] *vec
            long long p, connector
            pair[RouteSet_t.iterator, bool] status
            unsigned int miss_count = 0

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX
        route_set = new RouteSet_t()
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
                    thread_cost[connector] = penatly * thread_cost[connector]

                reverse(vec.begin(), vec.end())

                # To prevent runaway algorithms if we find N duplicate routes we should stop
                status = route_set.insert(vec)
                miss_count = miss_count + (not status.second)
                if miss_count > max_misses:
                    break
            else:
                break

        return route_set

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef pair[vector[long long] *, vector[long long] *] compute_frequency(RouteSet_t *route_set, vector[bool] &route_mask) noexcept nogil:
        """
        Compute a frequency map for each route with the route_mask applied.

        Each node at index i in the first returned vector has frequency at index i in the second vector.
        """
        cdef:
            vector[long long] *keys
            vector[long long] *counts
            vector[long long] link_union
            vector[long long].const_iterator union_iter
            vector[long long] *route

            # Scratch objects
            size_t length, count, i
            long long link
            bool route_present = False

        keys = new vector[long long]()
        counts = new vector[long long]()

        # When calculating the frequency of routes, we need to exclude those not in the mask.
        i = 0
        length = 0
        for route in d(route_set):
            # We do so here ...
            route_present = route_mask[inc(i)]
            if not route_present:
                continue

            length = length + route.size()
        link_union.reserve(length)

        i = 0
        for route in d(route_set):
            # ... and here.
            route_present = route_mask[inc(i)]
            if not route_present:
                continue

            link_union.insert(link_union.end(), route.begin(), route.end())

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

        return make_pair(keys, counts)

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[double] *compute_cost(RouteSet_t *route_set, double[:] cost_view) noexcept nogil:
        """Compute the cost each route."""
        cdef:
            vector[double] *cost_vec

            # Scratch objects
            double cost
            long long link, i

        cost_vec = new vector[double]()
        cost_vec.reserve(route_set.size())

        for route in d(route_set):
            cost = 0.0
            for link in d(route):
                cost = cost + cost_view[link]

            cost_vec.push_back(cost)

        return cost_vec

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[bool] *compute_mask(double cutoff_prob, vector[double] &total_cost) noexcept nogil:
        """
        Computes a binary logit between the minimum cost path and each path, if the total cost is greater than the
        minimum + the difference in utilities required to produce the cut-off probability then the route is excluded from
        the route set.
        """
        cdef:
            size_t i

            vector[bool] *route_mask = new vector[bool](total_cost.size())
            vector[double].const_iterator min = min_element(total_cost.cbegin(), total_cost.cend())
            double cutoff_cost = d(min) \
                + inverse_binary_logit(cutoff_prob, 0.0, 1.0)

        # The route mask should be True for the routes we wish to include.
        for i in range(total_cost.size()):
            d(route_mask)[i] = (total_cost[i] <= cutoff_cost)

        # Always include the min element. It should already be but I don't trust floating math to do this correctly.
        # But only if there actually was a min element (i.e. empty route set)
        if min != total_cost.cend():
            d(route_mask)[min - total_cost.cbegin()] = True

        return route_mask

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[double] *compute_path_overlap(
        RouteSet_t *route_set,
        pair[vector[long long] *, vector[long long] *] &freq_set,
        vector[double] &total_cost,
        vector[bool] &route_mask,
        double[:] cost_view
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
            vector[double] *path_overlap_vec  = new vector[double](route_set.size())

            # Scratch objects
            vector[long long].const_iterator link_iter
            double path_overlap
            long long link
            size_t i = 0

        for route in d(route_set):
            # Skip masked routes
            if not route_mask[i]:
                inc(i)
                continue

            path_overlap = 0.0
            for link in d(route):
                # We know the frequency table is ordered and contains every link in the union of the routes.
                # We want to find the index of the link, and use that to look up it's frequency
                link_iter = lower_bound(freq_set.first.begin(), freq_set.first.end(), link)

                # lower_bound returns freq_set.first.end() when no link is found.
                # This /should/ never happen.
                if link_iter == freq_set.first.end():
                    continue
                path_overlap = path_overlap + cost_view[link] \
                    / d(freq_set.second)[link_iter - freq_set.first.begin()]

            d(path_overlap_vec)[i] = path_overlap / total_cost[i]
            inc(i)

        return path_overlap_vec

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[double] *compute_prob(
        vector[double] &total_cost,
        vector[double] &path_overlap_vec,
        vector[bool] &route_mask,
        double beta
    ) noexcept nogil:
        """Compute a probability for each route in the route set based on the path overlap."""
        cdef:
            # Scratch objects
            vector[double] *prob_vec = new vector[double](total_cost.size())
            double inv_prob
            size_t i, j

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

                inv_prob = inv_prob + pow(path_overlap_vec[j] / path_overlap_vec[i], beta) \
                    * exp((total_cost[i] - total_cost[j]))  # Assuming theta=1.0

            d(prob_vec)[i] = 1.0 / inv_prob

        return prob_vec

    @cython.embedsignature(True)
    def link_loading(RouteChoiceSet self, matrix, generate_path_files: bool = False, cores: int = 0):
        """
        Apply link loading to the network using the demand matrix and the previously computed route sets.
        """
        if self.ods == nullptr \
           or self.link_union_set == nullptr \
           or self.prob_set == nullptr:
            raise ValueError("link loading requires Route Choice path_size_logit results")

        if not isinstance(matrix, AequilibraeMatrix):
            raise ValueError("`matrix` is not an AequilibraE matrix")

        cores = cores if cores > 0 else omp_get_max_threads()

        cdef:
            vector[vector[double] *] *path_files = <vector[vector[double] *] *>nullptr
            vector[double] *vec

        if generate_path_files:
            path_files = RouteChoiceSet.compute_path_files(
                d(self.ods),
                d(self.results),
                d(self.link_union_set),
                d(self.prob_set),
                cores,
            )

            # # FIXME, write out path files
            # tmp = []
            # for vec in d(path_files):
            #     tmp.append(d(vec))
            # print(tmp)

        link_loads = {}
        for i, name in enumerate(matrix.names):
            m = matrix.matrix_view if len(matrix.view_names) == 1 else matrix.matrix_view[:, :, i]

            ll = self.apply_link_loading_from_path_files(m, d(path_files)) \
                if generate_path_files else self.apply_link_loading(m)

            link_loads[name] = self.apply_link_loading_func(ll, cores)
            del ll

        if generate_path_files:
            for vec in d(path_files):
                del vec
            del path_files

        return link_loads

    cdef apply_link_loading_func(RouteChoiceSet self, vector[double] *ll, int cores):
        """Helper function for link_loading."""
        # push_back(0.0) and pop_back() are a hack to add a single element to the end of the link loading vector to
        # prevent a OOB access the assign_link_loads_cython method.
        ll.push_back(0.0)
        compressed = np.array(d(ll)).reshape(ll.size(), 1)
        actual = np.zeros((self.graph_compressed_id_view.shape[0], 1), dtype=np.float64)

        assign_link_loads_cython(
            actual,
            compressed,
            self.graph_compressed_id_view,
            cores
        )

        # Let's remove that last element we added, just in case
        ll.pop_back()

        return actual.reshape(-1), compressed.reshape(-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[vector[double] *] *compute_path_files(
        vector[pair[long long, long long]] &ods,
        vector[RouteSet_t *] &results,
        vector[vector[long long] *] &link_union_set,
        vector[vector[double] *] &prob_set,
        unsigned int cores
    ) noexcept nogil:
        """
        Computes the path files for the provided vector of RouteSets.

        Returns vector of vectors of link loads corresponding to each link in it's link_union_set.
        """
        cdef:
            vector[vector[double] *] *link_loads = new vector[vector[double] *](ods.size())
            vector[long long] *link_union
            vector[double] *loads
            vector[long long] *links

            vector[long long].const_iterator link_union_iter
            vector[long long].const_iterator link_iter

            size_t link_loc
            double prob
            long long i

        with parallel(num_threads=cores):
            for i in prange(ods.size()):
                link_union = link_union_set[i]
                loads = new vector[double](link_union.size(), 0.0)

                # We now iterate over all routes in the route_set, each route has an associated probability
                route_prob_iter = prob_set[i].cbegin()
                for route in d(results[i]):
                    prob = d(route_prob_iter)
                    inc(route_prob_iter)

                    if prob == 0.0:
                        continue

                    # For each link in the route, we need to assign the appropriate demand * prob Because the link union
                    # is known to be sorted, if the links in the route are also sorted we can just step along both
                    # arrays simultaneously, skipping elements in the link_union when appropriate. This allows us to
                    # operate on the link loads as a sparse map and avoid blowing up memory usage when using a dense
                    # formulation. This is also more cache efficient, the only downsides are that the code is
                    # harder to read and it requires sorting the route.

                    # NOTE: the sorting of routes is technically something that is already computed, during the
                    # computation of the link frequency we merge and sort all links, if we instead sorted then used an
                    # N-way merge we could reuse the sorted routes and the sorted link union.

                    # We copy the links in case the routes haven't already been saved
                    links = new vector[long long](d(route))
                    sort(links.begin(), links.end())

                    # links and link_union are sorted, and links is a subset of link_union
                    link_union_iter = link_union.cbegin()
                    link_iter = links.cbegin()

                    while link_iter != links.cend():
                        # Find the next location for the current link in links
                        while d(link_iter) != d(link_union_iter) and link_iter != links.cend():
                            inc(link_union_iter)

                        link_loc = link_union_iter - link_union.cbegin()
                        d(loads)[link_loc] = d(loads)[link_loc] + prob  # += here results in all zeros? Odd

                        inc(link_iter)

                    del links

                d(link_loads)[i] = loads

        return link_loads

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef vector[double] *apply_link_loading_from_path_files(
        RouteChoiceSet self,
        double[:, :] matrix_view,
        vector[vector[double] *] &path_files
    ) noexcept nogil:
        """
        Apply link loading from path files.

        Returns a vector of link loads indexed by compressed link ID.
        """
        cdef:
            vector[double] *loads
            vector[long long] *link_union
            long origin_index, dest_index
            double demand

            vector[double] *link_loads = new vector[double](self.num_links)

        for i in range(self.ods.size()):
            loads = path_files[i]
            link_union = d(self.link_union_set)[i]

            origin_index = self.nodes_to_indices_view[d(self.ods)[i].first]
            dest_index = self.nodes_to_indices_view[d(self.ods)[i].second]
            demand = matrix_view[origin_index, dest_index]

            for j in range(link_union.size()):
                link = d(link_union)[j]
                d(link_loads)[link] = d(link_loads)[link] + demand * d(loads)[j]

        return link_loads

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef vector[double] *apply_link_loading(self, double[:, :] matrix_view) noexcept nogil:
        """
        Apply link loading.

        Returns a vector of link loads indexed by compressed link ID.
        """
        cdef:
            RouteSet_t *route_set
            vector[double] *route_set_prob
            vector[double].const_iterator route_prob_iter
            long origin_index, dest_index
            double demand, prob, load

            vector[double] *link_loads = new vector[double](self.num_links)

        for i in range(self.ods.size()):
            route_set = d(self.results)[i]
            route_set_prob = d(self.prob_set)[i]

            origin_index = self.nodes_to_indices_view[d(self.ods)[i].first]
            dest_index = self.nodes_to_indices_view[d(self.ods)[i].second]
            demand = matrix_view[origin_index, dest_index]

            route_prob_iter = route_set_prob.cbegin()
            for route in d(route_set):
                prob = d(route_prob_iter)
                inc(route_prob_iter)

                load = prob * demand
                for link in d(route):
                    d(link_loads)[link] = d(link_loads)[link] + load  # += here results in all zeros? Odd

        return link_loads

    @cython.embedsignature(True)
    def select_link_loading(RouteChoiceSet self, matrix, select_links: Dict[str, FrozenSet[FrozenSet[int]]], cores: int = 0):
        """
        Apply link loading to the network using the demand matrix and the previously computed route sets.
        """
        if self.ods == nullptr \
           or self.link_union_set == nullptr \
           or self.prob_set == nullptr:
            raise ValueError("select link loading requires Route Choice path_size_logit results")

        if not isinstance(matrix, AequilibraeMatrix):
            raise ValueError("`matrix` is not an AequilibraE matrix")

        cores = cores if cores > 0 else omp_get_max_threads()

        cdef:
            vector[unordered_set[long long] *] *select_link_set
            vector[size_t] *select_link_set_length

            vector[vector[unordered_set[long long] *] *] select_link_sets
            vector[vector[size_t] *] select_link_set_lengths

            vector[double] *ll

        # Coerce the select link sets to their cpp structures ahead of time. We'll be using these a lot and they don't
        # change. We allocate a vector of select link sets. These select link sets a vector representing an OR set,
        # containing a unordered_set of links representing the AND set.
        select_link_sets.reserve(len(select_links))
        select_link_set_lengths.reserve(len(select_links))
        for or_set in select_links.values():
            select_link_set = new vector[unordered_set[long long] *](len(or_set))
            select_link_set_length = new vector[size_t](len(or_set))

            for i, and_set in enumerate(or_set):
                d(select_link_set)[i] = new unordered_set[long long](and_set)
                d(select_link_set_length)[i] = len(and_set)

            select_link_sets.push_back(select_link_set)
            select_link_set_lengths.push_back(select_link_set_length)

        link_loads = {}
        for i, name in enumerate(matrix.names):
            matrix_ll = {}
            m = matrix.matrix_view if len(matrix.view_names) == 1 else matrix.matrix_view[:, :, i]
            for i, k in enumerate(select_links.keys()):
                select_link_set = select_link_sets[i]
                select_link_set_length = select_link_set_lengths[i]

                coo = COO((self.zones, self.zones))

                ll = self.apply_select_link_loading(coo, m, d(select_link_set), d(select_link_set_length))
                res = self.apply_link_loading_func(ll, cores)

                # Because we converted the python sets to C++ ourselves we have to clean them up as well
                del ll

                matrix_ll[k] = (coo, res)
            link_loads[name] = matrix_ll


        # Clean up our coercion
        for select_link_set in select_link_sets:
            for cpp_and_set in d(select_link_set):
                del cpp_and_set
            del select_link_set

        for i in range(select_link_set_lengths.size()):
            del select_link_set_lengths[i]

        return link_loads

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef vector[double] *apply_select_link_loading(
        RouteChoiceSet self,
        COO sparse_mat,
        double[:, :] matrix_view,
        vector[unordered_set[long long] *] &select_link_set,
        vector[size_t] select_link_set_lengths
    ) noexcept nogil:
        """
        Apply select link loading.

        Returns a vector of link loads indexed by compressed link ID.
        """
        cdef:
            RouteSet_t *route_set
            vector[double] *route_set_prob
            vector[double].const_iterator route_prob_iter
            long origin_index, dest_index, o, d
            double demand, prob, load

            vector[double] *link_loads = new vector[double](self.num_links)

        # For each OD pair, if a route contains one or more links in a select link set, add that ODs demand to
        # a sparse matrix of Os to Ds

        # For each route, if it contains one or more links in a select link set, apply the link loading for
        # that route

        for i in range(self.ods.size()):
            route_set = d(self.results)[i]
            route_set_prob = d(self.prob_set)[i]

            origin_index = self.nodes_to_indices_view[d(self.ods)[i].first]
            dest_index = self.nodes_to_indices_view[d(self.ods)[i].second]
            demand = matrix_view[origin_index, dest_index]

            route_prob_iter = route_set_prob.cbegin()
            for route in d(route_set):
                prob = d(route_prob_iter)
                inc(route_prob_iter)
                load = prob * demand

                if RouteChoiceSet.is_in_select_link(d(route), select_link_set, select_link_set_lengths):
                    sparse_mat.append(origin_index, dest_index, load)
                    for link in d(route):
                        d(link_loads)[link] = d(link_loads)[link] + load  # += here results in all zeros? Odd

        return link_loads

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef bool is_in_select_link(
        vector[long long] &route,
        vector[unordered_set[long long] *] &select_link_set,
        vector[size_t] &select_link_set_lengths
    ) noexcept nogil:
        """
        Confirms if a given route satisfies the requirements of the select link set.

        **Assumes the route contains unique links**. If a link is present >1 and is in the select link set, this method
        will double count it.
        """

        cdef:
            bool set_present
            vector[size_t] link_counts
            size_t select_link_set_idx
            long long link
            unordered_set[long long] *and_set

        # We count the number of appearances of links within the AND set. Assuming we only see unique links, then if
        # that count goes to 0 we have seen all links present within the AND set. A shortest-path will contain only
        # unique links in paths. This may not be not be true for heuristics, but if the heuristic is that bad that it's
        # traversing a link twice I don't think we should be using it.
        link_counts.insert(link_counts.begin(), select_link_set_lengths.cbegin(), select_link_set_lengths.cend())

        # We iterate over an AND set first in hopes that a whole set is satisfied before checking another. This let's
        # just "short circuit" in a sense
        select_link_set_idx = 0
        for and_set in select_link_set:
            for link in route:
                if and_set.find(link) != and_set.end():
                    if pre_dec(link_counts[select_link_set_idx]) == 0:
                        return True
                else:
                    continue
            inc(select_link_set_idx)

        return False

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef shared_ptr[libpa.CTable] make_table_from_results(
        RouteChoiceSet self,
        vector[pair[long long, long long]] &ods,
        vector[RouteSet_t *] &route_sets,
        vector[vector[double] *] *cost_set,
        vector[vector_bool_ptr] *mask_set,
        vector[vector[double] *] *path_overlap_set,
        vector[vector[double] *] *prob_set
    ):
        """
        Construct an Arrow table from C++ stdlib structures.

        Note: this function directly utilises the Arrow C++ API, the Arrow Cython API is not sufficient.
        See `route_choice_set.pxd` for Cython declarations.

        Returns a shared pointer to a Arrow CTable. This should be wrapped in a Python table before use.
        Compressed link IDs are expanded to full network link IDs.
        """
        cdef:
            shared_ptr[libpa.CArray] paths
            shared_ptr[libpa.CArray] offsets

            libpa.CMemoryPool *pool = libpa.c_get_memory_pool()

            # Custom imports, these are declared in route_choice.pxd *not* libarrow.
            CUInt32Builder *path_builder = new CUInt32Builder(pool)
            CDoubleBuilder *cost_col = <CDoubleBuilder *>nullptr
            CBooleanBuilder *mask_col = <CBooleanBuilder *>nullptr
            CDoubleBuilder *path_overlap_col = <CDoubleBuilder *>nullptr
            CDoubleBuilder *prob_col = <CDoubleBuilder *>nullptr

            libpa.CInt32Builder *offset_builder = new libpa.CInt32Builder(pool)  # Must be Int32 *not* UInt32
            libpa.CUInt32Builder *o_col = new libpa.CUInt32Builder(pool)
            libpa.CUInt32Builder *d_col = new libpa.CUInt32Builder(pool)
            vector[shared_ptr[libpa.CArray]] columns
            shared_ptr[libpa.CDataType] route_set_dtype = libpa.pyarrow_unwrap_data_type(RouteChoiceSet.route_set_dtype)

            libpa.CResult[shared_ptr[libpa.CArray]] route_set_results

            int offset = 0
            size_t network_link_begin, network_link_end, link
            bint psl = (cost_set != nullptr and path_overlap_set != nullptr and prob_set != nullptr)

        # Origins, Destination, Route set, [Cost for route, Mask, Path_Overlap for route, Probability for route]
        columns.resize(7 if psl else 3)

        if psl:
            cost_col = new CDoubleBuilder(pool)
            mask_col = new CBooleanBuilder(pool)
            path_overlap_col = new CDoubleBuilder(pool)
            prob_col = new CDoubleBuilder(pool)

            for i in range(ods.size()):
                cost_col.AppendValues(d(d(cost_set)[i]))
                mask_col.AppendValues(d(d(mask_set)[i]))
                path_overlap_col.AppendValues(d(d(path_overlap_set)[i]))
                prob_col.AppendValues(d(d(prob_set)[i]))

        for i in range(ods.size()):
            route_set = route_sets[i]

            # Instead of constructing a "list of lists" style object for storing the route sets we instead will
            # construct one big array of link IDs with a corresponding offsets array that indicates where each new row
            # (path) starts.
            for route in d(route_set):
                o_col.Append(ods[i].first)
                d_col.Append(ods[i].second)

                offset_builder.Append(offset)

                for link in d(route):
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

        if psl:
            cost_col.Finish(&columns[3])
            mask_col.Finish(&columns[4])
            path_overlap_col.Finish(&columns[5])
            prob_col.Finish(&columns[6])

        cdef shared_ptr[libpa.CSchema] schema = libpa.pyarrow_unwrap_schema(
            RouteChoiceSet.psl_schema if psl else RouteChoiceSet.schema
        )
        cdef shared_ptr[libpa.CTable] table = libpa.CTable.MakeFromArrays(schema, columns)

        del path_builder
        del offset_builder
        del o_col
        del d_col

        if psl:
            del cost_col
            del mask_col
            del path_overlap_col
            del prob_col

        return table

    def get_results(self):  # Cython doesn't like this type annotation... -> pa.Table:
        """
        :Returns:
            **route sets** (:obj:`pyarrow.Table`): Returns a table of OD pairs to lists of link IDs for
                each OD pair provided (as columns). Represents paths from ``origin`` to ``destination``.
        """
        if self.results == nullptr or self.ods == nullptr:
            raise RuntimeError("Route Choice results not computed yet")

        table = libpa.pyarrow_wrap_table(
            self.make_table_from_results(
                d(self.ods),
                d(self.results),
                self.cost_set,
                self.mask_set,
                self.path_overlap_set,
                self.prob_set
            )
        )

        return table


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


cdef double inverse_binary_logit(double prob, double beta0, double beta1) noexcept nogil:
    if prob == 1.0:
        return INFINITY
    elif prob == 0.0:
        return -INFINITY
    else:
        return (log(prob / (1.0 - prob)) - beta0) / beta1
