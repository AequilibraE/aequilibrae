# cython: language_level=3str

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
directly into the output ccp. This is done allow declaring of the `()` operator, which is required and AFAIK not
possible in Cython. The hash is designed to dereference then hash order dependent vectors. One isn't provided by
stdlib. The comparator simply dereferences the pointer and uses the vector comparator. It's designed to store the
outputted paths. Heap allocated (needs to be returned).

removed_links: See route_choice.pxd for full type signature. It's an unordered set of pointers to unordered sets of link
IDs. Similarly to `route_set` is uses a custom hash function and comparator. This hash function is designed to be order
independent and should only use commutative operations. The comparator is the same. It's designed to store all of the
removed link sets we've seen before. This allows us to detected duplicated graphs.

rng: A custom imported version of std::linear_congruential_engine. libcpp doesn't provide one so we do. It should be
significantly faster than the std::mersenne_twister_engine without sacrificing much. We don't need amazing RNG, just
ok and fast. This is only used to shuffle the queue.

queue, next_queue: These are vectors of pointers to sets of removed links. We never need to push to the front of these so a
vector is best. We maintain two queues, one that we are currently iterating over, and one that we can add to, building
up with all the newly removed link sets. These two are swapped at the end of an iteration, next_queue is then
cleared. These store sets of removed links.

banned, next_banned: `banned` is the iterator variable for `queue`. `banned` is copied into `next_banned` where another
link can be added without mutating `banned`. If we've already seen this set of removed links `next_banned` is immediately
deallocated. Otherwise it's placed into `next_queue`.

vec: `vec` is a scratch variable to store pointers to new vectors, or rather, paths while we are building them. Each time a path
is found a new one is allocated, built, and stored in the route_set.

p, connector: Scratch variables for iteration.

Optimisations: As described in the paper, both optimisations have been implemented. The path finding operates on the
compressed graph and the queue is shuffled if its possible to fill the route set that iteration. The route set may not
be filled due to duplicate paths but we can't know that in advance so we shuffle anyway.

Any further optimisations should focus on the path finding, from benchmarks it dominates the run time (~98%). Since huge
routes aren't required small-ish things like the memcpy and banned link set copy aren't high priority.

"""

from aequilibrae import Graph

from libc.math cimport INFINITY, pow, exp
from libc.string cimport memcpy
from libc.limits cimport UINT_MAX
from libc.stdlib cimport abort
from libcpp cimport nullptr
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort, lower_bound
from cython.operator cimport dereference as deref, preincrement as inc
from cython.parallel cimport parallel, prange, threadid
cimport openmp

import numpy as np
import pyarrow as pa
from typing import List, Tuple
import itertools
import pathlib
import logging
import warnings

cimport numpy as np  # Numpy *must* be cimport'd BEFORE pyarrow.lib, there's nothing quite like Cython.
cimport pyarrow as pa
cimport pyarrow.lib as libpa
import pyarrow.dataset
import pyarrow.parquet as pq
from libcpp.memory cimport shared_ptr

from libc.stdio cimport fprintf, printf, stderr

# It would really be nice if these were modules. The 'include' syntax is long deprecated and adds a lot to compilation times
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
        pa.field("gamma", pa.float64(), nullable=False),
        pa.field("probability", pa.float64(), nullable=False),
    ])

    def __cinit__(self):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""
        results = <vector[RouteSet_t *] *>nullptr
        link_union_set = <vector[vector[long long] *] *>nullptr
        cost_set = <vector[vector[double] *] *>nullptr
        gamma_set = <vector[vector[double] *] *>nullptr
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
        self.num_nodes = graph.compact_num_nodes
        self.num_links = graph.compact_num_links
        self.zones = graph.num_zones
        self.block_flows_through_centroids = graph.block_centroid_flows

    def __dealloc__(self):
        """
        C level deallocation. For freeing memory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """
        self.deallocate_results()

    def deallocate_results(self):
        """
        Deallocate stored results, existing extracted results are not invalidated.
        """
        cdef:
            RouteSet_t *route_set
            vector[long long] *link_vec
            vector[double] *double_vec

        if self.results != nullptr:
            for route_set in deref(self.results):
                for link_vec in deref(route_set):
                    del link_vec
                del route_set
            del self.results

        if self.link_union_set != nullptr:
            for link_vec in deref(self.link_union_set):
                del link_vec
            del self.link_union_vec

        if self.cost_set != nullptr:
            for double_vec in deref(self.cost_set):
                del double_vec
            del self.cost_vec

        if self.gamma_set != nullptr:
            for double_vec in deref(self.gamma_set):
                del double_vec
            del self.gamma_vec

        if self.prob_set != nullptr:
            for double_vec in deref(self.prob_set):
                del double_vec
            del self.prob_vec

        if self.ods != nullptr:
            del self.ods

    @cython.embedsignature(True)
    def run(self, origin: int, destination: int, *args, **kwargs):
        """
        Compute the a route set for a single OD pair.

        Often the returned list's length is ``max_routes``, however, it may be limited by ``max_depth`` or if all
        unique possible paths have been found then a smaller set will be returned.

        Thin wrapper around ``RouteChoiceSet.batched``. Additional arguments are forwarded to ``RouteChoiceSet.batched``.

        :Arguments:
            **origin** (:obj:`int`): Origin node ID. Must be present within compact graph. Recommended to choose a centroid.
            **destination** (:obj:`int`): Destination node ID. Must be present within compact graph. Recommended to choose a centroid.

        :Returns:
            **route set** (:obj:`list[tuple[int, ...]]): Returns a list of unique variable length tuples of compact link IDs.
                                                         Represents paths from ``origin`` to ``destination``.
        """
        self.batched([(origin, destination)], *args, **kwargs)
        return [tuple(x) for x in self.get_results().column("route set").to_pylist()]

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
            seed: int = 0,
            cores: int = 0,
            a_star: bool = True,
            bfsle: bool = True,
            penalty: float = 0.0,
            where: Optional[str] = None,
            path_size_logit: bool = False,
            beta: float = 1.0,
            theta: float = 1.0,
    ):
        """
        Compute the a route set for a list of OD pairs.

        Often the returned list for each OD pair's length is ``max_routes``, however, it may be limited by ``max_depth`` or if all
        unique possible paths have been found then a smaller set will be returned.

        :Arguments:
            **ods** (:obj:`list[tuple[int, int]]`): List of OD pairs ``(origin, destination)``. Origin and destination node ID must be
                                                    present within compact graph. Recommended to choose a centroids.
            **max_routes** (:obj:`int`): Maximum size of the generated route set. Must be non-negative. Default of ``0`` for unlimited.
            **max_depth** (:obj:`int`): Maximum depth BFSLE can explore, or maximum number of iterations for link penalisation.
                                        Must be non-negative. Default of ``0`` for unlimited.
            **seed** (:obj:`int`): Seed used for rng. Must be non-negative. Default of ``0``.
            **cores** (:obj:`int`): Number of cores to use when parallelising over OD pairs. Must be non-negative. Default of ``0`` for all available.
            **bfsle** (:obj:`bool`): Whether to use Breadth First Search with Link Removal (BFSLE) over link penalisation. Default ``True``.
            **penalty** (:obj:`float`): Penalty to use for Link Penalisation. Must be ``> 1.0``. Not compatible with ``bfsle=True``.
            **where** (:obj:`str`): Optional file path to save results to immediately. Will return None.
        """
        cdef:
            long long o, d

        if max_routes == 0 and max_depth == 0:
            raise ValueError("Either `max_routes` or `max_depth` must be > 0")

        if max_routes < 0 or max_depth < 0:
            raise ValueError("`max_routes`, `max_depth`, and `cores` must be non-negative")

        if penalty != 0.0 and bfsle:
            raise ValueError("Link penalisation (`penalty` > 1.0) and `bfsle` cannot be enabled at once")

        if not bfsle and penalty <= 1.0:
            raise ValueError("`penalty` must be > 1.0. `penalty=1.1` is recommended")

        if path_size_logit and (beta < 0 or theta <= 0):
            raise ValueError("`beta` must be >= 0 and `theta` > 0 for path sized logit model")

        for o, d in ods:
            if self.nodes_to_indices_view[o] == -1:
                raise ValueError(f"Origin {o} is not present within the compact graph")
            if self.nodes_to_indices_view[d] == -1:
                raise ValueError(f"Destination {d} is not present within the compact graph")

        cdef:
            long long origin_index, dest_index, i
            unsigned int c_max_routes = max_routes
            unsigned int c_max_depth = max_depth
            unsigned int c_seed = seed
            unsigned int c_cores = cores if cores > 0 else openmp.omp_get_num_threads()

            vector[pair[long long, long long]] c_ods

            # A* (and Dijkstra's) require memory views, so we must allocate here and take slices. Python can handle this memory
            double [:, :] cost_matrix = np.empty((c_cores, self.cost_view.shape[0]), dtype=float)
            long long [:, :] predecessors_matrix = np.empty((c_cores, self.num_nodes + 1), dtype=np.int64)
            long long [:, :] conn_matrix = np.empty((c_cores, self.num_nodes + 1), dtype=np.int64)
            long long [:, :] b_nodes_matrix = np.broadcast_to(self.b_nodes_view, (c_cores, self.b_nodes_view.shape[0])).copy()

            # This matrix is never read from, it exists to allow using the Dijkstra's method without changing the
            # interface.
            long long [:, :] _reached_first_matrix

            vector[RouteSet_t *] *results
            size_t max_results_len, batch_len, j

        # self.a_star = a_star

        if self.a_star:
            _reached_first_matrix = np.zeros((c_cores, 1), dtype=np.int64)  # Dummy array to allow slicing
        else:
            _reached_first_matrix = np.zeros((c_cores, self.num_nodes + 1), dtype=np.int64)

        set_ods = set(ods)
        if len(set_ods) != len(ods):
            warnings.warn(f"Duplicate OD pairs found, dropping {len(ods) - len(set_ods)} OD pairs")

        if where is not None:
            checkpoint = Checkpoint(where, self.schema, partition_cols=["origin id"])
            batches = list(Checkpoint.batches(list(set_ods)))
            max_results_len = <size_t>max(len(batch) for batch in batches)
        else:
            batches = [list(set_ods)]
            max_results_len = len(set_ods)

        results = new vector[RouteSet_t *](max_results_len)

        cdef:
            RouteSet_t *route_set
            pair[vector[long long] *, vector[long long] *] freq_pair
            vector[long long] *link_union_scratch = <vector[long long] *>nullptr
            vector[vector[long long] *] *link_union_set = <vector[vector[long long] *] *>nullptr
            vector[vector[double] *] *cost_set = <vector[vector[double] *] *>nullptr
            vector[vector[double] *] *gamma_set = <vector[vector[double] *] *>nullptr
            vector[vector[double] *] *prob_set = <vector[vector[double] *] *>nullptr

        if path_size_logit:
            link_union_set = new vector[vector[long long] *](max_results_len)
            cost_set = new vector[vector[double] *](max_results_len)
            gamma_set = new vector[vector[double] *](max_results_len)
            prob_set = new vector[vector[double] *](max_results_len)

        self.deallocate_results()  # We have be storing results from a previous run

        for batch in batches:
            c_ods = batch  # Convert the batch to a cpp vector, this isn't strictly efficient but is nicer
            batch_len = c_ods.size()
            results.resize(batch_len)  # We know we've allocated enough size to store all max length batch but we resize to a smaller size when not needed

            if path_size_logit:
                # we may clear these objects because it's either:
                # - the first iteration and they contain no elements, thus no memory to leak
                # - the internal objects were freed by the previous iteration
                link_union_set.clear()
                cost_set.clear()
                gamma_set.clear()
                prob_set.clear()

                link_union_set.resize(batch_len)
                cost_set.resize(batch_len)
                gamma_set.resize(batch_len)
                prob_set.resize(batch_len)

            with nogil, parallel(num_threads=c_cores):
                # The link union needs to be allocated per thread as scratch space, as its of unknown length we can't allocated a matrix of them.
                # Additionally getting them to be reused between batches is complicated, instead we just get a new one each batch
                if path_size_logit:
                    link_union_scratch = new vector[long long]()

                for i in prange(batch_len):
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
                            cost_matrix[threadid()],
                            predecessors_matrix[threadid()],
                            conn_matrix[threadid()],
                            b_nodes_matrix[threadid()],
                            _reached_first_matrix[threadid()],
                            c_seed,
                        )
                    else:
                        route_set = RouteChoiceSet.link_penalisation(
                            self,
                            origin_index,
                            dest_index,
                            c_max_routes,
                            c_max_depth,
                            cost_matrix[threadid()],
                            predecessors_matrix[threadid()],
                            conn_matrix[threadid()],
                            b_nodes_matrix[threadid()],
                            _reached_first_matrix[threadid()],
                            penalty,
                            c_seed,
                        )

                    if path_size_logit:
                        link_union_scratch.clear()
                        freq_pair = RouteChoiceSet.compute_frequency(route_set, deref(link_union_scratch))
                        deref(link_union_set)[i] = freq_pair.first
                        deref(cost_set)[i] = RouteChoiceSet.compute_cost(route_set, self.cost_view)
                        deref(gamma_set)[i] = RouteChoiceSet.compute_gamma(route_set, freq_pair, deref(deref(cost_set)[i]), self.cost_view)
                        deref(prob_set)[i] = RouteChoiceSet.compute_prob(deref(deref(cost_set)[i]), deref(deref(gamma_set)[i]), beta, theta)
                        del freq_pair.second  # While we need the unique sorted links (.first), we don't need the frequencies (.second)

                    deref(results)[i] = route_set

                    if self.block_flows_through_centroids:
                        blocking_centroid_flows(
                            1,  # Always unblocking
                            origin_index,
                            self.zones,
                            self.graph_fs_view,
                            b_nodes_matrix[threadid()],
                            self.b_nodes_view,
                        )

                if path_size_logit:
                    del link_union_scratch

            if where is not None:
                table = libpa.pyarrow_wrap_table(RouteChoiceSet.make_table_from_results(c_ods, deref(results), cost_set, gamma_set, prob_set))

                # Once we've made the table all results have been copied into some pyarrow structure, we can free our inner internal structures
                if path_size_logit:
                    for j in range(batch_len):
                        del deref(link_union_set)[j]
                        del deref(cost_set)[j]
                        del deref(gamma_set)[j]
                        del deref(prob_set)[j]

                for j in range(batch_len):
                    for route in deref(deref(results)[j]):
                        del route
                    del deref(results)[j]

                checkpoint.write(table)
                del table
            else:
                pass  # where is None ==> len(batches) == 1, i.e. there was only one batch and we should keep everything in memory

        # Here we decide if we wish to preserve our results for later saving/link loading
        if where is not None:
            # We're done with everything now, we can free the outer internal structures
            del results
            if path_size_logit:
                del link_union_set
                del cost_set
                del gamma_set
                del prob_set
        else:
            self.results = results
            self.link_union_set = link_union_set
            self.cost_set = cost_set
            self.gamma_set = gamma_set
            self.prob_set = prob_set

            # Copy the c_ods vector, it was provided by the auto Cython conversion and is allocated on the stack,
            # we should copy it to keep it around
            self.ods = new vector[pair[long long, long long]](c_ods)

            # self.link_union ?? This could be saved as a partial results from the computation above, although it isn't easy to get out rn

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
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        unsigned int seed
    ) noexcept nogil:
        """Main method for route set generation. See top of file for commentary."""
        cdef:
            RouteSet_t *route_set
            LinkSet_t removed_links
            minstd_rand rng

            # Scratch objects
            vector[unordered_set[long long] *] queue
            vector[unordered_set[long long] *] next_queue
            unordered_set[long long] *banned
            unordered_set[long long] *new_banned
            vector[long long] *vec
            long long p, connector

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX

        queue.push_back(new unordered_set[long long]()) # Start with no edges banned
        route_set = new RouteSet_t()
        rng.seed(seed)

        # We'll go at most `max_depth` iterations down, at each depth we maintain a queue of the next set of banned edges to consider
        for depth in range(max_depth):
            if route_set.size() >= max_routes or queue.size() == 0:
                break

            # If we could potentially fill the route_set after this depth, shuffle the queue
            if queue.size() + route_set.size() >= max_routes:
                shuffle(queue.begin(), queue.end(), rng)

            for banned in queue:
                # Copying the costs back into the scratch costs buffer. We could keep track of the modifications and reverse them as well
                memcpy(&thread_cost[0], &self.cost_view[0], self.cost_view.shape[0] * sizeof(double))

                for connector in deref(banned):
                    thread_cost[connector] = INFINITY

                RouteChoiceSet.path_find(self, origin_index, dest_index, thread_cost, thread_predecessors, thread_conn, thread_b_nodes, _thread_reached_first)

                # Mark this set of banned links as seen
                removed_links.insert(banned)

                # If the destination is reachable we must build the path and readd
                if thread_predecessors[dest_index] >= 0:
                    vec = new vector[long long]()
                    # Walk the predecessors tree to find our path, we build it up in a cpp vector because we can't know how long it'll be
                    p = dest_index
                    while p != origin_index:
                        connector = thread_conn[p]
                        p = thread_predecessors[p]
                        vec.push_back(connector)

                    for connector in deref(vec):
                        # This is one area for potential improvement. Here we construct a new set from the old one, copying all the elements
                        # then add a single element. An incremental set hash function could be of use. However, the since of this set is
                        # directly dependent on the current depth and as the route set size grows so incredibly fast the depth will rarely get
                        # high enough for this to matter.
                        # Copy the previously banned links, then for each vector in the path we add one and push it onto our queue
                        new_banned = new unordered_set[long long](deref(banned))
                        new_banned.insert(connector)
                        # If we've already seen this set of removed links before we already know what the path is and its in our route set
                        if removed_links.find(new_banned) != removed_links.end():
                            del new_banned
                        else:
                            next_queue.push_back(new_banned)

                    # The deduplication of routes occurs here
                    route_set.insert(vec)
                    if route_set.size() >= max_routes:
                        break

            queue.swap(next_queue)
            next_queue.clear()

        # We may have added more banned link sets to the queue then found out we hit the max depth, we should free those
        for banned in queue:
            del banned

        # We should also free all the sets in removed_links, we don't be needing them
        for banned in removed_links:
            del banned

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
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        double penatly,
        unsigned int seed
    ) noexcept nogil:
        cdef:
            RouteSet_t *route_set

            # Scratch objects
            vector[long long] *vec
            long long p, connector

        max_routes = max_routes if max_routes != 0 else UINT_MAX
        max_depth = max_depth if max_depth != 0 else UINT_MAX
        route_set = new RouteSet_t()
        memcpy(&thread_cost[0], &self.cost_view[0], self.cost_view.shape[0] * sizeof(double))

        for depth in range(max_depth):
            if route_set.size() >= max_routes:
                break

            RouteChoiceSet.path_find(self, origin_index, dest_index, thread_cost, thread_predecessors, thread_conn, thread_b_nodes, _thread_reached_first)

            if thread_predecessors[dest_index] >= 0:
                vec = new vector[long long]()
                # Walk the predecessors tree to find our path, we build it up in a cpp vector because we can't know how long it'll be
                p = dest_index
                while p != origin_index:
                    connector = thread_conn[p]
                    p = thread_predecessors[p]
                    vec.push_back(connector)

                for connector in deref(vec):
                    thread_cost[connector] *= penatly

                route_set.insert(vec)
            else:
                break

        return route_set

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef pair[vector[long long] *, vector[long long] *] compute_frequency(RouteSet_t *route_set, vector[long long] &link_union) noexcept nogil:
        cdef:
            vector[long long] *keys
            vector[long long] *counts

            # Scratch objects
            size_t length, count
            long long link, i

        link_union.clear()

        keys = new vector[long long]()
        counts = new vector[long long]()

        length = 0
        for route in deref(route_set):
            length = length + route.size()
        link_union.reserve(length)

        for route in deref(route_set):
            link_union.insert(link_union.end(), route.begin(), route.end())

        sort(link_union.begin(), link_union.end())

        union_iter = link_union.begin()
        while union_iter != link_union.end():
            count = 0
            link = deref(union_iter)
            while link == deref(union_iter):
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
        cdef:
            vector[double] *cost_vec

            # Scratch objects
            double cost
            long long link, i

        cost_vec = new vector[double]()
        cost_vec.reserve(route_set.size())

        for route in deref(route_set):
            cost = 0.0
            for link in deref(route):
                cost = cost + cost_view[link]

            cost_vec.push_back(cost)

        return cost_vec

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[double] *compute_gamma(
        RouteSet_t *route_set,
        pair[vector[long long] *, vector[long long] *] &freq_set,
        vector[double] &total_cost,
        double[:] cost_view
    ) noexcept nogil:
        """
        Notation changes:
            i: j
            a: link
            t_a: cost_view
            c_i: total_costs
            A_i: route
            sum_{k in R}: delta_{a,k}: freq_set
        """
        cdef:
            vector[double] *gamma_vec

            # Scratch objects
            vector[long long].const_iterator link_iter
            double gamma
            long long link, j
            size_t i

        gamma_vec = new vector[double]()
        gamma_vec.reserve(route_set.size())

        j = 0
        for route in deref(route_set):
            gamma = 0.0
            for link in deref(route):
                # We know the frequency table is ordered and contains every link in the union of the routes.
                # We want to find the index of the link, and use that to look up it's frequency
                link_iter = lower_bound(freq_set.first.begin(), freq_set.first.end(), link)

                gamma = gamma + cost_view[link] / deref(freq_set.second)[link_iter - freq_set.first.begin()]

            gamma_vec.push_back(gamma / total_cost[j])

            j = j + 1

        return gamma_vec

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef vector[double] *compute_prob(
        vector[double] &total_cost,
        vector[double] &gamma_vec,
        double beta,
        double theta
    ) noexcept nogil:
        cdef:
            # Scratch objects
            vector[double] *prob_vec
            double inv_prob
            long long route_set_idx
            size_t i, j

        prob_vec = new vector[double]()
        prob_vec.reserve(total_cost.size())

        # Beware when refactoring the below, the scale of the costs may cause floating point errors. Large costs will lead to NaN results
        for i in range(total_cost.size()):
            inv_prob = 0.0
            for j in range(total_cost.size()):
                inv_prob = inv_prob + pow(gamma_vec[j] / gamma_vec[i], beta) * exp(-theta * (total_cost[j] - total_cost[i]))

            prob_vec.push_back(1.0 / inv_prob)

        return prob_vec

    def link_loading(self, double[:, :] matrix_view):
        if self.ods == nullptr \
           or self.link_union_set == nullptr \
           or self.prob_set == nullptr:
            raise ValueError("link loading requires Route Choice path_size_logit results")

        cdef:
            vector[double] *loads
            vector[double] *route_set_prob
            vector[double] *collective_link_loads = new vector[double](self.num_links)  # FIXME FREE ME
            vector[vector[double] *] *link_loads = new vector[vector[double] *](self.ods.size())  # FIXME FREE ME

            vector[long long] *link_union
            vector[long long].const_iterator link_union_iter

            vector[long long] *links
            vector[long long].const_iterator link_iter

            vector[double].const_iterator prob_iter

            RouteSet_t *route_set
            double demand, load, prob
            size_t length
            long origin_index, dest_index
            int i

        fprintf(stderr, "starting link loading\n")
        with nogil:
            with parallel(num_threads=1):
                # The link union needs to be allocated per thread as scratch space, as its of unknown length we can't allocated a matrix of them.
                # Additionally getting them to be reused between batches is complicated, instead we just get a new one each batch
                fprintf(stderr, "core: %d\n", threadid())

                for i in prange(self.ods.size()):
                    fprintf(stderr, "od idx: %d, %d has demand: %f\n", origin_index, dest_index, demand)

                    route_set = deref(self.results)[i]
                    fprintf(stderr, "got route set\n")
                    link_union = deref(self.link_union_set)[i]
                    fprintf(stderr, "got link union\n")
                    route_set_prob = deref(self.prob_set)[i]
                    fprintf(stderr, "got route set probsk\n")

                    fprintf(stderr, "making new loads vector\n")
                    loads = new vector[double](link_union.size(), 0.0)  # FIXME FREE ME

                    fprintf(stderr, "starting route iteration\n")
                    # We now iterate over all routes in the route_set, each route has an associated probability
                    route_prob_iter = route_set_prob.cbegin()
                    for route in deref(route_set):
                        prob = deref(route_prob_iter)
                        inc(route_prob_iter)

                        if prob == 0.0:
                            continue

                        # For each link in the route, we need to assign the appropriate demand * prob
                        # Because the link union is known to be sorted, if the links in the route are also sorted we can just step
                        # along both arrays simultaneously, skipping elements in the link_union when appropriate. This allows us
                        # to operate on the link loads as a sparse map and avoid blowing up memory usage when using a dense formulation.
                        # This is also incredibly cache efficient, the only downsides are that the code is harder to read
                        # and it requires sorting the route. NOTE: the sorting of routes is technically something that is already
                        # computed, during the computation of the link frequency we merge and sort all links, if we instead sorted
                        # then used an N-way merge we could reuse the sorted routes and the sorted link union.
                        links = new vector[long long](deref(route))  # we copy the links in case the routes haven't already been saved  # FIXME FREE ME
                        sort(links.begin(), links.end())

                        # links and link_union are sorted, and links is a subset of link_union
                        link_union_iter = link_union.cbegin()
                        link_iter = links.cbegin()

                        # fprintf(stderr, "starting link iteration\n")
                        while link_iter != links.cend():
                            # Find the next location for the current link in links
                            while deref(link_iter) != deref(link_union_iter):
                                inc(link_union_iter)

                            fprintf(stderr, "adding load of %f to link %d because link %d is in route\n", load, deref(link_union_iter), deref(link_iter))
                            deref(loads)[link_union_iter - link_union.cbegin()] = deref(loads)[link_union_iter - link_union.cbegin()] + prob

                            inc(link_iter)

                    deref(link_loads)[i] = loads
                    with gil:
                        print("path file:", origin_index, dest_index, deref(loads))

            for i in range(self.ods.size()):
                loads = deref(link_loads)[i]
                link_union = deref(self.link_union_set)[i]

                origin_index = self.nodes_to_indices_view[deref(self.ods)[i].first]
                dest_index = self.nodes_to_indices_view[deref(self.ods)[i].second]
                demand = matrix_view[origin_index, dest_index]

                for j in range(link_union.size()):
                    deref(collective_link_loads)[deref(link_union)[j]] = deref(collective_link_loads)[deref(link_union)[j]] + demand * deref(loads)[j]
            with gil:
                print("link loads:", deref(collective_link_loads))


    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef shared_ptr[libpa.CTable] make_table_from_results(
        vector[pair[long long, long long]] &ods,
        vector[RouteSet_t *] &route_sets,
        vector[vector[double] *] *cost_set,
        vector[vector[double] *] *gamma_set,
        vector[vector[double] *] *prob_set
    ):
        cdef:
            shared_ptr[libpa.CArray] paths
            shared_ptr[libpa.CArray] offsets

            libpa.CMemoryPool *pool = libpa.c_get_memory_pool()

            # Custom imports, these are declared in route_choice.pxd *not* libarrow.
            CUInt32Builder *path_builder = new CUInt32Builder(pool)
            CDoubleBuilder *cost_col = <CDoubleBuilder *>nullptr
            CDoubleBuilder *gamma_col = <CDoubleBuilder *>nullptr
            CDoubleBuilder *prob_col = <CDoubleBuilder *>nullptr

            libpa.CInt32Builder *offset_builder = new libpa.CInt32Builder(pool)  # Must be Int32 *not* UInt32
            libpa.CUInt32Builder *o_col = new libpa.CUInt32Builder(pool)
            libpa.CUInt32Builder *d_col = new libpa.CUInt32Builder(pool)
            vector[shared_ptr[libpa.CArray]] columns
            shared_ptr[libpa.CDataType] route_set_dtype = libpa.pyarrow_unwrap_data_type(RouteChoiceSet.route_set_dtype)

            libpa.CResult[shared_ptr[libpa.CArray]] route_set_results

            int offset = 0
            bint psl = (cost_set != nullptr and gamma_set != nullptr and prob_set != nullptr)

        # Origins, Destination, Route set, [Cost for route, Gamma for route, Probability for route]
        columns.resize(6 if psl else 3)

        if psl:
            cost_col = new CDoubleBuilder(pool)
            gamma_col = new CDoubleBuilder(pool)
            prob_col = new CDoubleBuilder(pool)

            for i in range(ods.size()):
                cost_col.AppendValues(deref(deref(cost_set)[i]))
                gamma_col.AppendValues(deref(deref(gamma_set)[i]))
                prob_col.AppendValues(deref(deref(prob_set)[i]))

        for i in range(ods.size()):
            route_set = route_sets[i]

            # Instead of construction a "list of lists" style object for storing the route sets we instead will construct one big array of link ids
            # with a corresponding offsets array that indicates where each new row (path) starts.
            for route in deref(route_set):
                o_col.Append(ods[i].first)
                d_col.Append(ods[i].second)

                offset_builder.Append(offset)
                path_builder.AppendValues(route.crbegin(), route.crend())

                offset += route.size()

        path_builder.Finish(&paths)

        offset_builder.Append(offset)  # Mark the end of the array in offsets
        offset_builder.Finish(&offsets)

        route_set_results = libpa.CListArray.FromArraysAndType(route_set_dtype, deref(offsets.get()), deref(paths.get()), pool, shared_ptr[libpa.CBuffer]())

        o_col.Finish(&columns[0])
        d_col.Finish(&columns[1])
        columns[2] = deref(route_set_results)

        if psl:
            cost_col.Finish(&columns[3])
            gamma_col.Finish(&columns[4])
            prob_col.Finish(&columns[5])

        cdef shared_ptr[libpa.CSchema] schema = libpa.pyarrow_unwrap_schema(RouteChoiceSet.psl_schema if psl else RouteChoiceSet.schema)
        cdef shared_ptr[libpa.CTable] table = libpa.CTable.MakeFromArrays(schema, columns)

        del path_builder
        del offset_builder
        del o_col
        del d_col

        if psl:
            del cost_col
            del gamma_col
            del prob_col

        return table

    def get_results(self):  # Cython doesn't like this type annotation... -> pa.Table:
        """
        :Returns:
            **route sets** (:obj:`pyarrow.Table`): Returns a table of OD pairs to lists of compact link IDs for
                each OD pair provided (as columns). Represents paths from ``origin`` to ``destination``. None if ``where`` was not None.
        """
        if self.results == nullptr or self.ods == nullptr:
            raise ValueError("Route Choice results not computed yet")

        table = libpa.pyarrow_wrap_table(
            RouteChoiceSet.make_table_from_results(
                deref(self.ods),
                deref(self.results),
                self.cost_set,
                self.gamma_set,
                self.prob_set
            )
        )

        return table


@cython.embedsignature(True)
cdef class Checkpoint:
    """
    A small wrapper class to write a dataset partition by partition
    """

    def __init__(self, where, schema, partition_cols = None):
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
