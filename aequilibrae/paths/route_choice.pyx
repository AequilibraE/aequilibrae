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
directly into the output ccp. This is done allow declaring an the `()` operator, which is required and AFAIK not
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

from libc.math cimport INFINITY
from libc.string cimport memcpy
from libc.limits cimport UINT_MAX
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref
from cython.parallel cimport parallel, prange, threadid

import numpy as np
from typing import List, Tuple

# It would really be nice if these were modules. The 'include' syntax is long deprecated and adds a lot to compilation times
include 'basic_path_finding.pyx'

cdef class RouteChoiceSet:
    """
    Route choice implemented via breadth first search with link removal (BFS-LE) as described in Rieser-Schüssler,
    Balmer, and Axhausen, 'Route Choice Sets for Very High-Resolution Data'
    """

    def __cinit__(self):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""
        pass

    def __init__(self, graph: Graph):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        # self.heuristic = HEURISTIC_MAP[self.res._heuristic]
        self.cost_view = graph.compact_cost
        self.graph_fs_view = graph.compact_fs
        self.b_nodes_view = graph.compact_graph.b_node.values  # FIXME: Why does path_computation copy this?
        self.nodes_to_indices_view = graph.compact_nodes_to_indices
        tmp = graph.lonlat_index.loc[graph.compact_all_nodes]
        self.lat_view = tmp.lat.values
        self.lon_view = tmp.lon.values
        self.ids_graph_view = graph.compact_graph.id.values
        self.num_nodes = graph.compact_num_nodes

    def __dealloc__(self):
        """
        C level deallocation. For freeing memory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """
        pass

    def run(self, origin: int, destination: int, max_routes: int = 0, max_depth: int = 0, seed: int = 0):
        return self.batched([(origin, destination)], max_routes=max_routes, max_depth=max_depth, seed=seed)[(origin, destination)]

    # Bounds checking doesn't really need to be disabled here but the warning is annoying
    @cython.boundscheck(False)
    def batched(self, ods: List[Tuple[int, int]], max_routes: int = 0, max_depth: int = 0, seed: int = 0, cores: int = 1):
        cdef:
            vector[pair[long long, long long]] c_ods = ods

            # A* (and Dijkstra's) require memory views, so we must allocate here and take slices. Python can handle this memory
            double [:, :] cost_matrix = np.empty((cores, self.cost_view.shape[0]), dtype=float)
            long long [:, :] predecessors_matrix = np.empty((cores, self.num_nodes + 1), dtype=np.int64)
            long long [:, :] conn_matrix = np.empty((cores, self.num_nodes + 1), dtype=np.int64)

            vector[RouteSet_t *] *results = new vector[RouteSet_t *](len(ods))
            long long origin_index, dest_index, i

        if max_routes == 0 and max_depth == 0:
            raise ValueError("Either `max_routes` or `max_depth` must be >= 0")

        if max_routes < 0 or max_depth < 0 or cores < 0:
            raise ValueError("`max_routes`, `max_depth`, and `cores` must be non-negative")

        cdef:
            unsigned int c_max_routes = max_routes
            unsigned int c_max_depth = max_depth
            unsigned int c_seed = seed
            unsigned int c_cores = cores
            long long o, d

        for o, d in ods:
            if self.nodes_to_indices_view[o] == -1:
                raise ValueError(f"Origin {o} is not present within the compact graph")
            if self.nodes_to_indices_view[d] == -1:
                raise ValueError(f"Destination {d} is not present within the compact graph")

        with nogil, parallel(num_threads=c_cores):
            for i in prange(c_ods.size()):
                origin_index = self.nodes_to_indices_view[c_ods[i].first]
                dest_index = self.nodes_to_indices_view[c_ods[i].second]
                deref(results)[i] = RouteChoiceSet.generate_route_set(
                    self,
                    origin_index,
                    dest_index,
                    c_max_routes,
                    c_max_depth,
                    cost_matrix[threadid()],
                    predecessors_matrix[threadid()],
                    conn_matrix[threadid()],
                    c_seed
                )

        # Build results into python objects using Cythons auto-conversion from vector[long long] to list then to tuple (for set operations)
        res = []
        for result in deref(results):
            links = []
            for route in deref(result):
                links.append(tuple(deref(route)))
                del route
            res.append(links)

        del results
        return dict(zip(ods, res))

    def _generate_line_strins(self, project, graph, results):
        """Debug method"""
        import geopandas as gpd
        import shapely

        links = project.network.links.data.set_index("link_id")
        df = []
        for od, route_set in results.items():
            for route in route_set:
                df.append(
                    (
                        *od,
                        shapely.MultiLineString(
                            links.loc[
                                graph.graph[graph.graph.__compressed_id__.isin(route)].link_id
                            ].geometry.to_list()
                        ),
                    )
                )

        df = gpd.GeoDataFrame(df, columns=["origin", "destination", "geometry"])
        df.set_geometry("geometry")
        df.to_file("test1.gpkg", layer='routes', driver="GPKG")

    cdef void path_find(RouteChoiceSet self, long origin_index, long dest_index, double [:] thread_cost, long long [:] thread_predecessors, long long [:] thread_conn) noexcept nogil:
        path_finding_a_star(
            origin_index,
            dest_index,
            thread_cost,
            self.b_nodes_view,
            self.graph_fs_view,
            self.nodes_to_indices_view,
            self.lat_view,
            self.lon_view,
            thread_predecessors,
            self.ids_graph_view,
            thread_conn,
            EQUIRECTANGULAR  # FIXME: enum import failing due to redefinition
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.initializedcheck(False)
    cdef RouteSet_t *generate_route_set(
        RouteChoiceSet self,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
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

                RouteChoiceSet.path_find(self, origin_index, dest_index, thread_cost, thread_predecessors, thread_conn)

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
