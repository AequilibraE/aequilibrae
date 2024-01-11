# cython: language_level=3str
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

"""This module aims to implemented the BFS-LE algorithm as described in Rieser-Schüssler, Balmer, and Axhausen,
'Route Choice Sets for Very High-Resolution Data'.

A rough overview of the algorithm is as follows.
    1. Prepare the initial graph, this is depth 0 with no links removed.
    2. Find a short path, P. If P is empty stop, else add P to the path set.
    3. For all links p in P, remove p from E, compounding with the previously removed links.
    4. De-duplicate the sub-graphs, we only care about unique sub-graphs.
    5. Go to 2.

Thoughts:

    - Path set: One issue is that the path set can't be stored as a simple sub-graph. This is because it is the union of
      multiple paths. The union of two partially separate paths may create paths that are no in the paths
      themselves. Instead I believe we can store the paths as (compressed) tries, operating on the common prefix
      links/nodes. Each branch in the trie would encode a branch in the choice of route. This has the benefit of being
      very small to store and iterating over all choices is some traversal of the tree. Downsides of this are, insertion
      and search scale with the length of the paths. Each lookup requires a linear time search of the existing tree. As
      each link in the path is removed the number of branches scales with the length of the path times the degree of the
      vertices.

      Another option is hash maps, just throw hash maps at it. Upsides of this is they are much more generic, no special
      methods required and we'll likely be able to use something off the shelf. Downsides are that their performance is
      largely dependent on the hash function. We'll need to use the set of removed edges as the key, which the path as
      the value. This means we can't compresse the paths. Choosing a good hash function may be tough, because link/node
      ids can be arbitrarily large we'll have to consider overflows, though an non-naive function should handle this
      fine. We'll also want to avoid modulo, wikipedia says using a multiply-shift scheme with a Mersenne prime like
      2^61 - 1 should work well. Although we have variable length paths, fixed length vector hashing can be applied and
      padded to blocks of our paths.

    - Removed link set: This set suffers from the similar issues as the path set as order doesn't matter. I think a hash
      or tree set is about the only way to go about this. Since order doesn't matter a trie doesn't make sense but a
      tree set using sorted node/link ids could work.

      Another option is a bit map. For a million link network, the bitmap would "only" be 125kB. Membership checks and
      addition and removal from this set would be incredily fast, faster than anything else, however comparison may
      suffer. We could hash this bitmap but if we're hashing it we might as well just hash the removal set.

      We could also nest the hash sets, essientially building up a hash set of previously seen hashes.

    - Hash functions: We're looking for a "incremental (multi)set hash function", because we don't need it to be secure
      at all some commutative binary operation should work e.g. +, *, ^. Whether or not they make good hash functions remains to be
      seen.

    - Graph modification: Actually removing a link from the graph would require modification of the existing
      methods. Instead we can just require that the cost of a like is float or double and set it to INFINITY, that way
      the algorithms will never consider the link as viable.

Current front runner: Hash/tree set of removed link with index to a noe in the path set suffix trie. Whether its worth incrementally
building the trie or not should be tested.

"""

from aequilibrae import Graph
from aequilibrae.paths.results import PathResults
from libc.math cimport INFINITY
from libc.string cimport memcpy
from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

import numpy as np

# It would really be nice if these were modules. The 'include' syntax is long deprecated and adds a lot to compilation times
include 'basic_path_finding.pyx'

cdef class RouteChoice:
    """
    Route choice implemented via breadth first search with link removal (BFS-LE) as described in Rieser-Schüssler,
    Balmer, and Axhausen, 'Route Choice Sets for Very High-Resolution Data'
    """

    # cdef int num
    def __cinit__(self):
        """C level init. For C memory allocation and initalisation. Called exactly once per object."""

    def __init__(self, graph: Graph):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        self.res = PathResults()
        self.res.prepare(graph)

        # self.heuristic = HEURISTIC_MAP[self.res._heuristic]
        self.cost_view = graph.cost
        self.graph_fs_view = graph.fs
        self.b_nodes_view = graph.graph.b_node.values  # FIXME: Why does path_computation copy this?
        self.nodes_to_indices_view = graph.nodes_to_indices
        self.lat_view = graph.lonlat_index.lat.values
        self.lon_view = graph.lonlat_index.lon.values
        self.predecessors_view = np.empty(graph.num_nodes + 1, dtype=np.int64)
        self.ids_graph_view = graph.graph.id.values
        self.conn_view = np.empty(graph.num_nodes + 1, dtype=np.int64)

    def __dealloc__(self):
        """
        C level deallocation. For freeing memeory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """
        pass

    def run(self, origin, destination, max_routes=0, max_depth=0):
        cdef:
            long origin_index = self.nodes_to_indices_view[origin]
            long dest_index = self.nodes_to_indices_view[destination]
            unsigned int c_max_routes = max_routes
            unsigned int c_max_depth = max_depth
            double [:] scratch_cost = np.empty(self.cost_view.shape[0])  # allocation of new memory view required gil
            unordered_set[vector[long long] *] *results
            unordered_map[unordered_set[long long] *, vector[long long] *].const_iterator results_iter
        with nogil:
            results = RouteChoice.generate_route_set(self, origin_index, dest_index, c_max_routes, c_max_depth, scratch_cost)

        res = []
        for x in deref(results):
            res.append(list(deref(x)))
            del x

        return res

    cdef void path_find(RouteChoice self, long origin_index, long dest_index, double [:] scratch_cost) noexcept nogil:
        path_finding_a_star(
            origin_index,
            dest_index,
            scratch_cost,
            self.b_nodes_view,
            self.graph_fs_view,
            self.nodes_to_indices_view,
            self.lat_view,
            self.lon_view,
            self.predecessors_view,
            self.ids_graph_view,
            self.conn_view,
            EQUIRECTANGULAR  # FIXME: enum import failing due to redefinition
        )

    cdef unordered_set[vector[long long] *] *generate_route_set(RouteChoice self, long origin_index, long dest_index, unsigned int max_routes, unsigned int max_depth, double [:] scratch_cost) noexcept nogil:
        """Main method for route set generation"""
        cdef:
            unordered_map[unordered_set[long long] *, vector[long long] *] *working_set
            unordered_map[unordered_set[long long] *, vector[long long] *].const_iterator set_iter
            vector[unordered_set[long long] *] queue
            vector[unordered_set[long long] *] next_queue
            unordered_set[long long] *banned
            unordered_set[long long] *new_banned
            unordered_set[vector[long long] *] *route_set
            vector[long long] *vec
            pair[unordered_set[long long] *, vector[long long] *] *pair_tmp
            pair[unordered_set[long long] *, vector[long long] *] x
            long long p, connector

        queue.push_back(new unordered_set[long long]()) # Start with no edges banned
        working_set = new unordered_map[unordered_set[long long] *, vector[long long] *]()
        route_set = new unordered_set[vector[long long] *]()

        # We'll go at most `max_depth` iterations down, at each depth we maintain a deque of the next set of banned edges to consider
        for depth in range(max_depth):
            for banned in queue:
                memcpy(&scratch_cost[0], &self.cost_view[0], self.cost_view.shape[0] * sizeof(double))

                for connector in deref(banned):
                    scratch_cost[connector] = INFINITY

                RouteChoice.path_find(self, origin_index, dest_index, scratch_cost)

                vec = new vector[long long]()
                if self.predecessors_view[dest_index] >= 0:
                    # Walk the predecessors tree to find our path, we build it up in a cpp vector because we can't know how long it'll be
                    p = dest_index
                    while p != origin_index:
                        connector = self.conn_view[p]
                        p = self.predecessors_view[p]
                        vec.push_back(connector)

                    # This element is already in our set, skip it
                    if route_set.find(vec) != route_set.end():
                        continue

                    pair_tmp = new pair[unordered_set[long long] *, vector[long long] *](banned, vec)
                    working_set.insert(deref(pair_tmp))  # We'll keep this route around for (potential) insertion later
                    del pair_tmp

                else:
                    pass  # Node was unreachable

            printf("%ld + %ld >= %d\n", working_set.size(), route_set.size(), max_routes)
            if working_set.size() + route_set.size() >= max_routes:
                printf("route set full (or close to full)\n")
                # Randomly insert enough elements to fill our route_set, we don't need to add anything else to our queue since we're done now
                # FIXME: Horrible assumption, never do this
                # The iteration order of the working set is probably close enough to random, let's just grab the first max_routes - route_set.size() elements
                printf("yeeting %ld entries\n", working_set.size() - (max_routes - route_set.size()))

                # Skipp the elements we wish to save
                set_iter = working_set.cbegin()
                for _ in range(max_routes - route_set.size()):
                    inc(set_iter)

                # Destrory the remaining elements
                working_set.erase(set_iter, working_set.cend())

                # From the top, add what's left
                for x in deref(working_set):
                    vec = x.second
                    route_set.insert(vec)

                break
            else:
                printf("route set not full (or close)\n")
                for x in deref(working_set):
                    banned = x.first
                    vec = x.second

                    route_set.insert(vec)
                    # Copy the previously banned links, then for each vector in the path we add one and push it onto our queue
                    for edge in deref(vec):
                        new_banned = new unordered_set[long long](deref(banned))
                        new_banned.insert(edge)
                        next_queue.push_back(new_banned)

            queue.swap(next_queue)
            next_queue.clear()
            working_set.clear()

        # We may have added more banned link sets to the queue then found out we hit the max depth, we should free those
        for banned in queue:
            del banned

        return route_set
