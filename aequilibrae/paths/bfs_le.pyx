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

Current front runner: Hash/tree set of removed link with prefix code to path set trie. Whether its worth incrementally
building the trie or not should be tested.

"""

from aequilibrae import Graph
from aequilibrae.paths.results import PathResults
from libc.stdio cimport printf, fprintf, stderr
from libc.math cimport INFINITY

import numpy as np

# It would really be nice if these were modules. The 'include' syntax is long deprecated
include 'basic_path_finding.pyx'

cpdef float cube(float x):
    return x * x * x

cdef class RouteChoice:
    """
    Route choice implemented via breadth first search with link removal (BFS-LE) as described in Rieser-Schüssler,
    Balmer, and Axhausen, 'Route Choice Sets for Very High-Resolution Data'
    """

    # cdef int num
    def __cinit__(self):
        """C level init. For C memory allocation and initalisation. Called exactly once per object."""
        print("cinit called")


    def __init__(self, graph: Graph):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        print("init called")
        self.res = PathResults()
        self.res.prepare(graph)

        # self.heuristic = HEURISTIC_MAP[self.res._heuristic]
        self.cost_view = graph.cost
        self.graph_fs_view = graph.fs
        self.b_nodes_view = graph.graph.b_node.values  # FIXME: Why does path_computation copy this?
        self.nodes_to_indices_view = graph.nodes_to_indices
        self.lat_view = graph.lonlat_index.lat.values
        self.lon_view = graph.lonlat_index.lon.values
        self.predecessors_view = self.res.predecessors
        self.ids_graph_view = graph.graph.id.values
        self.conn_view = self.res.connectors

    def __dealloc__(self):
        """
        C level deallocation. For freeing memeory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """
        print("Deallocating!")

    cdef void c_helloworld(RouteChoice self) noexcept nogil:
        printf("Hello world\n")

    cpdef helloworld(self):
        with nogil:
            RouteChoice.c_helloworld(self)

    def run(self, origin, destination, max_depth=0):
        cdef:
            long origin_index = self.nodes_to_indices_view[origin]
            long dest_index = self.nodes_to_indices_view[destination]
            unsigned int c_max_depth = max_depth
        with nogil:
            RouteChoice.generate_route_set(self, origin_index, dest_index, c_max_depth)


    cdef void generate_route_set(RouteChoice self, long origin_index, long dest_index, unsigned int max_depth) noexcept nogil:
        """Main method for route set generation, thread safe."""
        cdef long connector

        for depth in range(max_depth):
            path_finding_a_star(
                origin_index,
                dest_index,
                self.cost_view,
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
            if self.predecessors_view[dest_index] >= 0:
                printf("%ld is still reachable from %ld at depth %d\n", dest_index, origin_index, depth)
                connector = self.conn_view[dest_index]
                self.cost_view[connector] = INFINITY
                printf("%ld has been banned\n", connector)

                with gil:
                    p = n = dest_index
                    all_connectors = []
                    if p != origin_index:
                        while p != origin_index:
                            p = self.predecessors_view[p]
                            connector = self.conn_view[n]
                            all_connectors.append(connector + 1)
                            n = p
                    print("(", ", ".join(str(x) for x in all_connectors[::-1]), ")")

            else:
                printf("%ld is unreachable from %ld at depth %d\n", dest_index, origin_index, depth)
                break
