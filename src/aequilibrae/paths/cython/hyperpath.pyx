"""
An implementation of Spiess and Florian's hyperpath generating algorithm.

reference: Spiess, H. and Florian, M. (1989). Optimal strategies: A new
assignment model for transit networks. Transportation Research Part B 23(2),
83-102.

The label-setting and loading kernels are in ``paths/cpp/hyperpath.hpp``. This
file drives them over destinations in parallel.
"""

cimport cython
import numpy as np
from libc.math cimport INFINITY
from libc.stdint cimport int64_t, uint8_t, uint32_t
from libcpp.vector cimport vector

from cython.parallel import prange, threadid

from aequilibrae.utils.cython.bridge cimport Bridge
from aequilibrae.utils.cython.bar cimport Bar

DATATYPE_PY = np.float64

# Raises every edge cost to at least this many seconds, so labels strictly increase along attractive edges. It is
# well above the spacing of doubles around the largest labels in practice (a few days in seconds).
COST_FLOOR_PY = 1.0e-9

cdef extern from "hyperpath.hpp" namespace "aequilibrae::paths::cpp::hyperpath" nogil:
    cdef enum SkimOperator:
        SKIM_ADD
        SKIM_ANY_WAIT
        SKIM_RESET_AT_WAIT
        SKIM_COUNT_WAITS

    cdef cppclass CppGraph "aequilibrae::paths::cpp::hyperpath::Graph":
        size_t vertex_count
        size_t edge_count
        const uint32_t *csc_indptr
        const uint32_t *csc_edges
        const uint32_t *tail
        const uint32_t *head

    cdef cppclass CppCosts "aequilibrae::paths::cpp::hyperpath::Costs":
        const double *cost
        const double *frequency
        const double *wait_constant
        const double *headway_fraction
        double cost_floor

    cdef cppclass CppSkims "aequilibrae::paths::cpp::hyperpath::Skims":
        size_t field_count
        const double *edge_values
        const uint8_t *operators
        const uint8_t *waits

    cdef cppclass CppWorkspace "aequilibrae::paths::cpp::hyperpath::Workspace":
        CppWorkspace(size_t vertex_count, size_t edge_count, size_t field_count) except +
        vector[double] label
        vector[double] skims

    void compute_labels(
        const CppGraph &graph, const CppCosts &costs, const CppSkims &skims, uint32_t destination, CppWorkspace &ws
    ) noexcept
    double load(
        const CppGraph &graph,
        const CppCosts &costs,
        const uint32_t *origins,
        const double *demand,
        size_t count,
        double *edge_volume,
        double *unassigned,
        CppWorkspace &ws,
    ) noexcept

SKIM_ADD_PY = SKIM_ADD
SKIM_ANY_WAIT_PY = SKIM_ANY_WAIT
SKIM_RESET_AT_WAIT_PY = SKIM_RESET_AT_WAIT
SKIM_COUNT_WAITS_PY = SKIM_COUNT_WAITS


cdef class _Workspaces:
    """Per-thread kernel scratch, kept between assignments of the same graph."""

    cdef vector[CppWorkspace *] items
    cdef size_t vertex_count
    cdef size_t edge_count
    cdef size_t field_count

    cdef void ensure(self, size_t threads, size_t vertex_count, size_t edge_count, size_t field_count) except *:
        if (vertex_count, edge_count, field_count) != (self.vertex_count, self.edge_count, self.field_count):
            self.clear()
            self.vertex_count, self.edge_count, self.field_count = vertex_count, edge_count, field_count
        while self.items.size() < threads:
            self.items.push_back(new CppWorkspace(vertex_count, edge_count, field_count))

    cdef void clear(self) noexcept:
        cdef CppWorkspace *ws
        for ws in self.items:
            del ws
        self.items.clear()

    def __dealloc__(self):
        self.clear()


def csc_structure(tail: np.ndarray, head: np.ndarray, vertex_count: int):
    """Return the incoming-edge offsets per vertex and the edges ordered by head, keeping edge order within a head."""
    counts = np.bincount(head, minlength=vertex_count)
    indptr = np.zeros(vertex_count + 1, dtype=np.uint32)
    np.cumsum(counts, out=indptr[1:])
    return indptr, np.argsort(head, kind="stable").astype(np.uint32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def assign_destinations(
    const uint32_t[::1] csc_indptr,
    const uint32_t[::1] csc_edges,
    const uint32_t[::1] tail,
    const uint32_t[::1] head,
    const double[::1] cost,
    const double[::1] frequency,
    const double[::1] wait_constant,
    const double[::1] headway_fraction,
    double cost_floor,
    const double[:, ::1] skim_values,
    const uint8_t[::1] skim_operators,
    const uint8_t[::1] skim_waits,
    const uint32_t[::1] destinations,
    const int64_t[::1] destination_zones,
    const uint32_t[::1] demand_indptr,
    const uint32_t[::1] demand_origins,
    const double[::1] demand_values,
    double[::1] unassigned,
    double[:, ::1] block_volumes,
    double[::1] block_waits,
    const int64_t[::1] zone_origins,
    double[:, :, ::1] skim_matrix,
    bint skim_labels,
    double[::1] labels,
    int64_t labels_task,
    _Workspaces workspaces,
    int threads,
    Bridge bridge,
):
    """Compute the strategy towards each destination, skim it and load its demand.

    Destinations are split into one contiguous block per row of ``block_volumes``, and each block loads into its own
    row, so results depend on the number of blocks but not on which thread runs which block.
    """
    cdef:
        CppGraph graph
        CppCosts costs
        CppSkims skims
        size_t vertex_count = labels.shape[0]
        size_t edge_count = tail.shape[0]
        size_t field_count = skim_operators.shape[0]
        size_t tasks = destinations.shape[0]
        size_t zones = zone_origins.shape[0]
        size_t offset = 1 if skim_labels else 0
        int blocks = block_volumes.shape[0]
        int block  # openmp on windows requires iterator variable have signed type
        size_t task, first, last, z, k, v, start, end
        int64_t zone, origin
        double weighted_wait, label
        CppWorkspace *ws
        Bar bar

    graph.vertex_count = vertex_count
    graph.edge_count = edge_count
    graph.csc_indptr = &csc_indptr[0]
    graph.csc_edges = &csc_edges[0] if edge_count else NULL
    graph.tail = &tail[0] if edge_count else NULL
    graph.head = &head[0] if edge_count else NULL

    costs.cost = &cost[0] if edge_count else NULL
    costs.frequency = &frequency[0] if edge_count else NULL
    costs.wait_constant = &wait_constant[0] if vertex_count else NULL
    costs.headway_fraction = &headway_fraction[0] if vertex_count else NULL
    costs.cost_floor = cost_floor

    skims.field_count = field_count
    skims.edge_values = &skim_values[0, 0] if field_count and edge_count else NULL
    skims.operators = &skim_operators[0] if field_count else NULL
    skims.waits = &skim_waits[0] if field_count else NULL

    workspaces.ensure(threads, vertex_count, edge_count, field_count)
    bar = bridge.new_bar("{}/{} destinations processed", total=tasks)

    with nogil:
        for block in prange(blocks, schedule="static", chunksize=1, num_threads=threads):
            ws = workspaces.items[threadid()]
            first = <size_t>block * tasks // blocks
            last = (<size_t>block + 1) * tasks // blocks
            weighted_wait = 0.0
            for task in range(first, last):
                if bridge.should_stop():
                    break

                compute_labels(graph, costs, skims, destinations[task], ws[0])

                zone = destination_zones[task]
                if zone >= 0:
                    for z in range(zones):
                        origin = zone_origins[z]
                        if origin < 0:
                            continue  # the matrix starts at inf
                        label = ws.label[origin]
                        if not label < INFINITY:
                            continue  # unreachable, also left at inf
                        if skim_labels:
                            skim_matrix[z, zone, 0] = label
                        for k in range(field_count):
                            skim_matrix[z, zone, k + offset] = ws.skims[origin * field_count + k]

                start = demand_indptr[task]
                end = demand_indptr[task + 1]
                if end > start:
                    weighted_wait = weighted_wait + load(
                        graph,
                        costs,
                        &demand_origins[start],
                        &demand_values[start],
                        end - start,
                        &block_volumes[block, 0],
                        &unassigned[start],
                        ws[0],
                    )

                if <int64_t>task == labels_task:
                    for v in range(vertex_count):
                        labels[v] = ws.label[v]
                bar.inc()
            block_waits[block] = weighted_wait
