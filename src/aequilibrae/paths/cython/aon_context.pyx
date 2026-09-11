# cython: language_level=3
"""Reusable all-or-nothing assignment with optional skimming and select links."""

import operator
import numpy as np
from collections import namedtuple
from collections.abc import Mapping
cimport cython

from cython.parallel cimport prange, threadid
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from libcpp cimport bool as cpp_bool
from libcpp.algorithm cimport fill_n


from aequilibrae.utils.cython.array_allocations cimport array, const_array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view
from aequilibrae.paths.cython.aon_workspace cimport CppAoNWorkspace
from aequilibrae.paths.cython.dijkstra cimport RoutingContext, cpp_dijkstra, cpp_turn_dijkstra
from aequilibrae.paths.cython.graph_context cimport (
    GraphContext, NodeBasedContext, TurnBasedContext, CppNodeBasedContext, CppTurnBasedContext,
)
from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.search_results cimport (
    SearchResults, CppSearchResults, cpp_skim_fields, cpp_network_loading, cpp_sum_weighted_turn_costs,
    cpp_select_link_loading,
)


AoNOutputShape = namedtuple('AoNOutputShape', 'links, zones, classes, fields')

cdef class AoNOutputs:
    """Store link loads, selected trip demand, skims and turn costs for one run.

    Views are read-only and keep their buffers alive. Reuse or rotate outputs
    between runs; do not read an output while a run is writing it.
    """
    cdef double[:, :, ::1] skims_buffer
    cdef double[:, ::1] link_loads_buffer
    cdef double[:, :, ::1] select_link_loads_buffer
    cdef double[:, :, :, ::1] select_link_od_buffer
    cdef readonly tuple select_link_names
    cdef readonly size_t links, zones, classes, fields
    cdef readonly double turn_cost_total

    def __cinit__(self):
        self.link_loads_buffer = None
        self.skims_buffer = None
        self.select_link_loads_buffer = None
        self.select_link_od_buffer = None

    def __init__(self, links, zones, classes, fields, *, select_link_names=()):
        if self.link_loads_buffer is not None:
            raise RuntimeError("AoNOutputs cannot be reinitialized")
        links, zones, classes, fields = map(operator.index, (links, zones, classes, fields))
        if links < 0 or zones < 1 or classes < 1 or fields < 0:
            raise ValueError("output dimensions must be nonnegative, with at least one class and zone")

        self.select_link_names = tuple(select_link_names)
        if len(set(self.select_link_names)) != len(self.select_link_names):
            raise ValueError("select_link_names must be unique")

        self.links = links
        self.zones = zones
        self.classes = classes
        self.fields = fields

        self.link_loads_buffer = array[double]((links, classes), True, 0)
        if fields:
            self.skims_buffer = array[double]((zones, zones, fields), True, np.inf)

        if self.select_link_names:
            self.select_link_loads_buffer = array[double]((len(self.select_link_names), links, classes), True, 0)
            self.select_link_od_buffer = array[double]((len(self.select_link_names), zones, zones, classes), True, 0)

    @property
    def shape(self) -> AoNOutputShape:
        """Return the numbers of links, zones, demand classes and skim fields."""
        return AoNOutputShape(self.links, self.zones, self.classes, self.fields)

    @property
    def link_loads(self) -> np.ndarray:
        """Read-only loads for each link and demand class."""
        return readonly_view(self.link_loads_buffer)

    @property
    def skims(self) -> np.ndarray | None:
        """Read-only skims by origin, destination and field, or None."""
        if self.skims_buffer is not None:
            return readonly_view(self.skims_buffer)
        else:
            return None

    @property
    def select_link_loads(self):
        """Loads on all links of matching paths: [sets, links, classes], or None.

        The view is read-only and shares storage so reading it needs no copy.
        """
        return None if self.select_link_loads_buffer is None else readonly_view(self.select_link_loads_buffer)

    @property
    def select_link_od(self):
        """Matching trip demand: [sets, origins, destinations, classes], or None.

        The view is read-only and shares storage so reading it needs no copy.
        """
        return None if self.select_link_od_buffer is None else readonly_view(self.select_link_od_buffer)

    @property
    def total_turn_penalty(self):
        """Total turn cost weighted by demand."""
        return self.turn_cost_total

    def copy(self):
        """Return an independent copy of these outputs."""
        return self.copy_to(AoNOutputs(*self.shape, select_link_names=self.select_link_names))

    def copy_to(self, other: AoNOutputs):
        """Copy into matching output buffers and return them."""
        if not isinstance(other, AoNOutputs):
            raise TypeError("other must be of type AoNOutputs")

        if self.shape != other.shape:
            raise ValueError(f"output shapes do not match, got {other.shape}, expected {self.shape}")

        # Equal sizes are not enough: copying rows in a different set order
        # would give the caller valid numbers under the wrong names.
        if self.select_link_names != other.select_link_names:
            raise ValueError("output select-link shapes/names do not match (including set order)")

        if other is self:
            return other

        np.copyto(np.asarray(other.link_loads_buffer), np.asarray(self.link_loads_buffer))
        if self.skims_buffer is not None:
            np.copyto(np.asarray(other.skims_buffer), np.asarray(self.skims_buffer))

        if self.select_link_names:
            np.copyto(np.asarray(other.select_link_loads_buffer), np.asarray(self.select_link_loads_buffer))
            np.copyto(np.asarray(other.select_link_od_buffer), np.asarray(self.select_link_od_buffer))

        other.turn_cost_total = self.turn_cost_total

        return other


def copy_input(value, name, shape=None, dtype=np.float64):
    """Copy an input into a read-only, contiguous array with the requested shape."""
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    array.flags.writeable = False
    return array


def make_select_link_masks(selected_links, links):
    """Copy each named link set into a row of link flags.

    A trip matches a set when its path uses any link in that set. Flags give
    the tree pass a direct lookup without scanning a list for each state.
    Copying keeps later changes to the caller's sets out of assignment runs.
    """
    if selected_links is None:
        selected_links = {}
    if not isinstance(selected_links, Mapping):
        raise TypeError("selected_links must be a mapping of names to local link indices")

    masks = np.zeros((len(selected_links), links), dtype=np.bool_)

    for row, members in enumerate(selected_links.values()):
        for member in members:
            if isinstance(member, (bool, np.bool_)):
                raise TypeError("selected link indices must be integers, not booleans")

            link = operator.index(member)
            if not 0 <= link < links:
                raise ValueError("selected link index must be in [0, link_count)")

            masks[row, link] = True

    masks.flags.writeable = False
    return tuple(selected_links), masks


def choose_origins(origins, demand: np.ndarray, skimming: bool):
    """Validate an origin subset, or choose origins needed for demand and skims."""
    zones = demand.shape[0]
    if origins is None:
        if skimming:
            return np.arange(zones, dtype=np.uintp)
        return np.flatnonzero(np.any(demand != 0, axis=(1, 2))).astype(np.uintp)

    indices = np.asarray(origins)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("origins must be centroid indices")
    indices = np.sort(indices)

    if np.any(indices < 0) or np.any(indices >= zones):
        raise ValueError("origins must be centroid indices")
    if np.any(indices[1:] == indices[:-1]):
        raise ValueError("origins must be unique (workers write disjoint skim rows)")

    return np.asarray(indices, dtype=np.uintp)


def make_destination_masks(demand: np.ndarray, nodes: int, skimming: bool):
    """Build destination masks and counts, sharing one mask when skimming."""
    cdef cpp_bool[:, ::1] masks
    cdef Py_ssize_t origin, destination
    zones = demand.shape[0]

    # Skimming needs every centroid, so workers can share one mask.
    masks = array[cpp_bool]((1 if skimming else zones, nodes), True, False)
    if skimming:
        for destination in range(zones):
            masks[0, destination] = True
    else:
        destinations = np.any(demand != 0, axis=2)
        for origin in range(zones):
            for destination in range(zones):
                masks[origin, destination] = destinations[origin, destination]

    counts = np.count_nonzero(np.asarray(masks), axis=1).astype(np.uintp)
    counts.flags.writeable = False
    return masks, counts


cdef class PreparedAoN:
    """Prepare workers for repeated assignment runs on a fixed graph.

    Demand [zones, zones, classes] and link skim fields are copied once; costs
    are borrowed. Centroids are nodes 0..zones-1. Empty skim_fields skips skimming;
    skim_penalties selects fields that include turn costs. Use make_outputs()
    and run() for each iteration. selected_links maps names to directed-link
    indices in this context. Sets are copied once so later caller changes do
    not affect a run. A trip matches when it uses any link in a set; its demand
    is added to every link on its path. Do not change inputs or use these same
    objects elsewhere during a run.
    """
    cdef GraphContext context
    cdef list workers, link_fields

    cdef vector[CppSearchResults *] searches
    cdef vector[CppAoNWorkspace[double] *] workspaces
    cdef vector[const double *] field_pointers

    cdef const double[::1] costs_buffer
    cdef const double[:, :, ::1] demand_buffer
    cdef const cpp_bool[:, ::1] destination_masks_buffer
    cdef const size_t[::1] destination_counts_buffer, origins_buffer
    cdef const cpp_bool[::1] include_turn_costs
    cdef double[:, :, ::1] link_loads_buffer
    cdef double[::1] turn_costs_buffer
    cdef const cpp_bool[:, ::1] select_link_masks_buffer
    cdef double[:, :, :, ::1] select_link_loads_buffer
    cdef readonly tuple select_link_names
    cdef size_t select_link_count
    cdef size_t[:, ::1] blocked_heads
    cdef size_t zone_count, class_count, link_count, field_count
    cdef int cores
    cdef bint block_centroids

    def __cinit__(self):
        self.select_link_loads_buffer = None

    def __init__(
            self,
            context: NodeBasedContext | TurnBasedContext,
            demand,
            *,
            costs,
            cores=1,
            skim_fields=None,
            skim_penalties=None,
            block_centroids=False,
            origins=None,
            selected_links=None
    ):
        cdef const size_t[::1] context_heads
        cdef int core

        if self.context is not None:
            raise RuntimeError("PreparedAoN cannot be reinitialized")
        if not isinstance(context, (NodeBasedContext, TurnBasedContext)):
            raise TypeError("context must be a node or turn routing context")

        cores = operator.index(cores)
        if not 1 <= cores <= np.iinfo(np.int32).max:
            raise ValueError("cores must be positive and fit an OpenMP thread count")

        demand = copy_input(demand, "demand")
        if (demand.ndim != 3 or demand.shape[0] != demand.shape[1]
                or not 1 <= demand.shape[0] <= context.node_count or demand.shape[2] < 1):
            raise ValueError("demand must have shape (zones, zones, classes), zones <= node_count, classes >= 1")

        if block_centroids and isinstance(context, TurnBasedContext):
            raise ValueError("turn contexts must encode centroid blocking in their turn restrictions")

        # Copy fixed inputs once and retain their buffers.
        self.context = context
        self.cores = cores
        self.zone_count, self.class_count = demand.shape[0], demand.shape[2]
        self.link_count = context.link_count
        self.link_fields = [
            copy_input(field, "skim field", (self.link_count,))
            for field in (() if skim_fields is None else skim_fields)
        ]
        self.field_count = len(self.link_fields)
        self.select_link_names, self.select_link_masks_buffer = make_select_link_masks(selected_links, self.link_count)
        self.select_link_count = len(self.select_link_names)

        # Choose which skim fields include turn costs.
        if skim_penalties is None:
            self.include_turn_costs = array[cpp_bool](self.field_count, True, False)
        else:
            self.include_turn_costs = copy_input(skim_penalties, "skim_penalties", (self.field_count,), dtype=np.bool_)

        self.origins_buffer = choose_origins(origins, demand, self.field_count > 0)
        self.destination_masks_buffer, self.destination_counts_buffer = make_destination_masks(
            demand, context.node_count, self.field_count > 0
        )

        self.demand_buffer = demand

        # Worker totals are cleared at the start of each run.
        self.link_loads_buffer = array[double]((cores, self.link_count, self.class_count), True, 0)
        self.turn_costs_buffer = array[double](cores, True, 0)
        # Origins can load the same link at the same time. Give each worker
        # its own loads so workers do not need locks on each addition.
        if self.select_link_count:
            self.select_link_loads_buffer = array[double](
                (cores, self.select_link_count, self.link_count, self.class_count), True, 0
            )

        self.block_centroids = block_centroids and self.link_count > 0
        if self.block_centroids:
            self.blocked_heads = array[size_t]((cores, self.link_count), False, 0)
            context_heads = context.heads
            for core in range(cores):
                self.blocked_heads[core, :] = context_heads
        self.prepare_workers()
        self.update_costs(costs)

    cdef void prepare_workers(self) except *:
        """Allocate each worker's search results and scratch buffers."""
        cdef SearchResults worker
        cdef const double[::1] field

        self.workers = []
        self.searches.clear()
        self.workspaces.clear()
        self.field_pointers.clear()

        for _ in range(self.cores):
            worker = SearchResults(self.context)
            if self.field_count:
                worker.workspace.prepare_skims(self.field_count)
            worker.workspace.prepare_loading(self.class_count)
            if self.select_link_count:
                # Allocate now so the origin loop can run without Python setup.
                worker.workspace.prepare_select_links()

            self.workers.append(worker)
            self.searches.push_back(&worker.cpp)
            self.workspaces.push_back(&worker.workspace.cpp)

        for field in self.link_fields:
            self.field_pointers.push_back(const_array_pointer(field))

    @property
    def shape(self) -> AoNOutputShape:
        """Return the output dimensions required by this assignment."""
        return AoNOutputShape(self.link_count, self.zone_count, self.class_count, self.field_count)

    @property
    def origins(self):
        """Read-only indices of the origins processed by each run."""
        return readonly_view(self.origins_buffer)

    @property
    def destination_masks(self):
        """Read-only destination masks: one shared row or one row per origin."""
        return readonly_view(self.destination_masks_buffer)

    @property
    def destination_counts(self):
        """Number of requested destinations in each mask."""
        return readonly_view(self.destination_counts_buffer)

    @property
    def costs(self):
        """Read-only view of the current link costs, without copying."""
        return readonly_view(self.costs_buffer)

    @property
    def thread_outputs(self):
        """Read-only link loads and turn totals for each worker."""
        return readonly_view(self.link_loads_buffer), readonly_view(self.turn_costs_buffer)

    @property
    def select_link_masks(self):
        """Read-only link flags [sets, links], in select_link_names order."""
        return readonly_view(self.select_link_masks_buffer)

    @property
    def thread_select_link_loads(self):
        """Read-only worker loads [workers, sets, links, classes], or None."""
        return None if self.select_link_loads_buffer is None else readonly_view(self.select_link_loads_buffer)

    def update_costs(self, costs):
        """Retain new link costs without copying their data.

        Requires aligned, contiguous float64 values, one per link, with no NaN or
        negative costs. Changes between runs are visible; changes during a run
        are unsafe. Invalid inputs leave the previous buffer in place.
        """
        cdef const double[::1] view
        if costs is None:
            raise TypeError("costs must be a contiguous float64 buffer")
        view = costs
        if <size_t>view.shape[0] != self.link_count:
            raise ValueError("costs must have one value per context link")
        values = np.asarray(view)
        if not values.flags.aligned:
            raise ValueError("costs must be aligned")
        if np.any(np.isnan(values)) or np.any(values < 0):
            raise ValueError("costs must be nonnegative and must not contain NaN")
        if (np.shares_memory(values, self.link_loads_buffer) or np.shares_memory(values, self.turn_costs_buffer)
                or (self.select_link_count and np.shares_memory(values, self.select_link_loads_buffer))):
            raise ValueError("costs must not overlap worker scratch")
        self.costs_buffer = view

    def make_outputs(self):
        """Allocate matching output buffers for the caller to reuse."""
        return AoNOutputs(*self.shape, select_link_names=self.select_link_names)

    def run(self, AoNOutputs out not None):
        """Run assignment, overwrite out and return it without keeping a reference.

        Unreachable or skipped skims are infinity; searched diagonals are zero.
        Do not use this assignment or its output concurrently.
        """
        if out.shape != self.shape or out.select_link_names != self.select_link_names:
            raise ValueError("output shape does not match assignment shape (including select-link names/order)")

        # Clearing outputs must not erase costs that the searches still need.
        # Do this check before changing any output or worker buffer.
        if (np.shares_memory(self.costs_buffer, out.link_loads_buffer)
                or (self.field_count and np.shares_memory(self.costs_buffer, out.skims_buffer))
                or (self.select_link_count and (
                    np.shares_memory(self.costs_buffer, out.select_link_loads_buffer)
                    or np.shares_memory(self.costs_buffer, out.select_link_od_buffer)))):
            raise ValueError("costs must not overlap output buffers")

        if self.link_count:
            fill_n(&self.link_loads_buffer[0, 0, 0], <size_t>self.link_loads_buffer.size, 0)
        fill_n(&self.turn_costs_buffer[0], <size_t>self.turn_costs_buffer.size, 0)

        if self.select_link_count:
            if self.link_count:
                fill_n(&self.select_link_loads_buffer[0, 0, 0, 0], <size_t>self.select_link_loads_buffer.size, 0)
            # Skipped origins are not visited below. Clear their OD rows too,
            # so a reused output cannot retain trips from an earlier run.
            fill_n(&out.select_link_od_buffer[0, 0, 0, 0], <size_t>out.select_link_od_buffer.size, 0)

        if self.field_count:
            fill_n(&out.skims_buffer[0, 0, 0], <size_t>out.skims_buffer.size, <double>np.inf)

        if isinstance(self.context, NodeBasedContext):
            with nogil:
                assign_origins[NodeBasedContext](<NodeBasedContext>self.context, self, out)
        else:
            with nogil:
                assign_origins[TurnBasedContext](<TurnBasedContext>self.context, self, out)

        np.sum(np.asarray(self.link_loads_buffer), axis=0, out=np.asarray(out.link_loads_buffer))
        out.turn_cost_total = float(np.sum(np.asarray(self.turn_costs_buffer)))
        if self.select_link_count:
            np.sum(np.asarray(self.select_link_loads_buffer), axis=0, out=np.asarray(out.select_link_loads_buffer))

        return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void search_origin(
    RoutingContext context,
    size_t origin,
    const double *costs,
    size_t zones,
    size_t *blocked_heads,
    CppSearchResults &search
) noexcept nogil:
    """Find paths from one origin using the current costs and centroid blocking."""
    cdef size_t link
    cdef CppNodeBasedContext node_graph
    cdef CppTurnBasedContext turn_graph

    if RoutingContext is NodeBasedContext:
        node_graph = context.view()
        node_graph.costs = costs

        if blocked_heads != NULL:
            for link in range(node_graph.fs[zones]):
                blocked_heads[link] = origin
            for link in range(node_graph.fs[origin], node_graph.fs[origin + 1]):
                blocked_heads[link] = node_graph.heads[link]

            node_graph.heads = blocked_heads

        cpp_dijkstra[FourAryHeap](node_graph, origin, search)
    else:
        turn_graph = context.view()
        turn_graph.graph.costs = costs

        cpp_turn_dijkstra[FourAryHeap](turn_graph, origin, search)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void assign_origin(
    RoutingContext context,
    PreparedAoN prepared,
    AoNOutputs out,
    size_t origin,
    int thread_index,
) noexcept nogil:
    """Find paths, write skims and load demand for one origin."""
    cdef CppSearchResults search = prepared.searches[thread_index][0]
    cdef CppAoNWorkspace[double] *workspace = prepared.workspaces[thread_index]

    cdef:
        size_t zones = prepared.zone_count
        size_t classes = prepared.class_count
        size_t fields = prepared.field_count
        size_t mask_row = 0 if fields else origin
        size_t selection

        const double *demand = &prepared.demand_buffer[origin, 0, 0]
        double *skims = &out.skims_buffer[origin, 0, 0] if fields else NULL
        double *link_loads = &prepared.link_loads_buffer[thread_index, 0, 0] if prepared.link_count else NULL

    search.destination_mask = &prepared.destination_masks_buffer[mask_row, 0]
    search.destination_count = prepared.destination_counts_buffer[mask_row]

    search_origin(
        context,
        origin,
        const_array_pointer(prepared.costs_buffer),
        zones,
        &prepared.blocked_heads[thread_index, 0] if prepared.block_centroids else NULL,
        search
    )

    if fields:
        cpp_skim_fields[double](search, zones, prepared.field_pointers.data(), fields, workspace[0], skims)

    cpp_network_loading[double](search, zones, demand, classes, workspace[0], link_loads)

    # Reuse the same scratch for each set to keep memory use down. Each origin
    # runs on just one worker, so OD can go straight into its output row without
    # locks or a separate OD cube for every worker.
    for selection in range(prepared.select_link_count):
        cpp_select_link_loading[double](
            search, zones, demand, classes,
            &prepared.select_link_masks_buffer[selection, 0] if prepared.link_count else NULL,
            workspace[0], &out.select_link_od_buffer[selection, origin, 0, 0],
            &prepared.select_link_loads_buffer[thread_index, selection, 0, 0] if prepared.link_count else NULL,
        )

    if RoutingContext is TurnBasedContext:
        prepared.turn_costs_buffer[thread_index] += cpp_sum_weighted_turn_costs[double](
            search,
            zones,
            demand,
            classes,
            &prepared.include_turn_costs[0] if fields else NULL,
            fields,
            skims
        )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void assign_origins(RoutingContext context, PreparedAoN prepared, AoNOutputs out) noexcept nogil:
    """Assign the chosen origins across the prepared workers."""
    cdef Py_ssize_t origin_index

    for origin_index in prange(prepared.origins_buffer.shape[0], num_threads=prepared.cores, schedule="guided"):
        assign_origin(context, prepared, out, prepared.origins_buffer[origin_index], threadid())


def aon_parallel_context(matrix, graph, result, aux_result, cores, bridge=None):
    """Run the legacy Graph adapter; use PreparedAoN for repeated runs."""
    from aequilibrae.paths.cython.aon_graph import aon_parallel_context as legacy

    return legacy(matrix, graph, result, aux_result, cores, bridge)
