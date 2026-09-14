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
from aequilibrae.paths.cython.workspaces cimport AoNWorkspace, CppAoNWorkspace
from aequilibrae.paths.cython.dijkstra cimport RoutingContext, cpp_dijkstra, cpp_turn_dijkstra
from aequilibrae.paths.cython.graph_context cimport (
    GraphContext, NodeBasedContext, TurnBasedContext,
)
from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults, CppMutableSearchResults
from aequilibrae.paths.cython.queries cimport CppLoadingQuery, CppSearchQuery
from aequilibrae.paths.cython.skimming cimport cpp_skim_fields, cpp_sum_weighted_turn_costs
from aequilibrae.paths.cython.network_loading cimport cpp_network_loading, cpp_reduce_loading_outputs
from aequilibrae.paths.cython.outputs cimport LoadingOutputs, CppLoadingOutputs
from aequilibrae.paths.cython.select_link_loading cimport cpp_select_link_loading


AoNOutputShape = namedtuple('AoNOutputShape', 'links, zones, classes, fields')

cdef class AoNOutputs:
    """Store link loads, selected trip demand, skims and turn costs for one run.

    Views are read-only and keep their buffers alive. Reuse or rotate outputs
    between runs; do not read an output while a run is writing it.
    """
    cdef double[:, :, ::1] skims_buffer
    cdef readonly LoadingOutputs loading
    cdef double[:, :, ::1] select_link_loads_buffer
    cdef double[:, :, :, ::1] select_link_od_buffer
    cdef readonly tuple select_link_names
    cdef readonly size_t links, zones, classes, fields
    cdef readonly double turn_cost_total

    def __cinit__(self):
        self.skims_buffer = None
        self.select_link_loads_buffer = None
        self.select_link_od_buffer = None

    def __init__(self, links, zones, classes, fields, *, select_link_names=()):
        if self.loading is not None:
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

        self.loading = LoadingOutputs(links, classes)
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
        return self.loading.link_loads

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


def copy_input(value, name, shape=None, dtype=np.float64):
    """Copy an input into a read-only, contiguous array with the requested shape."""
    array = np.array(value, dtype=dtype, order="C", copy=True)

    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    array.flags.writeable = False
    return array


def borrow_input(value, name, shape=None):
    """Keep large numeric inputs alive without packing or copying them."""
    values = np.asarray(memoryview(value))

    if values.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64")
    if not values.flags.c_contiguous or not values.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    return values


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

    Demand [zones, zones, classes], link skim fields and costs are borrowed.
    Centroids are nodes 0..zones-1. Empty skim_fields skips skimming;
    skim_penalties selects fields that include turn costs. Use make_outputs()
    and run() for each iteration. selected_links maps names to directed-link
    indices in this context. Sets are copied once so later caller changes do
    not affect a run. A trip matches when it uses any link in a set; its demand
    is added to every link on its path. Do not change inputs or use these same
    objects elsewhere during a run. Origin selection and target masks are fixed
    at setup, so changes to demand must not introduce new search targets.
    """
    cdef GraphContext context
    cdef const cpp_bool[:, ::1] destination_masks_buffer
    cdef const cpp_bool[:, ::1] select_link_masks_buffer
    cdef const cpp_bool[::1] include_turn_costs
    cdef const double[:, :, ::1] demand_buffer
    cdef const size_t[::1] destination_counts_buffer
    cdef const size_t[::1] origins_buffer
    cdef double[:, :, :, ::1] select_link_loads_buffer
    cdef double[::1] turn_costs_buffer
    cdef int cores
    cdef list workers, worker_workspaces, worker_loading_outputs, link_fields
    cdef readonly tuple select_link_names
    cdef size_t select_link_count
    cdef size_t zone_count, class_count, link_count, field_count
    cdef vector[CppAoNWorkspace[double]] workspaces
    cdef vector[CppMutableSearchResults] searches
    cdef vector[CppLoadingOutputs[double]] loading_outputs
    cdef vector[const double *] field_pointers

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
        if self.context is not None:
            raise RuntimeError("PreparedAoN cannot be reinitialized")
        if not isinstance(context, (NodeBasedContext, TurnBasedContext)):
            raise TypeError("context must be a node or turn routing context")

        cores = operator.index(cores)
        if not 1 <= cores <= np.iinfo(np.int32).max:
            raise ValueError("cores must be positive and fit an OpenMP thread count")

        demand = borrow_input(demand, "demand")
        if (demand.ndim != 3 or demand.shape[0] != demand.shape[1]
                or not 1 <= demand.shape[0] <= context.node_count or demand.shape[2] < 1):
            raise ValueError("demand must have shape (zones, zones, classes), zones <= node_count, classes >= 1")

        # Independent cost bindings share topology, not routing scratch.
        self.context = context.with_costs(costs)
        if block_centroids:
            self.context.blocked_centroid_count = demand.shape[0]
        self.cores = cores
        self.zone_count, self.class_count = demand.shape[0], demand.shape[2]
        self.link_count = context.link_count
        self.link_fields = [
            borrow_input(field, "skim field", (self.link_count,))
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
        self.turn_costs_buffer = array[double](cores, True, 0)
        # Origins can load the same link at the same time. Give each worker
        # its own loads so workers do not need locks on each addition.
        if self.select_link_count:
            self.select_link_loads_buffer = array[double](
                (cores, self.select_link_count, self.link_count, self.class_count), True, 0
            )

        self.prepare_workers()

    cdef void prepare_workers(self) except *:
        """Allocate each worker's search results and scratch buffers."""
        cdef SearchResults worker
        cdef AoNWorkspace workspace
        cdef LoadingOutputs loading
        cdef const double[::1] field

        self.workers = []
        self.worker_workspaces = []
        self.worker_loading_outputs = []
        self.searches.clear()
        self.workspaces.clear()
        self.loading_outputs.clear()
        self.field_pointers.clear()

        for _ in range(self.cores):
            worker = SearchResults(self.context.node_count, self.context.state_count, self.link_count)
            workspace = AoNWorkspace(
                self.context.state_count,
                class_count=self.class_count,
                field_count=self.field_count if self.field_count else None,
                select_links=self.select_link_count > 0,
            )
            loading = LoadingOutputs(self.link_count, self.class_count)

            self.workers.append(worker)
            self.worker_workspaces.append(workspace)
            self.worker_loading_outputs.append(loading)
            # These views borrow fixed allocations. Retain their Cython owners
            # separately; do not take pointers to temporary view() return values.
            self.searches.push_back(worker.view())
            self.workspaces.push_back(workspace.view())
            self.loading_outputs.push_back(loading.view())

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
        return self.context.costs

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
        self.context.update_costs(costs)

    def make_outputs(self):
        """Allocate matching output buffers for the caller to reuse."""
        return AoNOutputs(*self.shape, select_link_names=self.select_link_names)

    def run(self, AoNOutputs out not None):
        """Run assignment, overwrite out and return it without keeping a reference.

        Unreachable or skipped skims are infinity; searched diagonals are zero.
        Do not use this assignment or its output concurrently.
        """
        cdef size_t worker
        if out.shape != self.shape or out.select_link_names != self.select_link_names:
            raise ValueError("output shape does not match assignment shape (including select-link names/order)")

        with nogil:
            for worker in range(self.loading_outputs.size()):
                self.loading_outputs[worker].reset()
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

        with nogil:
            cpp_reduce_loading_outputs[double](
                self.loading_outputs.data(), self.loading_outputs.size(), out.loading.view()
            )
        out.turn_cost_total = float(np.sum(np.asarray(self.turn_costs_buffer)))
        if self.select_link_count:
            np.sum(np.asarray(self.select_link_loads_buffer), axis=0, out=np.asarray(out.select_link_loads_buffer))

        return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void search_origin(
    RoutingContext context,
    const CppSearchQuery &query,
    CppMutableSearchResults search
) noexcept nogil:
    """Use the same complete search inputs as the standalone entry point."""
    if RoutingContext is NodeBasedContext:
        cpp_dijkstra[FourAryHeap](context.view(), query, search)
    else:
        cpp_turn_dijkstra[FourAryHeap](context.view(), query, search)


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
    cdef CppMutableSearchResults writable_search = prepared.searches[thread_index]
    cdef CppSearchResults search = writable_search.read_view()
    cdef CppAoNWorkspace[double] *workspace = &prepared.workspaces[thread_index]
    cdef CppSearchQuery search_query
    cdef CppLoadingQuery[double] loading_query
    cdef CppLoadingOutputs[double] selected_output

    cdef:
        size_t zones = prepared.zone_count
        size_t classes = prepared.class_count
        size_t fields = prepared.field_count
        size_t mask_row = 0 if fields else origin
        size_t selection

        const double *demand = &prepared.demand_buffer[origin, 0, 0]
        double *skims = &out.skims_buffer[origin, 0, 0] if fields else NULL

    search_query.node_count = search.node_count
    search_query.origin = origin
    search_query.target_count = prepared.destination_counts_buffer[mask_row]
    if search_query.target_count:
        search_query.target_mask = &prepared.destination_masks_buffer[mask_row, 0]
    search_origin(context, search_query, writable_search)

    if fields:
        cpp_skim_fields[double](search, zones, prepared.field_pointers.data(), fields, workspace[0].skimming, skims)

    loading_query.destination_count = zones
    loading_query.class_count = classes
    loading_query.demand = demand
    cpp_network_loading[double](search, loading_query, workspace[0].loading, prepared.loading_outputs[thread_index])

    # Reuse the same scratch for each set to keep memory use down. Each origin
    # runs on just one worker, so OD can go straight into its output row without
    # locks or a separate OD cube for every worker.
    selected_output.link_count = prepared.link_count
    selected_output.class_count = classes
    for selection in range(prepared.select_link_count):
        selected_output.link_loads = (
            &prepared.select_link_loads_buffer[thread_index, selection, 0, 0] if prepared.link_count else NULL
        )
        cpp_select_link_loading[double](
            search, loading_query,
            &prepared.select_link_masks_buffer[selection, 0] if prepared.link_count else NULL,
            workspace[0].select_link, workspace[0].loading,
            &out.select_link_od_buffer[selection, origin, 0, 0], selected_output,
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
