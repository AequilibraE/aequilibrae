# cython: language_level=3
"""Reusable all-or-nothing assignment with optional skimming and select links."""

import operator
from collections import namedtuple

import numpy as np
cimport cython

from cython.parallel cimport prange, threadid
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from libcpp cimport bool as cpp_bool
from libcpp.algorithm cimport fill_n

from aequilibrae.utils.cython.array_allocations cimport array
from aequilibrae.utils.cython.array_allocations import readonly_view
from aequilibrae.paths.cython.workspaces cimport AoNWorkspace, CppAoNWorkspace
from aequilibrae.paths.cython.dijkstra cimport RoutingContext, cpp_dijkstra, cpp_turn_dijkstra
from aequilibrae.paths.cython.context cimport (
    CppSelectLinkContext, CppSkimmingContext, GraphContext, NodeBasedContext,
    SelectLinkContext, SkimmingContext, TurnBasedContext,
)
from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults, CppMutableSearchResults
from aequilibrae.paths.cython.queries cimport CppLoadingQuery, CppSearchQuery
from aequilibrae.paths.cython.skimming cimport cpp_skimming
from aequilibrae.paths.cython.network_loading cimport (
    cpp_network_loading, cpp_reduce_loading_outputs, cpp_sum_weighted_turn_costs,
)
from aequilibrae.paths.cython.outputs cimport (
    LoadingOutputs, CppLoadingOutputs, SkimmingOutputs, SelectLinkOutputs,
    SelectLinkLoadingOutputs, CppSelectLinkLoadingOutputsView, CppSelectLinkODOriginView,
)
from aequilibrae.paths.cython.outputs import _validate_skim_names, _validate_selection_names
from aequilibrae.paths.cython.select_link_loading cimport (
    cpp_select_link_loading, cpp_reduce_select_link_loading_outputs,
)


AoNOutputShape = namedtuple('AoNOutputShape', 'links, zones, classes, fields')


cdef class AoNOutputs:
    """Store link loads, selected trip demand, skims and turn costs for one run.

    Views are read-only and keep their buffers alive. Reuse or rotate outputs
    between runs; do not read an output while a run is writing it.
    """
    cdef readonly SkimmingOutputs skimming
    cdef readonly tuple skim_names
    cdef readonly LoadingOutputs loading
    cdef readonly SelectLinkOutputs select_link
    cdef readonly tuple select_link_names
    cdef readonly size_t links, zones, classes, fields
    cdef readonly double turn_cost_total

    def __init__(
            self,
            links,
            zones,
            classes,
            *,
            skim_names=(),
            select_link_names=(),
            select_link_loads=True,
            select_link_od=True
    ):
        if self.loading is not None:
            raise RuntimeError("AoNOutputs cannot be reinitialized")

        links, zones, classes = map(operator.index, (links, zones, classes))
        if links < 0 or zones < 1 or classes < 1:
            raise ValueError("output dimensions must be nonnegative, with at least one class and zone")

        self.skim_names = _validate_skim_names(skim_names)
        fields = len(self.skim_names)

        self.select_link_names = _validate_selection_names(select_link_names)

        self.links = links
        self.zones = zones
        self.classes = classes
        self.fields = fields

        self.loading = LoadingOutputs(links, classes)
        if fields:
            self.skimming = SkimmingOutputs(zones, zones, self.skim_names)

        if self.select_link_names:
            self.select_link = SelectLinkOutputs(
                links, zones, classes, self.select_link_names, origin_count=zones,
                link_loads=select_link_loads, od=select_link_od,
            )

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
        """Read-only skims by origin, field and destination, or None."""
        return None if self.skimming is None else self.skimming.skims

    @property
    def select_link_loads(self):
        """Read-only full-path loads [sets, links, classes], or None."""
        return (None if self.select_link is None or self.select_link.loading is None
                else self.select_link.loading.link_loads)

    @property
    def select_link_od(self):
        """Read-only matching demand [origins, sets, destinations, classes], or None."""
        return (None if self.select_link is None or self.select_link.od is None
                else self.select_link.od.demand)

    @property
    def total_turn_penalty(self):
        """Total turn cost weighted by demand."""
        return self.turn_cost_total


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

    Demand [zones, zones, classes] and costs are borrowed. Skimming inputs are
    supplied as an independent SkimmingContext; None or no fields skips skimming.
    Centroids are nodes 0..zones-1. Use make_outputs() and run() for each
    iteration. selected_links is an independent SelectLinkContext. Selected OD
    and link outputs can be enabled separately at setup. A trip matches when
    it uses any link in a set; its demand is added to every link on its path.
    Do not change inputs or use these same
    objects elsewhere during a run. Origin selection and target masks are fixed
    at setup, so changes to demand must not introduce new search targets.
    """
    cdef GraphContext context
    cdef const cpp_bool[:, ::1] destination_masks_buffer
    cdef const double[:, :, ::1] demand_buffer
    cdef const size_t[::1] destination_counts_buffer
    cdef const size_t[::1] origins_buffer
    cdef double[::1] turn_costs_buffer
    cdef int cores
    cdef list workers, worker_workspaces, worker_loading_outputs, worker_selected_outputs
    cdef SkimmingContext skim_context
    cdef CppSkimmingContext[double] skim_inputs
    cdef readonly tuple select_link_names
    cdef SelectLinkContext selection_context
    cdef CppSelectLinkContext selection_inputs
    cdef bint load_selected_links, write_selected_od
    cdef size_t select_link_count
    cdef size_t zone_count, class_count, link_count, field_count
    cdef vector[CppAoNWorkspace[double]] workspaces
    cdef vector[CppMutableSearchResults] searches
    cdef vector[CppLoadingOutputs[double]] loading_outputs
    cdef vector[CppSelectLinkLoadingOutputsView[double]] selected_outputs

    def __init__(
            self,
            context: NodeBasedContext | TurnBasedContext,
            demand,
            *,
            costs,
            cores=1,
            SkimmingContext skimming=None,
            block_centroids=False,
            origins=None,
            SelectLinkContext selected_links=None,
            select_link_loads=True,
            select_link_od=True,
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

        if skimming is not None:
            if skimming.link_count != self.link_count:
                raise ValueError("skimming link_count does not match routing context")

            # The view borrows field pointers. Keep their owner alive across
            # repeated runs instead of preparing another pointer table here.
            self.skim_context = skimming
            self.skim_inputs = skimming.view()

        self.field_count = self.skim_inputs.field_count
        if selected_links is not None:
            if selected_links.link_count != self.link_count:
                raise ValueError("selection link_count does not match routing context")
            self.selection_context = selected_links
            self.selection_inputs = selected_links.view()

        self.select_link_names = () if selected_links is None else selected_links.set_names
        self.select_link_count = self.selection_inputs.set_count
        self.load_selected_links = self.select_link_count > 0 and bool(select_link_loads)
        self.write_selected_od = self.select_link_count > 0 and bool(select_link_od)

        self.origins_buffer = choose_origins(origins, demand, self.field_count > 0)
        self.destination_masks_buffer, self.destination_counts_buffer = make_destination_masks(
            demand, context.node_count, self.field_count > 0
        )

        self.demand_buffer = demand

        # Worker totals are cleared at the start of each run.
        self.turn_costs_buffer = array[double](cores, True, 0)

        self.prepare_workers()

    cdef void prepare_workers(self) except *:
        """Allocate each worker's search results and scratch buffers."""
        cdef SearchResults worker
        cdef AoNWorkspace workspace
        cdef LoadingOutputs loading
        cdef SelectLinkLoadingOutputs selected

        self.workers = []
        self.worker_workspaces = []
        self.worker_loading_outputs = []
        self.worker_selected_outputs = []
        self.searches.clear()
        self.workspaces.clear()
        self.loading_outputs.clear()
        self.selected_outputs.clear()

        # Objective and turn-only skims read labels directly. Only link fields
        # need a scratch column for each search state.
        skim_width = None
        if self.skim_inputs.needs_state_sums():
            skim_width = self.skim_inputs.additive_field_count

        for _ in range(self.cores):
            worker = SearchResults(self.context.node_count, self.context.state_count, self.link_count)
            workspace = AoNWorkspace(
                self.context.state_count,
                class_count=self.class_count,
                field_count=skim_width,
                select_links=self.load_selected_links or self.write_selected_od,
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

            if self.load_selected_links:
                # OD-only analysis allocates no worker link accumulators.
                selected = SelectLinkLoadingOutputs(
                    self.link_count, self.class_count, self.select_link_names
                )
                self.worker_selected_outputs.append(selected)
                self.selected_outputs.push_back(selected.view())

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

    def update_costs(self, costs):
        """Retain new link costs without copying their data.

        Requires aligned, contiguous float64 values, one per link, with no NaN or
        negative costs. Changes between runs are visible; changes during a run
        are unsafe. Invalid inputs leave the previous buffer in place.
        """
        self.context.update_costs(costs)

    def make_outputs(self):
        """Allocate matching output buffers for the caller to reuse."""
        return AoNOutputs(
            self.link_count, self.zone_count, self.class_count,
            skim_names=() if self.skim_context is None else self.skim_context.field_names,
            select_link_names=self.select_link_names,
            select_link_loads=self.load_selected_links, select_link_od=self.write_selected_od,
        )

    def run(self, AoNOutputs out not None):
        """Run assignment, overwrite out and return it without keeping a reference.

        Unreachable or skipped skims are infinity; searched diagonals are zero.
        Do not use this assignment or its output concurrently.
        """
        cdef size_t worker

        # Check names as well as dimensions before clearing any caller output.
        skim_names = () if self.skim_context is None else self.skim_context.field_names
        if (
            out.shape != self.shape
            or out.select_link_names != self.select_link_names
            or out.skim_names != skim_names
        ):
            raise ValueError("output shape and ordered names must match assignment")

        if self.select_link_count:
            if ((out.select_link.loading is not None) != self.load_selected_links
                    or (out.select_link.od is not None) != self.write_selected_od):
                raise ValueError("selected output components must match assignment configuration")

        for worker in range(self.loading_outputs.size()):
            self.loading_outputs[worker].reset()

        for worker in range(self.selected_outputs.size()):
            self.selected_outputs[worker].reset()

        fill_n(&self.turn_costs_buffer[0], <size_t>self.turn_costs_buffer.size, 0)

        if self.write_selected_od:
            # Reset skipped OD rows too. Link totals are reset by reduction.
            out.select_link.od.reset()

        if self.field_count:
            # A subset run never visits the other origins. Clear their old
            # values too, so they remain marked as missing in reused outputs.
            out.skimming.reset()

        cdef bint node_based = isinstance(self.context, NodeBasedContext)

        with nogil:
            if node_based:
                assign_origins[NodeBasedContext](<NodeBasedContext>self.context, self, out)
            else:
                assign_origins[TurnBasedContext](<TurnBasedContext>self.context, self, out)

            cpp_reduce_loading_outputs[double](
                self.loading_outputs.data(), self.loading_outputs.size(), out.loading.view()
            )

            if self.load_selected_links:
                cpp_reduce_select_link_loading_outputs[double](
                    self.selected_outputs.data(), self.selected_outputs.size(),
                    out.select_link.loading.view(),
                )

        out.turn_cost_total = float(np.sum(np.asarray(self.turn_costs_buffer)))

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
    cdef CppSelectLinkLoadingOutputsView[double] selected_loads
    cdef CppSelectLinkODOriginView[double] selected_od

    cdef:
        size_t zones = prepared.zone_count
        size_t classes = prepared.class_count
        size_t fields = prepared.field_count
        size_t mask_row = 0 if fields else origin

        const double *demand = &prepared.demand_buffer[origin, 0, 0]

    search_query.node_count = search.node_count
    search_query.origin = origin
    search_query.target_count = prepared.destination_counts_buffer[mask_row]
    if search_query.target_count:
        search_query.target_mask = &prepared.destination_masks_buffer[mask_row, 0]

    search_origin(context, search_query, writable_search)

    if fields:
        # Only this worker writes this origin's contiguous output block.
        # The view borrows that block without packing a temporary.
        cpp_skimming[double](
            search,
            prepared.skim_inputs,
            workspace[0].skimming,
            out.skimming.view().origin(origin),
        )

    loading_query.destination_count = zones
    loading_query.class_count = classes
    loading_query.demand = demand

    cpp_network_loading[double](
        search, loading_query, workspace[0].loading, prepared.loading_outputs[thread_index]
    )

    # Only the origin's worker writes this OD block. Link accumulators stay
    # worker-local until reduction. Default views omit either output.
    if prepared.load_selected_links:
        selected_loads = prepared.selected_outputs[thread_index]
    if prepared.write_selected_od:
        selected_od = out.select_link.od.view().origin(origin)
    cpp_select_link_loading[double](
        search, loading_query, prepared.selection_inputs,
        workspace[0].select_link, workspace[0].loading, selected_loads, selected_od,
    )

    # Demand-weighted turn totals are independent of which skims were requested.
    prepared.turn_costs_buffer[thread_index] += cpp_sum_weighted_turn_costs[double](
        search, loading_query
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void assign_origins(RoutingContext context, PreparedAoN prepared, AoNOutputs out) noexcept nogil:
    """Assign the chosen origins across the prepared workers."""
    cdef Py_ssize_t origin_index

    for origin_index in prange(prepared.origins_buffer.shape[0], num_threads=prepared.cores, schedule="guided"):
        assign_origin(context, prepared, out, prepared.origins_buffer[origin_index], threadid())
