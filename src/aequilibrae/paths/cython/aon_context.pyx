# cython: language_level=3
"""Assignment orchestration over independent routing and operation components."""

import operator

import numpy as np
cimport cython

from cython.parallel cimport prange, threadid
from libcpp.vector cimport vector
from libcpp cimport bool as cpp_bool

from aequilibrae.utils.cython.array_allocations cimport array
from aequilibrae.utils.cython.array_allocations import readonly_view
from aequilibrae.paths.cython.workspaces cimport AoNWorkspace
from aequilibrae.paths.cython.dijkstra cimport cpp_dijkstra, cpp_turn_dijkstra
from aequilibrae.paths.cython.context cimport (
    GraphContext,
    NodeBasedContext,
    TurnBasedContext,
    SelectLinkContext,
    SkimmingContext,
)
from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.skimming cimport cpp_skimming
from aequilibrae.paths.cython.network_loading cimport (
    cpp_network_loading,
    cpp_reduce_loading_outputs,
    cpp_sum_weighted_turn_costs,
)
from aequilibrae.paths.cython.outputs cimport (
    AoNOutputs,
    CppAoNOutputsView,
    LoadingOutputs,
    SelectLinkLoadingOutputs,
    CppSelectLinkODOriginView,
)
from aequilibrae.paths.cython.select_link_loading cimport (
    cpp_select_link_loading,
    cpp_reduce_select_link_loading_outputs,
)


def borrow_input(value, name):
    """Retain packed demand without copying or changing caller writeability."""
    values = np.asarray(memoryview(value))

    if values.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64")
    if not values.flags.c_contiguous or not values.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")

    return values


def choose_origins(origins, demand: np.ndarray, skimming: bool):
    """Choose origins once; each OD output row must have just one writer."""
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
        raise ValueError("origins must be unique (workers write disjoint OD rows)")

    return np.asarray(indices, dtype=np.uintp)


def make_destination_masks(demand: np.ndarray, nodes: int, skimming: bool):
    """Prepare fixed targets, sharing one all-centroid mask when skimming."""
    cdef cpp_bool[:, ::1] masks
    cdef Py_ssize_t origin, destination

    zones = demand.shape[0]
    mask_rows = 1 if skimming else zones
    masks = array[cpp_bool]((mask_rows, nodes), True, False)

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


cdef class _AoNWorker:
    """Keep one worker's owners together for the lifetime of their cached views."""

    cdef SearchResults search
    cdef AoNWorkspace workspace
    cdef LoadingOutputs loading
    cdef SelectLinkLoadingOutputs selected_loading

    def __init__(
        self,
        GraphContext context,
        class_count,
        skim_width,
        selection_names,
        load_selected_links,
        write_selected_od,
    ):
        self.search = SearchResults(context.node_count, context.state_count, context.link_count)
        self.workspace = AoNWorkspace(
            context.state_count,
            class_count=class_count,
            field_count=skim_width,
            select_links=load_selected_links or write_selected_od,
        )
        self.loading = LoadingOutputs(context.link_count, class_count)
        if load_selected_links:
            self.selected_loading = SelectLinkLoadingOutputs(
                context.link_count, class_count, selection_names,
            )

    cdef CppAoNWorkerView view(self) noexcept nogil:
        cdef CppAoNWorkerView worker

        worker.search = self.search.view()
        worker.workspace = self.workspace.view()
        worker.loading = self.loading.view()

        if self.selected_loading is not None:
            worker.selected_loading = self.selected_loading.view()

        return worker


cdef class PreparedAoN:
    """Reuse assignment workers with a caller-configured routing context.

    Demand [zones, zones, classes], topology, restrictions, origins and targets
    stay fixed. Only routing costs change between iterations: update the supplied
    context, or write valid costs into its borrowed buffer, before run(). The
    driver retains that context directly and refreshes its view each run.

    Skimming and selection inputs are independent owners with fixed configuration
    and buffer bindings. Ordinary loading and weighted turn accounting always run.
    This is not a general multi-origin executor.
    Do not mutate inputs or use a worker or output concurrently with a run.
    """

    cdef GraphContext context
    cdef const double[:, :, ::1] demand_buffer
    cdef const cpp_bool[:, ::1] destination_masks_buffer
    cdef const size_t[::1] destination_counts_buffer, origins_buffer
    cdef SkimmingContext skim_context
    cdef SelectLinkContext selection_context
    cdef bint load_selected_links, write_selected_od
    cdef size_t zone_count, class_count
    cdef int cores

    cdef list workers
    cdef vector[CppAoNOrigin] origin_queries
    cdef vector[CppAoNWorkerView] worker_views
    cdef CppAoNInputs inputs

    # The standalone reductions require contiguous tables of component views.
    # These tables borrow the same worker owners; they allocate no load buffers.
    cdef vector[CppLoadingOutputs[double]] loading_outputs
    cdef vector[CppSelectLinkLoadingOutputsView[double]] selected_outputs

    def __init__(
        self,
        context,
        demand,
        *,
        cores=1,
        SkimmingContext skimming=None,
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
        if (
            demand.ndim != 3
            or demand.shape[0] != demand.shape[1]
            or not 1 <= demand.shape[0] <= context.node_count
            or demand.shape[2] < 1
        ):
            raise ValueError("demand must have shape (zones, zones, classes), zones <= node_count, classes >= 1")

        if skimming is not None and skimming.link_count != context.link_count:
            raise ValueError("skimming link_count does not match routing context")
        if selected_links is not None and selected_links.link_count != context.link_count:
            raise ValueError("selection link_count does not match routing context")

        self.context = context
        self.cores = cores
        self.demand_buffer = demand
        self.zone_count = demand.shape[0]
        self.class_count = demand.shape[2]

        self.skim_context = skimming
        self.selection_context = selected_links

        if skimming is not None:
            self.inputs.skimming = skimming.view()
        if selected_links is not None:
            self.inputs.selection = selected_links.view()

        self.load_selected_links = self.inputs.selection.set_count > 0 and bool(select_link_loads)
        self.write_selected_od = self.inputs.selection.set_count > 0 and bool(select_link_od)

        self.origins_buffer = choose_origins(origins, demand, self.inputs.skimming.field_count > 0)
        self.destination_masks_buffer, self.destination_counts_buffer = make_destination_masks(
            demand, context.node_count, self.inputs.skimming.field_count > 0,
        )

        self.__prepare_origins()
        self.__prepare_workers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void __prepare_origins(self) except *:
        """Bind each origin's demand and targets once, without Python row owners."""
        cdef size_t origin, mask_row
        cdef CppAoNOrigin query

        for origin in self.origins_buffer:
            query = CppAoNOrigin()
            mask_row = 0 if self.inputs.skimming.field_count else origin
            query.search.node_count = self.context.node_count
            query.search.origin = origin
            query.search.target_count = self.destination_counts_buffer[mask_row]
            if query.search.target_count:
                query.search.target_mask = &self.destination_masks_buffer[mask_row, 0]

            # An explicitly requested origin with no demand keeps the full-search
            # query. Automatic origin selection omits these rows without skims.
            query.loading.destination_count = self.zone_count
            query.loading.class_count = self.class_count
            query.loading.demand = &self.demand_buffer[origin, 0, 0]
            self.origin_queries.push_back(query)

        self.inputs.origins = self.origin_queries.data()
        self.inputs.origin_count = self.origin_queries.size()

    cdef void __prepare_workers(self) except *:
        """Allocate only the scratch and accumulators required by each worker."""
        cdef _AoNWorker worker
        cdef CppAoNWorkerView view

        # Label-only skimming reads search labels and needs no state-sum buffer.
        skim_width = None
        if self.inputs.skimming.needs_state_sums():
            skim_width = self.inputs.skimming.additive_field_count

        selection_names = ()
        if self.selection_context is not None:
            selection_names = self.selection_context.set_names

        self.workers = []
        for _ in range(self.cores):
            worker = _AoNWorker(
                self.context,
                self.class_count,
                skim_width,
                selection_names,
                self.load_selected_links,
                self.write_selected_od,
            )
            self.workers.append(worker)

            view = worker.view()
            self.worker_views.push_back(view)
            self.loading_outputs.push_back(view.loading)
            if self.load_selected_links:
                self.selected_outputs.push_back(view.selected_loading)

    @property
    def origins(self):
        return readonly_view(self.origins_buffer)

    @property
    def destination_masks(self):
        return readonly_view(self.destination_masks_buffer)

    @property
    def destination_counts(self):
        return readonly_view(self.destination_counts_buffer)

    def make_outputs(self):
        """Allocate a matching group of outputs, caller must hold on to it."""
        skim_names = () if self.skim_context is None else self.skim_context.field_names
        selection_names = () if self.selection_context is None else self.selection_context.set_names

        return AoNOutputs(
            self.context.link_count,
            self.zone_count,
            self.class_count,
            skim_names=skim_names,
            select_link_names=selection_names,
            select_link_loads=self.load_selected_links,
            select_link_od=self.write_selected_od,
        )

    cdef void validate_outputs(self, AoNOutputs out) except *:
        """Check actual components before clearing caller output or worker totals."""
        if (
            out.loading.link_count != self.context.link_count
            or out.loading.class_count != self.class_count
        ):
            raise ValueError("loading output dimensions must match assignment")

        if (out.skimming is not None) != (self.inputs.skimming.field_count > 0):
            raise ValueError("skimming output presence must match assignment")

        if out.skimming is not None:
            if (
                out.skimming.origin_count != self.zone_count
                or out.skimming.destination_count != self.zone_count
                or out.skimming.field_names != self.skim_context.field_names
            ):
                raise ValueError("skimming output dimensions and ordered names must match assignment")

        if (out.select_link is not None) != (self.load_selected_links or self.write_selected_od):
            raise ValueError("selected output components must match assignment configuration")

        if out.select_link is None:
            return

        if (
            (out.select_link.loading is not None) != self.load_selected_links
            or (out.select_link.od is not None) != self.write_selected_od
        ):
            raise ValueError("selected output components must match assignment configuration")

        if self.load_selected_links:
            if (
                out.select_link.loading.link_count != self.context.link_count
                or out.select_link.loading.class_count != self.class_count
                or out.select_link.loading.set_names != self.selection_context.set_names
            ):
                raise ValueError("selected loading dimensions and ordered names must match assignment")

        if self.write_selected_od:
            if (
                out.select_link.od.origin_count != self.zone_count
                or out.select_link.od.destination_count != self.zone_count
                or out.select_link.od.class_count != self.class_count
                or out.select_link.od.set_names != self.selection_context.set_names
            ):
                raise ValueError("selected OD dimensions and ordered names must match assignment")

    def run(self, AoNOutputs out not None):
        """Replace this iteration's outputs, reusing all prepared worker storage."""
        cdef size_t worker
        cdef CppAoNOutputsView output
        cdef CppNodeBasedContext nodes
        cdef CppTurnBasedContext turns
        cdef double turn_cost_total = 0
        cdef bint node_based = isinstance(self.context, NodeBasedContext)

        self.validate_outputs(out)
        output = out.view()

        # Routing costs may have been rebound, we get a new view
        if node_based:
            nodes = (<NodeBasedContext>self.context).view()
        else:
            turns = (<TurnBasedContext>self.context).view()

        with nogil:
            for worker in range(self.worker_views.size()):
                self.worker_views[worker].reset()

            # Clear skipped rows as well. Link accumulators are replaced by the
            # reductions below, so they do not need another full reset here.
            output.skimming.reset()
            output.selected_od.reset()

            if node_based:
                assign_origins(nodes, self.inputs, self.worker_views.data(), self.cores, output)
            else:
                assign_origins(turns, self.inputs, self.worker_views.data(), self.cores, output)

            cpp_reduce_loading_outputs[double](
                self.loading_outputs.data(),
                self.loading_outputs.size(),
                output.loading,
            )
            if self.load_selected_links:
                cpp_reduce_select_link_loading_outputs[double](
                    self.selected_outputs.data(),
                    self.selected_outputs.size(),
                    output.selected_loading,
                )

            for worker in range(self.worker_views.size()):
                turn_cost_total += self.worker_views[worker].turn_cost_total

        out.turn_cost_total = turn_cost_total
        return out


cdef void assign_origin(
    const RoutingView &routing,
    const CppAoNOrigin &query,
    const CppAoNInputs &inputs,
    CppAoNWorkerView &worker,
    const CppAoNOutputsView &output,
) noexcept nogil:
    """Compose operations on one search, without consulting Python owners."""
    cdef CppSearchResults search = worker.search.read_view()
    cdef CppSelectLinkODOriginView[double] selected_od
    cdef size_t origin = query.search.origin

    if RoutingView is CppNodeBasedContext:
        cpp_dijkstra[FourAryHeap](routing, query.search, worker.search)
    else:
        cpp_turn_dijkstra[FourAryHeap](routing, query.search, worker.search)

    if inputs.skimming.field_count:
        cpp_skimming[double](
            search,
            inputs.skimming,
            worker.workspace.skimming,
            output.skimming.origin(origin),
        )

    cpp_network_loading[double](search, query.loading, worker.workspace.loading, worker.loading)

    # OD rows have one writer. Selected link accumulators remain worker-local
    # until reduction, and can reuse the ordinary loading scratch sequentially.
    if output.selected_od.set_count:
        selected_od = output.selected_od.origin(origin)

    cpp_select_link_loading[double](
        search,
        query.loading,
        inputs.selection,
        worker.workspace.select_link,
        worker.workspace.loading,
        worker.selected_loading,
        selected_od,
    )

    worker.turn_cost_total += cpp_sum_weighted_turn_costs[double](search, query.loading)


cdef void assign_origins(
    const RoutingView &routing,
    const CppAoNInputs &inputs,
    CppAoNWorkerView *workers,
    int cores,
    const CppAoNOutputsView &output,
) noexcept nogil:
    cdef Py_ssize_t index

    # MSVC requires the OpenMP loop counter generated by prange to be signed.
    # origin_count cannot exceed the NumPy-backed demand dimension (Py_ssize_t).
    for index in prange(<Py_ssize_t>inputs.origin_count, num_threads=cores, schedule="guided"):
        assign_origin(routing, inputs.origins[index], inputs, workers[threadid()], output)
